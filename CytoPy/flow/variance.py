#!/usr/bin.env/python
# -*- coding: utf-8 -*-
"""
Before we perform any detailed analysis and/or classification of our
single cell data, it is valuable to assess the inter-sample variation
that could be arising from biological differences, but also technical
variation introduced by batch effects. This module contains multiple functions
for visualising univariate and multivatiate differences between
FileGroups in the same experiment. Additionally we have the SimilarityMatrix
class, that generates a heatmap of pairwise statistical distance's, allow
us to group similar FileGroups.


Copyright 2020 Ross Burton

Permission is hereby granted, free of charge, to any person
obtaining a copy of this software and associated documentation
files (the "Software"), to deal in the Software without restriction,
including without limitation the rights to use, copy, modify,
merge, publish, distribute, sublicense, and/or sell copies of the
Software, and to permit persons to whom the Software is furnished
to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included
in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""

from ..data.experiment import Experiment, FileGroup
from ..feedback import progress_bar, vprint
from ..flow import transform as transform_module
from .dim_reduction import dimensionality_reduction
from .sampling import density_dependent_downsampling, faithful_downsampling, uniform_downsampling
from .transform import apply_transform, Transformer
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KernelDensity
from scipy.spatial.distance import jensenshannon as jsd
from scipy.stats import entropy as kl
from scipy.cluster import hierarchy
from scipy.spatial import distance
from collections import defaultdict
from KDEpy import FFTKDE
from warnings import warn
import matplotlib.pyplot as plt
from matplotlib import cm
import seaborn as sns
import pandas as pd
import numpy as np
import harmonypy
import logging
import math

np.random.seed(42)

COLOURS = list(cm.get_cmap("tab20").colors) + list(cm.get_cmap("tab20b").colors) + list(cm.get_cmap("tab20c").colors)

__author__ = "Ross Burton"
__copyright__ = "Copyright 2020, CytoPy"
__credits__ = ["Ross Burton", "Simone Cuff", "Andreas Artemiou", "Matthias Eberl"]
__license__ = "MIT"
__version__ = "1.0.0"
__maintainer__ = "Ross Burton"
__email__ = "burtonrj@cardiff.ac.uk"
__status__ = "Production"


def load_and_sample(experiment: Experiment,
                    population: str,
                    sample_size: int or float,
                    sample_ids: list or None = None,
                    sampling_method: str or None = "uniform",
                    transform: str or None = "logicle",
                    features: list or None = None,
                    transform_kwargs: dict or None = None,
                    **kwargs) -> pd.DataFrame and Transformer:
    """
    Load sample data from experiment and return a Pandas DataFrame. Individual samples
    identified by "sample_id" column

    Parameters
    ----------
    experiment: Experiment
    sample_ids: list
    sample_size: int or float (optional)
        Total number of events to sample from each file
    sampling_method: str
    transform: str (optional)
    features: list
    transform_kwargs: dict (optional)
    population: str
    kwargs:
        Additional keyword arguments for sampling method

    Returns
    -------
    Pandas.DataFrame and Transformer
    """
    transform_kwargs = transform_kwargs or {}
    sample_ids = sample_ids or experiment.list_samples()
    files = [experiment.get_sample(s) for s in sample_ids]
    data = list()
    for f in progress_bar(files, verbose=True):
        df = _sample_filegroup(filegroup=f,
                               sample_size=sample_size,
                               sampling_method=sampling_method,
                               population=population,
                               **kwargs)
        df["sample_id"] = f.primary_id
        data.append(df)
    data = pd.concat(data)
    if transform is not None:
        assert features is not None, "Must provide features for transform"
        data, transformer = apply_transform(data=data, features=features, method=transform,
                                            return_transformer=True, **transform_kwargs)
        return data, transformer
    return data, None


def bw_optimisation(data: pd.DataFrame,
                    features: list,
                    kernel: str = "gaussian",
                    bandwidth: tuple = (0.01, 0.1, 10),
                    cv: int = 10,
                    verbose: int = 0) -> float:
    """
    Using GridSearchCV and the Scikit-Learn implementation of KDE, find the optimal
    bandwidth for the given data using grid search cross-validation

    Parameters
    ----------
    data: pd.DataFrame
    features: features
    kernel: str (default="gaussian")
    bandwidth: tuple (default=(0.01, 0.1, 20))
        Linear search space for bandwidth (min, max, increments)
    cv: int (default=10)
        Number of k-folds
    verbose: int (default=0)

    Returns
    -------
    float
    """
    bandwidth = np.linspace(*bandwidth)
    kde = KernelDensity(kernel=kernel)
    grid = GridSearchCV(estimator=kde,
                        param_grid={"bandwidth": bandwidth},
                        cv=cv,
                        n_jobs=-1,
                        verbose=verbose)
    grid.fit(data[features])
    return grid.best_params_.get("bandwidth")


def calculate_ref_sample(data: pd.DataFrame,
                         features: list or None = None,
                         verbose: bool = True) -> str:
    """

    This is performed as described in Li et al paper (https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5860171/) on
    DeepCyTOF: for every 2 samples i, j compute the euclidean norm of the difference between their covariance matrics
    and then select the sample with the smallest average distance to all other samples.

    This is an optimised version of supervised.ref.calculate_red_sample that leverages the multi-processing library
    to speed up operations

    Parameters
    ----------
    data: dict
    features: list, optional
    verbose: bool, (default=True)
        Feedback

    Returns
    -------
    str
        Sample ID of reference sample
    """
    feedback = vprint(verbose)
    feedback('Calculate covariance matrix for each sample...')
    # Calculate covar for each
    data = data.dropna(axis=1, how="any")
    features = features or list(data.columns)
    covar = {k: np.cov(v[features], rowvar=False) for k, v in data.groupby(by="sample_id")}
    feedback('Search for sample with smallest average euclidean distance to all other samples...')
    # Make comparisons
    sample_ids = list(covar.keys())
    n = len(sample_ids)
    norms = np.zeros(shape=[n, n])
    ref_ind = None
    for i, sample_i in enumerate(sample_ids):
        for j, sample_j in enumerate(sample_ids):
            cov_diff = covar.get(sample_i) - covar.get(sample_j)
            norms[i, j] = np.linalg.norm(cov_diff, ord='fro')
            norms[j, i] = norms[i, j]
            avg = np.mean(norms, axis=1)
            ref_ind = np.argmin(avg)
    return sample_ids[int(ref_ind)]


def _sample_filegroup(filegroup: FileGroup,
                      population: str,
                      sample_size: int or float = 5000,
                      sampling_method: str or None = None,
                      **kwargs) -> pd.DataFrame:
    """
    Given a FileGroup and the name of the desired population, load the
    population with transformations applied and downsample if necessary.

    Parameters
    ----------
    filegroup: FileGroup
    population: str
    sample_size: int or float (optional)
    sampling_method: str (optional)
    kwargs:
        Down-sampling keyword arguments

    Returns
    -------
    Pandas.DataFrame
    """
    data = filegroup.load_population_df(population=population,
                                        transform=None)
    if sampling_method == "uniform":
        return uniform_downsampling(data=data, sample_size=sample_size)
    if sampling_method == "density":
        return density_dependent_downsampling(data=data, sample_size=sample_size, **kwargs)
    if sampling_method == "faithful":
        return faithful_downsampling(data=data, **kwargs)
    return data


def marker_variance(data: pd.DataFrame,
                    reference: str,
                    comparison_samples: list or None = None,
                    markers: list or None = None,
                    figsize: tuple = (10, 10),
                    xlim: tuple or None = None,
                    verbose: bool = True,
                    kernel: str = "gaussian",
                    kde_bw: str or float = "silverman",
                    **kwargs):
    """
    Compare the kernel density estimates for each marker in the associated experiment for the given
    comparison samples. The estimated distributions of the comparison samples will be plotted against
    the reference sample.

    Parameters
    ----------
    data: Pandas.DataFrame
        DataFrame as generated from load_and_sample
    reference: str
        Reference sample to plot in the background
    comparison_samples: list
        List of valid sample IDs for the associated experiment
    markers: list (optional)
        List of markers to include (defaults to all available markers)
    figsize: figsize (default=(10,10))
    xlim: tuple (optional)
        x-axis limits
    verbose: bool (default=True)
    kernel: str (default="gaussian")
    kde_bw: str or float (default="silverman")
    kwargs: dict
        Additional kwargs passed to Matplotlib.Axes.plot call

    Returns
    -------
    matplotlib.Figure
    """
    assert reference in data.sample_id.unique(), "Reference absent from given data"
    comparison_samples = comparison_samples or [x for x in data.sample_id.unique() if x != reference]
    fig = plt.figure(figsize=figsize)
    markers = markers or data.get(reference).columns.tolist()
    i = 0
    nrows = math.ceil(len(markers) / 3)
    fig.suptitle(f'Per-channel KDE, Reference: {reference}', y=1.02)
    for marker in progress_bar(markers, verbose=verbose):
        i += 1
        ax = fig.add_subplot(nrows, 3, i)
        x, y = (FFTKDE(kernel=kernel,
                       bw=kde_bw)
                .fit(data[data.sample_id == reference][marker].values)
                .evaluate())
        ax.plot(x, y, color="b", **kwargs)
        ax.fill_between(x, 0, y, facecolor="b", alpha=0.2)
        ax.set_title(marker)
        if xlim:
            ax.set_xlim(xlim)
        for comparison_sample_id in comparison_samples:
            df = data[data.sample_id == comparison_sample_id]
            if marker not in df.columns:
                warn(f"{marker} missing from {comparison_sample_id}, this marker will be ignored")
            else:
                x, y = (FFTKDE(kernel=kernel,
                               bw=kde_bw)
                        .fit(df[marker].values)
                        .evaluate())
                ax.plot(x, y, color="r", **kwargs)
                if ax.get_legend() is not None:
                    ax.get_legend().remove()
        ax.set(aspect="auto")
    fig.tight_layout()
    return fig


def dim_reduction_grid(data: pd.DataFrame,
                       reference: str,
                       features: list,
                       comparison_samples: list or None = None,
                       figsize: tuple = (10, 10),
                       method: str = 'PCA',
                       kde: bool = False,
                       verbose: bool = True,
                       dim_reduction_kwargs: dict or None = None):
    """
    Generate a grid of embeddings using a valid dimensionality reduction technique, in each plot a reference sample
    is shown in blue and a comparison sample in red. The reference sample is conserved across all plots.

    Parameters
    ------------
    data: Pandas.DataFrame
        DataFrame as generated from load_and_sample
    reference: str
        Reference sample to plot in the background
    comparison_samples: list
        List of samples to compare to reference (blue)
    features: list
        List of features to use for dimensionality reduction
    figsize: tuple, (default=(10,10))
        Size of figure
    method: str, (default='PCA')
        Method to use for dimensionality reduction (see flow.dim_reduction)
    dim_reduction_kwargs: dict
        Additional keyword arguments passed to CytoPy.dim_reduction.dimensionality_reduction
    kde: bool, (default=False)
        If True, overlay with two-dimensional PDF estimated by KDE
    verbose: bool (default=True)

    Returns
    -------
    None
        Plot printed to stdout
    """
    assert reference in data.sample_id.unique(), "Reference absent from given data"
    data = data.dropna(axis=1, how="any")
    comparison_samples = comparison_samples or [x for x in data.sample_id.unique() if x != reference]
    dim_reduction_kwargs = dim_reduction_kwargs or {}
    fig = plt.figure(figsize=figsize)
    nrows = math.ceil(len(comparison_samples) / 3)
    reference_df = data[data.sample_id == reference].copy()
    assert all([f in reference_df.columns for f in features]), \
        f'Invalid features; valid are: {reference_df.columns}'
    reference_df, reducer = dimensionality_reduction(reference_df.reset_index(),
                                                     features=features,
                                                     method=method,
                                                     n_components=2,
                                                     return_reducer=True,
                                                     **dim_reduction_kwargs)
    i = 0
    fig.suptitle(f'{method}, Reference: {reference}', y=1.05)
    for sample_id in progress_bar(comparison_samples, verbose=verbose):
        i += 1
        ax = fig.add_subplot(nrows, 3, i)
        embeddings = reducer.transform(data[data.sample_id == sample_id].reset_index()[features])
        x = f'{method}1'
        y = f'{method}2'
        ax.scatter(reference_df[x], reference_df[y], c='blue', s=4, alpha=0.2)
        if kde:
            sns.kdeplot(reference_df[x], reference_df[y], c='blue', n_levels=100, ax=ax, shade=False)
        ax.scatter(embeddings[:, 0], embeddings[:, 1], c='red', s=4, alpha=0.1)
        if kde:
            sns.kdeplot(embeddings[:, 0], embeddings[:, 1], c='red',
                        n_levels=100, ax=ax, shade=False)
        ax.set_title(sample_id)
        ax.set_yticklabels([])
        ax.set_xticklabels([])
        ax.set(aspect='auto')
    fig.tight_layout()
    return fig


class SimilarityMatrix:
    """
    Class for assessing the degree of variation observed in a single experiment. This can be
    useful for determining the influence of batch effects in your cytometry experiment.

    Attributes
    -----------
    data: Pandas.DataFrame
        DataFrame as generated from load_and_sample
    reference: str
        Reference sample; this will be the dataframe used to establish the embedded space
        upon which data is projected to reduce dimensionality
    verbose: bool (default=True)
        Whether to provide feedback
    njobs: int (default=-1)
        Number of parallel jobs to run
    kde_kernel: str (default="gaussian")
        Kernel to use for KDE, for options see KDEpy.FFTKDE
    kde_bw: str or float (default="ISJ"
        Bandwidth/bandwidth estimation method to use for KDE. See KDEpy for options.
        Defaults too improved Sheather Jones (ISJ) algorithm, which does not assume normality
        and is robust to multimodal distributions. If you need to speed up results, change this to
        'silvermans' which is less accurate but less computationally intensive.
    kde_norm: int (default=2)
        p-norm for high-dimensional KDE calculation
    """

    def __init__(self,
                 data: pd.DataFrame,
                 reference: str,
                 verbose: bool = True,
                 kde_kernel: str = "gaussian",
                 kde_bw: str or float = "cv",
                 kde_norm: int = 2):
        assert reference in data.sample_id.unique(), "Invalid reference, not present in given data"
        self.verbose = verbose
        self.print = vprint(verbose)
        self.kde_cache = dict()
        self.kde_kernel = kde_kernel
        self.kde_norm = kde_norm
        self._kde_bw = "cv"
        self.reference = reference
        self.data = data.dropna(axis=1)
        self.kde_bw = kde_bw

    @property
    def kde_bw(self):
        return self._kde_bw

    @kde_bw.setter
    def kde_bw(self, x: str or float):
        if isinstance(x, str):
            assert x == "cv", f"kde_bw should be a float or have value 'cv'"
        else:
            assert isinstance(x, float), "kde_bw should be a float or 'cv'"
        self._kde_bw = x

    def clean_cache(self):
        """
        Clears the KDE cached results

        Returns
        -------
        None
        """
        self.kde_cache = {}

    def _estimate_pdf(self,
                      sample_id: str,
                      features: list,
                      df: pd.DataFrame,
                      **kwargs) -> None:
        """
        Given a sample ID and its events dataframe, estimate the PDF by KDE with the option
        to perform dimensionality reduction first. Resulting PDF is saved to kde_cache.

        Parameters
        ----------
        sample_id: str
        df: Pandas.DataFrame
        features: list

        Returns
        -------
        None
        """
        bw = self.kde_bw
        if bw == "cv":
            bw = bw_optimisation(data=df, features=features, **kwargs)
        df = df[features].copy().select_dtypes(include=['number'])
        kde = FFTKDE(kernel=self.kde_kernel, bw=bw, norm=self.kde_norm)
        self.kde_cache[sample_id] = np.exp(kde.fit(df.values).evaluate()[1])

    def _calc_divergence(self,
                         target_id: str,
                         distance_metric: str or callable = 'jsd') -> list:
        """
        Given the name of a sample contained within self.data, loop over kde_cache
        and calculate the statistical distance between this sample and all other
        samples contained within self.data

        Parameters
        ----------
        target_id: str
            Should be a valid sample ID for the associated experiment
        distance_metric: callable or str (default='jsd')
            Either a callable function to calculate the statistical distance or a string value; options are:
                * jsd: Jensson-shannon distance
                * kl:Kullback-Leibler divergence (entropy)

        Returns
        -------
        list
            List of statistical distances, with results given as a list of nested tuples of type: (sample ID, distance).
        """
        # Assign distance metric func
        metrics = {"kl": kl,
                   "jsd": jsd}
        if isinstance(distance_metric, str):
            assert distance_metric in ['jsd', 'kl'], \
                'Invalid divergence metric must be one of either jsd, kl, or a callable function]'
            distance_metric = metrics.get(distance_metric)
        return [(name, distance_metric(self.kde_cache.get(target_id), q))
                for name, q in self.kde_cache.items()]

    def _generate_reducer(self,
                          features: list,
                          n_components: int,
                          dim_reduction_method: str,
                          **kwargs):
        """
        Generate the dimension reduction object for producing lower dimension embeddings
        using the reference sample as the source for generating the low dimension space.

        Parameters
        ----------
        features: list
        n_components: int
        dim_reduction_method: str
        kwargs
            Additional keyword arguments passed to
            CytoPy.flow.dim_reduction.dimensionality_reduction

        Returns
        -------
        object
            Reducer
        """
        reference = self.data[self.data.sample_id == self.reference].copy()
        ref_embeddings, reducer = dimensionality_reduction(data=reference,
                                                           method=dim_reduction_method,
                                                           features=features,
                                                           return_reducer=True,
                                                           return_embeddings_only=True,
                                                           n_components=n_components,
                                                           **kwargs)
        return reducer

    def _dim_reduction(self,
                       reducer: object,
                       n_components: int,
                       features: list) -> dict:
        """
        Loop over each sample in self.data and generate low dimension embeddings

        Parameters
        ----------
        reducer: object
        n_components: int
        features: list

        Returns
        -------
        dict
            Dictionary of embeddings
        """
        embeddings = list()
        for sample_id, df in progress_bar(self.data.groupby(by="sample_id"), verbose=self.verbose):
            embeddings.append(reducer.transform(df[features].values))
        col_names = [f"embedding{i + 1}" for i in range(n_components)]
        embeddings = {k: pd.DataFrame(em, columns=col_names)
                      for k, em in zip(self.data.sample_id.unique(), embeddings)}
        return embeddings

    def _pairwise_stat_dist(self,
                            distance_metric: str) -> pd.DataFrame:
        """
        Looping over every sample in self.data, calculate the pairwise statistical
        distance from each sample PDF p, in relation to every other sample PDF q.
        Returns a symmetrical matrix of pairwise distances.

        Parameters
        ----------
        distance_metric: str

        Returns
        -------
        Pandas.DataFrame
        """
        distance_df = pd.DataFrame()
        for s in progress_bar(self.data.sample_id.unique(), verbose=self.verbose):
            distances = self._calc_divergence(target_id=s, distance_metric=distance_metric)
            name_distances = defaultdict(list)
            for n, d in distances:
                name_distances[n].append(d)
            name_distances = pd.DataFrame(name_distances)
            name_distances["sample_id"] = s
            distance_df = pd.concat([distance_df, name_distances])
        return distance_df

    def matrix(self,
               distance_metric: str or callable = 'jsd',
               features: None or list = None,
               dim_reduction_method: str = "PCA",
               dim_reduction_kwargs: dict or None = None,
               bw_optimisaiton_kwargs: dict or None = None) -> pd.DataFrame:
        """
        Generate a Pandas DataFrame containing a symmetrical matrix of
        pairwise statistical distances for every sample in self.data

        Parameters
        ----------
        distance_metric: callable or str (default='jsd')
            Either a callable function to calculate the statistical distance or a string value; options are:
                * jsd: Jensson-shannon distance
                * kl:Kullback-Leibler divergence (entropy)
        features: list (optional)
            List of markers to use in analysis. If not given, will use all available markers.
        dim_reduction_method: str (default="PCA")
            Dimension reduction method, see CytoPy.flow.dim_reduction. Set to None to not reduce first
        dim_reduction_kwargs: dict
            Keyword arguments for dimension reduction method, see CytoPy.flow.dim_reduction
        bw_optimisaiton_kwargs: dict
            Additional keyword arguments passed to CytoPy.flow.variance.bw_optimisation call

        Returns
        -------
        Pandas.DataFrame
        """
        # Set defaults
        dim_reduction_kwargs = dim_reduction_kwargs or {}
        bw_optimisaiton_kwargs = bw_optimisaiton_kwargs or {}
        if distance_metric == "kl":
            warn("Kullback-Leiber Divergence chosen as statistical distance metric, KL divergence "
                 "is an asymmetrical function and as such it is not advised to use this metric for the "
                 "similarity matrix'")

        features = features or self.data.columns.tolist()
        # Create the reducer
        n_components = dim_reduction_kwargs.get("n_components", 2)
        reducer = self._generate_reducer(features=features,
                                         n_components=n_components,
                                         dim_reduction_method=dim_reduction_method,
                                         **dim_reduction_kwargs)
        # Perform dim reduction
        self.print("...performing dimensionality reduction")
        embeddings = self._dim_reduction(reducer=reducer,
                                         features=features,
                                         n_components=n_components)
        # Estimate PDFs
        self.print("...estimate PDFs of embeddings")
        features = [f"embedding{i + 1}" for i in range(n_components)]
        for sample_id, df in progress_bar(embeddings.items()):
            self._estimate_pdf(sample_id=sample_id,
                               df=df,
                               features=features,
                               **bw_optimisaiton_kwargs)

        # Generate distance matrix
        self.print("...calculating pairwise statistical distances")
        return self._pairwise_stat_dist(distance_metric=distance_metric)

    def __call__(self,
                 distance_df: pd.DataFrame or None = None,
                 figsize: tuple = (12, 12),
                 distance_metric: str or callable = 'jsd',
                 clustering_method: str = 'average',
                 features: None or list = None,
                 dim_reduction_method: str = "PCA",
                 dim_reduction_kwargs: dict or None = None,
                 cluster_plot_kwargs: dict or None = None,
                 bw_optimisaiton_kwargs: dict or None = None):
        """
        Generate a heatmap of pairwise statistical distances with the axis clustered using
        agglomerative clustering.

        Parameters
        ----------
        figsize: tuple (default=(12,12))
            Figure size
        distance_metric: callable or str (default='jsd')
            Either a callable function to calculate the statistical distance or a string value; options are:
                * jsd: Jensson-shannon distance
                * kl:Kullback-Leibler divergence (entropy)
        clustering_method: str
            Method passed to call to scipy.cluster.heirachy
        features: list (optional)
            List of markers to use in analysis. If not given, will use all available markers.
        dim_reduction_method: str (default="PCA")
            Dimension reduction method, see CytoPy.flow.dim_reduction. Set to None to not reduce first
        dim_reduction_kwargs: dict
            Keyword arguments for dimension reduction method, see CytoPy.flow.dim_reduction
        cluster_plot_kwargs: dict
            Additional keyword arguments passed to Seaborn.clustermap call
        bw_optimisaiton_kwargs: dict
            Additional keyword arguments passed to CytoPy.flow.variance.bw_optimisation call

        Returns
        -------
        Array, Array, ClusterGrid
            Linkage array, ordered array of sample IDs and seaborn ClusterGrid object
        """
        # Set defaults
        dim_reduction_kwargs = dim_reduction_kwargs or {}
        cluster_plot_kwargs = cluster_plot_kwargs or {}
        bw_optimisaiton_kwargs = bw_optimisaiton_kwargs or {}
        if distance_df is None:
            distance_df = self.matrix(distance_metric=distance_metric,
                                      features=features,
                                      dim_reduction_method=dim_reduction_method,
                                      dim_reduction_kwargs=dim_reduction_kwargs,
                                      bw_optimisaiton_kwargs=bw_optimisaiton_kwargs)
        # Perform hierarchical clustering
        r = distance_df.drop('sample_id', axis=1).values
        c = distance_df.drop('sample_id', axis=1).T.values
        row_linkage = hierarchy.linkage(distance.pdist(r), method=clustering_method)
        col_linkage = hierarchy.linkage(distance.pdist(c), method=clustering_method)

        if distance_metric == 'jsd':
            center = 0.5
        else:
            center = 0
        g = sns.clustermap(distance_df.set_index('sample_id'),
                           row_linkage=row_linkage,
                           col_linkage=col_linkage,
                           method=clustering_method,
                           center=center,
                           cmap="vlag",
                           figsize=figsize,
                           **cluster_plot_kwargs)
        ax = g.ax_heatmap
        ax.set_xlabel('')
        ax.set_ylabel('')
        return row_linkage, distance_df.sample_id.values, g


def generate_groups(linkage_matrix: np.array,
                    sample_ids: list or np.array,
                    n_groups: int):
    """
    Given the output of SimilarityMatrix (that is the linkage matrix and ordered list of sample
    IDs) and a desired number of groups, return a Pandas DataFrame of sample IDs and assigned group ID, generated by
    cutting the linkage matrix in such a way that the desired number of groups are generated.
    Parameters
    ----------
    linkage_matrix: np.array
        Linkage matrix generated from EvaluateBatchEffects.similarity_matrix (using SciPy.cluster.hierarchy.linkage)
    sample_ids: list or np.array
        Ordered list of sample IDs generated from EvaluateBatchEffects.similarity_matrix
    n_groups: int
        Desired number of groups
    Returns
    -------
    Pandas.DataFrame
    """
    groups = pd.DataFrame({'sample_id': sample_ids,
                           'group': list(map(lambda x: x + 1,
                                             hierarchy.cut_tree(linkage_matrix, n_groups).flatten()))})
    groups = groups.sort_values('group')
    return groups


class Harmony:
    """
    Perform batch-effect correction using the Harmony algorithm, first described by Korsunsky et al [1] and
    implemented in Python by Kamil Slowikowski [2]. The user should provide an ordered dictionary of sampled
    data as generated by CytoPy.flow.variance.load_and_sample.

    L2 normalisation is recommend prior to running   Harmony, this can be performed by calling 'normalisation'.
    After running the harmony algorithm, the resulting corrected samples can be saved to an Experiment by using the
    'save' method. If L2 normalisation has been performed then this will be reversed prior to saving, however,
    if a transform was applied to the data prior to initialising the Harmony object (logicle transform is
    recommended for Flow Cytometry data, for example), the data will be saved WITH THE TRANSFORM APPLIED. This
    means future actions should take this into account i.e. do not further transform the data when applying
    gates, clustering, or supervised classification algorithms

    [1] Korsunsky, I., Millard, N., Fan, J. et al. Fast, sensitive and accurate integration of single-cell data
    with Harmony. Nat Methods 16, 1289–1296 (2019). https://doi.org/10.1038/s41592-019-0619-0
    [2] https://github.com/slowkow/harmonypy
    """

    def __init__(self,
                 experiment: Experiment,
                 population: str,
                 features: list,
                 sample_size: int or float,
                 sample_ids: list or None = None,
                 sampling_method: str or None = "uniform",
                 transform: str = "logicle",
                 transform_kwargs: dict or None = None,
                 sample_kwargs: dict or None = None,
                 logging_level=None):
        """
        Parameters
        ----------
        data: Pandas.DataFrame
            Can be generated using the CytoPy.flow.variance.load_and_sample function
        """
        sample_kwargs = sample_kwargs or {}
        self.data, self.transformer = load_and_sample(experiment=experiment,
                                                      population=population,
                                                      sample_size=sample_size,
                                                      sample_ids=sample_ids,
                                                      sampling_method=sampling_method,
                                                      transform=transform,
                                                      features=features,
                                                      transform_kwargs=transform_kwargs,
                                                      **sample_kwargs)
        self.data = self.data.dropna(axis=1, how="any")
        self.features = [x for x in features if x in self.data.columns]
        self.meta = self.data[["sample_id"]]
        self.harmony = None
        self._norms = None
        self._logging_level = logging_level
        if logging_level:
            logging.getLogger("harmonypy").setLevel(logging_level)

    @property
    def logging_level(self):
        return self._logging_level

    @logging_level.setter
    def logging_level(self, value):
        logging.getLogger("harmonypy").setLevel(value)
        self._logging_level = value

    def normalisation(self):
        """
        Perform L2 normalisation of columns (norms are stored internally in _norms parameter)

        Returns
        -------
        None
        """
        normaliser = transform_module.Normalise()
        self.data = normaliser(self.data, self.features)
        self._norms = normaliser._norms

    def run(self, **kwargs):
        """
        Run the harmony algorithm (see https://github.com/slowkow/harmonypy for details). Resulting object
        is stored in 'harmony' attribute

        Parameters
        ----------
        kwargs:
            Additional keyword arguments passed to harmonypy.run_harmony

        Returns
        -------
        Harmony
        """
        data = self.data[self.features].astype(float)
        self.harmony = harmonypy.run_harmony(data_mat=data.values,
                                             meta_data=self.meta,
                                             vars_use="sample_id",
                                             **kwargs)
        return self

    def hyperparameter_search(self,
                              param_grid: list,
                              **kwargs):
        kwargs = kwargs or {}
        kwargs["ci"] = kwargs.get("ci", "sd")
        kwargs["estimator"] = kwargs.get("estimator", np.median)
        kwargs["capsize"] = kwargs.get("capsize", .2)
        lisi_values = dict()
        for params in progress_bar(param_grid):
            lisi_values[str(params)] = self.run(**params).batch_lisi()
        lisi_values = (pd.DataFrame({k: v.reshape(-1) for k, v in lisi_values.items()})
                       .melt(var_name="Params", value_name="LISI"))
        return sns.pointplot(data=lisi_values, y="LISI", x="Params", **kwargs)

    def batch_lisi(self):
        return harmonypy.lisi.compute_lisi(self.batch_corrected()[self.features].values,
                                           metadata=self.meta,
                                           label_colnames=["sample_id"])

    def batch_lisi_distribution(self, **kwargs):
        before = harmonypy.lisi.compute_lisi(self.data[self.features].values,
                                             metadata=self.meta,
                                             label_colnames=["sample_id"])
        data = pd.DataFrame({"Before": before.reshape(-1),
                             "After": self.batch_lisi().reshape(-1)})
        data = data.melt(var_name="Data", value_name="LISI")
        return sns.histplot(data=data, x="LISI", hue="Data", **kwargs)

    def batch_corrected(self):
        """
        Generates a Pandas DataFrame of batch corrected values. If L2 normalisation was performed prior to
        this, it is reversed. Additional column 'batch_id' identifies rows.

        Returns
        -------
        Pandas.DataFrame
        """
        assert self.harmony is not None, "Call 'run' first"
        corrected = pd.DataFrame(self.harmony.Z_corr.T, columns=self.features)
        corrected["sample_id"] = self.meta.sample_id.values
        return corrected

    def save(self,
             experiment: Experiment,
             prefix: str = "Corrected_",
             subject_mappings: dict or None = None):
        """
        Saved the batch corrected data to an Experiment with each biological specimen (batch) saved
        to an individual FileGroup

        Parameters
        ----------
        experiment: Experiment
        prefix: str (default="Corrected_")
            Prefix added to sample ID when creating new FileGroup
        subject_mappings: dict, optional
            If provided, key values should match batch_id and value the Subject to associate the new
            FileGroup to

        Returns
        -------
        None
        """
        assert self.harmony is not None, "Call 'run' first"
        subject_mappings = subject_mappings or {}
        for sample_id, df in progress_bar(self.batch_corrected().groupby("sample_id"),
                                          verbose=True,
                                          total=self.meta.sample_id.nunique()):
            if self.transformer is not None:
                df = self.transformer.inverse_scale(data=df, features=self.features)
            experiment.add_dataframes(sample_id=str(prefix) + str(sample_id),
                                      primary_data=df[self.features],
                                      mappings=[{"channel": x, "marker": x} for x in self.features],
                                      verbose=False,
                                      subject_id=subject_mappings.get(sample_id, None))


def create_experiment(project,
                      features: list,
                      experiment_name: str,
                      data_directory: str) -> Experiment:
    markers = [{"name": x, "regex": f"^{x}$", "case": 0, "permutations": ""}
               for x in features]
    channels = [{"name": x, "regex": f"^{x}$", "case": 0, "permutations": ""}
                for x in features]
    mappings = [(x, x) for x in features]
    panel_definition = {"markers": markers, "channels": channels, "mappings": mappings}
    return project.add_experiment(experiment_id=experiment_name,
                                  data_directory=data_directory,
                                  panel_definition=panel_definition)
