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
from .dim_reduction import dimensionality_reduction
from .sampling import density_dependent_downsampling, faithful_downsampling, uniform_downsampling
from .transforms import scaler
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KernelDensity
from scipy.spatial.distance import jensenshannon as jsd
from scipy.stats import entropy as kl
from scipy.cluster import hierarchy
from scipy.spatial import distance
from collections import defaultdict, OrderedDict
from KDEpy import FFTKDE
from warnings import warn
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import math
np.random.seed(42)

__author__ = "Ross Burton"
__copyright__ = "Copyright 2020, CytoPy"
__credits__ = ["Ross Burton", "Simone Cuff", "Andreas Artemiou", "Matthias Eberl"]
__license__ = "MIT"
__version__ = "1.0.0"
__maintainer__ = "Ross Burton"
__email__ = "burtonrj@cardiff.ac.uk"
__status__ = "Production"


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


def _common_features(data: OrderedDict) -> list:
    """
    Given an OrderedDict (as generated from load_and_sample, return list of column
    names common to all DataFrames.

    Parameters
    ----------
    data: OrderedDict

    Returns
    -------
    list
    """
    features = [df.columns.tolist() for df in data.values()]
    return list(set(features[0]).intersection(*features))


def calculate_ref_sample(data: OrderedDict,
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
    features = _common_features(data=data)
    covar = {k: np.cov(v[features], rowvar=False) for k, v in data.items()}
    feedback('Search for sample with smallest average euclidean distance to all other samples...')
    # Make comparisons
    sample_ids = list(data.keys())
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


def scale_data(data: OrderedDict,
               method: str or None = "standard",
               **kwargs) -> OrderedDict:
    """
    Given a dictionary of events data as generated by `load_and_sample`,
    scale the data using a valid scale method (see CytoPy.flow.transforms.scaler)

    Parameters
    ----------
    data: dict
    method: str (default="standard")
    kwargs: dict
        Keywords passed to scaler

    Returns
    -------
    dict
    """
    scaled = OrderedDict()
    for k, df in data.items():
        scaled[k] = pd.DataFrame(scaler(data=df.values,
                                        scale_method=method,
                                        return_scaler=False,
                                        **kwargs),
                                 columns=df.columns)
    return scaled


def load_and_sample(experiment: Experiment,
                    population: str,
                    sample_size: int or float,
                    sample_ids: list or None = None,
                    sampling_method: str or None = "uniform",
                    transform: str or None = "logicle",
                    **kwargs) -> OrderedDict:
    """
    Load sample data from experiment and return as a dictionary of Pandas DataFrames.

    Parameters
    ----------
    experiment: Experiment
    sample_ids: list
    sample_size: int or float (optional)
        Total number of events to sample from each file
    sampling_method: str
    transform: str (optional)
    population: str
    kwargs:
        Additional keyword arguments for sampling method

    Returns
    -------
    OrderedDict
    """
    sample_ids = sample_ids or experiment.list_samples()
    files = [experiment.get_sample(s) for s in sample_ids]
    data = OrderedDict()
    for f in progress_bar(files, verbose=True):
        data[f.primary_id] = _sample_filegroup(filegroup=f,
                                               sample_size=sample_size,
                                               sampling_method=sampling_method,
                                               transform=transform,
                                               population=population,
                                               **kwargs)
    return data


def _sample_filegroup(filegroup: FileGroup,
                      population: str,
                      transform: str or None,
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
    transform: str (optional)
    sample_size: int or float (optional)
    sampling_method: str (optional)

    Returns
    -------
    Pandas.DataFrame
    """
    data = filegroup.load_population_df(population=population,
                                        transform=transform)
    if sampling_method == "uniform":
        return uniform_downsampling(data=data, sample_size=sample_size)
    if sampling_method == "density":
        return density_dependent_downsampling(data=data, sample_size=sample_size, **kwargs)
    if sampling_method == "faithful":
        return faithful_downsampling(data=data, **kwargs)
    return data


def marker_variance(data: OrderedDict,
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
    data: OrderedDict
        Ordered dictionary as generated from load_and_sample
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
    assert reference in data.keys(), "Reference absent from given data"
    comparison_samples = comparison_samples or [x for x in data.keys() if x != reference]
    fig = plt.figure(figsize=figsize)
    markers = markers or data.get(reference).columns.tolist()
    i = 0
    nrows = math.ceil(len(markers) / 3)
    fig.suptitle(f'Per-channel KDE, Reference: {reference}', y=1.05)
    for marker in progress_bar(markers, verbose=verbose):
        i += 1
        ax = fig.add_subplot(nrows, 3, i)
        x, y = (FFTKDE(kernel=kernel,
                       bw=kde_bw)
                .fit(data.get(reference)[marker].values)
                .evaluate())
        ax.plot(x, y, color="b", **kwargs)
        ax.fill_between(x, 0, y, facecolor="b", alpha=0.2)
        ax.set_title(f'Total variance in {marker}')
        if xlim:
            ax.set_xlim(xlim)
        for comparison_sample_id in comparison_samples:
            if comparison_sample_id not in data.keys():
                warn(f"{comparison_sample_id} is not a valid ID")
                continue
            if marker not in data.get(comparison_sample_id).columns:
                warn(f"{marker} missing from {comparison_sample_id}, this marker will be ignored")
            else:
                x, y = (FFTKDE(kernel=kernel,
                               bw=kde_bw)
                        .fit(data.get(comparison_sample_id)[marker].values)
                        .evaluate())
                ax.plot(x, y, color="r", **kwargs)
                if ax.get_legend() is not None:
                    ax.get_legend().remove()
        ax.set(aspect="auto")
    fig.tight_layout()
    return fig


def dim_reduction_grid(data: OrderedDict,
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
    data: OrderedDict
        Ordered dictionary as generated from load_and_sample
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
    assert reference in data.keys(), "Reference absent from given data"
    comparison_samples = comparison_samples or [x for x in data.keys() if x != reference]
    dim_reduction_kwargs = dim_reduction_kwargs or {}
    fig = plt.figure(figsize=figsize)
    nrows = math.ceil(len(comparison_samples) / 3)
    reference_df = data.get(reference).copy()
    reference_df['label'] = 'Target'
    assert all([f in reference_df.columns for f in features]), \
        f'Invalid features; valid are: {reference_df.columns}'
    reference_df, reducer = dimensionality_reduction(reference_df,
                                                     features=features,
                                                     method=method,
                                                     n_components=2,
                                                     return_reducer=True,
                                                     **dim_reduction_kwargs)
    i = 0
    fig.suptitle(f'{method}, Reference: {reference}', y=1.05)
    for sample_id, df in progress_bar(data.items(), verbose=verbose):
        if sample_id == reference:
            continue
        if not all([f in df.columns for f in features]):
            warn(f'Features missing from {sample_id}, skipping')
            continue
        i += 1
        df['label'] = 'Comparison'
        ax = fig.add_subplot(nrows, 3, i)
        embeddings = reducer.transform(df[features])
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
    data: OrderedDict
        Ordered dictionary as produced by load_and_sample function
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
                 data: OrderedDict,
                 reference: str,
                 verbose: bool = True,
                 kde_kernel: str = "gaussian",
                 kde_bw: str or float = "cv",
                 kde_norm: int = 2):
        assert reference in data.keys(), "Invalid reference, not present in given data"
        self.verbose = verbose
        self.print = vprint(verbose)
        self.kde_cache = dict()
        self.kde_kernel = kde_kernel
        self.kde_norm = kde_norm
        self._kde_bw = "cv"
        self.reference = reference
        self.data = data
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
        reference = self.data.get(self.reference)
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
        data = [x[features] for x in self.data.values()]
        embeddings = list()
        for df in progress_bar(data, verbose=self.verbose):
            embeddings.append(reducer.transform(df.values))
        col_names = [f"embedding{i + 1}" for i in range(n_components)]
        embeddings = {k: pd.DataFrame(em, columns=col_names)
                      for k, em in zip(self.data.keys(), embeddings)}
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
        for s in progress_bar(self.data.keys(), verbose=self.verbose):
            distances = self._calc_divergence(target_id=s,
                                              distance_metric=distance_metric)
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

        features = features or self.data.get(self.reference).columns.tolist()
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
