from ..data.experiment import Experiment
from ..feedback import progress_bar, vprint
from .dim_reduction import dimensionality_reduction
from .transforms import scaler
from scipy.spatial.distance import jensenshannon as jsd
from scipy.stats import entropy as kl
from scipy.cluster import hierarchy
from scipy.spatial import distance
from multiprocessing import Pool, cpu_count
from collections import defaultdict
from KDEpy import FFTKDE
from functools import partial
from typing import Dict
from warnings import warn
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import math


def covar_euclidean_norm(data: Dict[str, pd.DataFrame],
                         verbose: bool = True):
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
    covar = {k: np.cov(v, rowvar=False) for k, v in data.items()}
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


def scale_data(data: Dict[str, pd.DataFrame],
               method: str or None = "standard",
               **kwargs):
    """
    Given a dictionary of events data as generated by `load_and_sample`, scale the data using
    a valid scale method (see CytoPy.flow.transforms.scaler)

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
    return {k: pd.DataFrame(scaler(data=v.values,
                                   scale_method=method,
                                   return_scaler=False,
                                   **kwargs),
                            columns=v.columns) for k, v in data.items()}


def kde_wrapper(data: pd.DataFrame,
                kernel: str,
                bw: float or str):
    return FFTKDE(kernel=kernel, bw=bw).fit(data.values)()[1]


class EvaluateBatchEffects:
    """
    Class for assessing the degree of variation observed in a single experiment. This can be
    useful for determining the influence of batch effects in your cytometry experiment.

    Parameters
    -----------
    experiment: Experiment
        Experiment to investigate
    root_population: str
        Population to sample from e.g. T cells or Monocytes
    samples: list (optional)
        Samples to include in investigation (if None, uses all samples)
    transform: str (default="logicle")
        How to transform the data prior to processing
    verbose: bool (default=True)
        Whether to provide feedback
    njobs: int (default=-1)
        Number of parallel jobs to run
    kde_kernel: str (default="gaussian")
        Kernel to use for KDE, for options see KDEpy.FFTKDE
    kde_bw: str or float (default="ISJ"
        Bandwidth/bandwidth estimation method to use for KDE. See KDEpy for options.
        Defaults too mproved Sheather Jones (ISJ) algorithm, which does not assume normality
        and is robust to multimodal distributions. If you need to speed up results, change this to
        'silvermans' which is less accurate but less computaitonally intensive.
    """

    def __init__(self,
                 experiment: Experiment,
                 root_population: str,
                 samples: list or None = None,
                 reference_sample: str or None = None,
                 transform: str = 'logicle',
                 verbose: bool = True,
                 njobs: int = -1,
                 kde_kernel: str = "gaussian",
                 kde_bw: str or float = "ISJ"):
        self.experiment = experiment
        self.transform = transform
        self.root_population = root_population
        self.verbose = verbose
        self.print = vprint(verbose)
        self.kde_cache = dict()
        self.njobs = njobs
        self.kde_kernel = kde_kernel
        self.kde_bw = kde_bw
        if self.njobs < 0:
            self.njobs = cpu_count()
        if samples is None:
            self.sample_ids = experiment.list_samples()
        else:
            for x in samples:
                assert x in experiment.list_samples(), f"Invalid sample ID; {x} not found for given experiment"
            self.sample_ids = samples
        self.reference_id = reference_sample or self._calc_ref_sample()

    def clean_cache(self):
        """
        Clears the KDE cached results

        Returns
        -------
        None
        """
        self.kde_cache = {}

    def load_and_sample(self,
                        sample_n: int = 10000):
        """
        Load sample data from experiment and return as a dictionary of Pandas DataFrames.

        Parameters
        ----------
        sample_n: int
            Total number of events to sample from each file

        Returns
        -------
        dict
        """
        _load = partial(load_population,
                        experiment=self.experiment,
                        population=self.root_population,
                        sample_n=sample_n,
                        transform=self.transform,
                        indexed=True)
        with Pool(self.njobs) as pool:
            data = list(progress_bar(pool.imap(_load, self.sample_ids),
                                     verbose=self.verbose,
                                     total=len(self.sample_ids)))
        return {x[0]: x[1] for x in data}

    def _calc_ref_sample(self,
                         sample_n: int = 1000):
        """
        Estimates a valid reference sample, see CytoPy.batch_effects.covar_euclidean_norm.

        Parameters
        ----------
        sample_n: int (default=1000)

        Returns
        -------
        str
        """
        self.print("--- Calculating Reference Sample ---")
        return covar_euclidean_norm(data=self.load_and_sample(sample_n=sample_n),
                                    verbose=self.verbose)

    def marker_variance(self,
                        comparison_samples: list,
                        sample_n: int,
                        markers: list or None = None,
                        figsize: tuple = (10, 10),
                        xlim: tuple or None = None,
                        **kwargs):
        """
        Compare the kernel density estimates for each marker in the associated experiment for the given
        comparison samples. The estimated distributions of the comparison samples will be plotted against
        the reference sample.

        Parameters
        ----------
        comparison_samples: list
            List of valid sample IDs for the associated experiment
        sample_n: int
            Number of events to sample from each prior to KDE
        markers: list (optional)
            List of markers to include (defaults to all available markers)
        figsize: figsize (default=(10,10))
        xlim: tuple (optional)
            x-axis limits
        kwargs: dict
            Additional kwargs passed to Seaborn.kdeplot call

        Returns
        -------
        matplotlib.Figure
        """
        fig = plt.figure(figsize=figsize)
        data = self.load_and_sample(sample_n=sample_n)
        assert all([x in self.sample_ids for x in comparison_samples]), \
            f'One or more invalid sample IDs; valid IDs include: {self.sample_ids}'
        if markers is None:
            markers = data.get(self.reference_id).columns.tolist()
        i = 0
        nrows = math.ceil(len(markers) / 3)
        fig.suptitle(f'Per-channel KDE, Reference: {self.reference_id}', y=1.05)
        for marker in progress_bar(markers, verbose=self.verbose):
            i += 1
            ax = fig.add_subplot(nrows, 3, i)
            ax = sns.kdeplot(data.get(self.reference_id)[marker], shade=True, color="b", ax=ax, **kwargs)
            ax.set_title(f'Total variance in {marker}')
            if xlim:
                ax.set_xlim(xlim)
            for comparison_sample_id in comparison_samples:
                if marker not in data.get(comparison_sample_id).columns:
                    warn(f"{marker} missing from {comparison_sample_id}, this marker will be ignored")
                else:
                    ax = sns.kdeplot(data.get(comparison_sample_id)[marker],
                                     color='r', shade=False, alpha=0.5, ax=ax)
                    ax.get_legend().remove()
            ax.set(aspect="auto")
        fig.tight_layout()
        return fig

    def dim_reduction_grid(self,
                           features: list,
                           sample_n: int,
                           figsize: tuple = (10, 10),
                           method: str = 'PCA',
                           kde: bool = False,
                           scale: bool = True,
                           dim_reduction_kwargs: dict or None = None,
                           scale_kwargs: dict or None = None):
        """
        Generate a grid of embeddings using a valid dimensionality reduction technique, in each plot a reference sample
        is shown in blue and a comparison sample in red. The reference sample is conserved across all plots.
        reference_id: str
            This sample will appear in red as a comparison
        comparison_samples: list
            List of samples to compare to reference (blue)
        features: list
            List of features to use for dimensionality reduction
        figsize: tuple, (default=(10,10))
            Size of figure
        method: str, (default='PCA')
            Method to use for dimensionality reduction (see flow.dim_reduction)
        kde: bool, (default=False)
            If True, overlay with two-dimensional PDF estimated by KDE
        Returns
        -------
        None
            Plot printed to stdout
        """
        dim_reduction_kwargs = dim_reduction_kwargs or {}
        scale_kwargs = scale_kwargs or {}
        fig = plt.figure(figsize=figsize)
        nrows = math.ceil(len(self.sample_ids) / 3)
        data = self.load_and_sample(sample_n=sample_n)
        if scale:
            data = scale_data(data=data, **scale_kwargs)
        self.print('Plotting...')
        reference = data.pop(self.reference_id)
        reference['label'] = 'Target'
        assert all([f in reference.columns for f in features]), f'Invalid features; valid are: {reference.columns}'
        reference, reducer = dimensionality_reduction(reference,
                                                      features=features,
                                                      method=method,
                                                      n_components=2,
                                                      return_reducer=True,
                                                      **dim_reduction_kwargs)
        i = 0
        fig.suptitle(f'{method}, Reference: {self.reference_id}', y=1.05)
        for s in progress_bar(self.sample_ids, verbose=self.verbose):
            i += 1
            df = data.get(s)
            df['label'] = 'Comparison'
            ax = fig.add_subplot(nrows, 3, i)
            if not all([f in df.columns for f in features]):
                print(f'Features missing from {s}, skipping')
                continue
            embeddings = reducer.transform(df[features])
            x = f'{method}_0'
            y = f'{method}_1'
            ax.scatter(reference[x], reference[y], c='blue', s=4, alpha=0.2)
            if kde:
                sns.kdeplot(reference[x], reference[y], c='blue', n_levels=100, ax=ax, shade=False)
            ax.scatter(embeddings[:, 0], embeddings[:, 1], c='red', s=4, alpha=0.1)
            if kde:
                sns.kdeplot(embeddings[:, 0], embeddings[:, 1], c='red', n_levels=100, ax=ax, shade=False)
            ax.set_title(s)
            ax.set_yticklabels([])
            ax.set_xticklabels([])
            ax.set(aspect='auto')
        fig.tight_layout()
        fig.show()

    def _estimate_pdf(self,
                      target_id: str,
                      target: pd.DataFrame,
                      features: list,
                      reduce_first: bool = False,
                      dim_reduction_method: str = "UMAP",
                      dim_reduction_kwargs: dict or None = None):
        """
        Given a sample ID and its events dataframe, estimate the PDF by KDE with the option
        to perform dimensionality reduction first. Resulting PDF is saved to kde_cache.

        Parameters
        ----------
        target_id: str
        target: Pandas.DataFrame
        features: list
        reduce_first: bool (default=False)
        dim_reduction_method: str (default="UMAP")
        dim_reduction_kwargs: dict (optional)

        Returns
        -------
        None
        """
        dim_reduction_kwargs = dim_reduction_kwargs or {}
        target = target[features].copy().select_dtypes(include=['number'])
        reduced_features = None
        if reduce_first:
            self.print("...performing dimensionality reduction on target")
            target = dimensionality_reduction(target,
                                              features=features,
                                              method=dim_reduction_method,
                                              return_reducer=False,
                                              return_embeddings_only=False,
                                              **dim_reduction_kwargs)
            reduced_features = [f"{dim_reduction_method}_{i}"
                                for i in range(dim_reduction_kwargs.get("n_components"))]
        self.print("...estimating PDF for target")
        if reduced_features is not None:
            target = target[reduced_features]
        self.kde_cache[target_id] = FFTKDE(kernel=self.kde_kernel, bw=self.kde_bw).fit(target.values)()[1]

    def calc_divergence(self,
                        target_id: str,
                        features: list,
                        sample_n: int = 5000,
                        data: dict or None = None,
                        distance_metric: str or callable = 'jsd',
                        reduce_first: bool = False,
                        dim_reduction_method: str = "UMAP",
                        dim_reduction_kwargs: dict or None = None,
                        scale: bool = False,
                        scale_kwargs: dict or None = None) -> np.array:
        """
        Given some target sample ID, estimate the PDF (with the option to perform dimensionality reduction first) and
        calculate the statistical distance between the target and the PDF of all other samples in the associated experiment.

        Parameters
        ----------
        target_id: str
            Should be a valid sample ID for the associated experiment
        features: list
            Markers to be included in multidimensional KDE
        sample_n: int (default=5000)
            Number of events to sample for calculation
        data: dict (optional)
            If provided, data will be sourced from this dictionary rather than making a call to `load_and_sample`. If given
            then value given for sample is ignored.
        distance_metric: callable or str (default='jsd')
            Either a callable function to calculate the statistical distance or a string value; options are:
                * jsd: Jensson-shannon distance
                * kl:Kullback-Leibler divergence (entropy)
        reduce_first: bool (default=False)
            If True, dimensionality reduction performed first using the method specified by dim_reduction_method
        dim_reduction_method: str (default="UMAP")
            Dimension reduction method, see CytoPy.flow.dim_reduction
        dim_reduction_kwargs: dict
            Keyword arguments for dimension reduction method, see CytoPy.flow.dim_reduction
        scale: bool (default=False)
            Whether to scale data prior to calculation
        scale_kwargs
            Keyword arguments for CytoPy.flow.transform.scaler

        Returns
        -------
        list
            List of statistical distances, with results given as a list of nested tuples of type: (sample ID, distance).
        """
        # Set defaults
        if "n_components" not in dim_reduction_kwargs:
            dim_reduction_kwargs["n_components"] = 2
        scale_kwargs = scale_kwargs or {}
        # Assign distance metric func
        metrics = {"kl": kl,
                   "jsd": jsd}
        if isinstance(distance_metric, str):
            assert distance_metric in ['jsd', 'kl'], 'Invalid divergence metric must be one of either jsd, kl, or a callable function]'
            distance_metric = metrics.get(distance_metric)
        # Load data and scale if necessary
        data = data or self.load_and_sample(sample_n=sample_n)
        if scale:
            data = scale_data(data=data, **scale_kwargs)
        # Calculate PDF of target, cache result
        self.print("Calculating PDF for target...")
        if target_id not in self.kde_cache.keys():
            self._estimate_pdf(target_id=target_id,
                               target=data.get(target_id),
                               features=features,
                               reduce_first=reduce_first,
                               dim_reduction_method=dim_reduction_method,
                               dim_reduction_kwargs=dim_reduction_kwargs)
        self.print("Calculate PDF for all other samples and calculate distance from target...")
        # Perform dim reduction is requested
        if reduce_first:
            reduction_f = partial(dimensionality_reduction,
                                  features=features,
                                  method=dim_reduction_method,
                                  return_reducer=False,
                                  return_embeddings_only=False,
                                  **dim_reduction_kwargs)
            with Pool(self.njobs) as pool:
                self.print("...performing dimensionality reduction on comparisons")
                samples_df = list(progress_bar(pool.imap(reduction_f, data.values()),
                                               verbose=self.verbose,
                                               total=len(list(data.values()))))
        self.print("...estimating PDF for comparisons")
        kde_f = partial(kde_wrapper,
                        kernel=self.kde_kernel,
                        bw=self.kde_bw)
        with Pool(self.njobs) as pool:
            q_ = list(progress_bar(pool.imap(kde_f, samples_df),
                                   verbose=self.verbose,
                                   total=len(samples_df)))
        for name, q in zip(data.keys(), q_):
            self.kde_cache[name] = q
        return [(name, distance_metric(self.kde_cache.get(target_id), q)) for name, q in self.kde_cache.items()]

    def similarity_matrix(self,
                          sample_n: int,
                          figsize: tuple = (12, 12),
                          distance_metric: str or callable = 'jsd',
                          clustering_method: str = 'average',
                          features: None or list = None,
                          reduce_first: bool = True,
                          dim_reduction_method: str = "UMAP",
                          scale: bool = False,
                          dim_reduction_kwargs: dict or None = None,
                          scale_kwargs: dict or None = None,
                          cluster_plot_kwargs: dict or None = None):
        """
        Generate a heatmap of pairwise statistical distances with the axis clustered using agglomerative clustering.

        Parameters
        ----------
        sample_n: int
            Number of events to sample for analysis
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
        reduce_first: bool (default=False)
            If True, dimensionality reduction performed first using the method specified by dim_reduction_method
        dim_reduction_method: str (default="UMAP")
            Dimension reduction method, see CytoPy.flow.dim_reduction
        dim_reduction_kwargs: dict
            Keyword arguments for dimension reduction method, see CytoPy.flow.dim_reduction
        scale: bool (default=False)
            Whether to scale data prior to calculation
        scale_kwargs
            Keyword arguments for CytoPy.flow.transform.scaler
        cluster_plot_kwargs: dict
            Additional keyword arguments passed to Seaborn.clustermap call

        Returns
        -------
        Array, Array, ClusterGrid
            Linkage array, ordered array of sample IDs and seaborn ClusterGrid object
        """
        # Set defaults
        dim_reduction_kwargs = dim_reduction_kwargs or {}
        scale_kwargs = scale_kwargs or {}
        cluster_plot_kwargs = cluster_plot_kwargs or {}
        if distance_metric == "kl":
            warn("Kullback-Leiber Divergence chosen as statistical distance metric, KL divergence "
                 "is an asymmetrical function and as such it is not advised to use this metric for the "
                 "similarity matrix'")
        # Fetch data and scale if necessary
        data = self.load_and_sample(sample_n=sample_n)
        if scale:
            data = scale_data(data=data, **scale_kwargs)
        features = features or data.get(list(data.keys())[0]).columns.tolist()
        distance_df = pd.DataFrame()

        # Generate distance matrix
        for s in progress_bar(data.keys(), verbose=self.verbose):
            distances = self.calc_divergence(target_id=s,
                                             features=features,
                                             data=data,
                                             distance_metric=distance_metric,
                                             reduce_first=reduce_first,
                                             dim_reduction_method=dim_reduction_method,
                                             dim_reduction_kwargs=dim_reduction_kwargs,
                                             scale=scale,
                                             scale_kwargs=scale_kwargs)
            name_distances = defaultdict(list)
            for n, d in distances:
                name_distances[n].append(d)
            name_distances = pd.DataFrame(name_distances)
            name_distances["sample_id"] = s
            distance_df = pd.concat([distance_df, name_distances])

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
