from ..data.experiments import Experiment
from ..feedback import progress_bar, vprint
from ..utilities import indexed_parallel_func, hellinger_dist
from .density_estimation import multivariate_kde
from .dim_reduction import dimensionality_reduction
from .transforms import scaler
from .gating_tools import load_population
from scipy.spatial.distance import jensenshannon as jsd
from scipy.stats import entropy as kl
from scipy.cluster import hierarchy
from scipy.spatial import distance
from multiprocessing import Pool, cpu_count
from collections import defaultdict
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


def _transform(indexed_data: (str, np.array),
               reducers: dict):
    return reducers.get(indexed_data[0]).transform(indexed_data[1])


def _check_ref_n(embeddings: Dict[str, dict],
                 reference: str):
    for n in embeddings.keys():
        if reference not in embeddings.get(n).keys():
            warn(f"Reference sample is missing sample N {n} so comparisons will not be missing in some cases.")


def _jsd_n_comparison(pdfs: Dict[str, dict],
                      reference: str):

    jsd_comparisons = defaultdict(dict)
    _check_ref_n(embeddings=pdfs, reference=reference)
    # For each sample N, compare each PDF to the equivalent PDF of the reference sample
    for n in pdfs.keys():
        q = pdfs.get(n).get(reference)
        for _id, p in pdfs.get(n).items():
            if _id == reference:
                continue
            jsd_comparisons[_id][n] = [jsd(p, q)]
    # Wrangle into a dataframe
    df = pd.DataFrame()
    for sample_id in jsd_comparisons.keys():
        df_ = pd.DataFrame(jsd_comparisons.get(sample_id)).melt(var_name="n", value_name="JSD(p, q)")
        df_["Sample ID"] = sample_id
        df = pd.concat([df, df_])
    # Plot results
    fig, ax = plt.subplots(figsize=(10, 10))
    sns.lineplot(x="n", y="JSD(p, q)", hue="Sample ID", ci=None, ax=ax, alpha=.75)
    sns.scatterplot(x="n", y="JSD(p, q)", s=10, alpha=.5, c="black", ax=ax)
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    return ax


def _visual_n_comparison(embeddings: Dict[str, dict]):
    # Wrangle dictionary to have sample ID as primary key
    sample_embeddings = defaultdict(dict)
    for n in embeddings.keys():
        for _id in embeddings.get(n).keys():
            sample_embeddings[_id][n] = embeddings.get(n).get(_id)
    plots = dict()
    for sample_id in sample_embeddings.keys():
        total_n = len(sample_embeddings.get(sample_id).keys())
        fig, axes = plt.subplots(1, total_n, figsize=(10, 5))
        fig.suptitle(sample_id)
        for i, n in enumerate(sample_embeddings.get(sample_id).keys()):
            axes[0, i].scatter(x=sample_embeddings.get(sample_id).get(n)[:, 0],
                               y=sample_embeddings.get(sample_id).get(n)[:, 1],
                               c="#5982d4",
                               s=3.5,
                               alpha=0.5)
            axes[0, i].set_title(f"Sample n={n}")
        plots[sample_id] = (fig, axes)
    return plots


class EvaluateBatchEffects:
    def __init__(self,
                 experiment: Experiment,
                 root_population: str,
                 samples: list or None = None,
                 reference_sample: str or None = None,
                 transform: str = 'logicle',
                 verbose: bool = True,
                 njobs: int = -1):
        self.experiment = experiment
        self.transform = transform
        self.root_population = root_population
        self.verbose = verbose
        self.print = vprint(verbose)
        self.kde_cache = dict()
        self.njobs = njobs
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
        self.kde_cache = {}

    def load_and_sample(self,
                        sample_n: int = 10000):
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

    @staticmethod
    def scale_data(data: Dict[str, pd.DataFrame],
                   scale: str or None = "standard",
                   **kwargs):
        return {k: pd.DataFrame(scaler(data=v.values,
                                       scale_method=scale,
                                       return_scaler=False,
                                       **kwargs),
                                columns=v.columns) for k, v in data.items()}

    def _calc_ref_sample(self,
                         sample_n: int = 1000):
        self.print("--- Calculating Reference Sample ---")
        return covar_euclidean_norm(data=self.load_and_sample(sample_n=sample_n),
                                    verbose=self.verbose)

    def select_optimal_sample_n(self,
                                method: str = "jsd",
                                sample_range: list or None = None,
                                scale: str or None = "standard",
                                dimensionality_reduction_method: str = "UMAP",
                                scaler_kwargs: dict or None = None,
                                dim_reduction_kwargs: dict or None = None,
                                kde_kwargs: dict or None = None):
        scaler_kwargs = scaler_kwargs or dict()
        dim_reduction_kwargs = dim_reduction_kwargs or dict()
        kde_kwargs = kde_kwargs or dict()
        assert method in ["jsd", "visual"], "Method should be either 'jsd' or 'visual'"
        sample_range = sample_range or np.arange(1000, 11000, 1000)
        self.print("=============================================")
        self.print("Finding optimal sample N")
        self.print("---------------------------------------------")
        self.print("Warning: this process is computationally intensive. Although CytoPy will leverage "
                   "all available cores to reduce the time cost, this might take some time to complete.")
        largest_sample = sample_range[-1:]
        sample_range = sample_range[:-1]

        self.print(f"Fitting chosen reducer {dimensionality_reduction_method} using largest chosen sample size: "
                   f"{largest_sample}...")
        self.print("...collecting data...")
        data = self.load_and_sample(sample_n=largest_sample)
        if scale:
            self.print("...scaling data...")
            data = self.scale_data(data=data, scale=scale, **scaler_kwargs)

        _dim_reduction = partial(indexed_parallel_func,
                                 func=dimensionality_reduction,
                                 features=list(data.values())[0].columns.values,
                                 method=dimensionality_reduction_method,
                                 n_components=2,
                                 return_embeddings_only=True,
                                 return_reducer=True,
                                 **dim_reduction_kwargs)
        self.print("...fitting data...")
        with Pool(self.njobs) as pool:
            indexed_data = [(k, v) for k, v in data.items()]
            indexed_embeddings = list(progress_bar(pool.imap(_dim_reduction, indexed_data),
                                                   verbose=self.verbose,
                                                   total=len(indexed_data)))
        reducers = {x[0]: x[2] for x in indexed_embeddings}
        embeddings = dict()
        embeddings[largest_sample] = {x[0]: x[1] for x in indexed_embeddings}
        self.print("---------------------------------------------")
        self.print("Applying transform to remaining sample n's...")
        for n in sample_range:
            self.print(f"Sample n = {n}...")
            self.print("...collecting data...")
            self.load_and_sample(sample_n=largest_sample)
            if scale:
                self.print("...scaling data...")
                self.scale_data(scale=scale, **scaler_kwargs)
            self.print("...transforming data...")
            _apply_transform = partial(_transform,
                                       reducers=reducers)
            with Pool(self.njobs) as pool:
                indexed_data = [(k, v) for k, v in data.items()]
                indexed_embeddings = list(progress_bar(pool.imap(_apply_transform, indexed_data),
                                                       verbose=self.verbose,
                                                       total=len(indexed_data)))
            embeddings[n] = {x[0]: x[1] for x in indexed_embeddings}
        if method == "jsd":
            self.print("...estimating PDF for embeddings...")
            kde_func = partial(indexed_parallel_func,
                               func=multivariate_kde,
                               **kde_kwargs)
            pdfs = dict()
            for n in progress_bar(embeddings.keys(), verbose=self.verbose):
                indexed_data = [(_id, data) for _id, data in embeddings.get(n).items()]
                with Pool(self.njobs) as pool:
                    indexed_pdfs = list(progress_bar(pool.imap(kde_func, indexed_data),
                                                     verbose=self.verbose,
                                                     total=len(indexed_data)))
                pdfs[n] = {x[0]: x[1] for x in indexed_pdfs}
            return _jsd_n_comparison(pdfs=pdfs,
                                     reference=self.reference_id)
        return _visual_n_comparison(embeddings=embeddings)

    def marker_variance(self,
                        comparison_samples: list,
                        sample_n: int,
                        markers: list or None = None,
                        figsize: tuple = (10, 10),
                        xlim: tuple or None = None,
                        **kwargs):
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
            data =self.scale_data(data=data, **scale_kwargs)
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

    def calc_divergence(self,
                        target_id: str,
                        comparisons: list,
                        features: list,
                        sample_n: int,
                        distance_metric: str or callable = 'jsd',
                        reduce_first: bool = True,
                        dim_reduction_method: str = "UMAP",
                        dim_reduction_kwargs: dict or None = None,
                        scale: bool = False,
                        scale_kwargs: dict or None = None,
                        kde_kwargs: dict or None = None) -> np.array:
        # Set defaults
        dim_reduction_kwargs = dim_reduction_kwargs or {}
        if "n_components" not in dim_reduction_kwargs:
            dim_reduction_kwargs["n_components"] = 2
        scale_kwargs = scale_kwargs or {}
        kde_kwargs = kde_kwargs or {}
        # Assign distance metric func
        metrics = {"kl": kl,
                   "jsd": jsd,
                   "hellinger": hellinger_dist}
        if type(distance_metric) == str:
            assert distance_metric in ['jsd', 'kl', 'hellinger'], 'Invalid divergence metric must be one of ' \
                                                                  '[jsd, kl, hellinger]'
            distance_metric = metrics.get(distance_metric)
        # Load data and scale if necessary
        data = self.load_and_sample(sample_n=sample_n)
        if scale:
            data = self.scale_data(data=data, **scale_kwargs)
        # Calculate PDF of target, cache result
        self.print("Calculating PDF for target...")
        if target_id not in self.kde_cache.keys():
            target = data.get(target_id)[features]
            target = target.select_dtypes(include=['number']).values
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
            self.kde_cache[target_id] = multivariate_kde(data=target,
                                                         features=reduced_features or features,
                                                         **kde_kwargs)
        self.print("Calculate PDF for all other samples and calculate distance from target...")
        # Fetch data of comparison samples if not in cache
        samples_df = [(name, df.select_dtypes(include='number').values) for name, df in data.items()
                      if name in comparisons and name not in self.kde_cache.keys()]
        # Perform dim reduction is requested
        if reduce_first:
            reduction_f = partial(indexed_parallel_func,
                                  features=features,
                                  method=dim_reduction_method,
                                  return_reducer=False,
                                  return_embeddings_only=False,
                                  **dim_reduction_kwargs)
            with Pool(self.njobs) as pool:
                self.print("...performing dimensionality reduction on comparisons")
                samples_df = list(progress_bar(pool.imap(reduction_f, samples_df),
                                               verbose=self.verbose,
                                               total=len(samples_df)))
        self.print("...estimating PDF for comparisons")
        kde_f = partial(indexed_parallel_func, func=multivariate_kde, **kde_kwargs)
        with Pool(self.njobs) as pool:
            q_ = list(progress_bar(pool.imap(kde_f, samples_df),
                                   verbose=self.verbose,
                                   total=len(samples_df)))
        for name, q in q_:
            self.kde_cache[name] = q
        return [(name, distance_metric(self.kde_cache.get(target_id), q)) for name, q in self.kde_cache.items()]

    def similarity_matrix(self,
                          sample_n: int,
                          exclude: list or None = None,
                          figsize: tuple = (12, 12),
                          distance_metric: str or callable = 'jsd',
                          clustering_method: str = 'average',
                          features: None or list = None,
                          reduce_first: bool = True,
                          dim_reduction_method: str = "UMAP",
                          scale: bool = False,
                          dim_reduction_kwargs: dict or None = None,
                          scale_kwargs: dict or None = None,
                          kde_kwargs: dict or None = None,
                          cluster_plot_kwargs: dict or None = None):
        # Set defaults
        exclude = exclude or []
        samples = [s for s in self.sample_ids if s not in exclude]
        dim_reduction_kwargs = dim_reduction_kwargs or {}
        scale_kwargs = scale_kwargs or {}
        kde_kwargs = kde_kwargs or {}
        cluster_plot_kwargs = cluster_plot_kwargs or {}
        if distance_metric == "kl":
            warn("Kullback-Leiber Divergence chosen as statistical distance metric, KL divergence "
                 "is an asymmetrical function and as such it is not advised to use this metric for the "
                 "similarity matrix'")
        # Fetch data and scale if necessary
        data = self.load_and_sample(sample_n=sample_n)
        if scale:
            data = self.scale_data(data=data, **scale_kwargs)
        features = features or data.get(list(data.keys())[0]).columns.tolist()
        distance_df = pd.DataFrame()

        # Generate distance matrix
        for s in progress_bar(samples, verbose=self.verbose):
            distances = self.calc_divergence(target_id=s,
                                             comparisons=samples,
                                             features=features,
                                             sample_n=sample_n,
                                             distance_metric=distance_metric,
                                             reduce_first=reduce_first,
                                             dim_reduction_method=dim_reduction_method,
                                             dim_reduction_kwargs=dim_reduction_kwargs,
                                             scale=scale,
                                             scale_kwargs=scale_kwargs,
                                             kde_kwargs=kde_kwargs)
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