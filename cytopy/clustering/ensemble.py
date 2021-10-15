import logging
import math
import os
import pickle
from collections import defaultdict
from typing import Callable
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple
from typing import Type
from typing import Union

import numpy as np
import pandas as pd
import seaborn as sns
from ClusterEnsembles import ClusterEnsembles
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.ticker import MaxNLocator
from sklearn.base import ClusterMixin

from .clustering import Clustering
from .clustering import ClusteringError
from .clustering import ClusterMethod
from .clustering import remove_null_features
from .metrics import comparison_matrix
from .metrics import init_internal_metrics
from .metrics import InternalMetric
from .plotting import clustered_heatmap
from .plotting import plot_cluster_membership
from cytopy.data.experiment import Experiment
from cytopy.feedback import add_processing_animation
from cytopy.feedback import progress_bar
from cytopy.plotting.single_cell_plot import discrete_palette
from cytopy.utils.dim_reduction import DimensionReduction

logger = logging.getLogger(__name__)


def save_cache(func: Callable):
    def wrapper(obj, *args, **kwargs):
        output = func(obj, *args, **kwargs)
        # Save cache
        cache = {
            "experiment": obj.experiment.id,
            "features": obj.features,
            "sample_ids": obj.sample_ids,
            "root_population": obj.root_population,
            "transform": obj.transform,
            "transform_kwargs": obj.transform_kwargs,
            "verbose": obj.verbose,
            "population_prefix": obj.population_prefix,
            "data": obj.data,
            "clustering_permutations": obj.clustering_permutations,
        }
        with open(obj.cache_location, "wb") as f:
            logger.info(f"Saving results to {obj.cache_location}")
            pickle.dump(cache, file=f)
        return output

    return wrapper


def load_cache(path: str) -> Dict:
    with open(path, "rb") as f:
        return pickle.load(f)


class EnsembleClustering(Clustering):
    """
    The EnsembleClustering class provides a toolset for applying multiple clustering algorithms to a
    dataset, reviewing the clustering results of each algorithm, comparing their performance, and then
    forming a consensus for the clustering results that draws from the results of all the algorithms
    combined.

    Unlike the SingleClustering class, the EnsembleClustering class only supports global clustering and
    will merge multiple FileGroups (samples) of an Experiment and treat as a single feature space. Therefore
    it is important to address batch effects prior to applying ensemble clustering.

    Clustering algorithms are applied using the 'cluster' class and like SingleClustering, require that a
    valid method is given (either 'flowsom', 'phenograph', the name of a Scikit-Learn clustering method,
    or a ClusterMethod class). Each clustering result will have a name and the clustering labels and meta
    data are stored in the 'clustering_permutations' attribute. The results of the individual clustering can
    be observed using the 'plot' and 'heatmap' methods, and the performance of all clustering algorithms
    is accessible through the 'performance' attribute.

    The outputs of clustering algorithms can also be contrasted by comparing their mutual information or rand
    index (after adjusting for chance):
    * Adjusted mutual information: measures the agreement between clustering results where the ground truth
    clustering is expected to be unbalanced with possibly small clusters
    * Adjusted rand index: a measure of similarity between clustering results where the ground truth is
    expected to contain mostly equal sized clusters

    The above are accessed using the 'comparison' method returning a clustered matrix of the pairwise
    metrics.

    To obtain a consensus multiple 'finishing' techniques can be applied. All but one use a co-occurrence matrix
    (quantifies the number of times a pair of observations cluster together, for all observations in the dataset):

    * Clustering co-occurrence: the simplest solution is that we cluster the co-occurrence matrix ensuring that
    clusters are obtained that encapsulate data points that co-cluster robustly across methods
    * Majority vote: using the co-occurrence matrix, cluster assignment is made by majority vote to extract
    only consistent clusters
    * Graph closure: by treating the co-occurrence matrix as an adjacency matrix, find the complete subgraphs within
    the matrix bia k-cliques and percolation
    * Mixture model: a probabilistic model of consensus using a finite mixture of multinomial distributions
    in a space of clustering results. This method assumes that the number of clusters is predetermined and therefore
    the methods above may be preferred.

    Once multiple clustering methods have been applied, you can use the 'co_occurrence_matrix' method to generate
    a CoMatrix object that will provide access to co-occurrence clustering, majority vote, and graph closure
    for final label generation. Use the 'mixture_model' method to obtain a MixtureModel object to obtain final
    labels using the multivariate mixture models.
    """

    def __init__(
        self,
        cache: str,
        experiment: Experiment = None,
        features: List[str] = None,
        sample_ids: Optional[List[str]] = None,
        root_population: str = "root",
        transform: str = "logicle",
        transform_kwargs: Optional[Dict] = None,
        verbose: bool = True,
        population_prefix: str = "ensemble",
        random_state: int = 42,
    ):
        if os.path.isfile(cache):
            cached_data = load_cache(path=cache)
            logger.info(f"Loading EnsembleClustering from {cache}")
            super().__init__(
                experiment=Experiment.objects(id=cached_data["experiment"]).get(),
                features=cached_data["features"],
                sample_ids=cached_data["sample_ids"],
                root_population=cached_data["root_population"],
                transform=cached_data["transform"],
                transform_kwargs=cached_data["transform_kwargs"],
                verbose=cached_data["verbose"],
                population_prefix=cached_data["population_prefix"],
                data=cached_data["data"],
                random_state=random_state,
            )
            self.clustering_permutations = cached_data["clustering_permutations"]
        else:
            logger.info(f"Creating new EnsembleClustering object connected to {experiment.experiment_id}")
            super().__init__(
                experiment=experiment,
                features=features,
                sample_ids=sample_ids,
                root_population=root_population,
                transform=transform,
                transform_kwargs=transform_kwargs,
                verbose=verbose,
                population_prefix=population_prefix,
                random_state=random_state,
            )
            self.clustering_permutations = dict()
        self.cache_location = cache

    @save_cache
    def cluster(
        self,
        cluster_name: str,
        method: Union[str, ClusterMethod, ClusterMixin],
        overwrite_features: Optional[List[str]] = None,
        scale_method: Optional[str] = None,
        scale_kwargs: Optional[Dict] = None,
        dim_reduction: Optional[str] = None,
        dim_reduction_kwargs: Optional[Dict] = None,
        clustering_params: Optional[Dict] = None,
    ):
        clustering_params = clustering_params or {}
        dim_reduction_kwargs = dim_reduction_kwargs or {}
        overwrite_features = overwrite_features or self.features
        features = remove_null_features(self.data, features=overwrite_features)
        method = self._init_cluster_method(method=method, **clustering_params)
        data, features = self.scale_and_reduce(
            features=features,
            scale_method=scale_method,
            scale_kwargs=scale_kwargs,
            dim_reduction=dim_reduction,
            dim_reduction_kwargs=dim_reduction_kwargs,
        )

        logger.info(f"Running clustering: {cluster_name}")
        data = method.global_clustering(data=data, features=features)
        self.clustering_permutations[cluster_name] = {
            "labels": data["cluster_label"].values,
            "n_clusters": data["cluster_label"].nunique(),
            "features": features,
            "params": clustering_params,
            "scale_method": scale_method,
            "scale_params": scale_kwargs,
            "dim_reduction": dim_reduction,
            "dim_reduction_params": dim_reduction_kwargs,
        }
        logger.info("Clustering complete!")
        return self

    def comparison(self, method: str = "adjusted_mutual_info", **kwargs):
        kwargs["figsize"] = kwargs.get("figsize", (10, 10))
        kwargs["cmap"] = kwargs.get("cmap", "coolwarm")
        cluster_labels = {cluster_name: data["labels"] for cluster_name, data in self.clustering_permutations.items()}
        data = comparison_matrix(cluster_labels=cluster_labels, method=method)
        return sns.clustermap(
            data=data,
            **kwargs,
        )

    def _consensus(self, consensus_method: str, labels: np.ndarray, k: int, random_state: int = 42):
        if consensus_method == "cspa" and self.data.shape[0] > 5000:
            logger.warning("CSPA is not recommended when n>5000, consider a different method")
            return ClusterEnsembles.cspa(labels=labels, nclass=k)
        if consensus_method == "hgpa":
            return ClusterEnsembles.hgpa(labels=labels, nclass=k, random_state=random_state)
        if consensus_method == "mcla":
            return ClusterEnsembles.mcla(labels=labels, nclass=k, random_state=random_state)
        if consensus_method == "hbgf":
            return ClusterEnsembles.hbgf(labels=labels, nclass=k)
        if consensus_method == "nmf":
            return ClusterEnsembles.nmf(labels=labels, nclass=k, random_state=random_state)
        raise ClusteringError("Invalid consensus method, must be one of: cdpa, hgpa, mcla, hbgf, or nmf")

    @save_cache
    @add_processing_animation(text="Computing consensus clustering")
    def consensus(self, key: str, consensus_method: str, k: int, random_state: int = 42):
        labels = np.array([x["labels"] for x in self.clustering_permutations.values()])
        self.clustering_permutations[key] = {
            "labels": self._consensus(
                consensus_method=consensus_method, k=k, labels=labels, random_state=random_state
            ),
            "n_clusters": k,
            "params": {},
            "scale_method": None,
            "scale_params": {},
            "dim_reduction": None,
            "dim_reduction_params": {},
        }
        return self

    def choose_k(
        self,
        k_range: Tuple[int, int],
        consensus_method: str,
        sample_size: int,
        resamples: int,
        random_state: int = 42,
        metrics: Optional[List[Union[InternalMetric, str]]] = None,
        return_data: bool = True,
        **kwargs,
    ):
        if sample_size > self.data.shape[0]:
            raise ClusteringError(f"Sample size cannot exceed size of data ({self.data.shape[0]})")
        logger.info("Sampling...")
        metrics = init_internal_metrics(metrics=metrics)
        labels = []
        data = []
        for _ in progress_bar(range(resamples), total=resamples):
            idx = np.random.randint(0, self.data.shape[0], sample_size)
            labels.append(np.array([np.array(x["labels"])[idx] for x in self.clustering_permutations.values()]))
            data.append(self.data.iloc[idx])
        k_range = np.arange(k_range[0], k_range[1] + 1)
        results = defaultdict(list)
        for k in k_range:
            logger.info(f"Calculating consensus with k={k}...")
            for la, df in progress_bar(zip(labels, data), total=len(data)):
                la = self._consensus(consensus_method=consensus_method, k=k, labels=la, random_state=random_state)
                results["K"].append(k)
                for m in metrics:
                    results[m.name].append(m(data=df, features=self.features, labels=la))
        results = pd.DataFrame(results).melt(id_vars="K", var_name="Metric", value_name="Value")
        facet_kws = kwargs.pop("facet_kws", {})
        facet_kws["sharey"] = facet_kws.get("sharey", False)
        g = sns.relplot(data=results, x="K", y="Value", kind="line", col="Metric", facet_kws=facet_kws, **kwargs)
        g.set_titles("{col_name}")
        for ax in g.axes:
            ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        if return_data:
            return g, results
        return g

    def plot(
        self,
        cluster_name: str,
        sample_size: Union[int, None] = 100000,
        sampling_method: str = "uniform",
        method: Union[str, Type] = "UMAP",
        dim_reduction_kwargs: dict or None = None,
        label: str = "cluster_label",
        discrete: bool = True,
        **kwargs,
    ):
        plot_data = self.data.copy()
        plot_data["cluster_label"] = self.clustering_permutations[cluster_name]["labels"]
        return plot_cluster_membership(
            data=plot_data,
            features=self.features,
            sample_size=sample_size,
            sampling_method=sampling_method,
            method=method,
            dim_reduction_kwargs=dim_reduction_kwargs,
            label=label,
            discrete=discrete,
            **kwargs,
        )

    def min_k(self):
        return min([x["n_clusters"] for x in self.clustering_permutations.values()])

    def max_k(self):
        return max([x["n_clusters"] for x in self.clustering_permutations.values()])

    def plot_all(
        self,
        sample_size: int = 100000,
        method: Union[str, Type] = "UMAP",
        dim_reduction_kwargs: dict or None = None,
        label: str = "cluster_label",
        col_wrap: int = 3,
        figsize: Tuple[int, int] = (10, 10),
        bins: Union[str, int] = "sqrt",
        hist_cmap: str = "jet",
        palette: Optional[Union[str, List[str]]] = None,
        **kwargs,
    ):

        kwargs["s"] = kwargs.get("s", 5)
        kwargs["edgecolors"] = kwargs.get("edgecolors", None)
        kwargs["linewidth"] = kwargs.get("linewidth", 0)
        palette = palette or discrete_palette(n=self.max_k())

        plot_data = self.data.sample(n=sample_size)
        dim_reduction_kwargs = dim_reduction_kwargs or {}
        reducer = DimensionReduction(method=method, **dim_reduction_kwargs)
        plot_data = reducer.fit_transform(data=plot_data, features=self.features)
        fig, axes = plt.subplots(
            math.ceil((len(self.clustering_permutations) + 1) / col_wrap), col_wrap, figsize=figsize
        )
        axes = axes.flatten()

        bins = bins if isinstance(bins, int) else int(np.sqrt(plot_data.shape[0]))
        axes[0].hist2d(plot_data[f"{method}1"], plot_data[f"{method}2"], bins=bins, cmap=hist_cmap, norm=LogNorm())
        axes[0].autoscale(enable=True)

        for i, (name, cluster_data) in enumerate(self.clustering_permutations.items()):
            i += 1
            plot_data["cluster_label"] = cluster_data["labels"][plot_data.index.values]
            plot_data["cluster_label"] = plot_data["cluster_label"].astype(str)
            sns.scatterplot(
                data=plot_data,
                x=f"{method}1",
                y=f"{method}2",
                hue=label,
                ax=axes[i],
                palette=palette,
                **kwargs,
            )
            axes[i].get_legend().remove()
            axes[i].set_title(f"{name} (n_clusters={cluster_data['n_clusters']})")
        if (len(self.clustering_permutations) + 1) < len(axes):
            n = len(axes)
            while n > (len(self.clustering_permutations) + 1):
                fig.delaxes(axes[n - 1])
                n = n - 1
        fig.tight_layout()
        return fig

    def heatmap(
        self,
        cluster_name: str,
        features: Optional[str] = None,
        sample_id: Optional[str] = None,
        meta_label: bool = True,
        **kwargs,
    ):
        plot_data = self.data.copy()
        plot_data["cluster_label"] = self.clustering_permutations[cluster_name]["labels"]
        plot_data = self.data.groupby("cluster_label")[self.features].median()
        features = features or self.features
        kwargs["col_cluster"] = kwargs.get("col_cluster", True)
        kwargs["figsize"] = kwargs.get("figsize", (10, 15))
        kwargs["standard_scale"] = kwargs.get("standard_scale", 1)
        kwargs["cmap"] = kwargs.get("cmap", "viridis")
        return clustered_heatmap(
            data=plot_data, features=features, sample_id=sample_id, meta_label=meta_label ** kwargs
        )

    def save(self, cluster_name: str, verbose: bool = True, parent_populations: Optional[Dict] = None):
        self.data["cluster_label"] = self.clustering_permutations[cluster_name]["labels"]
        super()._save(verbose=verbose, population_var="cluster_label", parent_populations=parent_populations)
        self.data.drop("cluster_label", axis=1, inplace=True)
        return self
