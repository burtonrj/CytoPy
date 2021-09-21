import logging
import pickle
from typing import Callable
from typing import Dict
from typing import List
from typing import Optional
from typing import Type
from typing import Union

import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.base import ClusterMixin
from sklearn.metrics import adjusted_mutual_info_score
from sklearn.metrics import adjusted_rand_score

from ...data.experiment import Experiment
from ...feedback import progress_bar
from ..dim_reduction import dimension_reduction_with_sampling
from .clustering import Clustering
from .clustering import ClusteringError
from .clustering import ClusterMethod
from .clustering import remove_null_features
from .metrics import init_metrics
from .metrics import Metric
from .plotting import clustered_heatmap
from .plotting import plot_cluster_membership

logger = logging.getLogger(__name__)


def adjusted_score(a: List[int], b: List[int], method: str):
    methods = {
        "adjusted_mutual_info": adjusted_mutual_info_score,
        "adjusted_rand_score": adjusted_rand_score,
    }
    try:
        return methods[method](a, b)
    except KeyError:
        ValueError(f"Method must be one of {methods.keys()}")


def comparison_matrix(clustering_permutations: Dict, method: str = "adjusted_mutual_info") -> pd.DataFrame:
    labels = {cluster_name: data["labels"] for cluster_name, data in clustering_permutations.items()}
    data = pd.DataFrame(columns=list(labels.keys()), index=list(labels.keys()), dtype=float)
    names = list(labels.keys())
    for n1 in progress_bar(names):
        for n2 in names:
            if np.isnan(data.loc[n1, n2]):
                mi = float(adjusted_score(labels[n1], labels[n2], method=method))
                data.at[n1, n2] = mi
                data.at[n2, n1] = mi
    return data


def select_nclass(
    data: pd.DataFrame,
    k: List[int],
    consensus_method: str = "hbgf",
    metrics: Optional[List[Metric]] = None,
    resample: int = 20,
    sample_size: int = 1000,
    **kwargs,
) -> sns.FacetGrid:
    return sns.lineplot(**kwargs)


def valid_labels(func: Callable):
    def wrapper(self, cluster_labels: Union[str, List[int]], *args, **kwargs):
        if isinstance(cluster_labels, str):
            assert cluster_labels in self.clustering_permutations.keys(), "Invalid cluster name"
            cluster_labels = self.clustering_permutations[cluster_labels]["labels"]
            return func(self, cluster_labels, *args, **kwargs)
        if len(cluster_labels) != self.data.shape[0]:
            raise ClusteringError(
                f"cluster_idx does not match the number of events. Did you use a valid "
                f"finishing technique? {len(cluster_labels)} != {self.data.shape[0]}"
            )
        return func(self, cluster_labels, *args, **kwargs)

    return wrapper


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
        experiment: Experiment = None,
        features: List[str] = None,
        sample_ids: Optional[List[str]] = None,
        root_population: str = "root",
        transform: str = "logicle",
        transform_kwargs: Optional[Dict] = None,
        verbose: bool = True,
        population_prefix: str = "ensemble",
        random_state: int = 42,
        metrics: Optional[List[Union[str, Metric]]] = None,
    ):
        logger.info(f"Creating new EnsembleClustering object with connection to {experiment.experiment_id}")
        np.random.seed(random_state)
        super().__init__(
            experiment=experiment,
            features=features,
            sample_ids=sample_ids,
            root_population=root_population,
            transform=transform,
            transform_kwargs=transform_kwargs,
            verbose=verbose,
            population_prefix=population_prefix,
        )
        self.metrics = init_metrics(metrics=metrics)
        self._performance = dict()
        self.clustering_permutations = dict()

    @property
    def performance(self):
        if len(self._performance) == 0:
            raise ClusteringError("Add clusters before accessing metrics")
        return pd.DataFrame(self._performance)

    def cache(self, path: str):
        obj_data = {
            "performance": self._performance,
            "clustering_permutations": self.clustering_permutations,
            "metrics": self.metrics,
            "data": self.data,
        }
        with open(path, "wb") as f:
            pickle.dump(obj_data, f)

    def load(self, path: str):
        with open(path, "rb") as f:
            obj_data = pickle.load(f)
        self._performance = obj_data["performance"]
        self.clustering_permutations = obj_data["clustering_permutations"]
        self.metrics = obj_data["metrics"]
        self.data = obj_data["data"]

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
        method = self._init_cluster_method(method=method, metrics=self.metrics, **clustering_params)
        data, scaler = self.scale_data(features=features, scale_method=scale_method, scale_kwargs=scale_kwargs)
        if dim_reduction is not None:
            data, _ = dimension_reduction_with_sampling(
                data=self.data, features=features, method=dim_reduction, **dim_reduction_kwargs
            )
            features = [x for x in data.columns if dim_reduction in x]

        logger.info(f"Running clustering: {cluster_name}")
        data, _ = method.global_clustering(data=data, features=features, evaluate=False)
        self.clustering_permutations[cluster_name] = {
            "labels": data["cluster_label"].values,
            "n_clusters": data["cluster_label"].nunique(),
            "params": clustering_params,
            "scalar": scaler,
        }
        logger.info(f"Calculating performance metrics for {cluster_name}")
        self._performance[cluster_name] = {
            metric.name: metric(data, features, data["cluster_label"].values) for metric in self.metrics
        }
        logger.info("Clustering complete!")

    def co_occurrence_matrix(self, index: Optional[str] = None):
        return CoMatrix(
            data=self.data, features=self.features, clustering_permutations=self.clustering_permutations, index=index
        )

    def comparison(self, method: str = "adjusted_mutual_info", **kwargs):
        kwargs["figsize"] = kwargs.get("figsize", (10, 10))
        kwargs["cmap"] = kwargs.get("cmap", "coolwarm")
        data = comparison_matrix(clustering_permutations=self.clustering_permutations, method=method)
        return sns.clustermap(
            data=data,
            **kwargs,
        )

    def mixture_model(self):
        return MixtureModel(data=self.data, clustering_permuations=self.clustering_permutations)

    @valid_labels
    def plot(
        self,
        cluster_labels: Union[str, List[int]],
        sample_size: Union[int, None] = 100000,
        sampling_method: str = "uniform",
        method: Union[str, Type] = "UMAP",
        dim_reduction_kwargs: dict or None = None,
        label: str = "cluster_label",
        discrete: bool = True,
        **kwargs,
    ):
        data = self.data.copy()
        data["cluster_label"] = cluster_labels
        return plot_cluster_membership(
            data=data,
            features=self.features,
            sample_size=sample_size,
            sampling_method=sampling_method,
            method=method,
            dim_reduction_kwargs=dim_reduction_kwargs,
            label=label,
            discrete=discrete,
            **kwargs,
        )

    @valid_labels
    def heatmap(
        self,
        cluster_labels: Union[str, List[int]],
        features: Optional[str] = None,
        sample_id: Optional[str] = None,
        meta_label: bool = True,
        **kwargs,
    ):
        plot_data = self.data.copy()
        plot_data["cluster_label"] = cluster_labels
        plot_data = self.data.groupby("cluster_label")[self.features].median()
        features = features or self.features
        kwargs["col_cluster"] = kwargs.get("col_cluster", True)
        kwargs["figsize"] = kwargs.get("figsize", (10, 15))
        kwargs["standard_scale"] = kwargs.get("standard_scale", 1)
        kwargs["cmap"] = kwargs.get("cmap", "viridis")
        return clustered_heatmap(
            data=plot_data, features=features, sample_id=sample_id, meta_label=meta_label ** kwargs
        )

    @valid_labels
    def save(
        self, cluster_labels: Union[str, List[int]], verbose: bool = True, parent_populations: Optional[Dict] = None
    ):
        self.data["cluster_label"] = cluster_labels
        super().save(verbose=verbose, population_var="cluster_label", parent_populations=parent_populations)
