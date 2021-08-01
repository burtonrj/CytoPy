from ...data.experiment import Experiment, single_cell_dataframe
from ...data.population import Population
from ...feedback import progress_bar
from ..sampling import sample_dataframe_uniform_groups
from ..dim_reduction import DimensionReduction
from ..plotting import single_cell_plot
from ..transform import Scaler
from .main import remove_null_features, ClusteringError
from .ensemble_methods import CoMatrix, MixtureModel
from .mutual_info import MutualInfo
from .metrics import *
from collections import defaultdict
from typing import *
import seaborn as sns
import pandas as pd
import numpy as np
import logging

logger = logging.getLogger("clustering.ensemble")


def valid_labels(func: Callable):
    def wrapper(self, cluster_labels: List[int], *args, **kwargs):
        if len(cluster_labels) != self.data.shape[0]:
            raise ClusteringError(f"cluster_idx does not match the number of events. Did you use a valid "
                                  f"finishing technique? {len(cluster_labels)} != {self.data.shape[0]}")
        return func(self, cluster_labels, *args, **kwargs)

    return wrapper


class EnsembleClustering:

    default_metrics = {
        "ball_hall": BallHall,
        "baker_hubert_gamma_index": BakerHubertGammaIndex,
        "silhouette_coef": SilhouetteCoef,
        "davies_bouldin_index": DaviesBouldinIndex,
        "g_plus_index": GPlusIndex,
        "calinski_harabasz_score": CalinskiHarabaszScore
    }

    def __init__(self,
                 experiment: Experiment,
                 features: list,
                 sample_ids: list or None = None,
                 root_population: str = "root",
                 transform: Union[str, Dict] = "logicle",
                 transform_kwargs: dict or None = None,
                 verbose: bool = True,
                 population_prefix: str = "ensemble",
                 sample_size: Optional[int, float] = None,
                 sample_method: str = "uniform",
                 sampling_kwargs: Optional[Dict] = None,
                 random_state: int = 42,
                 metrics: Optional[List[Union[str, Metric]]] = None):
        logger.info(f"Creating new EnsembleClustering object with connection to {experiment.experiment_id}")
        np.random.seed(random_state)
        self.experiment = experiment
        self.verbose = verbose
        self.features = features
        self.transform = transform
        self.root_population = root_population
        self.graph = None
        self.metrics = None
        self._performance = dict()
        self.population_prefix = population_prefix
        self.clustering_permutations = dict()
        self._populate_metrics(metrics=metrics)

        logger.info(f"Obtaining data for clustering for population {root_population}")
        self.data = single_cell_dataframe(experiment=experiment,
                                          sample_ids=sample_ids,
                                          transform=transform,
                                          transform_kwargs=transform_kwargs,
                                          populations=root_population,
                                          sample_size=sample_size,
                                          sampling_level="file",
                                          sample_method=sample_method,
                                          sampling_kwargs=sampling_kwargs)
        self.data["cluster_label"] = None
        logging.info("Ready to cluster!")

    def _populate_metrics(self, metrics: Optional[List[Union[str, Metric]]]):
        try:
            for m in metrics:
                if isinstance(m, str):
                    self.metrics.append(self.default_metrics[m])
                else:
                    assert isinstance(m, Metric)
        except KeyError:
            logger.error(f"Invalid metric, must be one of {self.default_metrics.keys()}")
            raise
        except AssertionError:
            logger.error(f"metrics must be a list of strings corresponding to default metrics "
                         f"({self.default_metrics.keys()}) and/or Metric objects")
            raise

    def cluster(self,
                cluster_name: str,
                func: Callable,
                overwrite_features: Optional[List[str]] = None,
                scale: Optional[str] = None,
                scale_kwargs: Optional[Dict] = None,
                **kwargs):
        features = remove_null_features(self.data, features=overwrite_features)
        scale_kwargs = scale_kwargs or {}
        scalar = None
        data = self.data
        if scale is not None:
            scalar = Scaler(scale, **scale_kwargs)
            data = scalar(data=data, features=features)
        logger.info(f"Running clustering: {cluster_name}")
        data, _, _ = func(data=data,
                          features=features,
                          global_clustering=True,
                          print_performance_metrics=False)
        self.clustering_permutations[cluster_name] = {"labels": data["cluster_label"].values,
                                                      "n_clusters": data["cluster_label"].nunique(),
                                                      "params": kwargs or {},
                                                      "scalar": scalar}
        logger.info(f"Calculating performance metrics for {cluster_name}")
        performance = [self._cluster_metric(data["cluster_label"].values, metric)
                       for metric in self._performance]
        self._performance[cluster_name] = {x[0]: x[1] for x in performance}
        logger.info("Clustering complete!")

    @property
    def performance(self):
        if self._performance is None:
            raise ClusteringError("Add clusters before accessing metrics")
        return pd.DataFrame(self._performance)

    def _cluster_metric(self, labels: np.ndarray, metric: Metric):
        return metric.name, metric(self.data, self.features, labels)

    def co_occurrence_matrix(self, index: Optional[str]):
        return CoMatrix(data=self.data, clusterings=self.clustering_permutations, index=index)

    def mutual_info(self, method: str = "adjusted"):
        return MutualInfo(clusterings=self.clustering_permutations, method=method)

    def mixture_model(self):
        return MixtureModel(data=self.data, clustering_permuations=self.clustering_permutations)

    @valid_labels
    def single_cell_plot(self,
                         cluster_labels: List[int],
                         sample_size: Union[int, None] = 100000,
                         sampling_method: str = "uniform",
                         method: Union[str, Type] = "UMAP",
                         dim_reduction_kwargs: dict or None = None,
                         discrete: bool = True,
                         **kwargs):
        plot_data = self.data.copy()
        plot_data["cluster_label"] = cluster_labels
        if sample_size is not None:
            if sampling_method == "uniform":
                plot_data = sample_dataframe_uniform_groups(data=self.data,
                                                            group_id="sample_id",
                                                            sample_size=sample_size)
            else:
                if sample_size < self.data.shape[0]:
                    plot_data = self.data.sample(sample_size)
        dim_reduction_kwargs = dim_reduction_kwargs or {}
        reducer = DimensionReduction(method=method,
                                     n_components=2,
                                     **dim_reduction_kwargs)
        df = reducer.fit_transform(data=plot_data, features=self.features)
        return single_cell_plot(data=df,
                                x=f"{method}1",
                                y=f"{method}2",
                                label="cluster_label",
                                discrete=discrete,
                                **kwargs)

    @valid_labels
    def clustered_heatmap(self,
                          cluster_labels: List[int],
                          features: Optional[List] = None,
                          **kwargs):
        plot_data = self.data.copy()
        plot_data["cluster_label"] = cluster_labels
        plot_data = self.data.groupby("cluster_label")[self.features].median()
        features = features or self.features
        plot_data[features] = plot_data[features].apply(pd.to_numeric)
        kwargs = kwargs or {}
        kwargs["col_cluster"] = kwargs.get("col_cluster", True)
        kwargs["figsize"] = kwargs.get("figsize", (10, 15))
        kwargs["standard_scale"] = kwargs.get("standard_scale", 1)
        kwargs["cmap"] = kwargs.get("cmap", "viridis")
        return sns.clustermap(plot_data[features], **kwargs)

    def _create_parent_populations(self,
                                   data: pd.DataFrame,
                                   parent_populations: Dict,
                                   verbose: bool = True):
        logger.info("Creating parent populations from clustering results")
        parent_child_mappings = defaultdict(list)
        for child, parent in parent_populations.items():
            parent_child_mappings[parent].append(child)

        for sample_id in progress_bar(data.sample_id.unique(), verbose=verbose):
            fg = self.experiment.get_sample(sample_id)
            sample_data = data[data.sample_id == sample_id].copy()

            for parent, children in parent_child_mappings.items():
                cluster_data = sample_data[sample_data["cluster_label"].isin(children)]
                if cluster_data.shape[0] == 0:
                    logger.warning(f"No clusters found for {sample_id} to generate requested parent {parent}")
                    continue
                parent_population_name = parent if self.population_prefix is None \
                    else f"{self.population_prefix}_{parent}"
                pop = Population(population_name=parent_population_name,
                                 n=cluster_data.shape[0],
                                 parent=self.root_population,
                                 source="cluster",
                                 signature=cluster_data.mean().to_dict())
                pop.index = cluster_data.original_index.values
                fg.add_population(population=pop)
            fg.save()

    @valid_labels
    def save(self,
             cluster_labels: List[int],
             verbose: bool = True,
             parent_populations: Optional[Dict] = None):
        data = self.data.copy()
        data["cluster_label"] = cluster_labels
        if parent_populations is not None:
            self._create_parent_populations(data=data,
                                            parent_populations=parent_populations)
        parent_populations = parent_populations or {}

        for sample_id in progress_bar(data.sample_id.unique(), verbose=verbose):
            fg = self.experiment.get_sample(sample_id)
            sample_data = data[data.sample_id == sample_id].copy()

            for cluster_label, cluster in sample_data.groupby("cluster_label"):
                population_name = str(cluster_label) if self.population_prefix is None \
                    else f"{self.population_prefix}_{cluster_label}"
                parent = parent_populations.get(cluster_label, self.root_population)
                parent = parent if self.population_prefix is None or parent == self.root_population \
                    else f"{self.population_prefix}_{parent}"
                pop = Population(population_name=population_name,
                                 n=cluster.shape[0],
                                 parent=parent,
                                 source="cluster",
                                 signature=cluster.mean().to_dict())
                pop.index = cluster.original_index.values
                fg.add_population(population=pop)
            fg.save()
