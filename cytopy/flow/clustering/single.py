from typing import Dict
from typing import List
from typing import Optional
from typing import Type
from typing import Union

import numpy as np
import seaborn as sns
from sklearn.base import ClusterMixin

from . import metrics as cluster_metrics
from ...feedback import progress_bar
from ..dim_reduction import dimension_reduction_with_sampling
from .clustering import Clustering
from .clustering import ClusterMethod
from .clustering import remove_null_features
from .plotting import clustered_heatmap
from .plotting import plot_cluster_membership
from .plotting import plot_cluster_membership_sample
from .plotting import plot_meta_clusters


class SingleClustering(Clustering):
    """
    High-dimensional clustering offers the advantage of an unbiased approach
    to classification of single cells whilst also exploiting all available variables
    in your data (all your fluorochromes/isotypes). In cytopy, the clustering is
    performed on a Population of a FileGroup. The resulting clusters are saved
    as new Populations. We can compare the clustering results of many FileGroup's
    by 'clustering the clusters', to do this we summarise their clusters and perform meta-clustering.

    The Clustering class provides all the apparatus to perform high-dimensional clustering
    using any of the following functions from the cytopy.flow.clustering.main module:

    * sklearn_clustering - access any of the Scikit-Learn cluster/mixture classes for unsupervised learning;
      currently also provides access to HDBSCAN
    * phenograph_clustering - access to the PhenoGraph clustering algorithm
    * flowsom_clustering - access to the FlowSOM clustering algorithm

    In addition, meta-clustering (clustering or clusters) can be performed with any of the following from
    the same module:
    * sklearn_metaclustering
    * phenograph_metaclustering
    * consensus_metaclustering

    The Clustering class is algorithm agnostic and only requires that a function be
    provided that accepts a Pandas DataFrame with a column name 'sample_id' as the
    sample identifier, 'cluster_label' as the clustering results, and 'meta_label'
    as the meta clustering results. The function should also accept 'features' as
    a list of columns to use to construct the input space to the clustering algorithm.
    This function must return a Pandas DataFrame with the cluster_label/meta_label
    columns populated accordingly. It should also return two null value OR can optionally
    return a graph object, and modularity or equivalent score. These will be saved
    to the Clustering attributes.


    Parameters
    ----------
    experiment: Experiment
        Experiment to access for FileGroups to be clustered
    features: list
        Features (fluorochromes/cell markers) to use for clustering
    sample_ids: list, optional
        Name of FileGroups load from Experiment and cluster. If not given, will load all
        samples from Experiment.
    root_population: str (default="root")
        Name of the Population to use as input data for clustering
    transform: str (default="logicle")
        How to transform the data prior to clustering, see cytopy.flow.transform for valid methods
    transform_kwargs: dict, optional
        Additional keyword arguments passed to Transformer
    verbose: bool (default=True)
        Whether to provide output to stdout
    population_prefix: str (default='cluster')
        Prefix added to populations generated from clustering results

    Attributes
    ----------
    features: list
        Features (fluorochromes/cell markers) to use for clustering
    experiment: Experiment
        Experiment to access for FileGroups to be clustered
    metrics: float or int
        Metric values such as modularity score from Phenograph
    data: Pandas.DataFrame
        Feature space and clustering results. Contains features and additional columns:
        - sample_id: sample identifier
        - subject_id: subject identifier
        - cluster_label: cluster label (within sample)
        - meta_label: meta cluster label (between samples)
    """

    def cluster(
        self,
        method: Union[str, ClusterMethod, ClusterMixin],
        overwrite_features: Optional[List[str]] = None,
        metrics: Optional[List[Union[str, cluster_metrics.Metric]]] = None,
        evaluate: bool = False,
        **kwargs,
    ):
        overwrite_features = overwrite_features or self.features
        features = remove_null_features(self.data, features=overwrite_features)
        method = self._init_cluster_method(method=method, metrics=metrics, **kwargs)
        self.data, self.metrics = method.cluster(data=self.data, features=features, evaluate=evaluate)
        return self

    def global_clustering(
        self,
        method: Union[str, ClusterMethod, ClusterMixin],
        overwrite_features: Optional[List[str]] = None,
        metrics: Optional[List[Union[str, cluster_metrics.Metric]]] = None,
        evaluate: bool = False,
        scale_method: Optional[str] = None,
        scale_kwargs: Optional[Dict] = None,
        dim_reduction: Optional[str] = None,
        dim_reduction_kwargs: Optional[Dict] = None,
        clustering_params: Optional[Dict] = None,
    ):
        overwrite_features = overwrite_features or self.features
        features = remove_null_features(self.data, features=overwrite_features)

        dim_reduction_kwargs = dim_reduction_kwargs or {}
        data, scaler = self.scale_data(features=features, scale_method=scale_method, scale_kwargs=scale_kwargs)
        if dim_reduction is not None:
            data, _ = dimension_reduction_with_sampling(
                data=self.data, features=features, method=dim_reduction, **dim_reduction_kwargs
            )
            features = [x for x in data.columns if dim_reduction in x]

        clustering_params = clustering_params or {}
        method = self._init_cluster_method(method=method, metrics=metrics, **clustering_params)
        self.data, self.metrics = method.global_clustering(data=data, features=features, evaluate=evaluate)
        return self

    def meta_cluster(
        self,
        method: Union[str, ClusterMethod],
        overwrite_features: Optional[List[str]] = None,
        summary_method: str = "median",
        scale_method: str or None = None,
        scale_kwargs: dict or None = None,
        metrics: Optional[List[Union[str, cluster_metrics.Metric]]] = None,
        evaluate: bool = False,
        **kwargs,
    ):
        overwrite_features = overwrite_features or self.features
        features = remove_null_features(self.data, features=overwrite_features)
        method = self._init_cluster_method(method=method, metrics=metrics, **kwargs)
        self.data, self.metrics = method.meta_clustering(
            data=self.data,
            features=features,
            summary_method=summary_method,
            scale_method=scale_method,
            scale_kwargs=scale_kwargs,
            evaluate=evaluate,
            **kwargs,
        )

    def rename_meta_clusters(self, mappings: dict):
        """
        Given a dictionary of mappings, replace the current IDs stored
        in meta_label column of the data attribute with new IDs

        Parameters
        ----------
        mappings: dict
            Mappings; {current ID: new ID}

        Returns
        -------
        None
        """
        self.data["meta_label"].replace(mappings, inplace=True)

    def reset_meta_clusters(self):
        """
        Reset meta clusters to None

        Returns
        -------
        self
        """
        self.data["meta_label"] = None
        return self

    def plot(
        self,
        meta_clusters: bool = False,
        sample_id: Optional[str] = None,
        sample_size: Union[int, None] = 100000,
        sampling_method: str = "uniform",
        method: Union[str, Type] = "UMAP",
        dim_reduction_kwargs: dict or None = None,
        label: str = "cluster_label",
        discrete: bool = True,
        **kwargs,
    ):
        if meta_clusters:
            return plot_meta_clusters(
                data=self.data,
                features=self.features,
                colour_label=label,
                discrete=discrete,
                method=method,
                dim_reduction_kwargs=dim_reduction_kwargs,
                **kwargs,
            )
        if sample_id is None:
            return plot_cluster_membership(
                data=self.data,
                features=self.features,
                sample_size=sample_size,
                sampling_method=sampling_method,
                method=method,
                dim_reduction_kwargs=dim_reduction_kwargs,
                label=label,
                discrete=discrete,
                **kwargs,
            )
        return plot_cluster_membership_sample(
            data=self.data,
            features=self.features,
            sample_id=sample_id,
            method=method,
            dim_reduction_kwargs=dim_reduction_kwargs,
            label=label,
            discrete=discrete,
            **kwargs,
        )

    def heatmap(
        self, features: Optional[str] = None, sample_id: Optional[str] = None, meta_label: bool = True, **kwargs
    ):
        features = features or self.features
        return clustered_heatmap(
            data=self.data, features=features, sample_id=sample_id, meta_label=meta_label ** kwargs
        )

    def choose_k(
        self,
        max_k: int,
        cluster_n_param: str,
        method: Union[str, ClusterMethod],
        metric: cluster_metrics.Metric,
        overwrite_features: Optional[List[str]] = None,
        sample_id: Optional[str] = None,
        reduce_dimensions: bool = False,
        dim_reduction_kwargs: Optional[Dict] = None,
        clustering_params: Optional[Dict] = None,
    ):
        clustering_params = clustering_params or {}

        overwrite_features = overwrite_features or self.features
        features = remove_null_features(self.data, features=overwrite_features)
        data = (
            self.data
            if not reduce_dimensions
            else dimension_reduction_with_sampling(data=self.data, features=features, **dim_reduction_kwargs)
        )
        if sample_id is not None:
            data = data[data.sample_id == sample_id].copy()

        ylabel = metric.name
        x = list()
        y = list()
        for k in progress_bar(np.arange(1, max_k + 1, 1)):
            df = data.copy()
            clustering_params[cluster_n_param] = k
            method = self._init_cluster_method(method=method, **clustering_params)
            df = method.cluster(data=df, features=features, evaluate=False)
            x.append(k)
            y.append(metric(data=df, features=features, labels=df["cluster_label"]))
        ax = sns.lineplot(x=x, y=y, markers=True)
        ax.set_xlabel("K")
        ax.set_ylabel(ylabel)
        return ax
