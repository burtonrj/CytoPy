#!/usr/bin.env/python
# -*- coding: utf-8 -*-
"""
High-dimensional clustering offers the advantage of an unbiased approach
to classification of single cells whilst also exploiting all available variables
in your data (all your fluorochromes/isotypes). In cytopy, the clustering is
performed on a Population of a FileGroup. The resulting clusters are saved
as new Populations.

In CytoPy, we refer to three different types of clustering:
* Per-sample clustering, where each FileGroup (sample) is clustered individually
* Global clustering, where FileGroup's (sample's) are combined into the same space and clustering is
performed for all events - this is computationally expensive and requires that batch effects have been
minimised or corrected prior to clustering
* Meta-clustering, where the clustering results of individual FileGroup's are clustered to
match clusters between FileGroup's; essentially 'clustering the clusters'

In this module you will find the Clustering class, which is the apparatus to apply a
clustering method in cytopy and save the results to the database. We also
provide implementations of PhenoGraph, FlowSOM and provide access to any
of the clustering methods available through the Scikit-Learn API.

The Clustering class is algorithm agnostic and only requires that a function be
provided that accepts a Pandas DataFrame with a column name 'sample_id' as the
sample identifier, 'cluster_label' as the clustering results, and 'meta_label'
as the meta clustering results. The function should also accept 'features' as
a list of columns to use to construct the input space to the clustering algorithm.
This function must return a Pandas DataFrame with the cluster_label/meta_label
columns populated accordingly. It should also return two null value OR can optionally
return a graph object, and modularity or equivalent score. These will be saved
to the Clustering attributes.

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
import logging
import pickle
from collections import defaultdict
from typing import *

import numpy as np
import pandas as pd
import phenograph
import seaborn as sns
from sklearn.metrics import calinski_harabasz_score
from sklearn.metrics import davies_bouldin_score
from sklearn.metrics import silhouette_score

from . import metrics as cluster_metrics
from ...data.experiment import Experiment
from ...data.experiment import single_cell_dataframe
from ...data.population import Population
from ...data.subject import Subject
from ...feedback import progress_bar
from ..dim_reduction import dimension_reduction_with_sampling
from .consensus import ConsensusCluster
from .ensemble import CoMatrix
from .ensemble import comparison_matrix
from .ensemble import MixtureModel
from .flowsom import FlowSOM
from .plotting import clustered_heatmap
from .plotting import plot_cluster_membership
from .plotting import plot_cluster_membership_sample
from .plotting import plot_meta_clusters
from cytopy.flow.transform import Scaler

logger = logging.getLogger(__name__)


class ClusteringError(Exception):
    def __init__(self, message: str):
        logger.error(message)
        super().__init__(message)


def init_metrics(metrics: Optional[List[Union[str, cluster_metrics.Metric]]] = None):
    if metrics is None:
        return [x() for x in cluster_metrics.inbuilt_metrics.values()]
    metric_objs = list()
    try:
        for x in metrics:
            if isinstance(x, str):
                metric_objs.append(cluster_metrics.inbuilt_metrics[x]())
            else:
                assert isinstance(x, cluster_metrics.Metric)
                metric_objs.append(x)
    except KeyError:
        logger.error(f"Invalid metric, must be one of {cluster_metrics.inbuilt_metrics.keys()}")
        raise
    except AssertionError:
        logger.error(
            f"metrics must be a list of strings corresponding to default metrics "
            f"({cluster_metrics.inbuilt_metrics.keys()}) and/or Metric objects"
        )
        raise


def clustering_performance(data: pd.DataFrame, labels: list):
    for x in [
        "Clustering performance...",
        f"Silhouette coefficient: {silhouette_score(data.values, labels, metric='euclidean')}",
        f"Calinski-Harabasz index: {calinski_harabasz_score(data.values, labels)}",
        f"Davies-Bouldin index: {davies_bouldin_score(data.value, labels)}",
    ]:
        print(x)
        logger.info(x)


def remove_null_features(data: pd.DataFrame, features: Optional[List[str]] = None) -> list:
    """
    Check for null values in the dataframe.
    Returns a list of column names for columns with no missing values.

    Parameters
    ----------
    data: Pandas.DataFrame
    features: List[str], optional

    Returns
    -------
    List
        List of valid columns
    """
    features = features or data.columns.tolist()
    null_cols = data[features].isnull().sum()[data[features].isnull().sum() > 0].index.values
    if null_cols.size != 0:
        logger.warning(
            f"The following columns contain null values and will be excluded from " f"clustering analysis: {null_cols}"
        )
    return [x for x in features if x not in null_cols]


def assign_metalabels(data: pd.DataFrame, metadata: pd.DataFrame):
    """
    Given the original clustered data (data) and the meta-clustering results of
    clustering the clusters of this original data (metadata), assign the meta-cluster
    labels to the original data and return the modified dataframe with the meta cluster
    labels in a new column called 'meta_label'

    Parameters
    ----------
    data: Pandas.DataFrame
    metadata: Pandas.DataFrame

    Returns
    -------
    Pandas.DataFrame
    """
    data = data.drop("meta_label", axis=1)
    return pd.merge(
        data,
        metadata[["sample_id", "cluster_label", "meta_label"]],
        on=["sample_id", "cluster_label"],
    )


def summarise_clusters(
    data: pd.DataFrame,
    features: list,
    scale: str or None = None,
    scale_kwargs: dict or None = None,
    summary_method: str = "median",
):
    """
    Average cluster parameters along columns average to generated a centroid for
    meta-clustering

    Parameters
    ----------
    data: Pandas.DataFrame
        Clustering results to average
    features: list
        List of features to use when generating centroid
    summary_method: str (default='median')
        Average method, should be mean or median
    scale: str, optional
        Perform scaling of centroids; see cytopy.transform.Scaler
    scale_kwargs: dict, optional
        Additional keyword arguments passed to Scaler

    Returns
    -------
    Pandas.DataFrame

    Raises
    ------
    ValueError
        If invalid method provided
    """
    if summary_method == "median":
        data = data.groupby(["sample_id", "cluster_label"])[features].median().reset_index()
    elif summary_method == "mean":
        data = data.groupby(["sample_id", "cluster_label"])[features].mean().reset_index()
    else:
        raise ValueError("summary_method should be 'mean' or 'median'")
    scale_kwargs = scale_kwargs or {}
    if scale is not None:
        scaler = Scaler(method=scale, **scale_kwargs)
        data = scaler(data=data, features=features)
    return data


class Phenograph:
    def __init__(self, **params):
        params = params or {}
        self.params = params

    def fit_predict(self, data: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        communities, graph, q = phenograph.cluster(data, **self.params)
        return communities


class ClusterMethod:
    def __init__(
        self,
        klass: Type,
        params: Optional[Dict] = None,
        verbose: bool = True,
        metrics: Optional[List[Union[str, cluster_metrics.Metric]]] = None,
    ):
        params = params or {}
        self.verbose = verbose
        self.method = klass(**params)
        self.metrics = init_metrics(metrics=metrics)
        self.params = params

    def valid_method(self):
        try:
            fit_predict = getattr(self.method, "fit_predict", None)
            assert fit_predict is not None
            assert callable(fit_predict)
        except AssertionError:
            raise ClusteringError("Invalid Class as clustering method, must have function 'fit_predict'")

    def _cluster(self, data: pd.DataFrame, features: List[str]):
        return self.method.fit_predict(data[features])

    def cluster(self, data: pd.DataFrame, features: List[str], evaluate: bool = False):
        performance = {}
        data["cluster_label"] = None
        for _id, df in progress_bar(data.groupby("sample_id"), verbose=self.verbose):
            labels = self._cluster(data, features)
            data.loc[df.index, ["cluster_label"]] = labels
            if evaluate:
                performance[_id] = {metric.name: metric(df, features, labels) for metric in self.metrics}
        if evaluate:
            return data, pd.DataFrame(performance)
        return data, None

    def global_clustering(self, data: pd.DataFrame, features: List[str], evaluate: bool = False):
        data["cluster_label"] = self._cluster(data, features)
        if evaluate:
            return data, pd.DataFrame(
                {metric.name: [metric(data, features, data["cluster_label"].values)] for metric in self.metrics}
            )
        return data, None

    def meta_clustering(
        self,
        data: pd.DataFrame,
        features: List[str],
        summary_method: str = "median",
        evaluate: bool = False,
        scale_method: Optional[str] = None,
        scale_kwargs: Optional[Dict] = None,
    ):
        metadata = summarise_clusters(
            data=data, features=features, summary_method=summary_method, scale=scale_method, scale_kwargs=scale_kwargs
        )
        metadata["meta_label"] = self._cluster(metadata, features)
        data = assign_metalabels(data, metadata)
        if evaluate:
            return data, pd.DataFrame(
                {metric.name: [metric(data, features, data["meta_label"].values)] for metric in self.metrics}
            )
        return data, None


class Clustering:
    def __init__(
        self,
        experiment: Experiment,
        features: list,
        sample_ids: list or None = None,
        root_population: str = "root",
        transform: str = "logicle",
        transform_kwargs: dict or None = None,
        verbose: bool = True,
        population_prefix: str = "cluster",
    ):
        self.experiment = experiment
        self.verbose = verbose
        self.features = features
        self.transform = transform
        self.root_population = root_population
        self.metrics = None
        self.population_prefix = population_prefix

        logger.info(f"Obtaining data for clustering for population {root_population}")
        self.data = single_cell_dataframe(
            experiment=experiment,
            sample_ids=sample_ids,
            transform=transform,
            transform_kwargs=transform_kwargs,
            populations=root_population,
        ).to_pandas()
        self.data["meta_label"] = None
        self.data["cluster_label"] = None
        logger.info("Ready to cluster!")

    def _init_cluster_method(
        self,
        method: Union[str, ClusterMethod],
        metrics: Optional[List[Union[str, cluster_metrics.Metric]]] = None,
        **kwargs,
    ) -> ClusterMethod:
        if method == "phenograph":
            method = ClusterMethod(klass=Phenograph, params=kwargs, metrics=metrics, verbose=self.verbose)
        elif method == "flowsom":
            method = ClusterMethod(klass=FlowSOM, params=kwargs, metrics=metrics, verbose=self.verbose)
        elif method == "consensus":
            method = ClusterMethod(klass=ConsensusCluster, params=kwargs, metrics=metrics, verbose=self.verbose)
        elif isinstance(method, str):
            raise ValueError("If a string is given must be either 'phenograph', 'consensus' or 'flowsom'")
        elif not isinstance(method, ClusterMethod):
            method = ClusterMethod(klass=method, params=kwargs, metrics=metrics, verbose=self.verbose)
        if not isinstance(method, ClusterMethod):
            raise ValueError(
                "Must provide a valid string, a ClusterMethod object, or a valid Scikit-Learn like "
                "clustering class (must have 'fit_predict' method)."
            )
        return method

    def scale_data(self, features: List[str], scale_method: Optional[str] = None, scale_kwargs: Optional[Dict] = None):
        scale_kwargs = scale_kwargs or {}
        scalar = None
        data = self.data.copy()
        if scale_method is not None:
            scalar = Scaler(scale_method, **scale_kwargs)
            data = scalar(data=self.data, features=features)
        return data, scalar

    def elbow_plot(
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

    def reset_clusters(self):
        """
        Resets cluster and meta cluster labels to None

        Returns
        -------
        self
        """
        self.data["cluster_label"] = None
        self.data["meta_label"] = None
        return self

    def rename_clusters(self, sample_id: str, mappings: dict):
        """
        Given a dictionary of mappings, replace the current IDs stored
        in cluster_label column for a particular sample

        Parameters
        ----------
        sample_id: str
        mappings: dict
            Mappings; {current ID: new ID}

        Returns
        -------
        None
        """
        if sample_id != "all":
            idx = self.data[self.data.sample_id == sample_id].index
            self.data.loc[idx, "cluster_label"] = self.data.loc[idx]["cluster_label"].replace(mappings)
        else:
            self.data["cluster_label"] = self.data["cluster_label"].replace(mappings)

    def load_meta_variable(self, variable: str, verbose: bool = True, embedded: list or None = None):
        """
        Load a meta-variable for each Subject, adding this variable as a new column. If a sample
        is not associated to a Subject or the meta variable is missing from a Subject, value will be
        None.
        Parameters
        ----------
        variable: str
            Name of the meta-variable
        verbose: bool (default=True)
        embedded: list
            If the meta-variable is embedded, this should be a list of keys that
            preceed the variable

        Returns
        -------
        None
        """
        self.data[variable] = None
        for _id in progress_bar(self.data.subject_id.unique(), verbose=verbose):
            if _id is None:
                continue
            p = Subject.objects(subject_id=_id).get()
            try:
                if embedded is not None:
                    x = None
                    for key in embedded:
                        x = p[key]
                    self.data.loc[self.data.subject_id == _id, variable] = x[variable]
                else:
                    self.data.loc[self.data.subject_id == _id, variable] = p[variable]
            except KeyError:
                logger.warning(f"{_id} is missing meta-variable {variable}")
                self.data.loc[self.data.subject_id == _id, variable] = None

    def _create_parent_populations(self, population_var: str, parent_populations: Dict, verbose: bool = True):
        """
        Form parent populations from existing clusters

        Parameters
        ----------
        population_var: str
            Name of the cluster population variable i.e. cluster_label or meta_label
        parent_populations: Dict
            Dictionary of parent associations. Parent populations will be a merger of all child populations.
            Each child population intended to inherit from a parent that is not 'root' should be given as a
            key with the value being the parent to associate to.
        verbose: bool (default=True)
            Whether to provide feedback in the form of a progress bar

        Returns
        -------
        None
            Parent populations are saved to the FileGroup
        """
        logger.info("Creating parent populations from clustering results")
        parent_child_mappings = defaultdict(list)
        for child, parent in parent_populations.items():
            parent_child_mappings[parent].append(child)

        for sample_id in progress_bar(self.data.sample_id.unique(), verbose=verbose):
            fg = self.experiment.get_sample(sample_id)
            sample_data = self.data[self.data.sample_id == sample_id].copy()

            for parent, children in parent_child_mappings.items():
                cluster_data = sample_data[sample_data[population_var].isin(children)]
                if cluster_data.shape[0] == 0:
                    logger.warning(f"No clusters found for {sample_id} to generate requested parent {parent}")
                    continue
                parent_population_name = (
                    parent if self.population_prefix is None else f"{self.population_prefix}_{parent}"
                )
                pop = Population(
                    population_name=parent_population_name,
                    n=cluster_data.shape[0],
                    parent=self.root_population,
                    source="cluster",
                    signature=cluster_data.mean().to_dict(),
                )
                pop.index = cluster_data.Index.values
                fg.add_population(population=pop)
            fg.save()

    def save(
        self,
        verbose: bool = True,
        population_var: str = "meta_label",
        parent_populations: Optional[Dict] = None,
    ):
        """
        Clusters are saved as new Populations in each FileGroup in the attached Experiment
        according to the sample_id in data.

        Parameters
        ----------
        verbose: bool (default=True)
        population_var: str (default='meta_label')
            Variable in data that should be used to identify individual Populations
        parent_populations: Dict
            Dictionary of parent associations. Parent populations will be a merger of all child populations.
            Each child population intended to inherit from a parent that is not 'root' should be given as a
            key with the value being the parent to associate to.

        Returns
        -------
        None

        Raises
        ------
        ValueError
            If population_var is 'meta_label' and meta clustering has not been previously performed
        """
        if population_var == "meta_label":
            if self.data.meta_label.isnull().all():
                raise ValueError("Meta clustering has not been performed")

        if parent_populations is not None:
            self._create_parent_populations(population_var=population_var, parent_populations=parent_populations)
        parent_populations = parent_populations or {}

        for sample_id in progress_bar(self.data.sample_id.unique(), verbose=verbose):
            fg = self.experiment.get_sample(sample_id)
            sample_data = self.data[self.data.sample_id == sample_id].copy()

            for cluster_label, cluster in sample_data.groupby(population_var):
                population_name = (
                    str(cluster_label)
                    if self.population_prefix is None
                    else f"{self.population_prefix}_{cluster_label}"
                )
                parent = parent_populations.get(cluster_label, self.root_population)
                parent = (
                    parent
                    if self.population_prefix is None or parent == self.root_population
                    else f"{self.population_prefix}_{parent}"
                )
                pop = Population(
                    population_name=population_name,
                    n=cluster.shape[0],
                    parent=parent,
                    source="cluster",
                    signature=cluster.mean().to_dict(),
                )
                pop.index = cluster.Index.values
                fg.add_population(population=pop)
            fg.save()


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
        method: Union[str, ClusterMethod, Type],
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
        method: Union[str, ClusterMethod, Type],
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
        method: Union[str, ClusterMethod, Type],
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
        metrics: Optional[List[Union[str, cluster_metrics.Metric]]] = None,
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
        method: Union[str, ClusterMethod, Type],
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
