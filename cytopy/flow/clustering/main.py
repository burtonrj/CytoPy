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
from ..dim_reduction import DimensionReduction
from ..plotting import cluster_bubble_plot
from ..plotting import single_cell_plot
from ..sampling import sample_dataframe_uniform_groups
from .consensus import ConsensusCluster
from .flowsom import FlowSOM
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
    def __init__(self, params: Dict):
        self.params = params

    def fit_predict(self, data: pd.DataFrame, features: List[str]) -> np.ndarray:
        communities, graph, q = phenograph.cluster(data[features], **self.params)
        return communities


class ClusterMethod:
    def __init__(
        self,
        klass: Type,
        params: Dict,
        verbose: bool = True,
        metrics: Optional[List[Union[str, cluster_metrics.Metric]]] = None,
    ):
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
    graph: scipy.sparse
        NxN matrix representing a weighted graph. Populated by Phenograph method
    metrics: float or int
        Metric values such as modularity score from Phenograph
    data: Pandas.DataFrame
        Feature space and clustering results. Contains features and additional columns:
        - sample_id: sample identifier
        - subject_id: subject identifier
        - cluster_label: cluster label (within sample)
        - meta_label: meta cluster label (between samples)
    """

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
        logger.info(f"Creating new Clustering object with connection to {experiment.experiment_id}")
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
        kwargs = kwargs or {}
        if method == "phenograph":
            method = ClusterMethod(klass=Phenograph, params=kwargs, metrics=metrics, verbose=self.verbose)
        elif method == "flowsom":
            method = ClusterMethod(klass=FlowSOM, params=kwargs, metrics=metrics, verbose=self.verbose)
        elif isinstance(method, str):
            raise ValueError("If a string is given must be either 'phenograph' or 'flowsom'")
        elif not isinstance(method, ClusterMethod):
            method = ClusterMethod(klass=method, params=kwargs, metrics=metrics, verbose=self.verbose)
        if not isinstance(method, ClusterMethod):
            raise ValueError(
                "Must provide a valid string, a ClusterMethod object, or a valid Scikit-Learn like "
                "clustering class (must have 'fit_predict' method)."
            )
        return method

    def cluster(
        self,
        method: Union[str, ClusterMethod],
        overwrite_features: Optional[List[str]] = None,
        metrics: Optional[List[Union[str, cluster_metrics.Metric]]] = None,
        evaluate: bool = False,
        global_clustering: bool = False,
        **kwargs,
    ):
        overwrite_features = overwrite_features or self.features
        features = remove_null_features(self.data, features=overwrite_features)
        method = self._init_cluster_method(method=method, metrics=metrics, **kwargs)
        if global_clustering:
            self.data, self.metrics = method.global_clustering(data=self.data, features=features, evaluate=evaluate)
        else:
            self.data, self.metrics = method.cluster(data=self.data, features=features, evaluate=evaluate)
        return self

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

    def reset_meta_clusters(self):
        """
        Reset meta clusters to None

        Returns
        -------
        self
        """
        self.data["meta_label"] = None
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
        if sample_id is not "all":
            idx = self.data[self.data.sample_id == sample_id].index
            self.data.loc[idx, "cluster_label"] = self.data.loc[idx]["cluster_label"].replace(mappings)
        else:
            self.data["cluster_label"] = self.data["cluster_label"].replace(mappings)

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

    def single_cell_plot(
        self,
        sample_size: Union[int, None] = 100000,
        sampling_method: str = "uniform",
        method: Union[str, Type] = "UMAP",
        dim_reduction_kwargs: dict or None = None,
        label: str = "cluster_label",
        discrete: bool = True,
        **kwargs,
    ):
        """
        Plot the entire single cell dataframe (the data attribute). WARNING: this can be computationally
        expensive so if you have limited resource, try specifying a sample size first.

        Parameters
        ----------
        sample_size: int, optional (default=100000)
        sampling_method: str (default="uniform")
            Can either be uniform or absolute:
            * uniform: take a uniform sample of the same size from each sample
            * absolute: sample uniformly from the whole dataframe without accounting for individual
            biological sample size
        method: Union[str, Type]
            Dimensionality reduction technique; available in-built methods are: UMAP, PCA, PHATE, KernelPCA or tSNE.
            (see cytopy.flow.dimension_reduction)
        dim_reduction_kwargs: dict, optional
            Additional keyword arguments passed to dimension reduction (see cytopy.flow.dim_reduction)
        label: str, (default='cluster_label')
            How to colour single cells
        discrete: bool (default=True)
            If True, label is treated as a discrete variable. If False, continuous colourmap will be applied.
        kwargs:
            Additional keyword arguments passed to cytopy.flow.plotting.single_cell_plot

        Returns
        -------
        Matplotlib.Axes
        """
        plot_data = self.data
        if sample_size is not None:
            if sampling_method == "uniform":
                plot_data = sample_dataframe_uniform_groups(
                    data=self.data, group_id="sample_id", sample_size=sample_size
                )
            else:
                if sample_size < self.data.shape[0]:
                    plot_data = self.data.sample(sample_size)

        dim_reduction_kwargs = dim_reduction_kwargs or {}
        reducer = DimensionReduction(method=method, n_components=2, **dim_reduction_kwargs)
        df = reducer.fit_transform(data=plot_data, features=self.features)
        return single_cell_plot(
            data=df,
            x=f"{method}1",
            y=f"{method}2",
            label=label,
            discrete=discrete,
            **kwargs,
        )

    def plot_sample_clusters(
        self,
        sample_id: str,
        method: Union[str, Type] = "UMAP",
        dim_reduction_kwargs: dict or None = None,
        label: str = "cluster_label",
        discrete: bool = True,
        **kwargs,
    ):
        """
        Generate a single cell plot (see cytopy.flow.plotting.single_cell_plot) for a single sample,
        with cells coloured by cluster membership (default)

        Parameters
        ----------
        sample_id: str
        method: Union[str, Type]
            Dimensionality reduction technique; available in-built methods are: UMAP, PCA, PHATE, KernelPCA or tSNE.
            (see cytopy.flow.dimension_reduction)
        dim_reduction_kwargs: dict, optional
            Additional keyword arguments passed to dimension reduction (see cytopy.flow.dim_reduction)
        label: str, (default='cluster_label')
            How to colour single cells
        discrete: bool (default=True)
            If True, label is treated as a discrete variable. If False, continuous colourmap will be applied.
        kwargs:
            Additional keyword arguments passed to cytopy.flow.plotting.single_cell_plot

        Returns
        -------
        Matplotlib.Axes
        """
        dim_reduction_kwargs = dim_reduction_kwargs or {}
        df = self.data[self.data.sample_id == sample_id].copy()
        reducer = DimensionReduction(method=method, n_components=2, **dim_reduction_kwargs)
        df = reducer.fit_transform(data=df, features=self.features)
        return single_cell_plot(
            data=df,
            x=f"{method}1",
            y=f"{method}2",
            label=label,
            discrete=discrete,
            **kwargs,
        )

    def plot_meta_clusters(
        self,
        colour_label: str = "meta_label",
        discrete: bool = True,
        method: str = "UMAP",
        dim_reduction_kwargs: dict or None = None,
        **kwargs,
    ):
        """
        Generate a cluster bubble plot (see cytopy.flow.plotting.cluster_bubble_plot) where each
        data point (bubble) is a single cluster centroid from a unique patient. Size of the data points represents
        the fraction of cells with membership to the sample relative to the total number of events
        in that sample. By default data points are coloured by meta label membership.

        Parameters
        ----------
        method: str
            Dimensionality reduction technique; available methods are: UMAP, PCA, PHATE, KernelPCA or tSNE
        dim_reduction_kwargs: dict, optional
            Additional keyword arguments passed to dimension reduction (see cytopy.flow.dim_reduction)
        colour_label: str, (default='meta_label')
            How to colour cluster centroids
        discrete: bool (default=True)
            If True, label is treated as a discrete variable. If False, continuous colourmap will be applied.
        kwargs:
            Additional keyword arguments passed to cytopy.flow.plotting.cluster_bubble_plot

        Returns
        -------
        Matplotlib.Axes
        """
        return cluster_bubble_plot(
            data=self.data,
            features=self.features,
            cluster_label="cluster_label",
            sample_label="sample_id",
            colour_label=colour_label,
            discrete=discrete,
            dim_reduction_method=method,
            dim_reduction_kwargs=dim_reduction_kwargs,
            **kwargs,
        )

    def clustered_heatmap(
        self,
        features: list,
        sample_id: str or None = None,
        meta_label: bool = True,
        **kwargs,
    ):
        """
        Generate a clustered heatmap (using Seaborn Clustermap function). This function is capable of producing
        different types of heatmaps depending on the input and the clustered dataframe (data attribute):

        * sample_id is None and meta_label is True, will plot the median intensity of each feature for
        each meta cluster
        * sample_id is None and meta_label is False, will plot the median intensity of each feature
        for each cluster label
        * if sample_id is not None, then will plot the median intensity for each feature for each cluster
        for the specified sample

        Default parameters passed to clustermap (overwrite using kwargs):
        * col_cluster = True
        * figsize = (10, 15)
        * standard_scale = 1
        * cmap = "viridis"

        Parameters
        ----------
        features: list
        sample_id: str, optional
        meta_label: bool (default=True)
        kwargs:
            Additional keyword arguments passed to Seaborn.clustermap

        Returns
        -------
        Seaborn.ClusterGrid
        """
        if sample_id is None and meta_label:
            data = self.data.groupby(["meta_label"])[self.features].median()
        elif sample_id is None and not meta_label:
            data = self.data.groupby(["cluster_label"])[self.features].median()
        else:
            data = self.data[self.data.sample_id == sample_id].groupby(["cluster_label"]).median()
        data[features] = data[features].apply(pd.to_numeric)
        kwargs = kwargs or {}
        kwargs["col_cluster"] = kwargs.get("col_cluster", True)
        kwargs["figsize"] = kwargs.get("figsize", (10, 15))
        kwargs["standard_scale"] = kwargs.get("standard_scale", 1)
        kwargs["cmap"] = kwargs.get("cmap", "viridis")
        return sns.clustermap(data[features], **kwargs)

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


def geo_mean(x):
    a = np.array(x)
    return a.prod() ** (1.0 / len(a))


def _assert_unique_label(x):
    assert len(x) == 1, "Chosen label is not unique within clusters"
    return x[0]


def _scatterplot_defaults(**kwargs):
    updated_kwargs = {k: v for k, v in kwargs.items()}
    defaults = {"edgecolor": "black", "alpha": 0.75, "linewidth": 2, "s": 5}
    for k, v in defaults.items():
        if k not in updated_kwargs.keys():
            updated_kwargs[k] = v
    return updated_kwargs
