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

from ...data.experiment import Experiment, load_population_data_from_experiment
from ...data.population import Population
from ...data.subject import Subject
from ...feedback import progress_bar
from ..dim_reduction import DimensionReduction
from ..plotting import single_cell_plot, cluster_bubble_plot
from ..transform import Scaler
from .consensus import ConsensusCluster
from .flowsom import FlowSOM
from sklearn.cluster import *
from typing import List, Dict, Union, Tuple, Type
from sklearn.metrics import calinski_harabasz_score, silhouette_score, davies_bouldin_score
import seaborn as sns
import pandas as pd
import numpy as np
import phenograph
import logging

__author__ = "Ross Burton"
__copyright__ = "Copyright 2020, cytopy"
__credits__ = ["Ross Burton", "Žiga Sajovic", "Simone Cuff", "Andreas Artemiou", "Matthias Eberl"]
__license__ = "MIT"
__version__ = "2.0.0"
__maintainer__ = "Ross Burton"
__email__ = "burtonrj@cardiff.ac.uk"
__status__ = "Production"
logger = logging.getLogger("Clustering")


def clustering_performance(data: pd.DataFrame,
                           labels: list):
    for x in ["Clustering performance...",
              f"Silhouette coefficient: {silhouette_score(data.values, labels, metric='euclidean')}",
              f"Calinski-Harabasz index: {calinski_harabasz_score(data.values, labels)}",
              f"Davies-Bouldin index: {davies_bouldin_score(data.values, labels)}"]:
        print(x)
        logger.info(x)


def sklearn_clustering(data: pd.DataFrame,
                       features: list,
                       verbose: bool,
                       method: str,
                       global_clustering: bool = False,
                       print_performance_metrics: bool = True,
                       **kwargs):
    """
    Perform high-dimensional clustering of single cell data using
    one of the Scikit-Learn unsupervised methods (from the cluster or mixture
    modules) or from a library following the Scikit-Learn template (currently
    we support HDBSCAN). Clustering is performed either on the entire dataframe
    (if global_clustering is True) or on each biological sample, in which case a
    column should be provided called 'sample_id' which this function will group on
    and perform clustering in turn. In both cases, the clustering labels are assigned
    to a new column named 'cluster_label'.

    Parameters
    ----------
    data: Pandas.DataFrame
    features: list
        Columns to perform clustering on
    verbose: bool
        If True, provides a progress bar when global_clustering is False
    method: str
        Name of a valid Scikit-learn cluster or mixture class, or 'HDBSCAN'
    global_clustering: bool (default=False)
        Whether to cluster the whole dataframe or group on 'sample_id' and cluster
        groups
    print_performance_metrics: bool = True
        Print Calinski-Harabasz Index, Silhouette Coefficient, and Davies-Bouldin Index
        (see https://scikit-learn.org/stable/modules/clustering.html#clustering-performance-evaluation)
    kwargs:
        Additional keyword arguments passed when initialising Scikit-learn model

    Returns
    -------
    Pandas.DataFrame and None and None
        Modified dataframe with clustering IDs assigned to the column 'cluster_label'

    Raises
    ------
    ValueError
        Invalid Scikit-Learn or equivalent class provided in method
    """
    if method not in globals().keys():
        logger.error("Not a recognised method from the Scikit-Learn cluster/mixture modules or HDBSCAN")
        raise ValueError("Not a recognised method from the Scikit-Learn cluster/mixture modules or HDBSCAN")

    model = globals()[method](**kwargs)
    if global_clustering:
        logger.info(f"Performing global clustering with the Scikit-Learn clustering algo {method} "
                    f"using features {features}")
        data["cluster_label"] = model.fit_predict(data[features])
        if print_performance_metrics:
            clustering_performance(data[features], data["cluster_label"].values)
        return data, None, None

    logger.info(f"Performing clustering with the Scikit-Learn clustering algo {method} using features {features}")
    for _id, df in progress_bar(data.groupby("sample_id"), verbose=verbose):
        logger.info(f"clustering {_id}")
        df["cluster_label"] = model.fit_predict(df[features])
        data.loc[df.index, ["cluster_label"]] = df["cluster_label"].values
        if print_performance_metrics:
            clustering_performance(df[features], df["cluster_label"].values)
        logger.info(f"clustering complete")
    return data, None, None


def phenograph_clustering(data: pd.DataFrame,
                          features: list,
                          verbose: bool,
                          global_clustering: bool = False,
                          print_performance_metrics: bool = True,
                          **kwargs):
    """
    Perform high-dimensional clustering of single cell data using the popular
    PhenoGraph algorithm (https://github.com/dpeerlab/PhenoGraph)

    Clustering is performed either on the entire dataframe (if global_clustering is True)
    or on each biological sample, in which case a column should be provided called 'sample_id'
    which this function will group on and perform clustering in turn. In both cases,
    the clustering labels are assigned to a new column named 'cluster_label'.

    Parameters
    ----------
    data: Pandas.DataFrame
    features: list
        Columns to peform clustering on
    verbose: bool
        If True, provides a progress bar when global_clustering is False
    global_clustering: bool (default=False)
        Whether to cluster the whole dataframe or group on 'sample_id' and cluster
        groups
    print_performance_metrics: bool = True
        Print Calinski-Harabasz Index, Silhouette Coefficient, and Davies-Bouldin Index
        (see https://scikit-learn.org/stable/modules/clustering.html#clustering-performance-evaluation)
    kwargs:
        Additional keyword arguments passed when calling phenograph.cluster

    Returns
    -------
    Pandas.DataFrame, scipy.sparse.base.spmatrix, float
        Modified dataframe with clustering IDs assigned to the column 'cluster_label', sparse graph
        matrix, and modularity score for communities (Q)
    """
    data["cluster_label"] = None

    if global_clustering:
        logger.info(f"Performing clustering with the PhenoGraph using features {features}")
        communities, graph, q = phenograph.cluster(data[features], **kwargs)
        data["cluster_label"] = communities
        if print_performance_metrics:
            clustering_performance(data[features], data["cluster_label"].values)
        logger.info("PhenoGraph clustering complete")
        return data, graph, q

    graphs = dict()
    q = dict()
    for _id, df in data.groupby("sample_id"):
        logger.info(f"clustering {_id}")
        communities, graph, q_ = phenograph.cluster(df[features], **kwargs)
        graphs[_id], q[_id] = graph, q_
        df["cluster_label"] = communities
        data.loc[df.index, ["cluster_label"]] = df.cluster_label
        if print_performance_metrics:
            clustering_performance(df[features], df["cluster_label"].values)
        logger.info("PhenoGraph clustering complete")
    return data, graphs, q


def _assign_metalabels(data: pd.DataFrame,
                       metadata: pd.DataFrame):
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
    return pd.merge(data, metadata[["sample_id", "cluster_label", "meta_label"]], on=["sample_id", "cluster_label"])


def summarise_clusters(data: pd.DataFrame,
                       features: list,
                       scale: str or None = None,
                       scale_kwargs: dict or None = None,
                       summary_method: str = "median"):
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


def sklearn_metaclustering(data: pd.DataFrame,
                           features: list,
                           method: str,
                           summary_method: str = "median",
                           verbose: bool = True,
                           print_performance_metrics: bool = True,
                           scale_method: str or None = None,
                           scale_kwargs: dict or None = None,
                           **kwargs):
    """
    Meta-clustering with a Scikit-learn clustering/mixture model algorithm. This function
    will summarise the clusters in 'data' (where cluster IDs should be contained in a column
    named 'cluster_label') and then 'cluster the clusters' using the given method.

    Parameters
    ----------
    data: Pandas.DataFrame
        Clustered data with columns for sample_id and cluster_label
    features: list
        Columns clustering is performed on
    method: str
        Name of a valid Scikit-learn cluster or mixture class, or 'HDBSCAN'
    summary_method: str (default="median")
        How to summarise the clusters for meta-clustering
    print_performance_metrics: bool = True
        Print Calinski-Harabasz Index, Silhouette Coefficient, and Davies-Bouldin Index
        (see https://scikit-learn.org/stable/modules/clustering.html#clustering-performance-evaluation)
    verbose: bool (default=True)
        Whether to provide feedback to stdout
    scale_method: str, optional
        Perform scaling of centroids; see cytopy.transform.Scaler
    scale_kwargs: dict, optional
        Additional keyword arguments passed to Scaler
    kwargs:
        Keyword arguments for initialising Scikit-learn class

    Returns
    -------
    Pandas.DataFrame and None and None
        Updated dataframe with a new column named 'meta_label' with the meta-clustering
        associations

    Raises
    ------
    ValueError
        Invalid Scikit-Learn or equivalent class provided in method
    """
    if method not in globals().keys():
        raise ValueError("Not a recognised method from the Scikit-Learn cluster/mixture modules or HDBSCAN")

    model = globals()[method](**kwargs)
    logger.info(f"Meta clustering with {method}")
    metadata = summarise_clusters(data, features, scale_method, scale_kwargs, summary_method)
    metadata["meta_label"] = model.fit_predict(metadata[features].values)

    if print_performance_metrics:
        clustering_performance(metadata[features], metadata["meta_label"].values)
    data = _assign_metalabels(data, metadata)
    logger.info("Meta-clustering complete")

    return data, None, None


def phenograph_metaclustering(data: pd.DataFrame,
                              features: list,
                              verbose: bool = True,
                              summary_method: str = "median",
                              scale_method: str or None = None,
                              scale_kwargs: dict or None = None,
                              print_performance_metrics: bool = True,
                              **kwargs):
    """
    Meta-clustering with a the PhenoGraph algorithm. This function
    will summarise the clusters in 'data' (where cluster IDs should be contained in a column
    named 'cluster_label') and then 'cluster the clusters' using the PhenoGraph.
    Parameters
    ----------
    data: Pandas.DataFrame
        Clustered data with columns for sample_id and cluster_label
    features: list
        Columns clustering is performed on
    summary_method: str (default="median")
        How to summarise the clusters for meta-clustering
    print_performance_metrics: bool = True
        Print Calinski-Harabasz Index, Silhouette Coefficient, and Davies-Bouldin Index
        (see https://scikit-learn.org/stable/modules/clustering.html#clustering-performance-evaluation)
    verbose: bool (default=True)
        Whether to provide feedback to stdout
    scale_method: str, optional
        Perform scaling of centroids; see cytopy.transform.Scaler
    scale_kwargs: dict, optional
        Additional keyword arguments passed to Scaler
    kwargs:
        Keyword arguments passed to phenograph.cluster

    Returns
    -------
    Pandas.DataFrame
        Updated dataframe with a new column named 'meta_label' with the meta-clustering
        associations
    """
    logger.info(f"Meta clustering with PhenoGraph")
    metadata = summarise_clusters(data, features, scale_method, scale_kwargs, summary_method)

    communities, graph, q = phenograph.cluster(metadata[features].values, **kwargs)
    metadata["meta_label"] = communities

    if print_performance_metrics:
        clustering_performance(metadata[features], metadata["meta_label"].values)

    data = _assign_metalabels(data, metadata)
    logger.info("PhenoGraph clustering complete")

    return data, graph, q


def consensus_metacluster(data: pd.DataFrame,
                          features: list,
                          cluster_class: object,
                          verbose: bool = True,
                          summary_method: str = "median",
                          scale_method: str or None = None,
                          scale_kwargs: dict or None = None,
                          smallest_cluster_n: int = 5,
                          largest_cluster_n: int = 15,
                          n_resamples: int = 10,
                          resample_proportion: float = 0.5,
                          print_performance_metrics: bool = True,
                          **kwargs):
    """
    Meta-clustering with the consensus clustering algorithm, as first described here:
    https://link.springer.com/content/pdf/10.1023%2FA%3A1023949509487.pdf. This function
    will summarise the clusters in 'data' (where cluster IDs should be contained in a column
    named 'cluster_label') and then 'cluster the clusters'. The optimal number of clusters is
    taken as a consensus amongst multiple rounds of clustering with random starts. The algorithm
    used for clustering should be given with 'cluster_class' and should have the Scikit-Learn
    signatures for clustering i.e. fit_predict method.

    Parameters
    ----------
    data: Pandas.DataFrame
        Clustered data with columns for sample_id and cluster_label
    features: list
        Columns clustering is performed on
    summary_method: str (default="median")
        How to summarise the clusters for meta-clustering
    cluster_class: object
        Scikit-learn (or alike) object with the method 'fit_predict'.
    verbose: bool (default=True)
        Whether to provide feedback to stdout
    smallest_cluster_n: int (default=5)
        Minimum number of clusters to search for in consensus clustering
    largest_cluster_n: int (default=15)
        Maximum number of clusters to search for in consensus clustering
    n_resamples: int (default=10)
        Number of resampling rounds in consensus clustering
    resample_proportion: float (default=0.5)
        Proportion of data to sample (with replacement) in each round of sampling
        in consensus clustering
    print_performance_metrics: bool = True
        Print Calinski-Harabasz Index, Silhouette Coefficient, and Davies-Bouldin Index
        (see https://scikit-learn.org/stable/modules/clustering.html#clustering-performance-evaluation)
    scale_method: str, optional
        Perform scaling of centroids; see cytopy.transform.Scaler
    scale_kwargs: dict, optional
        Additional keyword arguments passed to Scaler
    kwargs:
        Additional keyword arguments to pass to ConsensusCluster

    Returns
    -------
    Pandas.DataFrame
        Updated dataframe with a new column named 'meta_label' with the meta-clustering
        associations

    Raises
    ------
    ValueError
        If maximum number of meta clusters exceeds the maximum number of clusters identified in any
        one sample
    """
    metadata = summarise_clusters(data, features, scale_method, scale_kwargs, summary_method)
    if (metadata.shape[0] * resample_proportion) < largest_cluster_n:
        err = f"Maximum number of meta clusters (largest_cluster_n) is currently set to " \
              f"{largest_cluster_n} but there are only {metadata.shape[0] * resample_proportion} " \
              f"clusters to cluster in each sample. Either decrease largest_cluster_n or increase " \
              f"resample_proportion."
        logging.error(err)
        raise ValueError(err)

    logger.info("Performing consensus meta-clustering")
    consensus_clust = ConsensusCluster(cluster=cluster_class,
                                       smallest_cluster_n=smallest_cluster_n,
                                       largest_cluster_n=largest_cluster_n,
                                       n_resamples=n_resamples,
                                       resample_proportion=resample_proportion,
                                       **kwargs)

    consensus_clust.fit(metadata[features].values)
    metadata["meta_label"] = consensus_clust.predict_data(metadata[features])

    if print_performance_metrics:
        clustering_performance(metadata[features], metadata["meta_label"].values)

    data = _assign_metalabels(data, metadata)
    logger.info("consensus clustering complete")
    return data, None, None


def _flowsom_clustering(data: pd.DataFrame,
                        features: list,
                        verbose: bool,
                        meta_cluster_class: object,
                        init_kwargs: dict or None = None,
                        training_kwargs: dict or None = None,
                        meta_cluster_kwargs: dict or None = None):
    """
    Wrapper of the FlowSOM method (see cytopy.flow.clustering.flowsom for local
    implementation). Takes a dataframe to cluster and returns a trained FlowSOM
    object, with meta-clustering of SOM nodes performed.

    Parameters
    ----------
    data: Pandas.DataFrame
        Feature space
    features: list
        Columns to perform clustering on
    verbose: bool
        Whether to print output to stdout
    meta_cluster_class: object
        Scikit-learn (or alike) object with the method 'fit_predict'; used for
        consensus clustering of SOM nodes
    init_kwargs: dict, optional
        Additional initialisation keyword parameters for FlowSOM (see cytopy.flow.clustering.flowsom.FlowSOM)
    training_kwargs: dict, optional
        Additional training keyword parameters for FlowSOM
        (see cytopy.flow.clustering.flowsom.FlowSOM.train)
    meta_cluster_kwargs: dict, optional
        Additional meta_cluster keyword parameters for FlowSOM
        (see cytopy.flow.clustering.flowsom.FlowSOM.meta_cluster)

    Returns
    -------
    FlowSOM
    """
    init_kwargs = init_kwargs or {}
    training_kwargs = training_kwargs or {}
    meta_cluster_kwargs = meta_cluster_kwargs or {}
    cluster = FlowSOM(data=data,
                      features=features,
                      verbose=verbose,
                      **init_kwargs)
    cluster.train(**training_kwargs)
    cluster.meta_cluster(cluster_class=meta_cluster_class,
                         **meta_cluster_kwargs)
    return cluster


def flowsom_clustering(data: pd.DataFrame,
                       features: list,
                       verbose: bool,
                       meta_cluster_class: callable,
                       global_clustering: bool = False,
                       init_kwargs: dict or None = None,
                       training_kwargs: dict or None = None,
                       meta_cluster_kwargs: dict or None = None,
                       print_performance_metrics: bool = True):
    """
    Perform high-dimensional clustering of single cell data using the popular
    FlowSOM algorithm (https://pubmed.ncbi.nlm.nih.gov/25573116/). For details
    on the cytopy implementation of FlowSOM see cytopy.flow.clustering.flowsom.FlowSOM

    Clustering is performed either on the entire dataframe (if global_clustering is True)
    or on each biological sample, in which case a column should be provided called 'sample_id'
    which this function will group on and perform clustering in turn. In both cases,
    the clustering labels are assigned to a new column named 'cluster_label'.
    Parameters
    ----------
    data: Pandas.DataFrame
        Clustered data with columns for sample_id and cluster_label
    features: list
        Columns clustering is performed on
    verbose: bool
        Whether to print output to stdout
    meta_cluster_class: object
        Scikit-learn (or alike) object with the method 'fit_predict'; used for
        consensus clustering of SOM nodes
    global_clustering: bool (default=False)
        Whether to cluster the whole dataframe or group on 'sample_id' and cluster
        groups
    init_kwargs: dict, optional
        Additional initialisation keyword parameters for FlowSOM (see cytopy.flow.clustering.flowsom.FlowSOM)
    training_kwargs: dict, optional
        Additional training keyword parameters for FlowSOM
        (see cytopy.flow.clustering.flowsom.FlowSOM.train)
    meta_cluster_kwargs: dict, optional
        Additional meta_cluster keyword parameters for FlowSOM
        (see cytopy.flow.clustering.flowsom.FlowSOM.meta_cluster)
    print_performance_metrics: bool = True
        Print Calinski-Harabasz Index, Silhouette Coefficient, and Davies-Bouldin Index
        (see https://scikit-learn.org/stable/modules/clustering.html#clustering-performance-evaluation)

    Returns
    -------
    Pandas.DataFrame and None and None
        Modified dataframe with clustering IDs assigned to the column 'cluster_label'
    """
    if global_clustering:
        logger.info(f"Performing global clustering with FlowSOM on features {features}")
        cluster = _flowsom_clustering(data=data,
                                      features=features,
                                      verbose=verbose,
                                      meta_cluster_class=meta_cluster_class,
                                      init_kwargs=init_kwargs,
                                      training_kwargs=training_kwargs,
                                      meta_cluster_kwargs=meta_cluster_kwargs)
        data["cluster_label"] = cluster.predict()
        if print_performance_metrics:
            clustering_performance(data[features], data["cluster_label"].values)
        return data, None, None

    logger.info(f"Performing clustering with FlowSOM on features {features}")
    for _id, df in data.groupby("sample_id"):

        logger.info(f"clustering {_id}")
        cluster = _flowsom_clustering(data=df,
                                      features=features,
                                      verbose=verbose,
                                      meta_cluster_class=meta_cluster_class,
                                      init_kwargs=init_kwargs,
                                      training_kwargs=training_kwargs,
                                      meta_cluster_kwargs=meta_cluster_kwargs)

        df["cluster_label"] = cluster.predict()
        if print_performance_metrics:
            clustering_performance(df[features], df["cluster_label"].values)
        data.loc[df.index, ["cluster_label"]] = df.cluster_label
        logger.info("FlowSOM clustering complete")

    return data, None, None


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

    def __init__(self,
                 experiment: Experiment,
                 features: list,
                 sample_ids: list or None = None,
                 root_population: str = "root",
                 transform: str = "logicle",
                 transform_kwargs: dict or None = None,
                 verbose: bool = True,
                 population_prefix: str = "cluster"):
        logger.info(f"Creating new Clustering object with connection to {experiment.experiment_id}")
        self.experiment = experiment
        self.verbose = verbose
        self.features = features
        self.transform = transform
        self.root_population = root_population
        self.graph = None
        self.metrics = None
        self.population_prefix = population_prefix

        logger.info(f"Obtaining data for clustering for population {root_population}")
        self.data = load_population_data_from_experiment(experiment=experiment,
                                                         sample_ids=sample_ids,
                                                         transform=transform,
                                                         transform_kwargs=transform_kwargs,
                                                         population=root_population)
        self.data["meta_label"] = None
        self.data["cluster_label"] = None
        logging.info("Ready to cluster!")

    def _check_null(self,
                    features: Union[List[str], None] = None) -> list:
        """
        Internal method. Check for null values in the underlying dataframe.
        Returns a list of column names for columns with no missing values.

        Parameters
        ----------
        features: Union[List[str], None] (default=None)

        Returns
        -------
        List
            List of valid columns
        """
        features = features or self.features
        null_cols = (self.data[features]
                     .isnull()
                     .sum()
                     [self.data[features].isnull().sum() > 0]
                     .index
                     .values)
        if null_cols.size != 0:
            logger.warning(f'The following columns contain null values and will be excluded from '
                           f'clustering analysis: {null_cols}')
        return [x for x in features if x not in null_cols]

    def cluster(self,
                func: callable,
                overwrite_features: Union[List[str], None] = None,
                **kwargs):
        """
        Perform clustering with a suitable clustering function from cytopy.flow.clustering.main:
            * sklearn_clustering - access any of the Scikit-Learn cluster/mixture classes for unsupervised learning;
              currently also provides access to HDBSCAN
            * phenograph_clustering - access to the PhenoGraph clustering algorithm
            * flowsom_clustering - access to the FlowSOM clustering algorithm
        See documentation for specific function parameters. Parameters can be provided in kwargs. Results
        will be stored in self.data under the 'cluster_label' column. If the function given uses PhenoGraph,
        the sparse graph matrix will be stored in self.graph and the modularity score in self.metrics

        Parameters
        ----------
        func: callable
        overwrite_features: List[str], optional
            Uses this list of features instead of those in the 'features' attribute. This can be
            useful if the 'dimension_reduction' function has been used and you wish to perform global
            clustering on the latent variables.
        kwargs:
            Additional keyword arguments passed to the given clustering function

        Returns
        -------
        self
        """
        features = self._check_null(overwrite_features)
        self.data, self.graph, self.metrics = func(data=self.data,
                                                   features=features,
                                                   verbose=self.verbose,
                                                   **kwargs)
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

    def meta_cluster(self,
                     func: callable,
                     summary_method: str = "median",
                     scale_method: str or None = None,
                     scale_kwargs: dict or None = None,
                     **kwargs):
        """
        Perform meta-clustering using one of the meta-clustering functions from
        cytopy.flow.clustering.main:
        * sklearn_metaclustering
        * phenograph_metaclustering
        * consensus_metaclustering

        Or a valid function as defined in the developer docs.

        See documentation for specific function parameters. Parameters can be provided in kwargs. Results
        will be stored in self.data under the 'meta_label' column. If the function given uses PhenoGraph,
        the sparse graph matrix will be stored in self.graph and the modularity score in self.metrics.
        (Note: this overwrites existing values for 'graph' and 'metrics')

        Prior to meta-clustering, the clustered dataframe (stored in 'data') will be summarised. The summary
        of the clusters are then clustered again to group similar clusters. The summary method should either be
        'median' or 'mean', averaging the observations of each cluster.

        The data can also be normalised prior to having the summary method applied. The normalisation method
        should be provided in 'normalise' (see cytopy.flow.transform.scaler for valid methods)

        Parameters
        ----------
        func: callable
        summary_method: str (default="median")
        scale_method: str, optional
            Perform scaling of centroids; see cytopy.transform.Scaler
        scale_kwargs: dict, optional
            Additional keyword arguments passed to Scaler
        kwargs:
            Additional keyword arguments passed to func

        Returns
        -------
        None
        """
        features = self._check_null()
        self.data, self.graph, self.metrics = func(data=self.data,
                                                   features=features,
                                                   verbose=self.verbose,
                                                   summary_method=summary_method,
                                                   scale_method=scale_method,
                                                   scale_kwargs=scale_kwargs,
                                                   **kwargs)

    def rename_clusters(self,
                        sample_id: str,
                        mappings: dict):
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
        self.data[self.data.sample_id == sample_id]["cluster_label"].replace(mappings, inplace=True)

    def rename_meta_clusters(self,
                             mappings: dict):
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

    def load_meta_variable(self,
                           variable: str,
                           verbose: bool = True,
                           embedded: list or None = None):
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
        for _id in progress_bar(self.data.subject_id.unique(),
                                verbose=verbose):
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
                logger.warning(f'{_id} is missing meta-variable {variable}')
                self.data.loc[self.data.subject_id == _id, variable] = None

    def dimension_reduction(self,
                            method: Union[str, Type],
                            n_components: int = 2,
                            **kwargs):
        """
        Perform dimension reduction on associated data. This will generate new latent features
        that can be used for clustering. Latent variables will be stored as indexed columns in
        data attribute named according to the method e.g. UMAP1, UMAP2 etc

        Parameters
        ----------
        method: str
            Valid dimension reduction method (see cytopy.flow.dimension_reduction)
        n_components: int (default=2)
            Number of components to generate
        kwargs:
            Additional keyword arguments passed to DimensionReduction

        Returns
        -------
        None
        """
        reducer = DimensionReduction(method=method,
                                     n_components=n_components,
                                     **kwargs)
        self.data = reducer.fit_transform(data=self.data, features=self.features)

    def single_cell_plot(self,
                         sample_size: Union[int, None] = 100000,
                         sampling_method: str = "uniform",
                         method: Union[str, Type] = "UMAP",
                         dim_reduction_kwargs: dict or None = None,
                         label: str = "cluster_label",
                         discrete: bool = True,
                         **kwargs):
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
                plot_data = list()
                n = int(sample_size / self.data.sample_id.nunique())
                for _, sample_data in self.data.groupby("sample_id"):
                    if n >= sample_data.shape[0]:
                        plot_data.append(sample_data)
                    else:
                        plot_data.append(sample_data.sample(n))
                plot_data = pd.concat(plot_data)
            else:
                if sample_size < self.data.shape[0]:
                    plot_data = self.data.sample(sample_size)

        reducer = DimensionReduction(method=method,
                                     n_components=2,
                                     **dim_reduction_kwargs)
        df = reducer.fit_transform(data=plot_data, features=self.features)
        return single_cell_plot(data=df,
                                x=f"{method}1",
                                y=f"{method}2",
                                label=label,
                                discrete=discrete,
                                **kwargs)

    def plot_sample_clusters(self,
                             sample_id: str,
                             method: Union[str, Type] = "UMAP",
                             dim_reduction_kwargs: dict or None = None,
                             label: str = "cluster_label",
                             discrete: bool = True,
                             **kwargs):
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
        reducer = DimensionReduction(method=method,
                                     n_components=2,
                                     **dim_reduction_kwargs)
        df = reducer.fit_transform(data=df, features=self.features)
        return single_cell_plot(data=df,
                                x=f"{method}1",
                                y=f"{method}2",
                                label=label,
                                discrete=discrete,
                                **kwargs)

    def plot_meta_clusters(self,
                           colour_label: str = "meta_label",
                           discrete: bool = True,
                           method: str = "UMAP",
                           dim_reduction_kwargs: dict or None = None,
                           **kwargs):
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
        return cluster_bubble_plot(data=self.data,
                                   features=self.features,
                                   cluster_label="cluster_label",
                                   sample_label="sample_id",
                                   colour_label=colour_label,
                                   discrete=discrete,
                                   dim_reduction_method=method,
                                   dim_reduction_kwargs=dim_reduction_kwargs,
                                   **kwargs)

    def clustered_heatmap(self,
                          features: list,
                          sample_id: str or None = None,
                          meta_label: bool = True,
                          **kwargs):
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

    def save(self, verbose: bool = True, population_var: str = "meta_label"):
        """
        Clusters are saved as new Populations in each FileGroup in the attached Experiment
        according to the sample_id in data.

        Parameters
        ----------
        verbose: bool (default=True)
        population_var: str (default='meta_label')
            Variable in data that should be used to identify individual Populations

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
        for sample_id in progress_bar(self.data.sample_id.unique(), verbose=verbose):
            fg = self.experiment.get_sample(sample_id)
            sample_data = self.data[self.data.sample_id == sample_id].copy()
            for cluster_label, cluster in sample_data.groupby(population_var):
                population_name = str(cluster_label)
                if self.population_prefix is not None:
                    population_name = f"{self.population_prefix}_{cluster_label}"
                pop = Population(population_name=population_name,
                                 n=cluster.shape[0],
                                 parent=self.root_population,
                                 source="cluster",
                                 signature=cluster.mean().to_dict())
                pop.index = cluster.original_index.values
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
    defaults = {"edgecolor": "black",
                "alpha": 0.75,
                "linewidth": 2,
                "s": 5}
    for k, v in defaults.items():
        if k not in updated_kwargs.keys():
            updated_kwargs[k] = v
    return updated_kwargs
