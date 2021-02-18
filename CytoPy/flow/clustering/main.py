#!/usr/bin.env/python
# -*- coding: utf-8 -*-
"""
High-dimensional clustering offers the advantage of an unbiased approach
to classification of single cells whilst also exploiting all available variables
in your data (all your fluorochromes/isotypes). In CytoPy, the clustering is
performed on a Population of a FileGroup. The resulting clusters are saved
to that Population. Multiple clustering methods can be tried on a single
Population and be tracked with a 'tag'. Additionally, we can compare the
clustering results of many FileGroup's by 'clustering the clusters', to do
this we summarise their clusters and perform meta-clustering. In this module
you will find the Clustering class, which is the apparatus to apply a
clustering method in CytoPy and save the results to the database. We also
provide implementations of PhenoGraph, FlowSOM and provide access to any
of the clustering methods available throught the Scikit-Learn API.

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
from ...feedback import vprint, progress_bar
from ..dim_reduction import dimensionality_reduction
from ..plotting import single_cell_plot, cluster_bubble_plot
from ..transform import Scaler
from .consensus import ConsensusCluster
from .flowsom import FlowSOM
from sklearn.cluster import *
from sklearn.metrics import calinski_harabasz_score, silhouette_score, davies_bouldin_score
from warnings import warn
import seaborn as sns
import pandas as pd
import numpy as np
import phenograph

__author__ = "Ross Burton"
__copyright__ = "Copyright 2020, CytoPy"
__credits__ = ["Ross Burton", "Žiga Sajovic", "Simone Cuff", "Andreas Artemiou", "Matthias Eberl"]
__license__ = "MIT"
__version__ = "1.0.0"
__maintainer__ = "Ross Burton"
__email__ = "burtonrj@cardiff.ac.uk"
__status__ = "Production"


def clustering_performance(data: pd.DataFrame,
                           labels: list):
    print("Clustering performance...")
    print(f"Silhouette coefficient: {silhouette_score(data.values, labels, metric='euclidean')}")
    print(f"Calinski-Harabasz index: {calinski_harabasz_score(data.values, labels)}")
    print(f"Davies-Bouldin index: {davies_bouldin_score(data.values, labels)}")


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
    """
    assert method in globals().keys(), \
        "Not a recognised method from the Scikit-Learn cluster/mixture modules or HDBSCAN"
    model = globals()[method](**kwargs)
    if global_clustering:
        data["cluster_label"] = model.fit_predict(data[features])
        if print_performance_metrics:
            clustering_performance(data[features], data["cluster_label"].values)
        return data, None, None
    for _id, df in progress_bar(data.groupby("sample_id"), verbose=verbose):
        data.loc[df.index, ["cluster_label"]] = model.fit_predict(df[features])
        if print_performance_metrics:
            clustering_performance(df[features], df["cluster_label"].values)
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
    _print = vprint(verbose=verbose)
    data["cluster_label"] = None
    if global_clustering:
        communities, graph, q = phenograph.cluster(data[features], **kwargs)
        data["cluster_label"] = communities
        if print_performance_metrics:
            clustering_performance(data[features], data["cluster_label"].values)
        return data, graph, q
    graphs = dict()
    q = dict()
    for _id, df in data.groupby("sample_id"):
        _print(f"----- Clustering {_id} -----")
        communities, graph, q_ = phenograph.cluster(df[features], **kwargs)
        graphs[_id], q[_id] = graph, q_
        df["cluster_label"] = communities
        data.loc[df.index, ["cluster_label"]] = df.cluster_label
        if print_performance_metrics:
            clustering_performance(df[features], df["cluster_label"].values)
        _print("-----------------------------")
        _print("\n")
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


def _summarise_clusters(data: pd.DataFrame,
                        features: list,
                        scale: str or None = None,
                        scale_kwargs: dict or None = None,
                        summary_method: str = "median"):
    """

    Parameters
    ----------
    data
    features
    summary_method

    Returns
    -------

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
    kwargs:
        Keyword arguments for initialising Scikit-learn class

    Returns
    -------
    Pandas.DataFrame and None and None
        Updated dataframe with a new column named 'meta_label' with the meta-clustering
        associations
    """
    vprint_ = vprint(verbose)
    assert method in globals().keys(), \
        "Not a recognised method from the Scikit-Learn cluster/mixture modules or HDBSCAN"
    model = globals()[method](**kwargs)
    vprint_(f"------ {method} meta-clustering ------")
    vprint_("...summarising clusters")
    metadata = _summarise_clusters(data, features, scale_method, scale_kwargs, summary_method)
    vprint_("...clustering the clusters")
    metadata["meta_label"] = model.fit_predict(metadata[features].values)
    if print_performance_metrics:
        clustering_performance(metadata[features], metadata["meta_label"].values)
    vprint_("...assigning meta-labels")
    data = _assign_metalabels(data, metadata)
    vprint_("------ Complete ------")
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
    kwargs:
        Keyword arguments passed to phenograph.cluster

    Returns
    -------
    Pandas.DataFrame
        Updated dataframe with a new column named 'meta_label' with the meta-clustering
        associations
    """
    vprint_ = vprint(verbose)
    vprint_("----- Phenograph meta-clustering ------")
    metadata = _summarise_clusters(data, features, scale_method, scale_kwargs, summary_method)
    vprint_("...summarising clusters")
    vprint_("...clustering the clusters")
    communities, graph, q = phenograph.cluster(metadata[features].values, **kwargs)
    metadata["meta_label"] = communities
    if print_performance_metrics:
        clustering_performance(metadata[features], metadata["meta_label"].values)
    vprint_("...assigning meta-labels")
    data = _assign_metalabels(data, metadata)
    vprint_("------ Complete ------")
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
    kwargs:
        Additional keyword arguments to pass to ConsensusCluster
    Returns
    -------
    Pandas.DataFrame
        Updated dataframe with a new column named 'meta_label' with the meta-clustering
        associations
    """
    vprint_ = vprint(verbose)
    metadata = _summarise_clusters(data, features, scale_method, scale_kwargs, summary_method)
    assert (metadata.shape[0] * resample_proportion) > largest_cluster_n, \
        f"Maximum number of meta clusters (largest_cluster_n) is currently set to {largest_cluster_n} but there are " \
        f"only {metadata.shape[0] * resample_proportion} clusters to cluster in each sample. Either decrease " \
        f"largest_cluster_n or increase resample_proportion."
    vprint_("----- Consensus meta-clustering ------")
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
    return data, None, None


def _flowsom_clustering(data: pd.DataFrame,
                        features: list,
                        verbose: bool,
                        meta_cluster_class: object,
                        init_kwargs: dict or None = None,
                        training_kwargs: dict or None = None,
                        meta_cluster_kwargs: dict or None = None):
    """
    Wrapper of the FlowSOM method (see CytoPy.flow.clustering.flowsom for local
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
        Additional initialisation keyword parameters for FlowSOM (see CytoPy.flow.clustering.flowsom.FlowSOM)
    training_kwargs: dict, optional
        Additional training keyword parameters for FlowSOM
        (see CytoPy.flow.clustering.flowsom.FlowSOM.train)
    meta_cluster_kwargs: dict, optional
        Additional meta_cluster keyword parameters for FlowSOM
        (see CytoPy.flow.clustering.flowsom.FlowSOM.meta_cluster)

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
    on the CytoPy implementation of FlowSOM see CytoPy.flow.clustering.flowsom.FlowSOM

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
        Additional initialisation keyword parameters for FlowSOM (see CytoPy.flow.clustering.flowsom.FlowSOM)
    training_kwargs: dict, optional
        Additional training keyword parameters for FlowSOM
        (see CytoPy.flow.clustering.flowsom.FlowSOM.train)
    meta_cluster_kwargs: dict, optional
        Additional meta_cluster keyword parameters for FlowSOM
        (see CytoPy.flow.clustering.flowsom.FlowSOM.meta_cluster)
    print_performance_metrics: bool = True
        Print Calinski-Harabasz Index, Silhouette Coefficient, and Davies-Bouldin Index
        (see https://scikit-learn.org/stable/modules/clustering.html#clustering-performance-evaluation)

    Returns
    -------
    Pandas.DataFrame and None and None
        Modified dataframe with clustering IDs assigned to the column 'cluster_label'
    """
    if global_clustering:
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
    vprint_ = vprint(verbose)
    for _id, df in data.groupby("sample_id"):
        vprint_(f"----- Clustering {_id} -----")
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
        vprint_("\n")
    return data, None, None


class Clustering:
    """
    High-dimensional clustering offers the advantage of an unbiased approach
    to classification of single cells whilst also exploiting all available variables
    in your data (all your fluorochromes/isotypes). In CytoPy, the clustering is
    performed on a Population of a FileGroup. The resulting clusters are saved
    to that Population. Multiple clustering methods can be tried on a single
    Population and be tracked with a 'tag'. Additionally, we can compare the
    clustering results of many FileGroup's by 'clustering the clusters', to do
    this we summarise their clusters and perform meta-clustering.

    The Clustering class provides all the apparatus to perform high-dimensional clustering
    using any of the following functions from the CytoPy.flow.clustering.main module:

    * sklearn_clustering - access any of the Scikit-Learn cluster/mixture classes for unsupervised learning;
      currently also provides access to HDBSCAN
    * phenograph_clustering - access to the PhenoGraph clustering algorithm
    * flowsom_clustering - access to the FlowSOM clustering algorithm

    In addition, meta-clustering (clustering or clusters) can be performed with any of the following from
    the same module:
    * sklearn_metaclustering
    * phenograph_metaclustering
    * consensus_metaclustering

    Attributes
    ----------
    experiment: Experiment
        Experiment to access for FileGroups to be clustered
    tag: str
        General identify associated to all clusters produced by this object. Can be
        used to retrieve specific clusters from database at later date
    features: list
        Features (fluorochromes/cell markers) to use for clustering
    sample_ids: list, optional
        Name of FileGroups load from Experiment and cluster. If not given, will load all
        samples from Experiment.
    root_population: str (default="root")
        Name of the Population to use as input data for clustering
    transform: str (default="logicle")
        How to transform the data prior to clustering
    verbose: bool (default=True)
        Whether to provide output to stdout
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
        self.experiment = experiment
        self.verbose = verbose
        self.print = vprint(verbose)
        self.features = features
        self.transform = transform
        self.root_population = root_population
        self.graph = None
        self.metrics = None
        self.population_prefix = population_prefix
        self.print("Loading single cell data...")
        self.data = load_population_data_from_experiment(experiment=experiment,
                                                         sample_ids=sample_ids,
                                                         transform=transform,
                                                         transform_kwargs=transform_kwargs,
                                                         population=root_population)
        self.data["meta_label"] = None
        self.data["cluster_label"] = None
        self.print("Ready to cluster!")

    def _check_null(self) -> list:
        """
        Internal method. Check for null values in the underlying dataframe.
        Returns a list of column names for columns with no missing values.

        Returns
        -------
        List
            List of valid columns
        """
        null_cols = (self.data[self.features]
                     .isnull()
                     .sum()
                     [self.data[self.features].isnull().sum() > 0]
                     .index
                     .values)
        if null_cols.size != 0:
            warn(f'The following columns contain null values and will be excluded from '
                 f'clustering analysis: {null_cols}')
        return [x for x in self.features if x not in null_cols]

    def cluster(self,
                func: callable,
                **kwargs):
        """
        Perform clustering with a suitable clustering function from CytoPy.flow.clustering.main:
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
        kwargs:
            Additional keyword arguments passed to the given clustering function

        Returns
        -------
        None
        """
        features = self._check_null()
        self.data, self.graph, self.metrics = func(data=self.data,
                                                   features=features,
                                                   verbose=self.verbose,
                                                   **kwargs)

    def reset_clusters(self):
        self.data["cluster_label"] = None
        self.data["meta_label"] = None

    def reset_meta_clusters(self):
        self.data["meta_label"] = None

    def meta_cluster(self,
                     func: callable,
                     summary_method: str = "median",
                     scale_method: str or None = None,
                     scale_kwargs: dict or None = None,
                     **kwargs):
        """
        Perform meta-clustering using one of the meta-clustering functions from
        CytoPy.flow.clustering.main:
        * sklearn_metaclustering
        * phenograph_metaclustering
        * consensus_metaclustering

        See documentation for specific function parameters. Parameters can be provided in kwargs. Results
        will be stored in self.data under the 'meta_label' column. If the function given uses PhenoGraph,
        the sparse graph matrix will be stored in self.graph and the modularity score in self.metrics.
        (Note: this overwrites existing values for 'graph' and 'metrics')

        Prior to meta-clustering, the clustered dataframe (stored in 'data') will be summarised. The summary
        of the clusters are then clustered again to group similar clusters. The summary method should either be
        'median' or 'mean', averaging the observations of each cluster.

        The data can also be normalised prior to having the summary method applied. The normalisation method
        should be provided in 'normalise' (see CytoPy.flow.transform.scaler for valid methods)

        Parameters
        ----------
        func: callable
        summary_method: str (default="median")
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
                warn(f'{_id} is missing meta-variable {variable}')
                self.data.loc[self.data.subject_id == _id, variable] = None

    def plot_sample_clusters(self,
                             sample_id: str,
                             method: str = "UMAP",
                             dim_reduction_kwargs: dict or None = None,
                             label: str = "cluster_label",
                             discrete: bool = True,
                             **kwargs):
        dim_reduction_kwargs = dim_reduction_kwargs or {}
        df = self.data[self.data.sample_id == sample_id].copy()
        df = dimensionality_reduction(data=df,
                                      features=self.features,
                                      n_components=2,
                                      return_reducer=False,
                                      return_embeddings_only=False,
                                      method=method,
                                      **dim_reduction_kwargs)
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
                          **kwargs):
        if sample_id is None:
            data = self.data.groupby(["meta_label"])[self.features].median()
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
        if population_var == "meta_label":
            assert not self.data.meta_label.isnull().all(), "Meta clustering has not been performed"
        for sample_id in progress_bar(self.data.sample_id.unique(), verbose=verbose):
            fg = self.experiment.get_sample(sample_id)
            sample_data = self.data[self.data.sample_id == sample_id]
            for cluster_label, cluster in sample_data.groupby(population_var):
                population_name = cluster_label
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
    return a.prod()**(1.0/len(a))


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