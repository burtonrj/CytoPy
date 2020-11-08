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

from ...data.experiment import Experiment
from ...data.population import Cluster
from ...feedback import vprint, progress_bar
from ..explore import Explorer
from ..transforms import scaler
from .consensus import ConsensusCluster
from .flowsom import FlowSOM
from multiprocessing import Pool, cpu_count
from sklearn.cluster import *
from sklearn.mixture import *
from hdbscan import HDBSCAN
from functools import partial
from warnings import warn
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


def sklearn_clustering(data: pd.DataFrame,
                       features: list,
                       verbose: bool,
                       method: str,
                       global_clustering: bool = False,
                       **kwargs):
    """
    Perform high-dimensional clustering of single cell data using
    one of the Scikit-Learn unsupervised methods (from the cluster or mixture
    modules) or from a library following the Scikit-Learn template (currently
    we support HDBSCAN). Clustering is performed either on the entire dataframe
    (if global_clustering is True) or on each biological sample, in which case a
    column should be provided called 'sample_id' which this function will group on
    and perform clustering in turn. In both cases, the clustering labels are assigned
    to a new column named 'cluster_id'.

    Parameters
    ----------
    data: Pandas.DataFrame
    features: list
        Columns to peform clustering on
    verbose: bool
        If True, provides a progress bar when global_clustering is False
    method: str
        Name of a valid Scikit-learn cluster or mixture class, or 'HDBSCAN'
    global_clustering: bool (default=False)
        Whether to cluster the whole dataframe or group on 'sample_id' and cluster
        groups
    kwargs:
        Additional keyword arguments passed when initialising Scikit-learn model

    Returns
    -------
    Pandas.DataFrame, None, None
        Modified dataframe with clustering IDs assigned to the column 'cluster_id'
    """
    assert method in globals().keys(), \
        "Not a recognised method from the Scikit-Learn cluster/mixture modules or HDBSCAN"
    model = globals()[method](**kwargs)
    if global_clustering:
        data["cluster_id"] = model.fit_predict(data[features])
        return data, None, None
    for _id, df in progress_bar(data.groupby("sample_id"), verbose=verbose):
        data.loc[df.index, ["cluster_id"]] = model.fit_predict(data[features])
    return data, None, None


def phenograph_clustering(data: pd.DataFrame,
                          features: list,
                          verbose: bool,
                          global_clustering: bool = False,
                          **kwargs):
    """
    Perform high-dimensional clustering of single cell data using the popular
    PhenoGraph algorithm (https://github.com/dpeerlab/PhenoGraph)

    Clustering is performed either on the entire dataframe (if global_clustering is True)
    or on each biological sample, in which case a column should be provided called 'sample_id'
    which this function will group on and perform clustering in turn. In both cases,
    the clustering labels are assigned to a new column named 'cluster_id'.

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
    kwargs:
        Additional keyword arguments passed when calling phenograph.cluster

    Returns
    -------
    Pandas.DataFrame, scipy.sparse.base.spmatrix, float
        Modified dataframe with clustering IDs assigned to the column 'cluster_id', sparse graph
        matrix, and modularity score for communities (Q)
    """
    _print = vprint(verbose=verbose)
    data["cluster_id"] = None
    if global_clustering:
        communities, graph, q = phenograph.cluster(data[features], **kwargs)
        data["cluster_id"] = communities
        return data, graph, q
    graphs = dict()
    q = dict()
    for _id, df in data.groupby("sample_id"):
        _print(f"----- Clustering {_id} -----")
        communities, graph, q_ = phenograph.cluster(df[features], **kwargs)
        graphs[_id], q[_id] = graph, q_
        df["cluster_id"] = communities
        data.loc[df.index, ["cluster_id"]] = df.cluster_id
        _print("-----------------------------")
        _print("\n")
    return data, graphs, q


def _asign_metalabels(data: pd.DataFrame,
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
    return pd.merge(data, metadata[["sample_id", "cluster_id", "meta_label"]], on=["sample_id", "cluster_id"])


def _meta_preprocess(data: pd.DataFrame,
                     features: list,
                     summary_method: callable,
                     norm_method: str or None,
                     **kwargs):
    """
    Summarise the features of the dataframe by grouping on the cluster_id. The summary
    method will be used to describe each cluster (e.g. this could be numpy.mean method
    and would therefore return the mean of each cluster).

    Optionally, data can be normalised prior to applying summary method. To do so provide a
    valid scaling method (see CytoPy.flow.transform.scaler).

    Parameters
    ----------
    data: Pandas.DataFrame
        Clustered data with columns for sample_id and cluster_id
    features: list
        Columns clustering is performed on
    summary_method: callable
        Function to apply to each sample_id/cluster_id group to summarise the
        clusters for meta-clustering
    norm_method: str or None
        If provided, method used to normalise data prior to summarising
    kwargs:
        Additional keyword arguments passed to CytoPy.flow.transform.scaler

    Returns
    -------
    Pandas.DataFrame
        Summarised dataframe
    """
    if norm_method is not None:
        norm_method = partial(scaler, scale_method=norm_method, return_scaler=False, **kwargs)
        data = (data.groupby(["sample_id", "cluster_id"])[features]
                .apply(lambda x: pd.DataFrame(norm_method(x), columns=features))
                .reset_index())
    summary = list()
    for _id, df in data.groupby(["sample_id", "cluster_id"]):
        x = pd.DataFrame(df[features].apply(np.median, axis=0)).T
        x["sample_id"] = _id[0]
        x["cluster_id"] = _id[1]
        summary.append(x)
    metadata = pd.concat(summary)
    metadata["meta_label"] = None
    return metadata


def sklearn_metaclustering(data: pd.DataFrame,
                           features: list,
                           method: str,
                           summary_method: callable = np.median,
                           norm_method: str or None = "norm",
                           norm_kwargs: dict or None = None,
                           verbose: bool = True,
                           **kwargs):
    """
    Meta-clustering with a Scikit-learn clustering/mixture model algorithm. This function
    will summarise the clusters in 'data' (where cluster IDs should be contained in a column
    named 'cluster_id') and then 'cluster the clusters' using the given method.

    Parameters
    ----------
    data: Pandas.DataFrame
        Clustered data with columns for sample_id and cluster_id
    features: list
        Columns clustering is performed on
    method: str
        Name of a valid Scikit-learn cluster or mixture class, or 'HDBSCAN'
    summary_method: callable
        Function to apply to each sample_id/cluster_id group to summarise the
        clusters for meta-clustering
    norm_method: str or None
        If provided, method used to normalise data prior to summarising
    norm_kwargs: dict, optional
        Additional keyword arguments passed to CytoPy.flow.transform.scaler
    verbose: bool (default=True)
        Whether to provide feedback to stdout
    kwargs:
        Keyword arguments for initialising Scikit-learn class

    Returns
    -------
    Pandas.DataFrame, None, None
        Updated dataframe with a new column named 'meta_label' with the meta-clustering
        associations
    """
    vprint_ = vprint(verbose)
    assert method in globals().keys(), \
        "Not a recognised method from the Scikit-Learn cluster/mixture modules or HDBSCAN"
    model = globals()[method](**kwargs)
    vprint_(f"------ {method} meta-clustering ------")
    vprint_("...summarising clusters")
    norm_kwargs = norm_kwargs or {}
    metadata = _meta_preprocess(data, features, summary_method, norm_method, **norm_kwargs)
    vprint_("...clustering the clusters")
    metadata["meta_label"] = model.fit_predict(metadata[features].values)
    vprint_("...assigning meta-labels")
    data = _asign_metalabels(data, metadata)
    vprint_("------ Complete ------")
    return data, None, None


def phenograph_metaclustering(data: pd.DataFrame,
                              features: list,
                              verbose: bool = True,
                              summary_method: callable = np.median,
                              norm_method: str or None = "norm",
                              norm_kwargs: dict or None = None,
                              **kwargs):
    """
    Meta-clustering with a the PhenoGraph algorithm. This function
    will summarise the clusters in 'data' (where cluster IDs should be contained in a column
    named 'cluster_id') and then 'cluster the clusters' using the PhenoGraph.
    Parameters
    ----------
    data: Pandas.DataFrame
        Clustered data with columns for sample_id and cluster_id
    features: list
        Columns clustering is performed on
    summary_method: callable
        Function to apply to each sample_id/cluster_id group to summarise the
        clusters for meta-clustering
    norm_method: str or None
        If provided, method used to normalise data prior to summarising
    norm_kwargs: dict, optional
        Additional keyword arguments passed to CytoPy.flow.transform.scaler
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
    norm_kwargs = norm_kwargs or {}
    metadata = _meta_preprocess(data, features, summary_method, norm_method, **norm_kwargs)
    vprint_("...summarising clusters")
    vprint_("...clustering the clusters")
    communities, graph, q = phenograph.cluster(metadata[features].values, **kwargs)
    metadata["meta_label"] = communities
    vprint_("...assigning meta-labels")
    data = _asign_metalabels(data, metadata)
    vprint_("------ Complete ------")
    return data, graph, q


def consensus_metacluster(data: pd.DataFrame,
                          features: list,
                          cluster_class: object,
                          verbose: bool = True,
                          summary_method: callable = np.median,
                          norm_method: str or None = "norm",
                          smallest_cluster_n: int = 5,
                          largest_cluster_n: int = 50,
                          n_resamples: int = 10,
                          resample_proportion: float = 0.5,
                          **norm_kwargs):
    """
    Meta-clustering with the consensus clustering algorithm, as first described here:
    https://link.springer.com/content/pdf/10.1023%2FA%3A1023949509487.pdf. This function
    will summarise the clusters in 'data' (where cluster IDs should be contained in a column
    named 'cluster_id') and then 'cluster the clusters'. The optimal number of clusters is
    taken as a consensus amongst multiple rounds of clustering with random starts. The algorithm
    used for clustering should be given with 'cluster_class' and should have the Scikit-Learn
    signatures for clustering i.e. fit_predict method.

    Parameters
    ----------
    data: Pandas.DataFrame
        Clustered data with columns for sample_id and cluster_id
    features: list
        Columns clustering is performed on
    summary_method: callable
        Function to apply to each sample_id/cluster_id group to summarise the
        clusters for meta-clustering
    cluster_class: object
        Scikit-learn (or alike) object with the method 'fit_predict'.
    verbose: bool (default=True)
        Whether to provide feedback to stdout
    norm_method: str or None
        If provided, method used to normalise data prior to summarising
    smallest_cluster_n: int (default=5)
        Minimum number of clusters to search for in consensus clustering
    largest_cluster_n: int (default=50)
        Maximum number of clusters to search for in consensus clustering
    n_resamples: int (default=10)
        Number of resampling rounds in consensus clustering
    resample_proportion: float (default=0.5)
        Proportion of data to sample (with replacement) in each round of sampling
        in consensus clustering
    norm_kwargs:
        Additional keyword arguments passed to CytoPy.flow.transform.scaler
    Returns
    -------
    Pandas.DataFrame
        Updated dataframe with a new column named 'meta_label' with the meta-clustering
        associations
    """
    vprint_ = vprint(verbose)
    metadata = _meta_preprocess(data, features, summary_method, norm_method, **norm_kwargs)
    vprint_("----- Consensus meta-clustering ------")
    consensus_clust = ConsensusCluster(cluster=cluster_class,
                                       smallest_cluster_n=smallest_cluster_n,
                                       largest_cluster_n=largest_cluster_n,
                                       n_resamples=n_resamples,
                                       resample_proportion=resample_proportion)
    consensus_clust.fit(metadata[features].values)
    metadata["meta_label"] = consensus_clust.predict_data(metadata[features])
    data = _asign_metalabels(data, metadata)
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
                       meta_cluster_kwargs: dict or None = None):
    """
    Perform high-dimensional clustering of single cell data using the popular
    FlowSOM algorithm (https://pubmed.ncbi.nlm.nih.gov/25573116/). For details
    on the CytoPy implementation of FlowSOM see CytoPy.flow.clustering.flowsom.FlowSOM

    Clustering is performed either on the entire dataframe (if global_clustering is True)
    or on each biological sample, in which case a column should be provided called 'sample_id'
    which this function will group on and perform clustering in turn. In both cases,
    the clustering labels are assigned to a new column named 'cluster_id'.
    Parameters
    ----------
    data: Pandas.DataFrame
        Clustered data with columns for sample_id and cluster_id
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

    Returns
    -------
    Pandas.DataFrame, None, None
        Modified dataframe with clustering IDs assigned to the column 'cluster_id'
    """
    if global_clustering:
        cluster = _flowsom_clustering(data=data,
                                      features=features,
                                      verbose=verbose,
                                      meta_cluster_class=meta_cluster_class,
                                      init_kwargs=init_kwargs,
                                      training_kwargs=training_kwargs,
                                      meta_cluster_kwargs=meta_cluster_kwargs)
        data["cluster_id"] = cluster.predict()
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
        df["cluster_id"] = cluster.predict()
        data.loc[df.index, ["cluster_id"]] = df.cluster_id
    return data, None, None


def _load_data(experiment: Experiment,
               population: str,
               transform: str = "logicle",
               sample_ids: list or None = None,
               verbose: bool = True):
    """
    Load Population from samples in the given Experiment and generate a
    standard clustering dataframe that contains the columns 'sample_id',
    'cluster_id' and 'meta_label'.

    Parameters
    ----------
    experiment: Experiment
    population: str
    transform: str
    sample_ids: list, optional
    verbose: bool (default=True)

    Returns
    -------
    Pandas.DataFrame
    """
    sample_ids = sample_ids or list(experiment.list_samples())
    population_data = list()
    for _id in progress_bar(sample_ids, verbose=verbose):
        fg = experiment.get_sample(sample_id=_id)
        pop = fg.load_population_df(population=population,
                                    transform=transform,
                                    label_downstream_affiliations=True)
        pop["sample_id"] = _id
        population_data.append(pop)
    data = pd.concat([df.reset_index().rename({"index": "original_index"}, axis=1)
                      for df in population_data]).reset_index(drop=True)
    data["cluster_id"] = None
    data["meta_label"] = None
    return data


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
                 tag: str,
                 features: list,
                 sample_ids: list or None = None,
                 root_population: str = "root",
                 transform: str = "logicle",
                 verbose: bool = True):
        self.experiment = experiment
        self.verbose = verbose
        self.print = vprint(verbose)
        self.tag = tag
        self.features = features
        self.transform = transform
        self.root_population = root_population
        self.graph = None
        self.metrics = None
        self.print("Loading single cell data...")
        self.data = _load_data(experiment=experiment,
                               sample_ids=sample_ids,
                               transform=transform,
                               population=root_population)
        self._load_clusters()
        self.print("Ready to cluster!")

    def _load_clusters(self):
        """
        Search the associated Experiment for existing clusters with the same clustering tag.
        If found, populates the relevant rows of 'cluster_id' and 'meta_label' in self.data.

        Returns
        -------
        None
        """
        self.print("Loading existing clusters...")
        for sample_id in progress_bar(list(self.experiment.list_samples()), verbose=self.verbose):
            sample = self.experiment.get_sample(sample_id)
            pop = sample.get_population(self.root_population)
            for cluster in pop.clusters:
                if cluster.tag != self.tag:
                    continue
                idx = self.data[(self.data.sample_id == sample_id) &
                                (self.data.original_index.isin(cluster.index))]
                self.data.loc[idx, ["cluster_id"]] = cluster.cluster_id
                self.data.loc[idx, ["meta_label"]] = cluster.meta_label

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
        will be stored in self.data under the 'cluster_id' column. If the function given uses PhenoGraph,
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

    def meta_cluster(self,
                     func: callable,
                     summary_method: callable = np.median,
                     normalise: str or None = "norm",
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
        of the clusters are then clustered again to group similar clusters. The summary function can be given
        in 'summary_method' but defaults to Numpy.median function.

        The data can also be normalised prior to having the summary method applied. The normalisation method
        should be provided in 'normalise' (see CytoPy.flow.transform.scaler for valid methods)

        Parameters
        ----------
        func: callable
        summary_method: callable
        normalise: str (default="norm")
        kwargs:
            Additional keyword arguments passed to func

        Returns
        -------
        None
        """
        features = self._check_null()
        self.data, self.graph, self.metrics = func(data=self.data,
                                                   samples=list(self.experiment.list_samples()),
                                                   features=features,
                                                   verbose=self.verbose,
                                                   summary_method=summary_method,
                                                   norm_method=normalise,
                                                   **kwargs)

    def rename_meta_clusters(self,
                             mappings):
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

    def _cluster_counts(self):
        """
        Updates the data attribute with an additional column named 'cluster_size',
        which corresponds to the proportion of events in each cluster relative to
        the total events from the original FileGroup

        Returns
        -------
        None
        """
        updated_data = list()
        for _id, df in self.data.groupby("sample_id"):
            cluster_counts = df.cluster_id.value_counts().to_dict()
            df["cluster_size"] = df["cluster_id"].apply(lambda x: cluster_counts.get(x))
            df["cluster_size"] = df["cluster_size"] / df.shape[0]
            updated_data.append(df)
        self.data = pd.concat(updated_data).reset_index(drop=True)

    def explore(self):
        """
        Generate an Explorer object (see CytoPy.flow.explore.Explorer) using the
        associated data with clustering labels

        Returns
        -------
        Explorer
        """
        self._cluster_counts()
        data = self.data.copy()
        return Explorer(data=data)

    def save(self):
        """
        Save the results of the clustering analysis to the Population of each FileGroup
        clustered in this analysis.

        Returns
        -------
        None
        """
        for sample_id in self.data.sample_id:
            sample_df = self.data[self.data.sample_id == sample_id]
            fg = self.experiment.get_sample(sample_id)
            root = fg.get_population(self.root_population)
            for cluster_id in sample_df.cluster_id:
                idx = sample_df[sample_df.cluster_id == cluster_id].original_index.values
                root.add_cluster(Cluster(cluster_id=cluster_id,
                                         meta_label=sample_df[sample_df.cluster_id == cluster_id].meta_label.values[0],
                                         n=len(idx),
                                         index=idx,
                                         prop_of_events=len(idx) / sample_df.shape[0],
                                         tag=self.tag))
            fg.save()
