from ...data.clustering import ClusteringExperiment
from ..gating_tools import load_population
from ...feedback import vprint, progress_bar
from .consensus import ConsensusCluster
from .flowsom import FlowSOM
from multiprocessing import Pool, cpu_count
from functools import partial
from warnings import warn
import pandas as pd
import numpy as np
import phenograph


def phenograph_clustering(data: pd.DataFrame,
                          features: list,
                          verbose: bool,
                          global_clustering: bool = False,
                          **kwargs):
    if global_clustering:
        communities, graph, q = phenograph.cluster(data[features], **kwargs)
        data["cluster_id"] = communities
        return data, graph, q
    graphs = dict()
    q = dict()
    for _id in progress_bar(data.sample_id, verbose=verbose):
        sample_data = data[data.sample_id == _id]
        communities, graph, q_ = phenograph.cluster(sample_data[features], **kwargs)
        graphs[_id], q[_id] = graph, q_
        sample_data["cluster_id"] = communities
        data.loc[sample_data.index, ["cluster_id"]] = sample_data.cluster_id
    return data, graphs, q


def phenograph_metaclustering(data: pd.DataFrame,
                              features: list,
                              verbose: bool = False,
                              summary_method: callable = np.median,
                              **kwargs):
    metadata = data.groupby(["sample_id", "cluster_id"])[features].apply(summary_method).reset_index()
    metadata["meta_label"] = None
    communities, graph, q = phenograph.cluster(data[features], **kwargs)
    metadata["meta_label"] = communities
    for sample_id in data.sample_id:
        sample_df = data[data.sample_id == sample_id]
        for cluster_id in sample_df.cluster_id:
            meta_label = metadata.loc[(metadata.sample_id == sample_id) & (metadata.cluster_id == cluster_id),
                                      ["meta_label"]]
            data[sample_df[sample_df.cluster_id == cluster_id].index, ["meta_label"]] = meta_label
    return data, graph, q

def


class Cluster:
    def __init__(self,
                 clustering_experiment: ClusteringExperiment,
                 verbose: bool = True,
                 njobs: int = -1):
        self.ce = clustering_experiment
        self.verbose = verbose
        self.print = vprint(verbose)
        self.graph = None
        self.metrics = None
        self.njobs = njobs
        if njobs < 0:
            self.njobs = cpu_count()
        self.data = self._load_data()

    def _check_null(self) -> list:
        """
        Internal method. Check for null values in the underlying dataframe. Returns a list of column names for columns
        with no missing values.
        Returns
        -------
        List
            List of valid columns
        """
        null_cols = (self.data[self.ce.features]
                     .isnull()
                     .sum()
                     [self.data[self.ce.features].isnull().sum() > 0]
                     .index
                     .values)
        if null_cols.size != 0:
            warn(f'The following columns contain null values and will be excluded from '
                 f'clustering analysis: {null_cols}')
        return [x for x in self.ce.features if x not in null_cols]

    def _load_data(self):
        self.print("Loading data...")
        _load = partial(load_population,
                        experiment=self.ce.experiment,
                        population=self.ce.root_population,
                        transform=self.ce.transform_method,
                        indexed=True,
                        indexed_in_dataframe=True)
        with Pool(self.njobs) as pool:
            n = len(list(self.ce.experiment.list_samples()))
            data = list(progress_bar(pool.imap(_load, self.ce.experiment.list_samples()),
                                     verbose=self.verbose,
                                     total=n))
        data = pd.concat([df.reset_index().rename({"index": "original_index"}, axis=1) for df in data])
        self.print("Labelling clusters...")
        # Load existing clusters
        data["cluster_id"] = None
        data["meta_label"] = None
        for cluster in self.ce.clusters:
            idx = data[(data.sample_id == cluster.file.primary_id) & (data.original_index.isin(cluster.index))]
            data.loc[idx, ["cluster_id"]] = cluster.label
            data.loc[idx, ["meta_label"]] = cluster.meta_label
        self.print("Data loaded successfully!")
        return data

    def cluster(self,
                func: callable,
                samples: list or None = None,
                **kwargs):
        features = self._check_null()
        samples = samples or list(self.ce.experiment.list_samples())
        self.data, self.graph, self.metrics = func(data=self.data,
                                                   samples=samples,
                                                   features=features,
                                                   verbose=self.verbose,
                                                   **kwargs)

    def meta_cluster(self,
                     func: callable,
                     **kwargs):
        features = self._check_null()
        self.data, self.graph, self.metrics = func(data=self.data,
                                                   samples=list(self.ce.experiment.list_samples()),
                                                   features=features,
                                                   verbose=self.verbose,
                                                   **kwargs)

    def save(self):
        for sample_id in self.data.sample_id:
            sample_df = self.data[self.data.sample_id == sample_id]
            fg = self.ce.experiment.get_sample(sample_id)
            for cluster_id in sample_df.cluster_id:
                if self.ce.prefix:
                    cluster_id = "_".join([self.ce.prefix, cluster_id])
                self.ce.add_cluster(cluster_id=cluster_id,
                                    file=fg,
                                    meta_label=sample_df[sample_df.cluster_id == cluster_id].meta_label.values[0],
                                    cluster_idx=sample_df[sample_df.cluster_id == cluster_id].original_index.values,
                                    root_n=sample_df.shape[0])
        self.ce.save()




