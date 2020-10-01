from ...data.experiments import Experiment
from ...data.populations import Cluster
from ...feedback import vprint, progress_bar
from ..explore import Explorer
from ..gating_tools import Gating, load_population
from ..transforms import scaler
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


def _asign_metalabels(data: pd.DataFrame,
                      metadata: pd.DataFrame,
                      verbose: bool):
    for sample_id in progress_bar(data.sample_id, verbose=verbose):
        sample_df = data[data.sample_id == sample_id]
        for cluster_id in sample_df.cluster_id:
            meta_label = metadata.loc[(metadata.sample_id == sample_id) & (metadata.cluster_id == cluster_id),
                                      ["meta_label"]]
            data[sample_df[sample_df.cluster_id == cluster_id].index, ["meta_label"]] = meta_label
    return data


def _meta_preprocess(data: pd.DataFrame,
                     features: list,
                     summary_method: callable,
                     norm_method: str or None):
    if norm_method is not None:
        norm_method = partial(scaler, scale_method=norm_method, return_scaled=False)
        data = data.groupby(["sample_id", "cluster_id"])[features].apply(norm_method).reset_index()
    metadata = data.groupby(["sample_id", "cluster_id"])[features].apply(summary_method).reset_index()
    metadata["meta_label"] = None
    return metadata


def phenograph_metaclustering(data: pd.DataFrame,
                              features: list,
                              verbose: bool = True,
                              summary_method: callable = np.median,
                              norm_method: str or None = "norm",
                              **kwargs):
    vprint_ = vprint(verbose)
    metadata = _meta_preprocess(data, features, summary_method, norm_method)
    vprint_("----- Phenograph meta-clustering ------")
    communities, graph, q = phenograph.cluster(data[features], **kwargs)
    metadata["meta_label"] = communities
    vprint_("Assigning clusters...")
    data = _asign_metalabels(data, metadata, verbose)
    return data, graph, q


def consensus_metacluster(data: pd.DataFrame,
                          features: list,
                          cluster_class: callable,
                          verbose: bool = True,
                          summary_method: callable = np.median,
                          norm_method: str or None = "norm",
                          smallest_cluster_n: int = 5,
                          largest_cluster_n: int = 50,
                          n_resamples: int = 10,
                          resample_proportion: float = 0.5):
    vprint_ = vprint(verbose)
    metadata = _meta_preprocess(data, features, summary_method, norm_method)
    vprint_("----- Consensus meta-clustering ------")
    consensus_clust = ConsensusCluster(cluster=cluster_class,
                                       smallest_cluster_n=smallest_cluster_n,
                                       largest_cluster_n=largest_cluster_n,
                                       n_resamples=n_resamples,
                                       resample_proportion=resample_proportion)
    consensus_clust.fit(metadata[features].values)
    metadata["meta_label"] = consensus_clust.predict_data(metadata[features])
    data = _asign_metalabels(data, metadata, verbose)
    return data, None, None


def _flowsom_clustering(data: pd.DataFrame,
                        features: list,
                        verbose: bool,
                        meta_cluster_class: callable,
                        init_kwargs: dict or None = None,
                        training_kwargs: dict or None = None,
                        meta_cluster_kwargs: dict or None = None):
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


def flowsom(data: pd.DataFrame,
            features: list,
            verbose: bool,
            meta_cluster_class: callable,
            global_clustering: bool = False,
            init_kwargs: dict or None = None,
            training_kwargs: dict or None = None,
            meta_cluster_kwargs: dict or None = None):
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
    for _id in data.sample_id:
        vprint_(f"----- Clustering {_id} -----")
        sample_data = data[data.sample_id == _id]
        cluster = _flowsom_clustering(data=data,
                                      features=features,
                                      verbose=verbose,
                                      meta_cluster_class=meta_cluster_class,
                                      init_kwargs=init_kwargs,
                                      training_kwargs=training_kwargs,
                                      meta_cluster_kwargs=meta_cluster_kwargs)
        sample_data["cluster_id"] = cluster.predict()
        data.loc[sample_data.index, ["cluster_id"]] = sample_data.cluster_id
    return data, None, None


class Clustering:
    def __init__(self,
                 experiment: Experiment,
                 tag: str,
                 features: list,
                 root_population: str,
                 transform: str = "logicle",
                 verbose: bool = True,
                 njobs: int = -1):
        self.experiment = experiment
        self.verbose = verbose
        self.print = vprint(verbose)
        self.tag = tag
        self.features = features
        self.transform = transform
        self.root_population = root_population
        self.graph = None
        self.metrics = None
        self.njobs = njobs
        if njobs < 0:
            self.njobs = cpu_count()
        self.data = self._load_data()
        self._load_clusters()

    def _check_null(self) -> list:
        """
        Internal method. Check for null values in the underlying dataframe. Returns a list of column names for columns
        with no missing values.
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

    def _load_clusters(self):
        for sample_id in progress_bar(self.experiment.list_samples(), verbose=self.verbose):
            sample = self.experiment.get_sample(sample_id)
            pop = sample.get_population(self.root_population)
            for cluster in pop.clusters:
                idx = self.data[(self.data.sample_id == sample_id) & (self.data.original_index.isin(cluster.index))]
                self.data.loc[idx, ["cluster_id"]] = cluster.cluster_id
                self.data.loc[idx, ["meta_labek"]] = cluster.meta_label

    def _load_data(self):
        self.print("Loading data...")
        _load = partial(load_population,
                        experiment=self.experiment,
                        population=self.root_population,
                        transform=self.transform,
                        indexed=True,
                        indexed_in_dataframe=True)
        with Pool(self.njobs) as pool:
            n = len(list(self.experiment.list_samples()))
            data = list(progress_bar(pool.imap(_load, self.experiment.list_samples()),
                                     verbose=self.verbose,
                                     total=n))
        data = pd.concat([df.reset_index().rename({"index": "original_index"}, axis=1) for df in data])
        self.print("Labelling clusters...")
        # Load existing clusters
        data["cluster_id"] = None
        data["meta_label"] = None
        self.print("Data loaded successfully!")
        return data

    def cluster(self,
                func: callable,
                samples: list or None = None,
                **kwargs):
        features = self._check_null()
        samples = samples or list(self.experiment.list_samples())
        self.data, self.graph, self.metrics = func(data=self.data,
                                                   samples=samples,
                                                   features=features,
                                                   verbose=self.verbose,
                                                   **kwargs)

    def meta_cluster(self,
                     func: callable,
                     summary_method: callable = np.median,
                     normalise: str or None = "norm",
                     **kwargs):
        features = self._check_null()
        self.data, self.graph, self.metrics = func(data=self.data,
                                                   samples=list(self.experiment.list_samples()),
                                                   features=features,
                                                   verbose=self.verbose,
                                                   summary_method=summary_method,
                                                   norm_method=normalise,
                                                   **kwargs)

    def explore(self,
                include_population_labels: bool = True):
        data = self.data.copy()
        if include_population_labels:
            self.print("Populating with population labels...")
            labeled_data = pd.DataFrame()
            for _id in progress_bar(data.sample_id, verbose=self.verbose):
                g = Gating(experiment=self.experiment, sample_id=_id, include_controls=False)
                labeled_cells = (g.get_labelled_population_df(population_name=self.root_population,
                                                              transform=None)
                                 .reset_index()
                                 .rename({"index": "original_index",
                                          "label": "population_label"}, axis=1)
                                 ["original_index", "population_label"])
                sample_df = data[data.sample_id == _id].merge(labeled_cells, on="original_index")
                labeled_data = pd.concat([labeled_data, sample_df])
                data = labeled_data
        return Explorer(data=data)

    def save(self):
        for sample_id in self.data.sample_id:
            sample_df = self.data[self.data.sample_id == sample_id]
            fg = self.experiment.get_sample(sample_id)
            root = [p for p in fg.populations if p.population_name == self.root_population][0]
            for cluster_id in sample_df.cluster_id:
                idx = sample_df[sample_df.cluster_id == cluster_id].original_index.values
                root.clusters.append(Cluster(cluster_id=cluster_id,
                                             meta_label=sample_df[sample_df.cluster_id == cluster_id].meta_label.values[0],
                                             n=len(idx),
                                             index=idx,
                                             prop_of_events=len(idx)/sample_df.shape[0],
                                             tag=self.tag))
            fg.save()





