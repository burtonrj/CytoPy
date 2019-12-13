from immunova.data.fcs_experiments import FCSExperiment
from immunova.data.fcs import Cluster
from immunova.data.clustering import ClusteringExperiment
from immunova.flow.gating.actions import Gating
from immunova.flow.clustering.plotting import static, interative
from immunova.flow.utilities import progress_bar
import phenograph
import pandas as pd
import numpy as np


class ClusteringError(Exception):
    pass


class Clustering:
    """
    Perform clustering on a single fcs file group
    """
    def __init__(self, clustering_experiment: ClusteringExperiment):
        self.ce = clustering_experiment
        self.clusters = dict()
        self.data = None
        self.experiment = None
        self.sample_id = None

    def _has_data(self):
        if self.data is None:
            raise ClusteringError('Error: no sample is currently associated to this object, before proceeding first '
                                  'associate a sample and its data using the `load_data` method')

    def load_data(self, experiment: FCSExperiment, sample_id: str):
        fg = experiment.pull_sample(sample_id)
        root_p = [p for p in fg.populations if p.population_name == self.ce.root_population]
        if not root_p:
            raise ClusteringError(f'Error: {self.ce.root_population} does not exist for sample {sample_id}')
        if self.ce.clustering_uid in root_p[0].list_clustering_experiments():
            self._load_clusters(root_p[0])
        sample = Gating(experiment, sample_id, include_controls=False)
        transform = True
        if not self.ce.transform_method:
            transform = False
        self.data = sample.get_population_df(self.ce.root_population,
                                             transform=transform,
                                             transform_method=self.ce.transform_method)
        if not self.ce.features:
            self.data = self.data[self.ce.features]

    def _load_clusters(self, root_p):
        """
        Load existing clusters (associated to given cluster UID) associated to the root population of chosen sample.
        """
        clusters = root_p.pull_clusters(self.ce.clustering_uid)
        for c in clusters:
            self.clusters[c.cluster_id] = dict(n_events=c.n_events,
                                               prop_of_root=c.prop_of_root,
                                               index=c.load_index())

    def _add_cluster(self, name: str, indexes: np.array) -> None:
        """
        Internal function. Add new cluster to internal collection of clusters.
        :param name: Name of the cluster
        :param indexes: Indexes corresponding to events in cluster
        :return: None
        """
        self.clusters[name] = dict(index=indexes,
                                   n_events=len(indexes),
                                   prop_of_root=len(indexes)/self.data.shape[0])

    def save_clusters(self):
        """
        Clusters will be saved to the root population of associated sample
        """
        # Is the given UID unique?
        self._has_data()
        fg = self.experiment.pull_sample(self.sample_id)
        root_p = [p for p in fg.populations if p.population_name == self.ce.root_population][0]
        if self.ce.clustering_uid in root_p.list_clustering_experiments():
            raise ClusteringError(f'Error: a clustering experiment with UID {self.ce.clustering_uid} '
                                  f'has already been associated to the root population {self.ce.root_population}')
        clusters = list()
        for name, cluster_data in self.clusters.items():
            c = Cluster(cluster_id=name,
                        n_events=cluster_data['n_events'],
                        prop_of_root=cluster_data['prop_of_root'],
                        cluster_experiment=self.ce)
            c.save_index(cluster_data['index'])
            clusters.append(c)
        # Save the clusters
        root_p.clustering = clusters
        fg.save()

    def get_cluster_dataframe(self, cluster_id: str):
        return self.data[self.data.index.isin(self.clusters[cluster_id]['index'])]

    def cluster(self):
        self._has_data()
        if self.ce.method == 'PhenoGraph':
            self._phenograph()

    def _phenograph(self):
        params = {k: v for k, v in self.ce.parameters}
        communities, self.graph, self.q = phenograph.cluster(self.data, **params)
        for x in np.unique(communities):
            indices = np.where(communities == x)[0]
            label = x
            if self.ce.cluster_prefix is not None:
                label = f'{self.ce.cluster_prefix}_{x}'
            self._add_cluster(label, indices)

    def static_plot_clusters(self, method: str, title: str, save_path: str or None = None,
                             n_components: int = 2, sample_n: int or None = 100000,
                             sample_method: str = 'uniform', heatmap: str or None = None) -> None:
        """
        Generate static plots for clusters. Performs dimensionality reduction and creates a scatter plot with
        events in embedded space coloured by cluster association. If a value is given for the argument `heatmap` then
        a heatmap will also be generated; value can be one of `standard` or `cluster`, the former generates a standard
        heatmap whereas the later creates a heatmap where clusters are 'grouped' by similarity using single linkage.
        :param method: methodology for dimensionality reduction, can be one of: umap, tsne or pca
        :param title: figure title
        :param save_path: path to save figure (optional)
        :param n_components: number of components for dimensionality reduction
        :param sample_n: number of events to sample (default = 100000)
        :param sample_method: sampling method, can be 'uniform' or 'density' where density is density dependent
        downsampling
        :param heatmap: type of heatmap to generate (optional), can be either 'standard' or 'cluster'
        :return: None
        """
        static.plot_clusters(self.data, self.clusters, method=method,
                             sample_n=sample_n, sample_method=sample_method,
                             n_components=n_components, save_path=save_path,
                             title=title)
        if heatmap == 'standard':
            static.heatmap(self.data, self.clusters, title, save_path)
        if heatmap == 'cluster':
            static.clustermap(self.data, self.clusters, title, save_path)

    def interactive_plots(self, title: str, output_path: str) -> None:
        """
        Using the BokehJS library, generates an interactive UMAP plot accompanied with a heatmap.
        :param title: figure title
        :param output_path: path to output HTML file
        :return: None
        """
        interative.umap_heatmap(self.data, self.clusters, title, output_path)


class MetaCluster:
    def __init__(self, experiment: FCSExperiment, root_population: str, clustering_experiment: ClusteringExperiment,
                 samples: str or list = 'all'):
        self.experiment = experiment
        self.ce = clustering_experiment
        self.root_population = root_population
        if samples == 'all':
            samples = experiment.list_samples()
        self.clusters = self.load_clusters(samples)

    def load_clusters(self, samples: list):
        print('--------- Meta Clustering: Loading data ---------')
        print('Each sample will be fetched from the database and a summary matrix created. Each row of this summary '
              'matrix will be a vector describing the centroid (the median of each channel/marker) of each cluster. ')
        columns = self.ce.features + ['sample_id', 'cluster_id']
        clusters = pd.DataFrame(columns=columns)
        for s in progress_bar(samples):
            clustering = Clustering(clustering_experiment=self.ce)
            clustering.load_data(experiment=self.experiment, sample_id=s)
            for c_name in clustering.clusters.keys():
                c_data = clusters.get_cluster_dataframe(c_name)
                c_data = c_data.median()
                c_data['sample_id'], c_data['cluster_id'] = s, c_name
                clusters = clusters.append(c_data, ignore_index=True)
        return clusters
