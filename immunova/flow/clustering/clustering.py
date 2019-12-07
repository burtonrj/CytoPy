from immunova.data.fcs_experiments import FCSExperiment
from immunova.data.fcs import ClusteringExperiment, Cluster
from immunova.flow.gating.actions import Gating
from immunova.flow.clustering.plotting import static, interative
import numpy as np


class ClusteringError(Exception):
    pass


class Clustering:
    """
    Perform clustering on a single fcs file group
    """
    def __init__(self, clustering_uid: str, experiment: FCSExperiment, sample_id: str, root_population: str,
                 n_jobs: int = -1, transform: bool = True,
                 transform_method: str = 'logicle', features: str or list = 'all'):
        self.clustering_uid = clustering_uid
        self.n_jobs = n_jobs
        self.clusters = dict()
        self.parameters = dict()
        self.method = None
        self.transform_method = transform_method
        self.root_population = root_population
        self.experiment = experiment
        self.sample_id = sample_id
        fg = self.experiment.pull_sample(self.sample_id)
        root_p = [p for p in fg.populations if p.population_name == self.root_population][0]
        if clustering_uid in root_p.list_clustering_experiments():
            print(f'Loading existing clustering experiment...')
            self._load_clusters(root_p, clustering_uid)
        self._load_data(transform=transform, features=features)

    def _load_data(self, transform: bool, features: str or list):
        """
        Internal function. Load data from sample.
        :param transform: If True transformation method specified in the objects 'transform_method' parameter is
        applied
        :param features: list of features to be included; alternatively if value is 'all' then all features will be
        used
        """
        sample = Gating(self.experiment, self.sample_id, include_controls=False)
        self.data = sample.get_population_df(self.root_population,
                                             transform=transform,
                                             transform_method=self.transform_method)
        if self.data is None:
            raise ClusteringError(f'Error: was unable to retrieve data for {self.sample_id} invalid root population, '
                                  f'{sample.populations.keys()}')
        if features != 'all':
            self.data = self.data[features]

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

    def save(self):
        """
        Clusters will be saved to the root population of associated sample
        """
        # Is the given UID unique?
        fg = self.experiment.pull_sample(self.sample_id)
        root_p = [p for p in fg.populations if p.population_name == self.root_population][0]
        if self.clustering_uid in root_p.list_clustering_experiments():
            print(f'Error: a clustering experiment with UID {self.clustering_uid} has already been associated to '
                  f'the root population {self.root_population}')
            return

        # Create objects for saving
        params = [(k, v) for k, v in self.parameters.items()]
        exp = ClusteringExperiment(clustering_uid=self.clustering_uid,
                                   method=self.method,
                                   parameters=params,
                                   transform_method=self.transform_method)
        clusters = list()
        for name, cluster_data in self.clusters.items():
            c = Cluster(cluster_id=name,
                        n_events=cluster_data['n_events'],
                        prop_of_root=cluster_data['prop_of_root'])
            c.save_index(cluster_data['index'])
            clusters.append(c)
        # Save the clusters
        exp.clusters = clusters
        root_p.clustering = exp
        fg.save()

    def _load_clusters(self, root_p, clustering_uid: str):
        """
        Load existing clusters (associated to given cluster UID) associated to the root population of chosen sample.
        :param clustering_uid: unique identifier to associate to save clustering experiment
        """
        clustering = root_p.pull_clustering_experiment(clustering_uid)

        # Rebuild clustering information
        self.parameters = {k: v for k, v in clustering.parameters}
        self.method = clustering.method
        self.transform_method = clustering.transform_method

        for c in clustering.clusters:
            self.clusters[c.cluster_id] = dict(n_events=c.n_events,
                                               prop_of_root=c.prop_of_root,
                                               index=c.load_index())
