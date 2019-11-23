from immunova.flow.gating.actions import Gating
from immunova.flow.clustering.plotting import static, interative
import phenograph


class ClusteringError(Exception):
    pass


class Clustering:
    def __init__(self, experiment, sample_id, root_population, n_jobs=-1, transform: bool = True,
                 transform_method: str = 'logicle', features='all'):
        self.n_jobs = n_jobs
        self.clusters = dict()
        self.parameters = dict()
        self.method = None
        self.transform_method = transform_method
        self.root_population = root_population
        self.sample = Gating(experiment, sample_id)
        self.data = self.sample.get_population_df(root_population,
                                                  transform=transform,
                                                  transform_method=transform_method,
                                                  transform_features=features)[features]
        if self.data is None:
            raise ClusteringError(f'Error: was unable to retrieve data for {sample_id} invalid root population, '
                                  f'{g.populations.keys()}')

    def add_cluster(self, name, indicies):
        self.clusters[name] = dict(indicies=indicies,
                                   n_events=len(indicies),
                                   prop_of_root=len(indicies)/self.data.shape[0])

    def static_plot_clusters(self, method: str, title: str, save_path: str or None = None, n_components=2, sample_n=100000,
                             sample_method='uniform', heatmap=None):
        static.plot_clusters(self.data, self.clusters, method=method,
                             sample_n=sample_n, sample_method=sample_method,
                             n_components=n_components, save_path=save_path,
                             title=title)
        if heatmap == 'standard':
            static.heatmap(self.data, self.clusters, title, save_path)
        if heatmap == 'cluster':
            static.clustermap(self.data, self.clusters, title, save_pathgit )


    def interactive_plots(self):
        interative.bokeh_heatmap_umap(self.data, self.clusters)

    def save(self):
        pass

    def load(self):
        pass

gated_populations = self.sample.find_dependencies(self.root_population)
gated_populations = {pop: node for pop, node in self.sample.populations.items() if pop in gated_populations}