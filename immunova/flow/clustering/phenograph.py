from immunova.flow.clustering.clustering import Clustering, ClusteringError
import phenograph


class PhenoGraph(Clustering):
    def __init__(self, k=30,  **kwargs):
        super().__init__(**kwargs)
        self.k = k

    def cluster_sample(self, sample_id, features, root_population):
        if sample_id in self.cluster_cache.keys():
            return self.cluster_cache
        if sample_id not in self.experiment.list_samples():
            print(f'Invalid sample ID, must be one of: {self.experiment.list_samples()}')
            return None
        data = Gating(self.experiment, sample_id).get_population_df(root_population)
        if not all([x in data.columns for x in features]):
            print(f'Invalid Features, must be one of: {data.columns}')
            return None
        communities, graph, Q = phenograph.cluster(data[features])
        self.cluster_cache[sample_id] = dict(communities=communities, graph=graph, Q=Q)
        return self.cluster_cache[sample_id]
