from immunova.flow.clustering.clustering import Clustering, ClusteringError
import phenograph
import numpy as np


class PhenoGraph(Clustering):
    def __init__(self, k=30,  **kwargs):
        super().__init__(**kwargs)
        self.parameters['k'] = k
        self.method = 'phenograph'
        self.graph = None
        self.q = None

    def cluster(self):
        communities, self.graph, self.q = phenograph.cluster(self.data)
        for x in np.unique(communities):
            indices = np.where(communities == x)[0]
            self.add_cluster(x, indices)

