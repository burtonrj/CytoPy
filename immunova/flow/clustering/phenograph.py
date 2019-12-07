from immunova.flow.clustering.clustering import Clustering
import phenograph
import numpy as np


class PhenoGraph(Clustering):
    """
    Perform clustering using the PhenoGraph algorithm
    """
    def __init__(self, k=30,  **kwargs):
        super().__init__(**kwargs)
        self.parameters['k'] = k
        self.method = 'phenograph'
        self.graph = None
        self.q = None

    def cluster(self, prefix: str or None = None):
        """
        Apply clustering and assign the results to 'clusters' parameter. Clusters will be number 0 to n, where n is the
        total number of clusters. The user can provide a prefix for cluster names by specifying a value for the 'prefix'
        argument.
        :param prefix: prefix to assign to every cluster (optional)
        :return None
        """
        communities, self.graph, self.q = phenograph.cluster(self.data)
        for x in np.unique(communities):
            indices = np.where(communities == x)[0]
            label = x
            if prefix is not None:
                label = f'{prefix}_{x}'
            self._add_cluster(label, indices)
