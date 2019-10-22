from immunova.flow.gating.base import Gate
from immunova.flow.gating.utilities import centroid
from sklearn.cluster import DBSCAN
from sklearn.neighbors import KNeighborsClassifier, KDTree
import numpy as np
import pandas as pd
import collections
import hdbscan


class DensityBasedClustering(Gate):
    """
    Class for Density based spatial clustering for applications with noise. Implements both DBSCAN and HDBSCAN
    """
    def __init__(self, min_pop_size: int, **kwargs):
        """
        Constructor for DensityBasedClustering gating object
        :param min_pop_size: minimum population size for a population cluster
        :param kwargs: Gate constructor arguments (see immunova.flow.gating.base)
        """
        super().__init__(**kwargs)
        self.min_pop_size = min_pop_size

    def dbscan(self, distance_nn: int, nn: int = 10, core_only: bool = False):
        """
        Perform gating with dbscan algorithm
        :param distance_nn: nearest neighbour distance (smaller value will create tighter clusters)
        :param core_only: if True, only core samples in density clusters will be included
        :return: Updated child populations with events indexing complete
        """
        # If parent is empty just return the child populations with empty index array
        if self.empty_parent:
            return self.child_populations

        # Break data into workable chunks
        d = np.ceil(self.data.shape[0]/30000)
        chunksize = int(np.ceil(self.data.shape[0]/d))
        chunks = list()
        data = self.data.copy()
        while data.shape[0] > 0:
            sample = self.sampling(data, chunksize)
            data = data[~data.index.isin(sample.index)]
            chunks.append(sample)

        # Cluster each chunk!
        clustered_data = pd.DataFrame()
        for sample in chunks:
            model = DBSCAN(eps=distance_nn,
                           min_samples=self.min_pop_size,
                           algorithm='ball_tree',
                           n_jobs=-1)
            model.fit(sample[[self.x, self.y]])
            db_labels = model.labels_
            if core_only:
                non_core_mask = np.ones(len(db_labels), np.bool)
                non_core_mask[model.core_sample_indices_] = 0
                np.put(db_labels, non_core_mask, -1)
            sample['labels'] = db_labels
            clustered_data = pd.concat([clustered_data, sample])

        # Post clustering error checking
        if len(clustered_data['labels'].unique()) == 1:
            self.warnings.append('Failed to identify any distinct populations')

        n_children = len(self.child_populations.populations.keys())
        n_clusters = len(clustered_data['labels'].unique())
        if n_clusters != n_children:
            self.warnings.append(f'Expected {n_children} populations, '
                                 f'identified {n_clusters}; {len(clustered_data["labels"].unique())}')
        self.data = clustered_data
        population_predictions = self.__predict_pop_clusters()
        return self.__assign_clusters(population_predictions)

    def hdbscan(self, inclusion_threshold: float or None = None):
        """
        Perform gating with hdbscan algorithm
        :param inclusion_threshold: float value for minimum probability threshold for data inclusion; data below this
        threshold will be classed as noise (Optional)
        :return: Updated child populations with events indexing complete
        """
        sample = None
        # If parent is empty just return the child populations with empty index array
        if self.empty_parent:
            return self.child_populations
        if self.frac is not None:
            sample = self.sampling(self.data, 40000)
        # Cluster!
        model = hdbscan.HDBSCAN(core_dist_n_jobs=-1, min_cluster_size=self.min_pop_size, prediction_data=True)
        if sample is not None:
            model.fit(sample[[self.x, self.y]])
            self.data['labels'], self.data['label_strength'] = hdbscan.approximate_predict(model,
                                                                                           self.data[[self.x, self.y]])
        else:
            model.fit(self.data[[self.x, self.y]])
            self.data['labels'] = model.labels_
            self.data['label_strength'] = model.probabilities_
        # Post clustering checks
        if inclusion_threshold is not None:
            mask = self.data['label_strength'] < inclusion_threshold
            self.data.loc[mask, 'labels'] = -1
        # Predict clusters for child populations
        population_predictions = self.__predict_pop_clusters()
        return self.__assign_clusters(population_predictions)

    def __filter_by_centroid(self, target, neighbours):
        distances = []
        for _, label in neighbours:
            d = self.data[self.data['labels'] == label]
            c = centroid(d[[self.x, self.y]])
            distances.append((label, np.linalg.norm(target-c)))
        return min(distances, key=lambda x: x[1])[0]

    def __predict_pop_clusters(self):
        """
        Internal method. Predict which clusters the expected child populations belong to
        using the their target mediod
        :return: predictions {labels: [child population names]}
        """
        cluster_centroids = [(x, centroid(self.data[self.data['labels'] == x][[self.x, self.y]].values))
                             for x in [l for l in self.data['labels'].unique() if l != -1]]
        predictions = collections.defaultdict(list)
        tree = KDTree(self.data[[self.x, self.y]].values)
        for name in self.child_populations.populations.keys():
            target = np.array(self.child_populations.populations[name].properties['target'])
            dist, ind = tree.query(target.reshape(1, -1), k=int(self.data.shape[0]*0.01))
            neighbour_labels = set(self.data.iloc[ind[0]]['labels'].values)
            # If all neighbours are noise, assign target to noise
            if neighbour_labels == {-1}:
                predictions[-1].append(name)
                continue
            distance_to_centroids = [(x, np.linalg.norm(target-c)) for x, c in cluster_centroids]
            predictions[min(distance_to_centroids, key=lambda x: x[1])[0]].append(name)
        return predictions

    def __assign_clusters(self, population_predictions):
        """
        Internal method. Given child population cluster predictions, update index and geom of child populations
        :param population_predictions: predicted clustering assignments {label: [population names]}
        :return: child populations with updated index and geom values
        """
        for label, p_id in population_predictions.items():
            idx = self.data[self.data['labels'] == label].index.values
            if len(p_id) > 1:
                # Multiple child populations have been associated to one cluster. Use the child population weightings
                # to choose priority population
                weights = [{'name': name, 'weight': x.properties['weight']}
                           for name, x in self.child_populations.populations.items()
                           if name in p_id]
                priority_id = max(weights, key=lambda x: x['weight'])['name']
                self.warnings.append(f'Populations f{p_id} assigned to the same cluster {label};'
                                     f'prioritising {priority_id} based on weighting.')
                self.child_populations.populations[priority_id].update_index(idx=idx, merge_options='overwrite')
                for x in p_id:
                    self.child_populations.populations[x].update_geom(shape='cluster', x=self.x, y=self.y)
            elif label == -1:
                self.warnings.append(f'Population {p_id} assigned to noise (i.e. population not found)')
            else:
                self.child_populations.populations[p_id[0]].update_index(idx=idx, merge_options='overwrite')
            self.child_populations.populations[p_id[0]].update_geom(shape='cluster', x=self.x, y=self.y)
        return self.child_populations
