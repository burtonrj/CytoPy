from immunova.flow.gating.base import Gate
from immunova.flow.gating.utilities import centroid
from sklearn.cluster import DBSCAN
from sklearn.neighbors import KNeighborsClassifier, KDTree
import numpy as np
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
        # Break data into workable chunks
        d = np.ceil(self.data.shape[0]/30000)
        chunksize = int(np.ceil(self.data.shape[0]/d))
        chunks = list()
        data = self.data.copy()
        while data.shape[0] > 0:
            sample = self.sampling(data, chunksize)
            data = data[~data.index.isin(sample.index)]
            chunks.append(sample)

        knn_model = KNeighborsClassifier(n_neighbors=nn, weights='distance', n_jobs=-1)
        # If parent is empty just return the child populations with empty index array
        if self.empty_parent:
            return self.child_populations

        # Cluster!
        model = DBSCAN(eps=distance_nn,
                       min_samples=self.min_pop_size,
                       algorithm='ball_tree',
                       n_jobs=-1)
        if self.sample is not None:
            model.fit(self.sample[[self.x, self.y]])
        else:
            model.fit(self.data[[self.x, self.y]])
        db_labels = model.labels_

        # Post clustering error checking
        if core_only:
            non_core_mask = np.ones(len(db_labels), np.bool)
            non_core_mask[model.core_sample_indices_] = 0
            np.put(db_labels, non_core_mask, -1)
        if len(set(db_labels)) == 1:
            self.warnings.append('Failed to identify any distinct populations')
        n_children = len(self.child_populations.populations.keys())
        n_clusters = len(set([x for x in db_labels if x != -1]))
        if n_clusters != n_children:
            self.warnings.append(f'Expected {n_children} populations, '
                                 f'identified {n_clusters}; {set(db_labels)}')

        # Up-sample (if necessary)
        if self.sample is not None:
            self.__upsample(knn_model, db_labels)
        else:
            self.data['labels'] = db_labels
        population_predictions = self.__predict_pop_clusters()
        return self.__assign_clusters(population_predictions)

    def hdbscan(self, inclusion_threshold: float or None = None):
        """
        Perform gating with hdbscan algorithm
        :param inclusion_threshold: float value for minimum probability threshold for data inclusion; data below this
        threshold will be classed as noise (Optional)
        :return: Updated child populations with events indexing complete
        """
        # If parent is empty just return the child populations with empty index array
        if self.empty_parent:
            return self.child_populations
        # Cluster!
        model = hdbscan.HDBSCAN(core_dist_n_jobs=-1, min_cluster_size=self.min_pop_size, prediction_data=True)
        if self.sample is not None:
            model.fit(self.sample[[self.x, self.y]])
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

    def __upsample(self, model, labels):
        """
        Internal method. Perform up-sampling from clustering labels
        :param labels: labels from clustering analysis
        :return: None
        """
        # Fit model to sampled data and then assign labels to all of the data based on nearest neighbour fit
        self.sample['labels'] = labels
        model.fit(self.sample[[self.x, self.y]], labels)
        self.data['labels'] = model.predict(self.data[[self.x, self.y]])
        # Sometimes data in low density regions which should be assigned to 'noise' will be classified as belonging
        # to a cluster. Correct for this by testing each data points local density and compare to the local density
        # of noise
        # Calculate average distance to 10 nearest neighbours for noise
        noise = self.sample[self.sample['labels'] == -1]
        noisy_tree = KDTree(noise[[self.x, self.y]])
        noise_dist, _ = noisy_tree.query(noise[[self.x, self.y]], k=10)
        avg_noise_dist = np.mean([np.mean(x[1:]) for x in noise_dist])
        # For all data points, if >= average
        tree = KDTree(self.data[[self.x, self.y]])
        whole_dist, _ = tree.query(self.data[[self.x, self.y]], k=10)
        self.data['distance'] = [np.mean(x[1:]) for x in whole_dist]
        self.data['labels'] = np.where(self.data['distance'] >= avg_noise_dist, -1, self.data['labels'])

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
