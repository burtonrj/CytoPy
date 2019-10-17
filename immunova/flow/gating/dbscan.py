from immunova.flow.gating.base import Gate, GateError
from immunova.flow.gating.defaults import ChildPopulationCollection
from immunova.flow.gating.utilities import centroid
from sklearn.cluster import DBSCAN
from sklearn.neighbors import KNeighborsClassifier, KDTree
from sklearn.exceptions import NotFittedError
import pandas as pd
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
        self.sample = self.sampling(self.data, 40000)
        self.min_pop_size = min_pop_size

    def dbscan(self, distance_nn: int, nn: int = 10, core_only: bool = False):
        """
        Perform gating with dbscan algorithm
        :param distance_nn: nearest neighbour distance (smaller value will create tighter clusters)
        :param core_only: if True, only core samples in density clusters will be included
        :return: Updated child populations with events indexing complete
        """
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
        model.fit(self.sample[[self.x, self.y]], labels)
        self.data['labels'] = model.predict(self.data[[self.x, self.y]])

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
        predictions = collections.defaultdict(list)
        tree = KDTree(self.data[[self.x, self.y]].values)
        for name in self.child_populations.populations.keys():
            target = self.child_populations.populations[name].properties['target']
            dist, ind = tree.query(np.array(target).reshape(1, -1), k=int(self.data.shape[0]*0.05))
            neighbour_labels_counts = collections.Counter(self.data.iloc[ind[0]]['labels'].values).most_common()
            # Remove noisy neighbours
            neighbour_labels_counts = [x for x in neighbour_labels_counts if x[0] != -1]
            if len(neighbour_labels_counts) == 0:
                # Population is assigned to noise
                predictions[-1].append(name)
                continue
            # One neighbour class
            if len(neighbour_labels_counts) == 1:
                predictions[neighbour_labels_counts[0][0]].append(name)
                continue
            most_popular_neighbour = max(neighbour_labels_counts, key=lambda x: x[1])
            equivalents = [x for x in neighbour_labels_counts if x[1] == most_popular_neighbour[1]]
            if len(equivalents) == 1:
                # No popularity contest
                predictions[most_popular_neighbour[0]].append(name)
            else:
                # Select label with the closest centroid
                label_with_closest_centroid = self.__filter_by_centroid(target, equivalents)
                predictions[label_with_closest_centroid].append(name)
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
                weights = [self.child_populations.populations[x].properties['weight'] for x in
                           self.child_populations.populations.keys()]
                priority_i = weights.index(max(weights))
                self.warnings.append(f'Populations f{p_id} assigned to the same cluster {label};'
                                     f'prioritising {p_id[priority_i]} based on weighting.')
                self.child_populations.populations[p_id[priority_i]].update_index(idx=idx, merge_options='overwrite')
            elif label == -1:
                self.warnings.append(f'Population {p_id} assigned to noise (i.e. population not found)')
            else:
                self.child_populations.populations[p_id[0]].update_index(idx=idx, merge_options='overwrite')
            self.child_populations.populations[p_id[0]].update_geom(shape='cluster', x=self.x, y=self.y)
        return self.child_populations
