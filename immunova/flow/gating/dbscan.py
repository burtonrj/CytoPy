from immunova.flow.gating.base import Gate
from immunova.flow.gating.defaults import ChildPopulationCollection
from sklearn.cluster import DBSCAN
from sklearn.neighbors import KNeighborsClassifier
from sklearn.exceptions import NotFittedError
import pandas as pd
import numpy as np
import collections
import hdbscan


class DensityBasedClustering(Gate):
    """
    Class for Density based spatial clustering for applications with noise. Implements both DBSCAN and HDBSCAN
    """
    def __init__(self, data: pd.DataFrame, x: str, y: str, child_populations: ChildPopulationCollection,
                 min_pop_size: int, nn: int, frac: float or None = 0.2, downsample_method: str = 'uniform',
                 density_downsample_kwargs: dict or None = None):
        """
        Constructor for DensityBasedClustering gating object
        :param data: pandas dataframe representing compensated and transformed flow cytometry data
        :param x: name of X dimension
        :param y: name of Y dimension
        :param child_populations: child populations expected as output (ChildPopulationCollection; see docs for info)
        :param frac: percentage of events to sample for clustering analysis. If None then all event data is clustered
        (not recommended when n > 40000)
        :param min_pop_size: minimum population size for a population cluster
        :param nn: number of neighbours to use for up-sampling with K-nearest neighbours
        (see documentation for more info)
        :param downsample_method: methodology to use for down-sampling prior to clustering (either 'uniform' or
        'density')
        :param density_downsample_kwargs: arguments to pass to density dependent down-sampling function (if method
        is 'uniform' leave value as None)
        """
        super().__init__(data=data, x=x, y=y, child_populations=child_populations, frac=frac,
                         downsample_method=downsample_method, density_downsample_kwargs=density_downsample_kwargs)
        self.sample = self.sampling(self.data, 40000)
        self.nn = nn
        self.min_pop_size = min_pop_size

    @property
    def knn_model(self):
        return KNeighborsClassifier(n_neighbors=self.nn, weights='distance', n_jobs=-1)

    def dbscan(self, distance_nn: int, core_only: bool = False,):
        """
        Perform gating with dbscan algorithm
        :param distance_nn: nearest neighbour distance (smaller value will create tighter clusters)
        :param core_only: if True, only core samples in density clusters will be included
        :return: Updated child populations with events indexing complete
        """
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
            self.__upsample(db_labels)
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
            self.data['label'], self.data['label_strength'] = hdbscan.approximate_predict(model,
                                                                                          self.data[[self.x, self.y]])
        else:
            model.fit(self.data[[self.x, self.y]])
            self.data['label'] = model.labels_
            self.data['label_strength'] = model.probabilities_
        # Post clustering checks
        if inclusion_threshold is not None:
            mask = self.data['label_strength'] < inclusion_threshold
            self.data.loc[mask, 'label'] = -1
        # Predict clusters for child populations
        population_predictions = collections.defaultdict(list)
        for name in self.child_populations.populations.keys():
            label, _ = hdbscan.approximate_predict(model,
                                                   [self.child_populations.populations[name].properties['target']])
            population_predictions[label[0]].append(name)
        return self.__assign_clusters(population_predictions)

    def __upsample(self, labels):
        """
        Internal method. Perform up-sampling from clustering labels
        :param labels: labels from clustering analysis
        :return: None
        """
        self.knn_model.fit(self.sample[[self.x, self.y]], labels)
        self.data['labels'] = self.knn_model.predict(self.data)

    def __predict_pop_clusters(self):
        """
        Internal method. Using KNN model predict which clusters the expected child populations belong to
        using the their target mediod
        :return: predictions {labels: [child population names]}
        """
        predictions = collections.defaultdict(list)
        try:
            for name in self.child_populations.populations.keys():
                target = self.child_populations.populations[name].properties['target']
                label = self.knn_model.predict(np.reshape(target, (1, -1)))
                predictions[label[0]].append(name)
            return predictions
        except NotFittedError:
            # If sampling was performed the above code will work, otherwise a NotFittedError will be thrown. This is
            # because self.__upsample() (where fitted occurs) was never called. In this case we want to fit the knn
            # model to all our data (since we have performed clustering on all of our data in this case). So call
            # fit method and then call self.__predict_pop_clusters() again
            self.knn_model.fit(self.data[[self.x, self.y]], self.data['labels'])
            self.__predict_pop_clusters()

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
