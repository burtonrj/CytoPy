from immunova.flow.gating.base import Gate, GateError
from immunova.flow.gating.utilities import centroid, multi_centroid_calculation
from multiprocessing import Pool, cpu_count
from sklearn.cluster import DBSCAN, KMeans
from sklearn.neighbors import KDTree
from functools import partial, partialmethod
import pandas as pd
import numpy as np
import collections
import hdbscan
from datetime import datetime


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

    def _meta_clustering(self, clustered_chunks: list):
        """
        Given a list of clustered samples, fit a KMeans model to find 'meta-clusters' that will allow for merging
        of samples into a single dataframe
        :param clustered_chunks: list of pandas dataframes, each clustered using DBSCAN and cluster labels assigned
        to a column named 'labels'
        :return: reference dataframe that assigns each sample cluster to a meta-cluster
        """
        # Calculate K (number of meta clusters) as the number of expected populations OR the median number of
        # clusters found in each sample IF this median is less that the number of expected populations
        clustered_chunks_wo_noise = [x[x['labels'] != -1] for x in clustered_chunks]
        median_k = np.median([len(x['labels'].unique()) for x in clustered_chunks_wo_noise])
        if median_k < len(self.child_populations.populations.keys()):
            k = int(median_k)
        else:
            k = len(self.child_populations.populations.keys())

        # Calculate centroids of each cluster, this will be used as the training data
        pool = Pool(cpu_count())
        cluster_centroids = pd.concat(pool.map(multi_centroid_calculation, clustered_chunks_wo_noise))
        meta = KMeans(n_clusters=k, n_init=10, precompute_distances=True, random_state=42, n_jobs=-1)
        meta.fit(cluster_centroids[['x', 'y']].values)
        cluster_centroids['meta_cluster'] = meta.labels_
        return cluster_centroids

    @staticmethod
    def __meta_assignment(df, meta_clusters):
        df = df.copy()
        ref_df = meta_clusters[meta_clusters['chunk_idx'] == df['chunk_idx'].values[0]]
        df['labels'] = df['labels'].apply(lambda x:
                                          -1 if x == -1 else ref_df[ref_df['cluster'] == x]['meta_cluster'].values[0])
        return df

    def dbscan(self, distance_nn: int, core_only: bool = False):
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
        if self.downsample_method == 'density':
            if self.density_downsample_kwargs is not None:
                if not type(self.density_downsample_kwargs) == dict:
                    raise GateError('If applying density dependent down-sampling then a dictionary of '
                                    'keyword arguments is required as input for density_downsample_kwargs')
                kwargs = dict(sample_n=chunksize, features=[self.x, self.y], **self.density_downsample_kwargs)
                sampling_func = partial(self.density_dependent_downsample, **kwargs)
            else:
                sampling_func = partial(self.density_dependent_downsample, sample_n=chunksize,
                                        features=[self.x, self.y])
        else:
            sampling_func = partial(self.uniform_downsample, frac=None, sample_n=chunksize)

        for x in range(0, int(d)):
            if data.shape[0] <= chunksize:
                chunks.append(data)
                break
            sample = sampling_func(data=data)
            sample['chunk_idx'] = x
            data = data[~data.index.isin(sample.index)]
            chunks.append(sample)

        # Cluster each chunk!
        clustered_data = list()
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
            clustered_data.append(sample)

        # Perform meta clustering and merge clusters across samples
        meta_clusters = self._meta_clustering(clustered_data)
        data = pd.DataFrame()
        for i, sample in enumerate(clustered_data):
            ref = meta_clusters[meta_clusters['chunk_idx'] == i]
            f = partial(meta_assignment, ref_df=ref)
            sample['labels'] = sample['labels'].apply(f)
            data = pd.concat([data, sample])

        e = datetime.now()
        print(f'Error checking t:{(e-s).total_seconds()}...')
        # Post clustering error checking
        if len(data['labels'].unique()) == 1:
            self.warnings.append('Failed to identify any distinct populations')

        n_children = len(self.child_populations.populations.keys())
        n_clusters = len(data['labels'].unique())
        if n_clusters != n_children:
            self.warnings.append(f'Expected {n_children} populations, '
                                 f'identified {n_clusters}; {len(data["labels"].unique())}')
        self.data = data

        e = datetime.now()
        print(f'Population assignment t:{(e-s).total_seconds()}...')
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
