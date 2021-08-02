from sklearn import metrics as sklearn_metrics
from scipy.spatial import distance
from typing import *
import pandas as pd
import numpy as np
import math


class Metric:
    def __init__(self, name: str, desc: str, **kwargs):
        self.name = name
        self.description = desc
        self.kwargs = kwargs

    def __call__(self, data: pd.DataFrame, features: List[str], labels: List[int]):
        return self


class BallHall(Metric):
    def __init__(self, **kwargs):
        super().__init__(
            name="Ball Hall Index",
            desc="Ball-Hall Index is the mean of the mean dispersion across all clusters",
            **kwargs
        )

    def __call__(self, data: pd.DataFrame, features: List[str], labels: List[int]):
        data = data[features].copy()
        sum_total = 0

        n = len(np.unique(labels))
        # iterate through all the clusters
        for i in range(n):
            sum_distance = 0
            indices = [t for t, x in enumerate(labels) if x == i]
            cluster_member = data.values[indices, :]
            # compute the center of the cluster
            cluster_center = np.mean(cluster_member, 0)
            # iterate through all the members
            for member in cluster_member:
                sum_distance = sum_distance + math.pow(
                    distance.euclidean(member, cluster_center), 2
                )
            sum_total = sum_total + sum_distance / len(indices)
        # compute the validation
        return sum_total / n


class BakerHubertGammaIndex(Metric):
    def __init__(self, **kwargs):
        super().__init__(
            name="Baker-Hubert Gamma Index",
            desc="A measure of compactness, based on similarity between points in a cluster, "
            "compared to similarity with points in other clusters",
            **kwargs
        )

    def __call__(self, data: pd.DataFrame, features: List[str], labels: List[int]):
        splus = 0
        sminus = 0
        data_matrix = data[features].copy()
        pair_dis = distance.pdist(data_matrix)
        num_pair = len(pair_dis)
        temp = np.zeros((len(labels), 2))
        temp[:, 0] = labels
        vec_b = distance.pdist(temp)
        # iterate through all the pairwise comparisons
        for i in range(num_pair - 1):
            for j in range(i + 1, num_pair):
                if vec_b[i] > 0 and vec_b[j] == 0:
                    if pair_dis[i] < pair_dis[j]:
                        splus += 1
                    if pair_dis[i] > vec_b[j]:
                        sminus += 1
                if vec_b[i] == 0 and vec_b[j] > 0:
                    if pair_dis[j] < pair_dis[i]:
                        splus += 1
                    if pair_dis[j] > vec_b[i]:
                        sminus += 1
        return (splus - sminus) / (splus + sminus)


class SilhouetteCoef(Metric):
    def __init__(self, **kwargs):
        super().__init__(
            name="Silhouette Coefficient",
            desc="Compactness and connectedness combination that measures a ratio of within cluster "
            "distances to closest neighbors outside of cluster. This uses sklearn.metrics "
            "version of the Silhouette.",
            **kwargs
        )

    def __call__(self, data: pd.DataFrame, features: List[str], labels: List[int]):
        return sklearn_metrics.silhouette_score(
            data[features].values, labels=labels, **self.kwargs
        )


class DaviesBouldinIndex(Metric):
    def __init__(self, **kwargs):
        super().__init__(
            name="Davies-Bouldin index",
            desc="The average similarity between clusters. Similarity is defined as the "
            "ratio of within-cluster distances to between-cluster distances. Clusters further "
            "apart and less dispersed will result in a better score.",
            **kwargs
        )

    def __call__(self, data: pd.DataFrame, features: List[str], labels: List[int]):
        return sklearn_metrics.davies_bouldin_score(
            data[features].values, labels=labels
        )


class GPlusIndex(Metric):
    def __init__(self, **kwargs):
        super().__init__(
            name="G-plus index",
            desc="The proportion of discordant pairs among all the pairs of distinct points - "
            "a measure of connectedness",
            **kwargs
        )

    def __call__(self, data: pd.DataFrame, features: List[str], labels: List[int]):
        sminus = 0
        data_matrix = data[features].copy()
        pair_dis = distance.pdist(data_matrix)
        num_pair = len(pair_dis)
        temp = np.zeros((len(labels), 2))
        temp[:, 0] = labels
        vec_b = distance.pdist(temp)
        # iterate through all the pairwise comparisons
        for i in range(num_pair - 1):
            for j in range(i + 1, num_pair):
                if vec_b[i] > 0 and vec_b[j] == 0:
                    if pair_dis[i] > vec_b[j]:
                        sminus += 1
                if vec_b[i] == 0 and vec_b[j] > 0:
                    if pair_dis[j] > vec_b[i]:
                        sminus += 1
        return (2 * sminus) / (num_pair * (num_pair - 1))


class CalinskiHarabaszScore(Metric):
    def __init__(self, **kwargs):
        super().__init__(
            name="Calinski and Harabasz score",
            desc="The score is defined as ratio between the within-cluster dispersion and the "
            "between-cluster dispersion",
            **kwargs
        )

    def __call__(self, data: pd.DataFrame, features: List[str], labels: List[int]):
        return sklearn_metrics.calinski_harabasz_score(
            data[features].values, labels=labels
        )
