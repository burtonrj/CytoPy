import pandas as pd
from scipy.spatial import distance
from typing import *
import numpy as np
import math


class Metric:
    def __init__(self,
                 name: str,
                 desc: str):
        self.name = name
        self.description = desc

    def __call__(self, *args, **kwargs):
        return self


class BallHall(Metric):
    def __init__(self):
        super().__init__(name="Ball Hall Index",
                         desc="Ball-Hall Index is the mean of the mean dispersion across all clusters")

    def __call__(self, *args, **kwargs):
        return self

def ball_hall(data: pd.DataFrame,
              features: List[str],
              labels: List[int]):
    """
    Ball-Hall Index is the mean of the mean dispersion across all clusters
    """
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
            sum_distance = sum_distance + math.pow(distance.euclidean(member, cluster_center), 2)
        sum_total = sum_total + sum_distance / len(indices)
    # compute the validation
    return sum_total / n


def baker_hubert_gamma():
    """
    Baker-Hubert Gamma Index: A measure of compactness, based on similarity between points in a cluster, compared to similarity
    with points in other clusters
    """
    self.description = 'Gamma Index: a measure of compactness'
    splus = 0
    sminus = 0
    pairDis = distance.pdist(self.dataMatrix)
    numPair = len(pairDis)
    temp = np.zeros((len(self.classLabel), 2))
    temp[:, 0] = self.classLabel
    vecB = distance.pdist(temp)
    # iterate through all the pairwise comparisons
    for i in range(numPair - 1):
        for j in range(i + 1, numPair):
            if vecB[i] > 0 and vecB[j] == 0:
                # heter points smaller than homo points
                if pairDis[i] < pairDis[j]:
                    splus = splus + 1
                # heter points larger than homo points
                if pairDis[i] > vecB[j]:
                    sminus = sminus + 1
            if vecB[i] == 0 and vecB[j] > 0:
                # heter points smaller than homo points
                if pairDis[j] < pairDis[i]:
                    splus = splus + 1
                # heter points larger than homo points
                if pairDis[j] > vecB[i]:
                    sminus = sminus + 1
    # compute the fitness
    self.validation = (splus - sminus) / (splus + sminus)
    return self.validation


def silhouette_coef():
    """
    Silhouette: Compactness and connectedness combination that measures a ratio of within cluster distances to closest neighbors
    outside of cluster. This uses sklearn.metrics version of the Silhouette.
    """
    #scikit
    pass


def davies_bouldin_index():
    """
    The Davies-Bouldin index, the average of all cluster similarities.
    """
    #scikit
    pass


def g_plus_index(self):
    """
    The G_plus index, the proportion of discordant pairs among all the pairs of distinct point, a measure of connectedness
    """
    self.description = "The G_plus index, a measure of connectedness"
    sminus = 0
    pairDis = distance.pdist(self.dataMatrix)
    numPair = len(pairDis)
    temp = np.zeros((len(self.classLabel), 2))
    temp[:, 0] = self.classLabel
    vecB = distance.pdist(temp)
    # iterate through all the pairwise comparisons
    for i in range(numPair - 1):
        for j in range(i + 1, numPair):
            if vecB[i] > 0 and vecB[j] == 0:
                # heter points larger than homo points
                if pairDis[i] > vecB[j]:
                    sminus = sminus + 1
            if vecB[i] == 0 and vecB[j] > 0:
                # heter points larger than homo points
                if pairDis[j] > vecB[i]:
                    sminus = sminus + 1
    # return fitness
    self.validation = 2 * sminus / (numPair * (numPair - 1))
    return self.validation


def calinski_harabasz_score():
    # Scikit learn
    pass
