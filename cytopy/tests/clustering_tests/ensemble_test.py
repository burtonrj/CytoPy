import numpy as np
import pytest
from scipy import sparse
from sklearn.cluster import MiniBatchKMeans
from sklearn.datasets import make_blobs

from cytopy.flow.clustering import ensemble


def label_matrix(n_samples: int = 5000, n_features: int = 5, centers: int = 5):
    labels = []
    x, y = make_blobs(n_samples=n_samples, n_features=n_features, random_state=42, centers=centers)
    for k in [3, 5, 8, 12]:
        labels.append(MiniBatchKMeans(n_clusters=k).fit_predict(x))
    return np.array(labels)


def test_create_hypergraph():
    lm = label_matrix(n_samples=4000000, n_features=15, centers=8)
    incidence_matrix = ensemble.create_hypergraph(label_matrix=lm)
    assert isinstance(incidence_matrix, sparse.csc_matrix)
    assert incidence_matrix.shape[0] == 4000000


def test_cluster_based_similarity_partitioning():
    lm = label_matrix(n_samples=2000, n_features=5, centers=4)
    consensus_labels = ensemble.cluster_based_similarity_partitioning(label_matrix=lm, k=5)
    assert len(np.unique(consensus_labels)) == 5
