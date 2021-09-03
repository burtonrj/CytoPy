import math

import numpy as np
import pandas as pd
import polars as pl
import pytest
from scipy.spatial import distance
from sklearn.cluster import MiniBatchKMeans
from sklearn.datasets import make_blobs

from cytopy.flow.clustering import metrics


@pytest.fixture
def example_data():
    x, y = make_blobs(n_samples=100000, n_features=10, random_state=42, centers=5)
    return pd.DataFrame(x, columns=[f"f{i + 1}" for i in range(10)]), y


@pytest.fixture
def small_example():
    x, y = make_blobs(n_samples=100, n_features=2, random_state=42, centers=3)
    return pd.DataFrame(x, columns=[f"f{i + 1}" for i in range(2)]), y


def cluster(data: pd.DataFrame, n_clusters: int):
    kmeans = MiniBatchKMeans(n_clusters=n_clusters, batch_size=1000)
    return kmeans.fit_predict(data)


def test_center_dispersion(example_data):
    x, y = example_data
    dispersion = metrics.center_dispersion(pl.DataFrame(x))
    assert isinstance(dispersion, float)


def test_ball_hall(small_example, example_data):
    x, y = small_example
    n = len(np.unique(y))
    sum_total = 0
    # iterate through all the clusters
    for i in range(n):
        sum_distance = 0
        indices = [t for t, x in enumerate(y) if x == i]
        cluster_member = x.values[indices, :]
        # compute the center of the cluster
        cluster_center = np.mean(cluster_member, 0)
        # iterate through all the members
        for member in cluster_member:
            sum_distance = sum_distance + math.pow(distance.euclidean(member, cluster_center), 2)
        sum_total = sum_total + sum_distance / len(indices)
    # compute the validation
    truth = sum_total / n
    ball_hall = metrics.BallHall()
    bh = ball_hall(data=x, features=["f1", "f2"], labels=y)
    assert pytest.approx(truth, bh)
    x, y = example_data
    y_hat = cluster(data=x, n_clusters=15)
    bh = ball_hall(data=x, features=x.columns.tolist(), labels=y)
    bh_hat = ball_hall(data=x, features=x.columns.tolist(), labels=y_hat)
    assert isinstance(bh, float)
    assert bh_hat < bh


def test_baker_hubert_gamma_index(example_data):
    x, y = example_data
    bhgi = metrics.BakerHubertGammaIndex()
    i = bhgi(data=x, features=x.columns.tolist(), labels=y)
    assert isinstance(i, float)
