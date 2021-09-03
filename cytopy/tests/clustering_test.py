import numpy as np
import pandas as pd
import polars as pl
import pytest
from sklearn.cluster import MiniBatchKMeans
from sklearn.datasets import make_blobs

from cytopy.flow.clustering import metrics


@pytest.fixture
def example_data():
    x, y = make_blobs(n_samples=50000, n_features=10, random_state=42, centers=5)
    return pd.DataFrame(x, columns=[f"f{i + 1}" for i in range(10)]), y


def cluster(data: pd.DataFrame, n_clusters: int):
    kmeans = MiniBatchKMeans(n_clusters=n_clusters, batch_size=1000)
    return kmeans.fit_predict(data)


def test_center_dispersion(example_data):
    x, y = example_data
    dispersion = metrics.center_dispersion(pl.DataFrame(x))
    assert isinstance(dispersion, np.ndarray)
