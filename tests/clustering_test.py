import numpy as np
import pandas as pd
import pytest
from sklearn.cluster import KMeans

from cytopy.clustering import clustering
from cytopy.data.project import Project


def test_remove_null_features():
    nulls = pd.DataFrame(
        {"x": [1, 2, np.nan, 4], "y": [1, 2, 3, 4], "z": [np.nan, np.nan, np.nan, np.nan], "w": [1, 2, 3, 4]}
    )
    assert {"y", "w"} == set(clustering.remove_null_features(nulls))
    assert {"w"} == set(clustering.remove_null_features(nulls, features=["x", "z", "w"]))


@pytest.mark.parametrize("scale,method", [(None, "median"), ("minmax", "mean"), ("standard", "median")])
def test_summarise_clusters(small_blobs, scale, method):
    x, y = small_blobs
    x["sample_id"] = np.random.randint(4, size=x.shape[0])
    x["cluster_label"] = y
    data = clustering.summarise_clusters(data=x, features=["f1", "f2", "f3"], scale=scale, summary_method=method)
    assert isinstance(data, pd.DataFrame)
    assert set(data.columns) == {"f1", "f2", "f3"}
    assert data.shape[0] == len(x.groupby(["sample_id", "cluster_label"]))


def test_construct_cluster_method(small_blobs):
    clusterer = clustering.ClusterMethod(klass=KMeans, params={"n_clusters": 5}, verbose=True)
    assert len(clusterer.metrics) == len(clustering.cluster_metrics.inbuilt_metrics)
    assert clusterer.params == {"n_clusters": 5}


def test_construct_cluster_method_invalid_klass():
    class Dummy:
        def __init__(self, n_clusters: int):
            self.n_clusters = n_clusters

    with pytest.raises(clustering.ClusteringError):
        clustering.ClusterMethod(klass=Dummy, params={"n_clusters": 5}, verbose=True)


def test_cluster_method_cluster(small_blobs):
    x, y = small_blobs
    x["sample_id"] = np.random.randint(4, size=x.shape[0])
    clusterer = clustering.ClusterMethod(klass=KMeans, params={"n_clusters": 3}, verbose=True)
    x = clusterer.cluster(data=x, features=["f1", "f2", "f3"])
    assert "cluster_label" in x.columns
    for _, df in x.groupby("sample_id"):
        assert df["cluster_label"].nunique() == 3


def test_cluster_method_global_clustering(small_blobs):
    x, y = small_blobs
    x["sample_id"] = np.random.randint(4, size=x.shape[0])
    clusterer = clustering.ClusterMethod(klass=KMeans, params={"n_clusters": 3}, verbose=True)
    x = clusterer.global_clustering(data=x, features=["f1", "f2", "f3"])
    assert "cluster_label" in x.columns
    assert x["cluster_label"].nunique() == 3


def test_cluster_method_meta_clustering(small_blobs):
    x, y = small_blobs
    x["sample_id"] = np.random.randint(4, size=x.shape[0])
    clusterer = clustering.ClusterMethod(klass=KMeans, params={"n_clusters": 3}, verbose=True)
    x = clusterer.cluster(data=x, features=["f1", "f2", "f3"])
    x = clusterer.meta_clustering(data=x, features=["f1", "f2", "f3"])
    assert "meta_label" in x.columns
    for _, df in x.groupby("sample_id"):
        assert df["meta_label"].nunique() == 3


def test_construct_clustering():
    exp = Project.objects(project_id="test_project").get().get_experiment(experiment_id="test_exp")
    clusterer = clustering.Clustering(
        experiment=exp,
        features=exp.panel.list_channels(),
        root_population="root",
        transform="logicle",
        population_prefix="clustering_test",
    )
    assert isinstance(clusterer.data, pd.DataFrame)
    assert "cluster_label" in clusterer.data.columns
    assert "meta_label" in clusterer.data.columns
    assert "original_index" in clusterer.data.columns
    assert all([x in clusterer.data.sample_id for x in exp.list_samples()])


@pytest.fixture(scope="module")
def base_clustering():
    exp = Project.objects(project_id="test_project").get().get_experiment(experiment_id="test_exp")
    clusterer = clustering.Clustering(
        experiment=exp,
        features=exp.panel.list_channels(),
        root_population="root",
        transform="logicle",
        population_prefix="clustering_test",
    )
    return clusterer


@pytest.mark.parametrize("method", ["phenograph", "flowsom", "consensus", KMeans])
def test_clustering_init_cluster_method(method, base_clustering):
    method = base_clustering._init_cluster_method(method=method)
    assert isinstance(method, clustering.ClusterMethod)


def test_clustering_scale_data(base_clustering):
    data, scaler = base_clustering.scale_data(features=base_clustering.features, scale_method="minmax")
    assert isinstance(data, pd.DataFrame)
    assert isinstance(scaler, clustering.Scaler)
    assert (0 <= data[base_clustering.features] <= 1).all().all()