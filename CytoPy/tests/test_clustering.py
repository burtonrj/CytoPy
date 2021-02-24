from ..data.project import Project
from ..flow.clustering.main import *
import pytest
import h5py
import os

FEATURES = ["FS Lin", "SS Log", "IgG1-FITC", "IgG1-PE", "CD45-ECD", "IgG1-PC5", "IgG1-PC7"]


def dummy_data(example_experiment):
    return example_experiment.get_sample("test sample").load_population_df("root")


def multisample_experiment(example_experiment):
    for i in range(3):
        example_experiment.add_fcs_files(sample_id=f"test sample {i + 1}",
                                         primary=f"{os.getcwd()}/CytoPy/tests/assets/test.FCS",
                                         controls={"test_ctrl": f"{os.getcwd()}/CytoPy/tests/assets/test.FCS"},
                                         compensate=False)
    return example_experiment


def multi_sample_data(example_experiment):
    return load_population_data_from_experiment(experiment=multisample_experiment(example_experiment),
                                                population="root",
                                                transform="logicle",
                                                verbose=True)


def test_load_data(example_experiment):
    data = load_population_data_from_experiment(experiment=multisample_experiment(example_experiment),
                                                population="root",
                                                transform="logicle",
                                                verbose=True)
    assert isinstance(data, pd.DataFrame)
    assert all([x in data.columns for x in ["sample_id", "cluster_id", "meta_label", "original_index", "subject_id"]])
    assert all([all(pd.isnull(data[x])) for x in ["subject_id", "cluster_id", "meta_label"]])
    for _id in data.sample_id.unique():
        fg = example_experiment.get_sample(_id)
        df = fg.load_population_df(population="root",
                                   transform="logicle",
                                   label_downstream_affiliations=True)
        assert np.array_equal(df.index.values, data[data.sample_id == _id]["original_index"].values)
    assert data.shape[0] == 30000 * len(list(example_experiment.list_samples()))
    assert set(data["sample_id"].values) == set(list(example_experiment.list_samples()))


def test_sklearn_clustering_invalid_method(example_experiment):
    with pytest.raises(AssertionError) as err:
        sklearn_clustering(data=dummy_data(example_experiment),
                           features=FEATURES,
                           method="INVALID",
                           verbose=False)
    assert str(err.value) == "Not a recognised method from the Scikit-Learn cluster/mixture modules or HDBSCAN"


def test_sklearn_clustering(example_experiment):
    data = multi_sample_data(example_experiment)
    data, _, _ = sklearn_clustering(data=data,
                                    features=FEATURES,
                                    method="MiniBatchKMeans",
                                    verbose=True,
                                    n_clusters=5,
                                    batch_size=1000)
    assert "cluster_id" in data.columns
    for _id in data.sample_id.unique():
        df = data[data.sample_id == _id]
        assert len(df.cluster_id.unique()) > 1


def test_sklearn_global_clustering(example_experiment):
    data = multi_sample_data(example_experiment)
    data, _, _ = sklearn_clustering(data=data,
                                    features=FEATURES,
                                    method="MiniBatchKMeans",
                                    verbose=True,
                                    global_clustering=True,
                                    n_clusters=5,
                                    batch_size=1000)
    assert "cluster_id" in data.columns
    assert len(data.cluster_id.unique()) > 1


def test_phenograph_clustering(example_experiment):
    data = multi_sample_data(example_experiment)
    data, graph, q = phenograph_clustering(data=data,
                                           features=FEATURES,
                                           verbose=True,
                                           global_clustering=False)
    assert "cluster_id" in data.columns
    for _id in data.sample_id.unique():
        df = data[data.sample_id == _id]
        assert len(df.cluster_id.unique()) > 1


def test_phenograph_global_clustering(example_experiment):
    data = multi_sample_data(example_experiment)
    data, graph, q = phenograph_clustering(data=data,
                                           features=FEATURES,
                                           verbose=True,
                                           global_clustering=True)
    assert "cluster_id" in data.columns
    assert len(data.cluster_id.unique()) > 1


def test_sklearn_metaclustering_invalid(example_experiment):
    data = dummy_data(example_experiment)
    with pytest.raises(AssertionError) as err:
        sklearn_metaclustering(data=data,
                               features=FEATURES,
                               method="INVALID")
    assert str(err.value) == "Not a recognised method from the Scikit-Learn cluster/mixture modules or HDBSCAN"


def test_sklearn_metaclustering(example_experiment):
    data = multi_sample_data(example_experiment)
    clustered, _, _ = sklearn_clustering(data=data,
                                         features=FEATURES,
                                         verbose=True,
                                         method="MiniBatchKMeans",
                                         n_clusters=5,
                                         batch_size=1000,
                                         global_clustering=False)
    for norm_method, summary_method in zip(["norm", "standard", None], ["mean", "median", "median"]):
        meta, _, _ = sklearn_metaclustering(data=clustered,
                                            features=FEATURES,
                                            method="KMeans",
                                            norm_method=norm_method,
                                            summary_method=summary_method,
                                            n_clusters=5)
        assert len(meta.meta_label.unique()) == 5


def test_phenograph_metaclustering(example_experiment):
    data = multi_sample_data(example_experiment)
    clustered, _, _ = phenograph_clustering(data=data,
                                            features=FEATURES,
                                            verbose=True,
                                            global_clustering=False)
    for norm_method, summary_method in zip(["norm", "standard", None], ["mean", "median", "median"]):
        meta, _, _ = phenograph_metaclustering(data=clustered,
                                               features=FEATURES,
                                               norm_method=norm_method,
                                               summary_method=summary_method)
        assert len(meta.meta_label.unique()) > 1


def test_consensus_metaclustering_cluster_num_err(example_experiment):
    data = multi_sample_data(example_experiment)
    clustered, _, _ = sklearn_clustering(data=data,
                                         features=FEATURES,
                                         verbose=True,
                                         method="MiniBatchKMeans",
                                         n_clusters=5,
                                         batch_size=1000,
                                         global_clustering=False)
    with pytest.raises(AssertionError) as err:
        meta, _, _ = consensus_metacluster(data=data,
                                           features=FEATURES,
                                           norm_method="norm",
                                           summary_method="median",
                                           cluster_class=AgglomerativeClustering,
                                           largest_cluster_n=50,
                                           verbose=True)
    assert err.value


def test_consensus_metaclustering(example_experiment):
    data = multi_sample_data(example_experiment)
    clustered, _, _ = sklearn_clustering(data=data,
                                         features=FEATURES,
                                         verbose=True,
                                         method="MiniBatchKMeans",
                                         n_clusters=5,
                                         batch_size=1000,
                                         global_clustering=False)
    for norm_method, summary_method in zip(["norm", "standard", None], ["mean", "median", "median"]):
        meta, _, _ = consensus_metacluster(data=data,
                                           features=FEATURES,
                                           norm_method=norm_method,
                                           summary_method=summary_method,
                                           cluster_class=AgglomerativeClustering,
                                           smallest_cluster_n=2,
                                           largest_cluster_n=8,
                                           verbose=True)
        assert len(meta.meta_label.unique()) > 1


def test_flowsom_clustering(example_experiment):
    data = multi_sample_data(example_experiment)
    data, _, _ = flowsom_clustering(data=data,
                                    features=FEATURES,
                                    meta_cluster_class=AgglomerativeClustering,
                                    verbose=True,
                                    global_clustering=False,
                                    training_kwargs={"som_dim": (10, 10)})
    assert "cluster_id" in data.columns
    for _id in data.sample_id.unique():
        df = data[data.sample_id == _id]
        assert len(df.cluster_id.unique()) > 1


def test_flowsom_global_clustering(example_experiment):
    data = multi_sample_data(example_experiment)
    data, _, _ = flowsom_clustering(data=data,
                                    features=FEATURES,
                                    meta_cluster_class=AgglomerativeClustering,
                                    verbose=True,
                                    global_clustering=True,
                                    training_kwargs={"som_dim": (10, 10)})
    assert "cluster_id" in data.columns
    assert len(data.cluster_id.unique()) > 1


def test_init_clustering(example_experiment):
    exp = multisample_experiment(example_experiment)
    c = Clustering(experiment=exp,
                   tag="test",
                   features=FEATURES)
    assert c.data.shape[0] == 30000 * len(list(exp.list_samples()))


def test_clustering_check_null(example_experiment):
    exp = multisample_experiment(example_experiment)
    c = Clustering(experiment=exp,
                   tag="test",
                   features=FEATURES)
    features = c._check_null()
    assert set(features) == set(FEATURES)
    c.data.loc[0:5, "FS Lin"] = None
    features = c._check_null()
    assert set(features) != set(FEATURES)
    assert "FS Lin" not in features
    assert len(features) == len(FEATURES) - 1


def test_clustering_cluster(example_experiment):
    exp = multisample_experiment(example_experiment)
    c = Clustering(experiment=exp,
                   tag="test",
                   features=FEATURES)
    c.cluster(phenograph_clustering)
    assert len(c.data["cluster_id"].unique()) > 1


def test_clustering_meta_cluster(example_experiment):
    exp = multisample_experiment(example_experiment)
    c = Clustering(experiment=exp,
                   tag="test",
                   features=FEATURES)
    c.cluster(sklearn_clustering, method="MiniBatchKMeans", n_clusters=5, batch_size=1000)
    c.meta_cluster(sklearn_metaclustering, method="KMeans", n_clusters=5)
    assert len(c.data["meta_label"].unique()) == 5


def test_clustering_rename_meta_clusters(example_experiment):
    exp = multisample_experiment(example_experiment)
    c = Clustering(experiment=exp,
                   tag="test",
                   features=FEATURES)
    c.data.loc[0, "meta_label"] = "t1"
    c.data.loc[5, "meta_label"] = "t1"
    c.data.loc[10, "meta_label"] = "t2"
    c.data.loc[15, "meta_label"] = "t3"
    c.rename_meta_clusters({"t1": "n1", "t2": "n2"})
    assert c.data.loc[0, "meta_label"] == "n1"
    assert c.data.loc[5, "meta_label"] == "n1"
    assert c.data.loc[10, "meta_label"] == "n2"
    assert c.data.loc[15, "meta_label"] == "t3"
    assert c.data.loc[20, "meta_label"] is None


def test_clustering_cluster_counts(example_experiment):
    exp = multisample_experiment(example_experiment)
    c = Clustering(experiment=exp,
                   tag="test",
                   features=FEATURES)
    for _id in c.data.sample_id.unique():
        idx = c.data[c.data.sample_id == _id].index.values
        x = np.concatenate([np.array([f"c{i+1}" for _ in range(10000)]) for i in range(3)])
        c.data.loc[idx, "cluster_id"] = x
    c._cluster_counts()
    assert "cluster_size" in c.data.columns
    for i in range(3):
        df = c.data[c.data.cluster_id == f"c{i+1}"]
        assert len(df.cluster_size.unique()) == 1
        assert df.cluster_size.unique()[0] == pytest.approx(0.33, 0.1)


def test_add_cluster_and_save(example_experiment):
    exp = multisample_experiment(example_experiment)
    fg = exp.get_sample("test sample 1")
    fg.get_population("root").add_cluster(Cluster(cluster_id="test",
                                                  meta_label="meta_test",
                                                  n=10,
                                                  index=np.arange(0, 10),
                                                  prop_of_events=0.25,
                                                  tag="test_tag"))
    assert "test" in [c.cluster_id for c in fg.get_population("root").get_clusters(cluster_ids="test", tag="test_tag")]
    assert "test" in [c.cluster_id for c in fg.get_population("root").get_clusters(meta_labels="meta_test")]
    fg.save()
    with h5py.File(fg.h5path, "r") as f:
        assert "root" in f["clusters"].keys()
        assert f"test_test_tag" in f["clusters/root"].keys()
        assert np.array_equal(f["clusters/root/test_test_tag"][:], np.arange(0, 10))
    fg = (Project.objects(project_id="test").
          get()
          .get_experiment("test experiment")
          .get_sample("test sample 1"))
    assert len(fg.get_population("root").clusters) == 1
    assert np.array_equal(fg.get_population("root").clusters[0].index, np.arange(0, 10))


def test_clustering_save(example_experiment):
    exp = multisample_experiment(example_experiment)
    c = Clustering(experiment=exp,
                   tag="test",
                   features=FEATURES)
    c.cluster(sklearn_clustering, method="MiniBatchKMeans", n_clusters=5, batch_size=1000)
    c.meta_cluster(sklearn_metaclustering, method="KMeans", n_clusters=5)
    c.save()

    # Check HDF5 files
    for _id, df in c.data.groupby("sample_id"):
        fg = exp.get_sample(_id)
        with h5py.File(fg.h5path, "r") as f:
            assert "root" in f["clusters"].keys()
            for cluster_id in df.cluster_id.unique():
                assert f"{cluster_id}_test" in f["clusters/root"].keys()

    # Check Population document
    exp.reload()
    for _id, df in c.data.groupby("sample_id"):
        fg = exp.get_sample(_id)
        cluster_ids = list(df.cluster_id.unique())
        clusters = fg.get_population("root").get_clusters(cluster_ids=cluster_ids, tag="test")
        assert len(clusters) == len(cluster_ids)
        for cluster_id, cluster_df in df.groupby("cluster_id"):
            cluster = [c for c in clusters if c.cluster_id == str(cluster_id)][0]
            assert cluster.n == cluster_df.shape[0]
            assert np.array_equal(cluster.index, cluster_df.original_index.values)
            assert len(cluster_df.meta_label.unique()) == 1
            assert cluster.meta_label == str(cluster_df.meta_label.unique()[0])


def test_clustering_reload_clusters(example_experiment):
    exp = multisample_experiment(example_experiment)
    c = Clustering(experiment=exp,
                   tag="test",
                   features=FEATURES)
    c.cluster(sklearn_clustering, method="MiniBatchKMeans", n_clusters=10, batch_size=1000)
    c.meta_cluster(sklearn_metaclustering, method="KMeans", n_clusters=5)
    c.save()
    original_data = c.data.copy()
    original_data.meta_label = original_data.meta_label.astype(str)
    original_data.cluster_id = original_data.cluster_id.astype(str)
    original_data.index = original_data.index.astype(np.dtype("int64"))
    exp.reload()
    c = Clustering(experiment=exp,
                   tag="test",
                   features=FEATURES)
    for sample_id, sample_df in original_data.groupby("sample_id"):
        new_sample_df = c.data[c.data.sample_id == sample_id]
        assert new_sample_df.shape[0] == sample_df.shape[0]
        for cluster_id, cluster_df in sample_df.groupby("cluster_id"):
            new_cluster_df = new_sample_df[new_sample_df.cluster_id == cluster_id]
            assert new_cluster_df.shape[0] == cluster_df.shape[0]
            new_cluster_df = new_cluster_df.sort_values("FS Lin")
            cluster_df = cluster_df.sort_values("FS Lin")
            for x in cluster_df.columns:
                assert np.array_equal(new_cluster_df[x].values, cluster_df[x].values)
