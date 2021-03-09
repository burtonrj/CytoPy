from ..data.project import Project
from ..flow.clustering.main import *
import pytest
import h5py
import os

FEATURES = ["FS Lin", "SS Log", "IgG1-FITC", "IgG1-PE", "CD45-ECD", "IgG1-PC5", "IgG1-PC7"]


def dummy_data(example_populated_experiment):
    return example_populated_experiment.get_sample("test sample").load_population_df("root")


def multisample_experiment(example_populated_experiment):
    for i in range(3):
        example_populated_experiment.add_fcs_files(sample_id=f"test sample {i + 1}",
                                                   primary=f"{os.getcwd()}/CytoPy/tests/assets/test.FCS",
                                                   controls={
                                                       "test_ctrl": f"{os.getcwd()}/CytoPy/tests/assets/test.FCS"},
                                                   compensate=False)
    return example_populated_experiment


def multi_sample_data(example_populated_experiment):
    return load_population_data_from_experiment(experiment=multisample_experiment(example_populated_experiment),
                                                population="root",
                                                transform="logicle",
                                                verbose=True)


def test_load_data(example_populated_experiment):
    data = load_population_data_from_experiment(experiment=multisample_experiment(example_populated_experiment),
                                                population="root",
                                                transform="logicle",
                                                verbose=True)
    assert isinstance(data, pd.DataFrame)
    assert all([x in data.columns for x in ["sample_id", "original_index", "subject_id"]])
    assert data["subject_id"].isnull().all()
    for _id in data.sample_id.unique():
        fg = example_populated_experiment.get_sample(_id)
        df = fg.load_population_df(population="root",
                                   transform="logicle",
                                   label_downstream_affiliations=True)
        assert np.array_equal(df.index.values, data[data.sample_id == _id]["original_index"].values)
    assert data.shape[0] == 30000 * len(list(example_populated_experiment.list_samples()))
    assert set(data["sample_id"].values) == set(list(example_populated_experiment.list_samples()))


def test_sklearn_clustering_invalid_method(example_populated_experiment):
    with pytest.raises(AssertionError) as err:
        sklearn_clustering(data=dummy_data(example_populated_experiment),
                           features=FEATURES,
                           method="INVALID",
                           verbose=False)
    assert str(err.value) == "Not a recognised method from the Scikit-Learn cluster/mixture modules or HDBSCAN"


def test_sklearn_clustering(example_populated_experiment):
    data = multi_sample_data(example_populated_experiment)
    data["cluster_label"], data["meta_label"] = None, None
    data, _, _ = sklearn_clustering(data=data,
                                    features=FEATURES,
                                    method="MiniBatchKMeans",
                                    verbose=True,
                                    n_clusters=5,
                                    batch_size=1000)
    assert "cluster_label" in data.columns
    for _id in data.sample_id.unique():
        df = data[data.sample_id == _id]
        assert len(df.cluster_label.unique()) > 1


def test_sklearn_global_clustering(example_populated_experiment):
    data = multi_sample_data(example_populated_experiment)
    data["cluster_label"], data["meta_label"] = None, None
    data, _, _ = sklearn_clustering(data=data,
                                    features=FEATURES,
                                    method="MiniBatchKMeans",
                                    verbose=True,
                                    global_clustering=True,
                                    n_clusters=5,
                                    batch_size=1000)
    assert "cluster_label" in data.columns
    assert len(data.cluster_label.unique()) > 1


def test_phenograph_clustering(example_populated_experiment):
    data = multi_sample_data(example_populated_experiment)
    data["cluster_label"], data["meta_label"] = None, None
    data, graph, q = phenograph_clustering(data=data,
                                           features=FEATURES,
                                           verbose=True,
                                           global_clustering=False)
    assert "cluster_label" in data.columns
    for _id in data.sample_id.unique():
        df = data[data.sample_id == _id]
        assert len(df.cluster_label.unique()) > 1


def test_phenograph_global_clustering(example_populated_experiment):
    data = multi_sample_data(example_populated_experiment)
    data["cluster_label"], data["meta_label"] = None, None
    data, graph, q = phenograph_clustering(data=data,
                                           features=FEATURES,
                                           verbose=True,
                                           global_clustering=True)
    assert "cluster_label" in data.columns
    assert len(data.cluster_label.unique()) > 1


def test_sklearn_metaclustering_invalid(example_populated_experiment):
    data = dummy_data(example_populated_experiment)
    with pytest.raises(AssertionError) as err:
        sklearn_metaclustering(data=data,
                               features=FEATURES,
                               method="INVALID")
    assert str(err.value) == "Not a recognised method from the Scikit-Learn cluster/mixture modules or HDBSCAN"


def test_sklearn_metaclustering(example_populated_experiment):
    data = multi_sample_data(example_populated_experiment)
    data["cluster_label"], data["meta_label"] = None, None
    clustered, _, _ = sklearn_clustering(data=data,
                                         features=FEATURES,
                                         verbose=True,
                                         method="MiniBatchKMeans",
                                         n_clusters=5,
                                         batch_size=1000,
                                         global_clustering=False)
    for scale_method, summary_method in zip(["robust", "standard", None], ["mean", "median", "median"]):
        meta, _, _ = sklearn_metaclustering(data=clustered,
                                            features=FEATURES,
                                            method="KMeans",
                                            scale_method=scale_method,
                                            summary_method=summary_method,
                                            n_clusters=5)
        assert len(meta.meta_label.unique()) == 5


def test_phenograph_metaclustering(example_populated_experiment):
    data = multi_sample_data(example_populated_experiment)
    data["cluster_label"], data["meta_label"] = None, None
    clustered, _, _ = phenograph_clustering(data=data,
                                            features=FEATURES,
                                            verbose=True,
                                            global_clustering=False)
    for scale_method, summary_method in zip(["robust", "standard", None], ["mean", "median", "median"]):
        meta, _, _ = phenograph_metaclustering(data=clustered,
                                               features=FEATURES,
                                               scale_method=scale_method,
                                               summary_method=summary_method)
        assert len(meta.meta_label.unique()) > 1


def test_consensus_metaclustering_cluster_num_err(example_populated_experiment):
    data = multi_sample_data(example_populated_experiment)
    data["cluster_label"], data["meta_label"] = None, None
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
                                           scale_method="standard",
                                           summary_method="median",
                                           cluster_class=AgglomerativeClustering,
                                           largest_cluster_n=50,
                                           verbose=True)
    assert err.value


def test_consensus_metaclustering(example_populated_experiment):
    data = multi_sample_data(example_populated_experiment)
    data["cluster_label"], data["meta_label"] = None, None
    clustered, _, _ = sklearn_clustering(data=data,
                                         features=FEATURES,
                                         verbose=True,
                                         method="MiniBatchKMeans",
                                         n_clusters=5,
                                         batch_size=1000,
                                         global_clustering=False)
    for scale_method, summary_method in zip(["robust", "standard", None], ["mean", "median", "median"]):
        meta, _, _ = consensus_metacluster(data=data,
                                           features=FEATURES,
                                           scale_method=scale_method,
                                           summary_method=summary_method,
                                           cluster_class=AgglomerativeClustering(),
                                           smallest_cluster_n=2,
                                           largest_cluster_n=8,
                                           verbose=True)
        assert len(meta.meta_label.unique()) > 1


def test_flowsom_clustering(example_populated_experiment):
    data = multi_sample_data(example_populated_experiment)
    data["cluster_label"], data["meta_label"] = None, None
    data, _, _ = flowsom_clustering(data=data,
                                    features=FEATURES,
                                    meta_cluster_class=AgglomerativeClustering(),
                                    verbose=True,
                                    global_clustering=False,
                                    training_kwargs={"som_dim": (10, 10)})
    assert "cluster_label" in data.columns
    for _id in data.sample_id.unique():
        df = data[data.sample_id == _id]
        assert len(df.cluster_label.unique()) > 1


def test_flowsom_global_clustering(example_populated_experiment):
    data = multi_sample_data(example_populated_experiment)
    data["cluster_label"], data["meta_label"] = None, None
    data, _, _ = flowsom_clustering(data=data,
                                    features=FEATURES,
                                    meta_cluster_class=AgglomerativeClustering(),
                                    verbose=True,
                                    global_clustering=True,
                                    training_kwargs={"som_dim": (10, 10)})
    assert "cluster_label" in data.columns
    assert len(data.cluster_label.unique()) > 1


def test_init_clustering(example_populated_experiment):
    exp = multisample_experiment(example_populated_experiment)
    c = Clustering(experiment=exp,
                   features=FEATURES)
    assert c.data.shape[0] == 30000 * len(list(exp.list_samples()))


def test_clustering_check_null(example_populated_experiment):
    exp = multisample_experiment(example_populated_experiment)
    c = Clustering(experiment=exp,
                   features=FEATURES)
    features = c._check_null()
    assert set(features) == set(FEATURES)
    c.data.loc[0:5, "FS Lin"] = None
    features = c._check_null()
    assert set(features) != set(FEATURES)
    assert "FS Lin" not in features
    assert len(features) == len(FEATURES) - 1


def test_clustering_cluster(example_populated_experiment):
    exp = multisample_experiment(example_populated_experiment)
    c = Clustering(experiment=exp,
                   features=FEATURES)
    c.cluster(phenograph_clustering)
    assert len(c.data["cluster_label"].unique()) > 1


def test_clustering_meta_cluster(example_populated_experiment):
    exp = multisample_experiment(example_populated_experiment)
    c = Clustering(experiment=exp,
                   features=FEATURES)
    c.cluster(sklearn_clustering, method="MiniBatchKMeans", n_clusters=5, batch_size=1000)
    c.meta_cluster(sklearn_metaclustering, method="KMeans", n_clusters=5)
    assert len(c.data["meta_label"].unique()) == 5


def test_clustering_rename_meta_clusters(example_populated_experiment):
    exp = multisample_experiment(example_populated_experiment)
    c = Clustering(experiment=exp,
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


def test_clustering_save(example_populated_experiment):
    exp = multisample_experiment(example_populated_experiment)
    c = Clustering(experiment=exp,
                   features=FEATURES)
    c.cluster(sklearn_clustering, method="MiniBatchKMeans", n_clusters=5, batch_size=1000)
    c.meta_cluster(sklearn_metaclustering, method="KMeans", n_clusters=5)
    c.save()

    # Check HDF5 files
    for _id, df in c.data.groupby("sample_id"):
        cluster_sizes = df.meta_label.value_counts()
        fg = exp.get_sample(_id)
        for cluster_id in df.meta_label.unique():
            cluster_id = f"cluster_{cluster_id}"
            assert cluster_id in fg.list_populations()
            with h5py.File(fg.h5path, "r") as f:
                assert cluster_id in f["index"].keys()
                assert len(f[f"index/{cluster_id}/primary"]) == cluster_sizes.get(int(cluster_id
                                                                                      .replace("cluster_", "")))

