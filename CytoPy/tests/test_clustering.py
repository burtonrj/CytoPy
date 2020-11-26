from ..flow.clustering.main import *
from .test_gating_strategy import example_experiment
import pytest
import os

FEATURES = ["FS Lin", "SS Log", "IgG1-FITC", "IgG1-PE", "CD45-ECD", "IgG1-PC5", "IgG1-PC7"]


def dummy_data(example_experiment):
    return example_experiment.get_sample("test sample").load_population_df("root")


def multisample_experiment(example_experiment):
    for i in range(3):
        example_experiment.add_new_sample(sample_id=f"test sample {i}+1",
                                          primary_path=f"{os.getcwd()}/CytoPy/tests/assets/test.FCS",
                                          controls_path={"test_ctrl": f"{os.getcwd()}/CytoPy/tests/assets/test.FCS"},
                                          compensate=False)
    return example_experiment


def multi_sample_data(example_experiment):
    return load_data(experiment=multisample_experiment(example_experiment),
                     population="root",
                     transform="logicle",
                     verbose=True)


def test_load_data(example_experiment):
    data = load_data(experiment=multisample_experiment(example_experiment),
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
                                               summary_method=summary_method,
                                               )
        assert len(meta.meta_label.unique()) > 1
