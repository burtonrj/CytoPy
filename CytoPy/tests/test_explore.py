from ..data.read_write import FCSFile
from ..flow.clustering.main import sklearn_clustering, sklearn_metaclustering
from ..flow.transform import apply_transform
from ..flow.explore import *
import pandas as pd
import numpy as np
import pytest
import os


def random_mutate(data: pd.DataFrame):
    data = data.copy()
    for i in range(5):
        m = np.random.choice([f"marker_{i+1}" for i in range(data.shape[1])])
        if np.random.choice(100) % 2 == 1:
            data[m] = data[m].apply(lambda x: x + x * 0.05)
        else:
            data[m] = data[m].apply(lambda x: x - x * 0.05)
    return data


def assign_identifiers(data: pd.DataFrame,
                       sample_id: str,
                       subject_id: str):
    data["sample_id"] = sample_id
    data["subject_id"] = subject_id
    return data


@pytest.fixture()
def example_data():
    fcs = FCSFile(filepath=f"{os.getcwd()}/CytoPy/tests/assets/test.FCS")
    features = [f"marker_{i+1}" for i in range(fcs.event_data.shape[1])]
    data = pd.DataFrame(fcs.event_data, columns=features)
    data = [assign_identifiers(random_mutate(apply_transform(data, transform_method="logicle")),
                               subject_id=f"subject{i}",
                               sample_id=f"sample{i}")
            for i in range(10)]
    data = pd.concat(data).reset_index(drop=True)
    data["meta_label"] = None
    data, _, _ = sklearn_clustering(data=data,
                                    features=features,
                                    verbose=True,
                                    method="MiniBatchKMeans",
                                    n_clusters=10,
                                    batch_size=1000)
    data, _, _ = sklearn_metaclustering(data=data,
                                        summary_method="median",
                                        norm_method="norm",
                                        features=features,
                                        method="KMeans",
                                        n_clusters=5)
    return data


def assign_subject_var(subject: Subject,
                       var_name: str,
                       value: str or float):
    subject[var_name] = value
    return subject


@pytest.fixture(scope="module")
def create_dummy_subjects():
    subjects = [Subject(subject_id=f"subject{i}") for i in range(10)]
    subjects = list(map(lambda x: assign_subject_var(x, "discrete", np.random.choice(["x", "y"])), subjects))
    subjects = list(map(lambda x: assign_subject_var(x, "continuous", np.random.uniform(1, 10)), subjects))
    for i in range(5):
        subjects[i]["sometimes_missing"] = "value"
    list(map(lambda x: x.save(), subjects))


def test_init_explorer(example_data):
    e = Explorer(example_data)
    assert e.data is not None


def test_mask_data(example_data):
    e = Explorer(example_data)
    e.mask_data(e.data.subject_id == "subject1")
    assert e.data.shape[0] == 30000


def test_load_meta(example_data, create_dummy_subjects):
    e = Explorer(example_data)
    e.load_meta("discrete")
    assert "discrete" in e.data.columns
    assert set(e.data["discrete"].unique()) == {"x", "y"}
    e.load_meta("continuous")
    assert "continuous" in e.data.columns
    e.load_meta("sometimes_missing")
    for i in range(5):
        assert e.data[e.data.sample_id == f"sample{i}"]["sometimes_missing"].unique()[0] == "value"
    for i in range(5):
        assert e.data[e.data.sample_id == f"sample{5+i}"]["sometimes_missing"].isnull().all()


@pytest.mark.parametrize("method,n", [("PCA", 4), ("UMAP", 2), ("PHATE", 3)])
def test_dim_reduction(example_data, method, n):
    e = Explorer(example_data)
    e.dimenionality_reduction(method=method,
                              features=[x for x in e.data.columns if "marker" in x],
                              n_components=n)
    for i in range(n):
        assert f"{method}{i+1}" in e.data.columns
