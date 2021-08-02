from ..flow.dim_reduction import DimensionReduction
from typing import List
import seaborn as sns
import pandas as pd
import pytest


@pytest.mark.parametrize(
    "method", ["UMAP", "tSNE", "PCA", "KernelPCA", "MDS", "Isomap", "PHATE"]
)
def test_init(method):
    reducer = DimensionReduction(method=method, n_components=2)
    assert isinstance(reducer, DimensionReduction)


def test_init_error():
    with pytest.raises(KeyError):
        DimensionReduction(method="NotSupported", n_components=2)

    with pytest.raises(TypeError):
        DimensionReduction(method="UMAP", n_components=2, invalid_attribute="invalid")


def load_iris() -> (pd.DataFrame, List[str]):
    iris = sns.load_dataset("iris")
    features = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
    return iris, features


def test_fit():
    iris, features = load_iris()
    reducer = DimensionReduction(method="UMAP", n_components=2)
    reducer.fit(data=iris, features=features)
    assert reducer.embeddings.shape == (iris.shape[0], 2)


def test_fit_warning():
    iris, features = load_iris()
    reducer = DimensionReduction(method="PHATE", n_components=2, verbose=False)
    with pytest.warns(UserWarning):
        reducer.fit(data=iris, features=features)


def test_transform():
    iris, features = load_iris()
    reducer = DimensionReduction(method="UMAP", n_components=2, verbose=False)
    reducer.fit(data=iris, features=features)
    assert reducer.embeddings.shape == (iris.shape[0], 2)
    data = reducer.transform(data=iris, features=features)
    assert "UMAP1" in data.columns
    assert "UMAP2" in data.columns


def test_transform_warning():
    iris, features = load_iris()
    reducer = DimensionReduction(method="PHATE", n_components=2, verbose=False)
    with pytest.warns(UserWarning):
        reducer.transform(data=iris, features=features)


def test_fit_transform():
    iris, features = load_iris()
    reducer = DimensionReduction(method="UMAP", n_components=2, verbose=False)
    data = reducer.fit_transform(data=iris, features=features)
    assert reducer.embeddings.shape == (iris.shape[0], 2)
    assert "UMAP1" in data.columns
    assert "UMAP2" in data.columns
