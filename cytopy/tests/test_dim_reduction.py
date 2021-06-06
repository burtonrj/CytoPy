from ..flow.dim_reduction import DimensionReduction
import seaborn as sns
import pytest


@pytest.mark.parametrize("method",
                         ["UMAP", "tSNE", "PCA", "KernelPCA", "MDS", "Isomap", "PHATE"])
def test_init(method):
    reducer = DimensionReduction(method=method, n_components=2)
    assert isinstance(reducer, DimensionReduction)


def test_init_error():
    with pytest.raises(KeyError):
        DimensionReduction(method="NotSupported", n_components=2)

    with pytest.raises(AttributeError):
        DimensionReduction(method="UMAP", n_components=2, invalid_attribute="invalid")


def test_fit():
    iris = sns.load_dataset('iris')
    reducer = DimensionReduction(method="UMAP", n_components=2)
    features = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
    reducer.fit(data=iris, features=features)
    assert reducer.embeddings.shape == (iris.shape[0], 2)


def test_fit_warning():
    pass


def test_transform():
    pass


def test_transform_warning():
    pass


def test_fit_transform():
    pass
