import logging
from typing import Union

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier

from ..flow.cell_classifier.cell_classifier import BaseClassifier
from ..flow.cell_classifier.cell_classifier import CalibratedCellClassifier
from ..flow.cell_classifier.cell_classifier import CellClassifier
from ..flow.cell_classifier.cell_classifier import ClassifierError


logger = logging.getLogger(__name__)


def base_classifier(**kwargs):
    kwargs["features"] = kwargs.get("features", ["FSC.H", "SSC.H", "FL1.H", "FL2.H", "FL3.H", "FL4.H"])
    return BaseClassifier(
        target_populations=[f"population_{i}" for i in range(5)],
        population_prefix="test",
        **kwargs,
    )


def test_construct_base_classifier():
    """
    Should be able to build the base classifier
    """
    base = base_classifier(model=KNeighborsClassifier(), x=None, y=None)
    assert isinstance(base, BaseClassifier)


def test_base_classifier_class_error():
    """
    Passing a class instead of an object as model should raise a ClassifierError
    """
    with pytest.raises(ClassifierError):
        base_classifier(model=KNeighborsClassifier, x=None, y=None)


def test_base_classifier_invalid_model():
    """
    Passing a model without fit and predict methods should raise a ClassifierError
    """

    class DummyModel:
        def __init__(self, x):
            x = x

    dummy = DummyModel(x=None)
    with pytest.raises(ClassifierError):
        base_classifier(model=dummy, x=None, y=None)


def test_base_classifier_set_params():
    """
    Should be able to update model params
    """
    classifier = base_classifier(model=KNeighborsClassifier(), x=None, y=None)
    classifier.set_params(n_neighbors=100, weights="distance")
    assert classifier.model.get_params()["n_neighbors"] == 100
    assert classifier.model.get_params()["weights"] == "distance"


@pytest.mark.parametrize("method,sample_size", [("uniform", 0.1), ("density", 0.1), ("faithful", 0.1)])
def test_base_classifier_downsample(method: str, sample_size: Union[int, float]):
    iris = load_iris()
    x = pd.DataFrame(iris["data"], columns=iris["feature_names"])
    y = iris["target"]
    classifier = base_classifier(model=KNeighborsClassifier(), x=x, y=y, features=iris["feature_names"])
    classifier.downsample(method=method, sample_size=sample_size)
    assert isinstance(classifier.x, pd.DataFrame)
    assert isinstance(classifier.y, np.ndarray)
    assert classifier.x.shape[0] <= x.sample(frac=0.1).shape[0]
    assert len(classifier.y) <= (len(y) * 0.1)
