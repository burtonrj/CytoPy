import logging

import pytest
from sklearn.neighbors import KNeighborsClassifier

from ..flow.cell_classifier.cell_classifier import BaseClassifier
from ..flow.cell_classifier.cell_classifier import CalibratedCellClassifier
from ..flow.cell_classifier.cell_classifier import CellClassifier
from ..flow.cell_classifier.cell_classifier import ClassifierError


logger = logging.getLogger(__name__)


def base_classifier(**kwargs):
    return BaseClassifier(
        target_populations=[f"population_{i}" for i in range(5)],
        population_prefix="test",
        features=["FSC.H", "SSC.H", "FL1.H", "FL2.H", "FL3.H", "FL4.H"],
        **kwargs,
    )


def test_construct_base_classifier():
    """
    Should be able to build the base classifier
    """
    base = base_classifier(model=KNeighborsClassifier(), x=None, y=None)
    assert isinstance(base, BaseClassifier)
