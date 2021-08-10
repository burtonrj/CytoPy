import logging
import os
from typing import Union

import numpy as np
import pandas as pd
import pytest
from matplotlib.pyplot import Axes
from matplotlib.pyplot import Figure
from matplotlib.pyplot import show
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection._search import BaseSearchCV
from sklearn.neighbors import KNeighborsClassifier

from ..data.fcs import FileGroup
from ..data.project import Project
from ..flow.cell_classifier.cell_classifier import BaseClassifier
from ..flow.cell_classifier.cell_classifier import CalibratedCellClassifier
from ..flow.cell_classifier.cell_classifier import CellClassifier
from ..flow.cell_classifier.cell_classifier import ClassifierError
from ..flow.transform import LogicleTransformer
from ..flow.transform import Scaler
from .conftest import add_populations
from .conftest import ASSET_PATH

logger = logging.getLogger(__name__)


@pytest.fixture
def iris():
    data = load_iris()
    return pd.DataFrame(data["data"], columns=data["feature_names"]), data["target"], data["feature_names"]


@pytest.fixture
def experiment():
    project = Project.objects(project_id="test_project").get()
    return project.get_experiment(experiment_id="test_exp")


@pytest.fixture
def loaded_cell_classifier(experiment):
    classifier = cell_classifier(model=LogisticRegression())
    classifier.load_training_data(experiment=experiment, root_population="root", reference="001")
    return classifier


@pytest.fixture
def calibrated_cell_classifier(experiment):
    return CalibratedCellClassifier(
        model=KNeighborsClassifier(),
        experiment=experiment,
        training_id="001",
        features=["FSC.H", "SSC.H", "FL1.H", "FL2.H", "FL3.H", "FL4.H"],
        root_population="root",
        target_populations=[f"population_{i}" for i in range(5)],
        sample_size=1000,
    )


@pytest.fixture(scope="module", autouse=True)
def population_first_fg_populations():
    add_populations(["001"])


def base_classifier(**kwargs):
    kwargs["features"] = kwargs.get("features", ["FSC.H", "SSC.H", "FL1.H", "FL2.H", "FL3.H", "FL4.H"])
    return BaseClassifier(
        target_populations=[f"population_{i}" for i in range(5)],
        population_prefix="test",
        **kwargs,
    )


def cell_classifier(**kwargs):
    kwargs["features"] = kwargs.get("features", ["FSC.H", "SSC.H", "FL1.H", "FL2.H", "FL3.H", "FL4.H"])
    return CellClassifier(
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


@pytest.mark.parametrize(
    "method",
    [
        "downsample",
        "oversample",
        "scale",
        "compute_class_weights",
        "_fit",
        "fit_train_test_split",
        "fit_cv",
        "hyperparam_search",
        "fit",
        "plot_confusion_matrix",
    ],
)
def test_base_classifier_no_data_error(method):
    """
    Should raise AssertionError because x and y are not defined
    """
    classifier = base_classifier(model=KNeighborsClassifier(), x=None, y=None)
    method = getattr(classifier, method)
    with pytest.raises(AssertionError):
        method()


@pytest.mark.parametrize("method,sample_size", [("uniform", 0.1), ("density", 0.1), ("faithful", 0.1)])
def test_base_classifier_downsample(iris, method: str, sample_size: Union[int, float]):
    """
    Should be able to downsample x and y in Classifier
    """
    x, y, features = iris
    classifier = base_classifier(model=KNeighborsClassifier(), x=x, y=y, features=features)
    classifier.downsample(method=method, sample_size=sample_size)
    assert set(classifier.x.columns) == set(classifier.features)
    assert isinstance(classifier.x, pd.DataFrame)
    assert isinstance(classifier.y, np.ndarray)
    assert classifier.x.shape[0] <= x.sample(frac=0.1).shape[0]
    assert len(classifier.y) <= (len(y) * 0.1)


def test_base_classifier_transform(iris):
    x, y, features = iris
    classifier = base_classifier(model=KNeighborsClassifier(), x=x, y=y, features=features)
    classifier.transform(method="logicle")
    assert isinstance(classifier.transformer, LogicleTransformer)
    assert (classifier.x["sepal length (cm)"] < 0.2).all()


def test_base_classifier_oversample(iris):
    x, y, features = iris
    classifier = base_classifier(model=KNeighborsClassifier(), x=x, y=y, features=features)
    classifier.oversample(random_state=42)
    assert isinstance(classifier.x, pd.DataFrame)
    assert isinstance(classifier.y, np.ndarray)


def test_base_classifier_scale(iris):
    x, y, features = iris
    classifier = base_classifier(model=KNeighborsClassifier(), x=x, y=y, features=features)
    classifier.scale(method="minmax")
    assert isinstance(classifier.scaler, Scaler)
    assert isinstance(classifier.x, pd.DataFrame)
    assert isinstance(classifier.y, np.ndarray)
    for f in features:
        assert classifier.x[f].between(0, 1).all()


def test_base_classifier_class_weights_invalid_model(iris):
    x, y, features = iris
    classifier = base_classifier(model=KNeighborsClassifier(), x=x, y=y, features=features)
    with pytest.raises(AssertionError):
        classifier.compute_class_weights()


def test_base_classifier_class_weights(iris):
    x, y, features = iris
    classifier = base_classifier(model=LogisticRegression(), x=x, y=y, features=features)
    classifier.compute_class_weights()
    assert isinstance(classifier.class_weights, dict)
    assert set(y) == set(classifier.class_weights.keys())


def test_base_classifier_fit_and_predict(iris):
    x, y, features = iris
    classifier = base_classifier(model=LogisticRegression(), x=x, y=y, features=features)
    y_pred, y_score = classifier.fit()._predict(x=classifier.x)
    assert isinstance(y_pred, np.ndarray)
    assert isinstance(y_score, np.ndarray)


def test_base_classifier_fit_train_test_split(iris):
    x, y, features = iris
    classifier = base_classifier(model=LogisticRegression(), x=x, y=y, features=features)
    results, predictions = classifier.fit_train_test_split()
    assert isinstance(results, dict)
    assert isinstance(predictions, dict)
    assert isinstance(results["train"], dict)
    assert isinstance(results["test"], dict)
    assert isinstance(predictions["train"], dict)
    assert isinstance(predictions["test"], dict)


def test_base_classifier_fit_cv(iris):
    x, y, features = iris
    classifier = base_classifier(model=LogisticRegression(), x=x, y=y, features=features)
    training_results, testing_results = classifier.fit_cv()
    assert isinstance(training_results, list)
    assert isinstance(testing_results, list)
    assert all([isinstance(x, dict) for x in training_results])
    assert all([isinstance(x, dict) for x in testing_results])


def test_base_classifier_hyperparam_search(iris):
    x, y, features = iris
    classifier = base_classifier(model=KNeighborsClassifier(), x=x, y=y, features=features)
    optimizer = classifier.hyperparam_search(param_grid={"n_neighbors": [5, 10, 15, 20]})
    assert isinstance(optimizer, BaseSearchCV)


def test_construct_cellclassifier():
    classifier = cell_classifier(model=LogisticRegression())
    assert isinstance(classifier, CellClassifier)


def test_cellclassifier_load_training_data(experiment):
    classifier = cell_classifier(model=LogisticRegression())
    classifier.load_training_data(experiment=experiment, root_population="root", reference="001")
    assert isinstance(classifier.x, pd.DataFrame)
    assert isinstance(classifier.y, np.ndarray)
    assert set(classifier.features) == set(classifier.x.columns)
    correct_labels = pd.read_csv(os.path.join(ASSET_PATH, "gvhd_labels", "001.csv"))
    assert np.array_equal((classifier.y - 1), correct_labels.V1.values)


def test_cellclassifier_fit_train_test_split(loaded_cell_classifier):
    training_results, testing_results = loaded_cell_classifier.fit_cv()
    assert isinstance(training_results, list)
    assert isinstance(testing_results, list)
    assert all([isinstance(x, dict) for x in training_results])
    assert all([isinstance(x, dict) for x in testing_results])


def test_cellclassifier_fit_cv(loaded_cell_classifier):
    results, predictions = loaded_cell_classifier.fit_train_test_split()
    assert isinstance(results, dict)
    assert isinstance(predictions, dict)
    assert isinstance(results["train"], dict)
    assert isinstance(results["test"], dict)
    assert isinstance(predictions["train"], dict)
    assert isinstance(predictions["test"], dict)


def test_cellclassifier_predict(loaded_cell_classifier, experiment):
    loaded_cell_classifier.fit()
    fg, predictions = loaded_cell_classifier.predict(experiment=experiment, sample_id="002", root_population="root")
    assert isinstance(fg, FileGroup)
    assert isinstance(predictions, dict)
    assert isinstance(predictions["y_pred"], np.ndarray)
    assert isinstance(predictions["y_score"], np.ndarray)
    expected_pops = [f"{loaded_cell_classifier.population_prefix}_population_{i}" for i in range(5)]
    expected_pops.append(f"{loaded_cell_classifier.population_prefix}_Unclassified")
    assert all([x in fg.list_populations() for x in expected_pops])


def test_construct_calcellclassifier(calibrated_cell_classifier):
    assert isinstance(calibrated_cell_classifier, CalibratedCellClassifier)


def test_calcellclassifier_calibration(calibrated_cell_classifier):
    calibrated_cell_classifier.calibrate()
    assert isinstance(calibrated_cell_classifier.targets, pd.DataFrame)
    assert "001" not in calibrated_cell_classifier.targets.sample_id.values
    assert isinstance(calibrated_cell_classifier.x, pd.DataFrame)
    assert calibrated_cell_classifier.x.shape[0] == 1000
    assert isinstance(calibrated_cell_classifier.y, np.ndarray)
    assert len(calibrated_cell_classifier.y) == 1000
    calibrated_cell_classifier.calibrator.plot_overlay(n=1000, s=10)
    show()


def test_calcellclassifier_fit_train_test_split(calibrated_cell_classifier):
    calibrated_cell_classifier.calibrate()
    training_results, testing_results = calibrated_cell_classifier.fit_cv()
    assert isinstance(training_results, list)
    assert isinstance(testing_results, list)
    assert all([isinstance(x, dict) for x in training_results])
    assert all([isinstance(x, dict) for x in testing_results])


def test_calcellclassifier_fit_cv(calibrated_cell_classifier):
    calibrated_cell_classifier.calibrate()
    results, predictions = calibrated_cell_classifier.fit_train_test_split()
    assert isinstance(results, dict)
    assert isinstance(predictions, dict)
    assert isinstance(results["train"], dict)
    assert isinstance(results["test"], dict)
    assert isinstance(predictions["train"], dict)
    assert isinstance(predictions["test"], dict)


def test_calcellclassifier_hyperparam_search(calibrated_cell_classifier):
    calibrated_cell_classifier.calibrate()
    optimizer = calibrated_cell_classifier.hyperparam_search(param_grid={"n_neighbors": [5, 10, 15, 20]})
    assert isinstance(optimizer, BaseSearchCV)


def test_calcellclassifier_meta_learner_fit_cv(calibrated_cell_classifier):
    calibrated_cell_classifier.calibrate()
    calibrated_cell_classifier.fit()
    training_results, testing_results = calibrated_cell_classifier.meta_learner_fit_cv()
    assert isinstance(training_results, pd.DataFrame)
    assert isinstance(testing_results, pd.DataFrame)


def test_calcellclassifier_meta_learner_fit_test_train_split(calibrated_cell_classifier):
    calibrated_cell_classifier.calibrate()
    calibrated_cell_classifier.fit()
    performance, predictions = calibrated_cell_classifier.meta_learner_fit_train_test_split()
    assert isinstance(performance, pd.DataFrame)
    assert isinstance(predictions, dict)


def test_calcellclassifier_meta_learner_hyperparam_tuning(calibrated_cell_classifier):
    calibrated_cell_classifier.calibrate()
    calibrated_cell_classifier.fit()
    performance, optimizers = calibrated_cell_classifier.meta_learner_hyperparam_tuning(
        param_grid={"n_neighbors": [5, 10, 15, 20]}
    )
    assert isinstance(performance, pd.DataFrame)
    assert isinstance(optimizers, dict)
    assert all([isinstance(x, BaseSearchCV) for x in optimizers.values()])


@pytest.mark.parametrize("meta", [True, False])
def test_calcellclassifier_plot_learning_curve(calibrated_cell_classifier, meta: bool):
    calibrated_cell_classifier.calibrate()
    if meta:
        calibrated_cell_classifier.fit()
    else:
        pass


def test_calcellclassifier_feature_importance(calibrated_cell_classifier):
    pass


def test_calcellclassifier_meta_fit_predict(calibrated_cell_classifier):
    pass


def test_calcellclassifier_predict(calibrated_cell_classifier):
    pass
