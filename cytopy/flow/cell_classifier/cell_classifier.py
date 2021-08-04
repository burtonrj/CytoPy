#!/usr/bin.env/python
# -*- coding: utf-8 -*-
"""
This module contains the base class CellClassifier for using supervised classification methods,
trained on some labeled FileGroup (has existing Populations) to predict single cell classifications.

Copyright 2020 Ross Burton

Permission is hereby granted, free of charge, to any person
obtaining a copy of this software and associated documentation
files (the "Software"), to deal in the Software without restriction,
including without limitation the rights to use, copy, modify,
merge, publish, distribute, sublicense, and/or sell copies of the
Software, and to permit persons to whom the Software is furnished
to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included
in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""
from __future__ import annotations

import logging
from functools import wraps
from inspect import signature
from typing import *

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from imblearn.over_sampling import RandomOverSampler
from sklearn.base import ClassifierMixin
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import BaseCrossValidator
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from tensorflow.keras.models import Sequential

from . import utils
from ...data.experiment import Experiment
from ...data.experiment import FileGroup
from ...data.experiment import single_cell_dataframe
from ...data.population import Population
from ...feedback import progress_bar
from ...flow import sampling
from ...flow.transform import apply_transform
from ...flow.transform import Scaler
from ...flow.transform import Transformer
from ...flow.transform import TRANSFORMERS
from ...flow.variance import Harmony

__author__ = "Ross Burton"
__copyright__ = "Copyright 2020, cytopy"
__credits__ = ["Ross Burton", "Simone Cuff", "Andreas Artemiou", "Matthias Eberl"]
__license__ = "MIT"
__version__ = "2.0.0"
__maintainer__ = "Ross Burton"
__email__ = "burtonrj@cardiff.ac.uk"
__status__ = "Production"
logger = logging.getLogger("CellClassifier")

DEFAULT_METRICS = ["balanced_accuracy_score", "f1_weighted", "roc_auc_score"]


def check_data_init(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        assert args[0].x is not None, "Call 'load_training_data' prior to fit"
        return func(*args, **kwargs)

    return wrapper


def check_calibrated(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        assert args[0].targets is not None, "Calibrator not defined"
        return func(*args, **kwargs)

    return wrapper


def check_predictions(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        assert args[0].target_predictions is not None, "Call 'predict' first"
        return func(*args, **kwargs)

    return wrapper


class ClassifierError(Exception):
    def __init__(self, message: str):
        logger.error(message)
        super().__init__(message)


class BaseClassifier:
    def __init__(
        self,
        model: Union[ClassifierMixin, Sequential],
        target_populations: List[str],
        population_prefix: str,
        x: Optional[pd.DataFrame],
        y: Optional[Union[pd.Series, np.ndarray]],
        features: List[str],
        transformer: Optional[Transformer] = None,
        multi_label: bool = False,
    ):
        self.model = model
        self.target_populations = target_populations
        self.x = x
        self.y = y
        self.features = features
        self.transformer = transformer
        self.scaler = None
        self.multi_label = multi_label
        self.class_weights = None
        self.population_prefix = population_prefix

        if not hasattr(self.model, "fit") and not hasattr(self.model, "predict"):
            raise ClassifierError("At a minimum, the model must have method 'fit' and 'predict'")

    def set_params(self, **kwargs):
        """
        Convenient wrapper function for direct access to set_params method of model.
        If model does not contain 'set_params' method, this function has no effect.
        Parameters
        ----------
        kwargs:
            New parameters
        Returns
        -------
        self
        """
        _set_params = getattr(self.model, "set_params", None)
        if callable(_set_params):
            self.model.set_params(**kwargs)
        else:
            logger.warning("Model does not support 'set_params'.")
        return self

    @check_data_init
    def downsample(self, method: str, sample_size: Union[int, float], **kwargs):
        """
        Downsample training data
        Parameters
        ----------
        method: str
            Method used for down sampling, must be either 'uniform', 'density' or 'faithful'.
            See cytopy.flow.sampling for more details.
        sample_size: int or float
            Desired sample size
        kwargs:
            Additional keyword arguments passed to sampling method
        Returns
        -------
        self
        Raises
        ------
        ValueError
            If invalid method requested
        """
        x = self.x.copy()
        x["y"] = self.y
        if method == "uniform":
            x = sampling.uniform_downsampling(data=x, sample_size=sample_size, **kwargs)
        elif method == "density":
            x = sampling.density_dependent_downsampling(data=x, sample_size=sample_size, **kwargs)
        elif method == "faithful":
            x = sampling.faithful_downsampling(data=x, **kwargs)
        else:
            valid = ["uniform", "density", "faithful"]
            raise ValueError(f"Invalid method, must be one of {valid}")
        self.x, self.y = x[self.features], x["y"].values
        return self

    @check_data_init
    def transform(self, method: str, **kwargs):
        """
        Transform training data. Transformer used will be saved to the attribute 'transformer'
        Parameters
        ----------
        method: str
            Method for transformation, valid methods are: 'logicle', 'hyperlog', 'asinh' or 'log'.
            See cytopy.flow.transform for more details.
        kwargs:
            Additional keyword arguments passed to Transformer.
        Returns
        -------
        self
        """
        self.x, self.transformer = apply_transform(
            data=self.x,
            features=self.features,
            method=method,
            return_transformer=True,
            **kwargs,
        )
        return self

    @check_data_init
    def oversample(self, **kwargs):
        """
        Performs random oversampling using the RandomOverSampler class of Imbalance learn.
        Helpful for combating imbalanced data.
        Parameters
        ----------
        kwargs
            Additional keyword arguments passed to RandomOverSampler
        Returns
        -------
        self
        """
        kwargs["random_state"] = kwargs.get("random_state", 42)
        self.x, self.y = RandomOverSampler(**kwargs).fit_resample(self.x, self.y)
        return self

    @check_data_init
    def scale(self, method: str = "standard", **kwargs):
        """
        Scale training data. Many Scikit-Learn scalers supported, see cytopy.flow.transform.Scaler
        for details. Scaler object is saved to the scaler attribute.
        Parameters
        ----------
        method: str (default='standard')
        kwargs:
            Additional keyword arguments passed to Scaler
        Returns
        -------
        self
        """
        self.scaler = Scaler(method=method, **kwargs)
        self.x = self.scaler(data=self.x, features=self.features)
        return self

    @check_data_init
    def compute_class_weights(self):
        """
        Computes class weights per sample; saved to class_weights attribute and passed to
        fit 'sample_weight' parameter.
        Returns
        -------
        self
        Raises
        ------
        AssertionError
            If model does not support sample_weights in it's fit function
        """
        assert not self.multi_label, "Class weights not supported for multi-class classifiers"
        if self.model is not None:
            err = "Chosen model does not support use of class weights, perhaps try 'oversample' instead?"
            assert "sample_weight" in signature(self.model.fit).parameters.keys(), err
        self.class_weights = utils.auto_weights(y=self.y)
        return self

    @check_data_init
    def _fit(self, x: pd.DataFrame, y: np.ndarray, **kwargs):
        """
        Internal function for calling fit. Handles class weights.
        Parameters
        ----------
        x: Pandas.DataFrame
        y: numpy.ndarray
        kwargs
        Returns
        -------
        self
        """
        if self.class_weights is not None:
            self.model.fit(x, y, sample_weight=self.class_weights, **kwargs)
        else:
            self.model.fit(x, y, **kwargs)
        return self

    def _predict(self, x: pd.DataFrame, threshold: float = 0.5):
        """
        Internal function for calling predict.
        Parameters
        ----------
        x: Pandas.DataFrame
        threshold: float (default=0.5)
        Returns
        -------
        numpy.ndarray, numpy.ndarray or None
            Predictions, probabilities (if supported)
        """
        predict_proba = getattr(self.model, "predict_proba", None)
        if callable(predict_proba):
            return self.model.predict(x), self.model.predict_proba(x)
        return self.model.predict(x), None

    @check_data_init
    def fit_train_test_split(
        self,
        test_frac: float = 0.3,
        metrics: Optional[List[str]] = None,
        return_predictions: bool = True,
        train_test_split_kwargs: Optional[Dict] = None,
        **fit_kwargs,
    ):
        """
        Fits model to training data and performs validation on holdout data. Training and testing
        performed returned as dictionaries.
        Parameters
        ----------
        test_frac: float (default=0.3)
            What percentage of data to keep as holdout
        metrics: list, (default=["balanced_accuracy_score", "f1_weighted", "roc_auc_score"])
            List of metrics to report; should be string values pertaining to valid Scikit-Learn
            metric function e.g. 'balanced_accuracy' or 'roc_auc'
            (see https://scikit-learn.org/stable/modules/model_evaluation.html).
            Alternatively, list can contain callable functions with the key value in train/test
            dictionary being 'custom_metric_{index}'
        return_predictions: bool (default=True)
            If True, returns dictionary of predicted values: keys correspond to train/test
        train_test_split_kwargs: dict, optional
            Additional keyword arguments passed to train_test_split function
        fit_kwargs:
            Additional keyword arguments passed to fit call
        Returns
        -------
        dict, Optional[Dict]
            Training/testing performance (keys are 'train' and 'test'),
            training/testing predictions (keys are 'train' and 'test')
        Raises
        ------
        AssertionError
            Invalid metric supplied
        """
        metrics = metrics or DEFAULT_METRICS
        train_test_split_kwargs = train_test_split_kwargs or {}
        logger.info("Generating training and testing data")
        x_train, x_test, y_train, y_test = train_test_split(
            self.x, self.y, test_size=test_frac, **train_test_split_kwargs
        )
        logger.info("Training model")
        self._fit(x=x_train, y=y_train, **fit_kwargs)
        results = dict()
        y_hat = dict()
        for key, (X, y) in zip(["train", "test"], [[x_train, y_train], [x_test, y_test]]):
            logger.info(f"Evaluating {key}ing performance....")
            y_pred, y_score = self._predict(X)
            y_hat[key] = {"y_pred": y_pred, "y_score": y_score}
            results[key] = utils.calc_metrics(metrics=metrics, y_pred=y_pred, y_score=y_score, y_true=y)
        if return_predictions:
            return results, y_hat
        return results, None

    @check_data_init
    def fit_cv(
        self,
        cross_validator: Optional[BaseCrossValidator] = None,
        metrics: Optional[List[str]] = None,
        split_kwargs: Optional[Dict] = None,
        verbose: bool = True,
        **fit_kwargs,
    ):
        """
        Fit model with cross-validation. Note, this method can be used for standard cross validaiton
        or hyper parameter search by providing a valid class suhc as Scikit-Learn's GridSearchCV for example.
        Parameters
        ----------
        cross_validator: BaseCrossValidator (default=KFold)
            Scikit-learn cross validator or model selection method with cross-validation
            (https://scikit-learn.org/stable/modules/cross_validation.html#cross-validation-iterators)
            (https://scikit-learn.org/stable/modules/classes.html#hyper-parameter-optimizers)
        metrics: list, (default=["balanced_accuracy_score", "f1_weighted", "roc_auc_score"])
            List of metrics to report; should be string values pertaining to valid Scikit-Learn
            metric function e.g. 'balanced_accuracy' or 'roc_auc'
            (see https://scikit-learn.org/stable/modules/model_evaluation.html).
            Alternatively, list can contain callable functions with the key value in train/test
            dictionary being 'custom_metric_{index}'
        split_kwargs: dict, optional
            Additional keyword arguments passed to 'split' call on cross validator
        verbose: bool (default=True)
            Whether to print a progress bar
        fit_kwargs:
            Additional keyword arguments passed to fit call
        Returns
        -------
        List, List
            Training results (list of dictionaries; keys=metrics), testing results (list of dictionaries; keys=metrics)
        """
        metrics = metrics or DEFAULT_METRICS
        split_kwargs = split_kwargs or {}
        cross_validator = cross_validator or KFold(n_splits=10, random_state=42, shuffle=True)
        training_results = list()
        testing_results = list()
        for train_idx, test_idx in progress_bar(
            cross_validator.split(X=self.x, y=self.y, **split_kwargs),
            total=cross_validator.n_splits,
            verbose=verbose,
        ):
            x_train, x_test = self.x.loc[train_idx], self.x.loc[test_idx]
            y_train, y_test = self.y[train_idx], self.y[test_idx]
            self._fit(x=x_train, y=y_train, **fit_kwargs)
            y_pred, y_score = self._predict(x=x_train)
            training_results.append(
                utils.calc_metrics(metrics=metrics, y_pred=y_pred, y_score=y_score, y_true=y_train)
            )
            y_pred, y_score = self._predict(x_test)
            testing_results.append(utils.calc_metrics(metrics=metrics, y_pred=y_pred, y_score=y_score, y_true=y_test))
        return training_results, testing_results

    @check_data_init
    def fit(self, **kwargs):
        """
        Call fit on model using training data
        Parameters
        ----------
        kwargs:
            Additional keyword arguments passed to fit call
        Returns
        -------
        self
        """
        self._fit(x=self.x, y=self.y, **kwargs)
        return self


class CellClassifier(BaseClassifier):
    """
    Use supervised machine learning to predict the classification of single cell data. This is the base
    class used by SklearnCellClassifier and KerasCellClassifier. Training data should be provided in the form
    of a FileGroup with existing Populations. Supports multi-class and multi-label classification; if multi-label
    classification is chosen, the tree structure of training data is NOT conserved - all resulting populations
    will have the same parent population.
    Parameters
    ----------
    model
    features: list
        List of channels/markers to use as features in prediction
    target_populations: list
        List of populations from training data to predict
    multi_label: bool (default=False)
        If True, single cells can belong to more than one population. The tree structure of training data is
        NOT conserved - all resulting populations will have the same parent population.
    population_prefix: str (default="CellClassifier_")
        Prefix applied to populations generated

    Attributes
    ----------
    scaler: Scaler
        Scaler object
    transformer: Transformer
        Transformer object
    class_weights: dict
        Sample class weights; key is sample index, value is weight. Set by calling compute_class_weights.
    x: Pandas.DataFrame
        Training feature space
    y: numpy.ndarray
        Target labels
    features: list
    target_populations: list
    """

    def __init__(
        self,
        model: Union[ClassifierMixin, Sequential],
        features: list,
        target_populations: list,
        multi_label: bool = False,
        population_prefix: str = "CellClassifier_",
    ):
        super().__init__(
            model=model,
            x=None,
            y=None,
            features=features,
            transformer=None,
            multi_label=multi_label,
            population_prefix=population_prefix,
            target_populations=target_populations,
        )

    def load_training_data(self, experiment: Experiment, root_population: str, reference: str):
        """
        Load a FileGroup with existing Populations to serve as training data
        Parameters
        ----------
        experiment: Experiment
        root_population: str
        reference: str
            Name of the FileGroup to use as training data
        Returns
        -------
        self
        Raises
        ------
        AssertionError
            If FileGroup is invalid for reasons of missing target populations or
            an invalid hierarchy; populations must be downstream of the chosen
            root_population
        """
        ref = experiment.get_sample(reference)
        utils.assert_population_labels(ref=ref, expected_labels=self.target_populations)
        utils.check_downstream_populations(
            ref=ref,
            root_population=root_population,
            population_labels=self.target_populations,
        )
        if self.multi_label:
            self.x, self.y = utils.multilabel(
                ref=ref,
                root_population=root_population,
                population_labels=self.target_populations,
                features=self.features,
            )
        else:
            self.x, self.y = utils.singlelabel(
                ref=ref,
                root_population=root_population,
                population_labels=self.target_populations,
                features=self.features,
            )
        return self

    def _add_unclassified_population(
        self,
        x: pd.DataFrame,
        y_pred: np.ndarray,
        root_population: str,
        target: FileGroup,
    ):
        """
        Add a population for events without a classification (not a member of any target population)
        Parameters
        ----------
        x: Pandas.DataFrame
        y_pred: numpy.ndarray
        root_population: str
        target: FileGroup
        Returns
        -------
        None
        """
        idx = x.index.values[np.where(y_pred == 0)[0]]
        target.add_population(
            Population(
                population_name=f"{self.population_prefix}_Unclassified",
                source="classifier",
                index=idx,
                n=len(idx),
                parent=root_population,
                warnings=["supervised_classification"],
            )
        )

    def predict(
        self,
        experiment: Experiment,
        sample_id: str,
        root_population: str,
        threshold: float = 0.5,
        return_predictions: bool = True,
    ) -> Union[FileGroup, Dict] or Union[FileGroup, None]:
        """
        Calls predict on the root population of some unclassified FileGroup, generating
        new populations that will be immediate children of the chosen root population.
        Model must be trained prior to calling predict.
        Parameters
        ----------
        experiment: Experiment
        sample_id: str
        root_population: str
        threshold: float (default=0.5)
            Only relevant if multi_label is True. Labels will be assigned if probability is greater
            than or eaual to this threshold.
        return_predictions: bool (default=True)
            Return predicted labels and scores
        Returns
        -------
        FileGroup, Optional[Dict]
            Modified FileGroup with new populations, predictions as dictionary with keys 'y_pred' (predicted
            labels) and 'y_score' (probabilities)
        """
        target = experiment.get_sample(sample_id)
        x = target.load_population_df(population=root_population, transform=None)[self.features]
        if self.transformer is not None:
            x = self.transformer.scale(data=x, features=self.features)
        if self.scaler is not None:
            x = self.scaler(data=x, features=self.features)
        y_pred, y_score = self._predict(x=x, threshold=threshold)
        if not self.multi_label:
            self._add_unclassified_population(x=x, y_pred=y_pred, root_population=root_population, target=target)
        for i, pop in enumerate(self.target_populations):
            if self.multi_label:
                idx = x.index.values[np.where(y_pred[:, i + 1] == 1)[0]]
            else:
                idx = x.index.values[np.where(y_pred == i + 1)[0]]
            target.add_population(
                Population(
                    population_name=f"{self.population_prefix}_{pop}",
                    source="classifier",
                    index=idx,
                    n=len(idx),
                    parent=root_population,
                    warnings=["supervised_classification"],
                )
            )
        if return_predictions:
            return target, {"y_pred": y_pred, "y_score": y_score}
        return target, None

    def load_validation(self, experiment: Experiment, validation_id: str, root_population: str):
        """
        Load a FileGroup that has existing populations equivalent but not the same as the
        target populations e.g. identified by some other method (must share the same name as
        the target populations!). This will generate data from the FileGroup that can be used
        to validate the model.
        Parameters
        ----------
        experiment: Experiment
        validation_id: str
            Name of the FileGroup to load as validation data
        root_population: str
            Name of the root population from which the target populations descend
        Returns
        -------
        Pandas.DataFrame, numpy.ndarray
            Feature space, labels
        Raises
        ------
        AssertionError
            If target populations are missing in validation data
        """
        val = experiment.get_sample(validation_id)
        assert all(
            [x in val.tree.keys() for x in self.target_populations]
        ), f"Validation sample should contain the following populations: {self.target_populations}"
        if self.multi_label:
            x, y = utils.multilabel(
                ref=val,
                root_population=root_population,
                population_labels=self.target_populations,
                features=self.features,
            )
        else:
            x, y = utils.singlelabel(
                ref=val,
                root_population=root_population,
                population_labels=self.target_populations,
                features=self.features,
            )
        if self.transformer is not None:
            x = self.transformer.scale(data=x, features=self.features)
        if self.scaler is not None:
            x = self.scaler(data=x, features=self.features)
        return x, y

    def validate_classifier(
        self,
        experiment: Experiment,
        validation_id: str,
        root_population: str,
        threshold: float = 0.5,
        metrics: Optional[List[str]] = None,
        return_predictions: bool = True,
    ):
        """
        Load a FileGroup that has existing populations equivalent but not the same as the
        target populations e.g. identified by some other method (must share the same name as
        the target populations!). Data and labels will be generated (see load_validation) and
        then the model used to predict the labels of the validation data. The true labels
        will then be compared and validation performance reported.
        Parameters
        ----------
        experiment: Experiment
        validation_id: str
            Name of the FileGroup to load as validation data
        root_population: str
            Name of the root population from which the target populations descend
        threshold: float (default=0.5)
            Only relevant if multi_label is True. Labels will be assigned if probability is greater
            than or eaual to this threshold.
        return_predictions: bool (default=True)
            Return predicted labels and scores
        metrics: list, (default=["balanced_accuracy_score", "f1_weighted", "roc_auc_score"])
            List of metrics to report; should be string values pertaining to valid Scikit-Learn
            metric function e.g. 'balanced_accuracy' or 'roc_auc'
            (see https://scikit-learn.org/stable/modules/model_evaluation.html).
            Alternatively, list can contain callable functions with the key value in train/test
            dictionary being 'custom_metric_{index}'
        Returns
        -------
        dict, Optional[Dict]
            Validation performance, predictions as dictionary with keys 'y_pred' (predicted
            labels) and 'y_score' (probabilities)
        Raises
        ------
        AssertionError
            If target populations are missing in validation data
        """
        metrics = metrics or DEFAULT_METRICS
        x, y = self.load_validation(
            experiment=experiment,
            validation_id=validation_id,
            root_population=root_population,
        )
        y_pred, y_score = self._predict(x, threshold=threshold)
        results = utils.calc_metrics(metrics=metrics, y_true=y, y_pred=y_pred, y_score=y_score)
        if return_predictions:
            return results, {"y_pred": y_pred, "y_score": y_score}
        return results, None

    def plot_learning_curve(
        self,
        experiment: Optional[Experiment] = None,
        validation_id: Optional[str] = None,
        root_population: Optional[str] = None,
        ax: Optional[plt.Axes] = None,
        x_label: str = "Training examples",
        y_label: str = "Score",
        train_sizes: np.array or None = None,
        verbose: int = 1,
        **kwargs,
    ):
        x, y = self.x, self.y
        if validation_id is not None:
            assert all([x is not None for x in [experiment, root_population]]), (
                "For plotting learning curve for validation, must provide validation ID, experiment "
                "object, and root population"
            )
            x, y = self.load_validation(
                validation_id=validation_id,
                root_population=root_population,
                experiment=experiment,
            )
        return utils.plot_learning_curve(
            model=self.model,
            x=x,
            y=y,
            ax=ax,
            x_label=x_label,
            y_label=y_label,
            train_sizes=train_sizes,
            verbose=verbose,
            **kwargs,
        )


class CalibratedCellClassifier(BaseClassifier):
    def __init__(
        self,
        model: Union[ClassifierMixin, Sequential],
        experiment: Experiment,
        training_id: str,
        features: List[str],
        root_population: str,
        target_populations: List[str],
        meta_learner: Optional[ClassifierMixin, Sequential] = None,
        multi_label: bool = False,
        population_prefix: str = "CalibratedCellClassifier_",
        sample_size: int = 10000,
        sampling_method: str = "uniform",
        sampling_level: str = "file",
        sampling_kwargs: Optional[Dict] = None,
        transform: Optional[Union[str, Dict]] = "logicle",
        transform_kwargs: Optional[Dict] = None,
        targets: Optional[List[str]] = None,
        **harmony_kwargs,
    ):
        transform_kwargs = transform_kwargs or {}
        super().__init__(
            model=model,
            x=None,
            y=None,
            features=features,
            transformer=None if transform is None else TRANSFORMERS[transform](**transform_kwargs),
            multi_label=multi_label,
            population_prefix=population_prefix,
            target_populations=target_populations,
        )
        self.sample_size = sample_size
        if not experiment.sample_exists(sample_id=training_id):
            raise ValueError("Invalid training ID, does not exist for given experiment.")
        utils.assert_population_labels(experiment.get_sample(training_id), expected_labels=target_populations)
        utils.check_downstream_populations(
            ref=experiment.get_sample(training_id),
            root_population=root_population,
            population_labels=target_populations,
        )
        sample_ids = targets + [training_id] if targets is not None else None
        self.experiment = experiment
        self.root_population = root_population
        self.training_id = training_id
        self.data = single_cell_dataframe(
            experiment=experiment,
            populations=root_population,
            sample_size=sample_size,
            sampling_kwargs=sampling_kwargs,
            sampling_level=sampling_level,
            sampling_method=sampling_method,
            transform=transform,
            transform_kwargs=transform_kwargs,
            sample_ids=sample_ids,
        )
        self.calibrator = Harmony(
            data=self.data,
            features=features,
            transform=transform,
            transform_kwargs=transform_kwargs,
            **harmony_kwargs,
        )
        self.targets = None
        self.target_predictions = None
        self.meta_learner = self._default_meta(model=meta_learner)

    def _default_meta(self, model: Optional[Union[ClassifierMixin, Sequential]] = None):
        if model is None:
            return KNeighborsClassifier(n_neighbors=int(np.sqrt(self.sample_size)))
        else:
            return model

    @check_predictions
    def elbow(self):
        pass

    def calibrate(self):
        self.calibrator.run(var_use="sample_id")
        calibrated_data = self.calibrator.batch_corrected()
        calibrated_training_data = calibrated_data[calibrated_data.sample_id == self.training_id]
        self._setup_training_data(calibrated_training_data=calibrated_training_data)
        self.targets = calibrated_data[calibrated_data.sample_id != self.training_id]
        return self

    def _setup_training_data(self, calibrated_training_data: pd.DataFrame):
        reference = self.experiment.get_sample(sample_id=self.training_id)
        if self.multi_label:
            self.x, self.y = utils.multilabel(
                ref=reference,
                root_population=self.root_population,
                population_labels=self.target_populations,
                features=self.features,
                idx=calibrated_training_data["original_index"].values,
            )
        else:
            self.x, self.y = utils.singlelabel(
                ref=reference,
                root_population=self.root_population,
                population_labels=self.target_populations,
                features=self.features,
                idx=calibrated_training_data["original_index"].values,
            )

    @check_calibrated
    def predict(self, threshold: float = 0.5, verbose: bool = True):
        for target_id, target_df in progress_bar(self.targets.groupby("sample_id"), verbose=verbose):
            y_pred, y_score = self._predict(x=target_df[self.features], threshold=threshold)
            self.target_predictions[target_id] = {"y_pred": y_pred, "y_score": y_score}
        return self

    def meta_learner_fit_cv(self):
        pass

    def plot_learning_curve(
        self,
        meta: bool = False,
        target_id: Optional[str] = None,
        ax: Optional[plt.Axes] = None,
        x_label: str = "Training examples",
        y_label: str = "Score",
        train_sizes: np.array or None = None,
        verbose: int = 1,
        **kwargs,
    ):

        return utils.plot_learning_curve(
            model=self.model,
            x=x,
            y=y,
            ax=ax,
            x_label=x_label,
            y_label=y_label,
            train_sizes=train_sizes,
            verbose=verbose,
            **kwargs,
        )

    def _get_import_features(self):
        try:
            selector = SelectFromModel(estimator=self.model)
            selector.fit(self.x, self.y)
            return np.array(self.features)[selector.get_support()]
        except ValueError:
            return self.features

    def _map_predictions_to_target(self, y_pred: np.ndarray) -> (pd.DataFrame, np.ndarray, np.ndarray):
        original_space = self._target_data.loc[self._target_sample["original_index"].values]
        features = self.features if len(self.features) < 10 else self._get_import_features()
        clf = GridSearchCV(
            estimator=KNeighborsClassifier(),
            param_grid={
                "k": [
                    int(original_space.shape[0] * 0.01),
                    int(original_space.shape[0] * 0.05),
                    int(original_space.shape[0] * 0.1),
                ]
            },
        )
        clf.fit(original_space[features], y_pred)
        k = clf.best_params_["k"]
        meta_learner = KNeighborsClassifier(k=k)
        meta_learner.fit(original_space[features], y_pred)
        return meta_learner.predict(self._target_sample[features]), meta_learner.predict_proba(
            self._target_sample[features]
        )

    def predict(self) -> FileGroup:
        pass
