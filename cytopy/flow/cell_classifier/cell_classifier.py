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
import os
from collections import defaultdict
from functools import wraps
from inspect import isclass
from inspect import signature
from typing import *

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from imblearn.over_sampling import RandomOverSampler
from sklearn.base import ClassifierMixin
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import BaseCrossValidator
from sklearn.model_selection import KFold
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import train_test_split
from sklearn.model_selection._search import BaseSearchCV
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


logger = logging.getLogger(__name__)

DEFAULT_METRICS = ["balanced_accuracy_score", "f1_weighted", "roc_auc_score"]


def check_data_init(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        assert args[0].x is not None, "Call 'load_training_data' prior to fit"
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

        if isclass(model):
            raise ClassifierError(
                "Given model is a class not an object. Model should be constructed before "
                "passing to Classifier. Parameters can be changed using 'set_params'"
            )

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
        x = sampling.sample_dataframe(data=x, sample_size=sample_size, method=method, **kwargs)
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

    def _predict(self, x: pd.DataFrame):
        """
        Internal function for calling predict.
        Parameters
        ----------
        x: Pandas.DataFrame
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
        return fit_train_test_split(
            model=self.model,
            x=self.x,
            y=self.y,
            test_frac=test_frac,
            metrics=metrics,
            return_predictions=return_predictions,
            train_test_split_kwargs=train_test_split_kwargs,
            **fit_kwargs,
        )

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
        Fit model with cross-validation.

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
        return fit_cv(
            model=self.model,
            x=self.x,
            y=self.y,
            cross_validator=cross_validator,
            metrics=metrics,
            split_kwargs=split_kwargs,
            verbose=verbose,
            **fit_kwargs,
        )

    @check_data_init
    def hyperparam_search(
        self,
        param_grid: Dict,
        hyper_param_optimizer: Optional[BaseSearchCV] = None,
        cv: int = 5,
        verbose: int = 1,
        n_jobs: int = -1,
        fit_kwargs: Optional[Dict] = None,
        **kwargs,
    ) -> BaseSearchCV:
        return hyperparam_search(
            model=self.model,
            x=self.x,
            y=self.y,
            param_grid=param_grid,
            hyper_param_optimizer=hyper_param_optimizer,
            cv=cv,
            verbose=verbose,
            n_jobs=n_jobs,
            fit_kwargs=fit_kwargs,
            **kwargs,
        )

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

    @check_data_init
    def plot_confusion_matrix(
        self,
        cmap: str or None = None,
        figsize: tuple = (10, 5),
        x: pd.DataFrame or None = None,
        y: np.ndarray or None = None,
        **kwargs,
    ):
        """
        Wraps cytopy.flow.supervised.confusion_matrix_plots (see for more details).
        Given some feature space and target labels, use the model to generate a confusion
        matrix heatmap. If x and y are not provided, will use associated training data.

        Parameters
        ----------
        cmap: str (optional)
            Colour scheme
        figsize: tuple (default=(10, 5))
            Figure size
        x: Pandas.DataFrame (optional)
            Feature space. If not given, will use associated training data. To use a validation
            dataset, use the 'load_validation' method to get relevant data.
        y: numpy.ndarray (optional)
            Target labels. If not given, will use associated training data. To use a validation
            dataset, use the 'load_validation' method to get relevant data.
        kwargs:
            Additional keyword arguments passed to cytopy.flow.supervised.confusion_matrix_plots

        Returns
        -------
        Matplotlib.Figure

        Raises
        ------
        AssertionError
            Invalid x, y input
        """
        assert not sum([x is not None, y is not None]) == 1, "Cannot provide x without y and vice-versa"
        if x is None:
            x, y = self.x, self.y
        return utils.confusion_matrix_plots(
            classifier=self.model,
            x=x,
            y=y,
            class_labels=["Unclassified"] + self.target_populations,
            cmap=cmap,
            figsize=figsize,
            **kwargs,
        )

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

    def _add_populations(self, target: FileGroup, x: pd.DataFrame, y_pred: np.ndarray, root_population: str):
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

    def predict(
        self,
        experiment: Experiment,
        sample_id: str,
        root_population: str,
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
        y_pred, y_score = self._predict(x=x)
        self._add_populations(target=target, x=x, y_pred=y_pred, root_population=root_population)
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


def check_calibrated(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        assert args[0].targets is not None, "Calibrator not defined"
        return func(*args, **kwargs)

    return wrapper


def check_target_predictions(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        assert args[0].target_predictions is not None, "Call 'predict_meta_labels' first"
        return func(*args, **kwargs)

    return wrapper


def check_valid_target(func):
    @wraps(func)
    def wrapper(*args, target_id: str, **kwargs):
        assert target_id in args[0].targets.sample_id.values, "Invalid target_id"
        return func(*args, target_id, **kwargs)

    return wrapper


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
        self.transform = transform
        self.transform_kwargs = transform_kwargs
        self.meta_learner = self._default_meta(model=meta_learner)

    def _default_meta(self, model: Optional[Union[ClassifierMixin, Sequential]] = None):
        if model is None:
            return KNeighborsClassifier(n_neighbors=int(np.sqrt(self.sample_size)))
        else:
            return model

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
    def predict_meta_labels(self, verbose: bool = True):
        self.target_predictions = {}
        for target_id, target_df in progress_bar(self.targets.groupby("sample_id"), verbose=verbose):
            y_pred, y_score = self._predict(x=target_df[self.features])
            self.target_predictions[target_id] = {"y_pred": y_pred, "y_score": y_score}
        return self

    @check_calibrated
    @check_target_predictions
    @check_valid_target
    def load_target_data(self, target_id: str, features: Optional[List[str]] = None, return_all_data: bool = False):
        features = features or self.features
        target = self.experiment.get_sample(sample_id=target_id)
        original_x = target.load_population_df(
            population=self.root_population, transform=self.transform, transform_kwargs=self.transform_kwargs
        )[features]
        idx = self.targets[self.targets.sample_id == target_id]["original_index"].values
        assert len(idx) > 0
        calibrated_x = original_x.loc[idx]
        y = self.target_predictions[target_id]["y_pred"]
        if return_all_data:
            return calibrated_x, y, original_x
        return calibrated_x, y

    def meta_fit(self, target_id: str, features: Optional[List[str]] = None, **fit_kwargs):
        calibrated_x, y, original_x = self.load_target_data(
            target_id=target_id, return_all_data=True, features=features
        )
        self.meta_learner.fit(calibrated_x, y, **fit_kwargs)

    def meta_fit_predict(self, target_id: str, features: Optional[List[str]] = None, **fit_kwargs):
        calibrated_x, y, original_x = self.load_target_data(
            target_id=target_id, return_all_data=True, features=features
        )
        self.meta_learner.fit(calibrated_x, y, **fit_kwargs)
        return predict(model=self.meta_learner, x=original_x)

    def meta_predict(self, target_id: str, features: Optional[List[str]] = None):
        calibrated_x, y, original_x = self.load_target_data(
            target_id=target_id, return_all_data=True, features=features
        )
        return predict(model=self.meta_learner, x=original_x)

    def meta_learner_fit_cv(
        self,
        features: Optional[List[str]] = None,
        target_id: Optional[Union[str, List[str]]] = None,
        cross_validator: Optional[BaseCrossValidator] = None,
        metrics: Optional[List[str]] = None,
        split_kwargs: Optional[Dict] = None,
        verbose: bool = True,
        **fit_kwargs,
    ):
        target_id = target_id or self.targets["sample_id"].unique()
        if isinstance(target_id, str):
            target_id = [target_id]
        training_results, testing_results = list(), list()
        for _id in target_id:
            logger.info(f"Fitting to {_id}")
            x, y = self.load_target_data(target_id=_id, features=features)
            train, test = fit_cv(
                model=self.meta_learner,
                x=x,
                y=y,
                cross_validator=cross_validator,
                metrics=metrics,
                split_kwargs=split_kwargs,
                verbose=verbose,
                **fit_kwargs,
            )
            train, test = pd.DataFrame(train), pd.DataFrame(test)
            train["sample_id"] = _id
            test["sample_id"] = _id
            training_results.append(train)
            testing_results.append(test)
        return pd.concat(training_results).reset_index(drop=True), pd.concat(testing_results).reset_index(drop=True)

    def meta_learner_fit_train_test_split(
        self,
        features: Optional[List[str]] = None,
        target_id: Optional[Union[str, List[str]]] = None,
        test_frac: float = 0.3,
        metrics: Optional[List[str]] = None,
        return_predictions: bool = True,
        train_test_split_kwargs: Optional[Dict] = None,
        **fit_kwargs,
    ):
        target_id = target_id or self.targets["sample_id"].unique()
        if isinstance(target_id, str):
            target_id = [target_id]
        all_results, predictions = list(), dict()
        for _id in target_id:
            logger.info(f"Fitting to {_id}")
            x, y = self.load_target_data(target_id=_id, features=features)
            results, y_hat = fit_train_test_split(
                model=self.meta_learner,
                x=x,
                y=y,
                test_frac=test_frac,
                metrics=metrics,
                return_predictions=return_predictions,
                train_test_split_kwargs=train_test_split_kwargs,
                **fit_kwargs,
            )
            results = pd.DataFrame(results)
            results["sample_id"] = _id
            all_results.append(results)
            if return_predictions:
                predictions[_id] = y_hat
        return pd.concat(all_results).reset_index(drop=True), predictions

    def meta_learner_hyperparam_tuning(
        self,
        param_grid: Dict,
        features: Optional[List[str]] = None,
        target_id: Optional[Union[str, List[str]]] = None,
        hyper_param_optimizer: Optional[BaseSearchCV] = None,
        cv: int = 5,
        verbose: bool = True,
        n_jobs: int = -1,
        fit_kwargs: Optional[Dict] = None,
        **kwargs,
    ):
        target_id = target_id or self.targets.sample_id.unique()
        if isinstance(target_id, str):
            target_id = [target_id]
        logger.info("Fitting hyper-parameter optimizers")
        optimizers = dict()
        results = defaultdict(list)
        for _id in progress_bar(target_id, verbose=verbose):
            x, y = self.load_target_data(target_id=_id, features=features)
            optimizers[_id] = hyperparam_search(
                model=self.meta_learner,
                x=x,
                y=y,
                param_grid=param_grid,
                hyper_param_optimizer=hyper_param_optimizer,
                cv=cv,
                n_jobs=n_jobs,
                fit_kwargs=fit_kwargs,
                **kwargs,
            )
            results["sample_id"].append(_id)
            results["best_score"].append(optimizers[_id].best_score_)
            for param, best_value in optimizers[_id].best_params_.items():
                results[param].append(best_value)
        return pd.DataFrame(results), optimizers

    def plot_learning_curve(
        self,
        meta: bool = False,
        target_id: Optional[str] = None,
        features: Optional[List[str]] = None,
        ax: Optional[plt.Axes] = None,
        x_label: str = "Training examples",
        y_label: str = "Score",
        train_sizes: np.array or None = None,
        verbose: int = 1,
        **kwargs,
    ):
        try:
            if meta:
                x, y = self.load_target_data(target_id=target_id, features=features)
                return utils.plot_learning_curve(
                    model=self.meta_learner,
                    x=x,
                    y=y,
                    ax=ax,
                    x_label=x_label,
                    y_label=y_label,
                    train_sizes=train_sizes,
                    verbose=verbose,
                    **kwargs,
                )
            else:
                return utils.plot_learning_curve(
                    model=self.model,
                    x=self.x,
                    y=self.y,
                    ax=ax,
                    x_label=x_label,
                    y_label=y_label,
                    train_sizes=train_sizes,
                    verbose=verbose,
                    **kwargs,
                )
        except KeyError as e:
            raise ClassifierError(f"Could not locate target predictions, has predict been called?; {e}.")

    def feature_importance(self):
        try:
            selector = SelectFromModel(estimator=self.model)
            selector.fit(self.x, self.y)
            important_features = np.array(self.features)[selector.get_support()]
            return important_features, selector
        except ValueError:
            logger.info("Chosen model does not have coefficient weights or feature importance available.")
            return np.array(self.features), None

    def set_meta_params(self, params: Dict):
        try:
            self.meta_learner.set_params(params=params)
        except AttributeError:
            logger.error(
                "Chosen meta-learner does not have method 'set_params'; hyper-parameters will not be updated."
            )

    def meta_fit_predict_populations(
        self,
        target_id: Optional[Union[str, List[str]]] = None,
        features: Optional[List[str]] = None,
        params: Optional[Union[Dict, List[Dict]]] = None,
        verbose: bool = True,
        return_filegroup_only: bool = True,
        **fit_kwargs,
    ):
        target_id = target_id or self.targets.sample_id.unique()
        results = {}

        if isinstance(target_id, str):
            target_id = [target_id]

        if isinstance(target_id, list) and isinstance(params, list):
            if len(target_id) != len(params):
                raise ClassifierError("Number of target IDs does not match number of parameter dictionaries.")

        if isinstance(params, dict):
            self.set_meta_params(params=params)

        for i, _id in progress_bar(enumerate(target_id), verbose=verbose):

            if isinstance(params, list):
                self.set_meta_params(params=params[i])
            y_pred, y_score = self.meta_fit_predict(target_id=_id, features=features, **fit_kwargs)
            _, _, x = self.load_target_data(target_id=_id, return_all_data=True, features=features)
            fg = self.experiment.get_sample(sample_id=_id)
            self._add_populations(target=fg, x=x, y_pred=y_pred, root_population=self.root_population)
            if return_filegroup_only:
                results[_id] = fg
            else:
                results[_id] = {"filegroup": fg, "y_pred": y_pred, "y_score": y_score}
        return results


def predict(model: Union[ClassifierMixin, Sequential], x: pd.DataFrame):
    predict_proba = getattr(model, "predict_proba", None)
    if callable(predict_proba):
        return model.predict(x), model.predict_proba(x)
    return model.predict(x), None


def fit_cv(
    model: Union[ClassifierMixin, Sequential],
    x: pd.DataFrame,
    y: np.ndarray,
    cross_validator: Optional[BaseCrossValidator] = None,
    metrics: Optional[List[str]] = None,
    split_kwargs: Optional[Dict] = None,
    verbose: bool = True,
    **fit_kwargs,
):
    metrics = metrics or DEFAULT_METRICS
    split_kwargs = split_kwargs or {}
    cross_validator = cross_validator or KFold(n_splits=10, random_state=42, shuffle=True)
    training_results = list()
    testing_results = list()
    for train_idx, test_idx in progress_bar(
        cross_validator.split(X=x.values, y=y, **split_kwargs),
        total=cross_validator.n_splits,
        verbose=verbose,
    ):
        x_train, x_test = x.values[train_idx], x.values[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        model.fit(x_train, y_train, **fit_kwargs)
        y_pred, y_score = predict(model=model, x=x_train)
        training_results.append(utils.calc_metrics(metrics=metrics, y_pred=y_pred, y_score=y_score, y_true=y_train))
        y_pred, y_score = predict(model=model, x=x_test)
        testing_results.append(utils.calc_metrics(metrics=metrics, y_pred=y_pred, y_score=y_score, y_true=y_test))
    return training_results, testing_results


def fit_train_test_split(
    model: Union[ClassifierMixin, Sequential],
    x: pd.DataFrame,
    y: np.ndarray,
    test_frac: float = 0.3,
    metrics: Optional[List[str]] = None,
    return_predictions: bool = True,
    train_test_split_kwargs: Optional[Dict] = None,
    **fit_kwargs,
):
    metrics = metrics or DEFAULT_METRICS
    train_test_split_kwargs = train_test_split_kwargs or {}
    logger.info("Generating training and testing data")
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_frac, **train_test_split_kwargs)
    logger.info("Training model")
    model.fit(x_train, y_train, **fit_kwargs)
    results = dict()
    y_hat = dict()
    for key, (X, y) in zip(["train", "test"], [[x_train, y_train], [x_test, y_test]]):
        logger.info(f"Evaluating {key}ing performance....")
        y_pred, y_score = predict(model=model, x=X)
        y_hat[key] = {"y_pred": y_pred, "y_score": y_score}
        results[key] = utils.calc_metrics(metrics=metrics, y_pred=y_pred, y_score=y_score, y_true=y)
    if return_predictions:
        return results, y_hat
    return results, None


def hyperparam_search(
    model: Union[ClassifierMixin, Sequential],
    x: pd.DataFrame,
    y: np.ndarray,
    param_grid: Dict,
    hyper_param_optimizer: Optional[BaseSearchCV] = None,
    cv: int = 5,
    verbose: int = 1,
    n_jobs: int = -1,
    fit_kwargs: Optional[Dict] = None,
    **kwargs,
) -> BaseSearchCV:
    fit_kwargs = fit_kwargs or {}
    hyper_param_optimizer = hyper_param_optimizer or RandomizedSearchCV
    hyper_param_optimizer = hyper_param_optimizer(model, param_grid, cv=cv, verbose=verbose, n_jobs=n_jobs, **kwargs)
    hyper_param_optimizer.fit(x, y, **fit_kwargs)
    return hyper_param_optimizer
