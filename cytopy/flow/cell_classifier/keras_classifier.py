#!/usr/bin.env/python
# -*- coding: utf-8 -*-
"""
This module contains the base class KerasCellClassifier for using deep learning methods,
trained on some labeled FileGroup (has existing Populations), to predict single cell classifications.

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

from ..build_models import build_keras_model
from .cell_classifier import CellClassifier, check_data_init, check_model_init
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import History
from matplotlib.pyplot import Axes
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


class KerasCellClassifier(CellClassifier):
    """
    Use Keras deep learning models to predict the classification of single cell data.
    Training data should be provided in the form of a FileGroup with existing Populations.
    Supports multi-class and multi-label classification; if multi-label classification is chosen,
    the tree structure of training data is NOT conserved - all resulting populations
    will have the same parent population.

    Note, this class assumes you use the Keras Sequential API. Objects can be constructed using
    a pre-built model, or the model designed through the parameters 'optimizer', 'loss' and 'metrics,
    and then a model constructed using the 'build_model' method.

    Parameters
    ----------
    model: Sequential, optional
        Pre-compiled Keras Sequential model
    optimizer: str, optional
        Provide if you intend to compile a model with the 'build_model' method.
        See https://keras.io/api/optimizers/ for optimizers
    loss: str, optional
        Provide if you intend to compile a model with the 'build_model' method.
        See https://keras.io/api/losses/ for valid loss functions
    metrics: list, optional
        Provide if you intend to compile a model with the 'build_model' method.
        See https://keras.io/api/metrics/ for valid metrics
    features: list
        List of channels/markers to use as features in prediction
    target_populations: list
        List of populations from training data to predict
    multi_label: bool (default=False)
        If True, single cells can belong to more than one population. The tree structure of training data is
        NOT conserved - all resulting populations will have the same parent population.
    logging_level: int (default=logging.INFO)
        Level to log events at
    log: str, optional
        Path to log output to; if not given, will log to stdout
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
    logger: logging.Logger
    features: list
    target_populations: list
    """

    def __init__(
        self,
        model: Sequential or None = None,
        optimizer: str or None = None,
        loss: str or None = None,
        metrics: list or None = None,
        **kwargs
    ):
        self.optimizer = optimizer
        self.loss = loss
        self.metrics = metrics
        if model is not None:
            self.model = model
        else:
            if any([x is None for x in [optimizer, loss, metrics]]):
                raise ValueError(
                    "If model is not provided, must provide optimizer, loss and metrics, and "
                    "call 'build_model' prior to fit"
                )
        super().__init__(**kwargs)

    def build_model(
        self,
        layers: list,
        layer_params: list,
        input_shape: tuple or None = None,
        **compile_kwargs
    ):
        """
        If Sequential model is not constructed and provided at object construction, this method
        can be used to specify a sequential model to be built.

        Parameters
        ----------
        layers: list
            List of keras layer class names (see https://keras.io/api/layers/)
        layer_params: list
            List of parameters to use when constructing layers (order must match layers)
        input_shape: tuple, optional
            Shape of input data to first layer, if None, then passed as (N, ) where N is the number
            of features
        compile_kwargs:
            Additional keyword arguments passed when calling compile

        Returns
        -------
        self
        """
        if self.model is not None:
            raise ValueError("Model already defined.")
        input_shape = input_shape or (len(self.features),)
        self.model = build_keras_model(
            layers=layers,
            layer_params=layer_params,
            optimizer=self.optimizer,
            loss=self.loss,
            metrics=self.metrics,
            input_shape=input_shape,
            **compile_kwargs
        )
        return self

    @check_model_init
    def _predict(self, x: pd.DataFrame, threshold: float = 0.5):
        """
        Overrides parent _predict method to facilitate Keras predict methods. If multi_class is True,
        then threshold is used to assign labels using the predicted probabilities; positive association
        where probability exceeds threshold.

        Parameters
        ----------
        x: Pandas.DataFrame
            Feature space
        threshold: float (default=0.5)
            Threshold for positivity when multi_class is True

        Returns
        -------
        numpy.ndarray, numpy.ndarray
            Predicted labels, prediction probabilities
        """
        y_score = self.model.predict(x)
        if self.multi_label:
            y_pred = list(map(lambda yi: [int(i > threshold) for i in yi], y_score))
        else:
            y_pred = np.argmax(self.model.predict(x), axis=-1)
        return y_pred, y_score

    def _fit(
        self,
        x: pd.DataFrame,
        y: np.ndarray,
        epochs: int = 100,
        validation_x: pd.DataFrame or None = None,
        validation_y: np.ndarray or None = None,
        **kwargs
    ):
        """
        Overwrites the _fit method of CellClassifier to support Keras classifier.
        If a validation feature space and labels are provided, then these are passed
        to 'validation_data' of keras 'fit' method.

        Parameters
        ----------
        x: Pandas.DataFrame
            Training feature space to fit
        y: numpy.ndarray
            Training labels
        epochs: int (default=100)
            Number of training rounds
        validation_x: Pandas.DataFrame
            Validation feature space
        validation_y: numpy.ndarray
            Validation labels
        kwargs:
            Additional keyword arguments passed to fit method of Keras sequential API

        Returns
        -------
        Keras.callbacks.History
            Keras History object

        Raises
        ------
        AssertionError
            validation_y not provided but validation_x is
        """
        if validation_x is not None:
            assert (
                validation_y is not None
            ), "validation_y cannot be None if validation_x given"
            return self.model.fit(
                x,
                to_categorical(y),
                epochs=epochs,
                validation_data=(validation_x, validation_y),
                **kwargs
            )
        return self.model.fit(x, to_categorical(y), epochs=epochs, **kwargs)

    @check_model_init
    @check_data_init
    def fit(
        self,
        validation_frac: float or None = 0.3,
        train_test_split_kwargs: dict or None = None,
        epochs: int = 100,
        **kwargs
    ):
        """
        Fit the Keras model to the associated training data. If 'validation_frac' is provided,
        then a given proportion of the training data will be set apart and given to
        the 'validation_data' parameter to Keras fit method of the sequential API.
        The validation data is created using the train_test_split function from
        Scikit-Learn; additional keyword arguments can be provided as a dictionary
        with 'train_test_split_kwargs'.

        Parameters
        ----------
        validation_frac: float (optional; default=0.3)
            Proportion of training data to set aside for validation
        train_test_split_kwargs: dict (optional)
            Additional keyword arguments for train_test_split function from Scikit-Learn
        epochs: int (default=100)
            Number of training rounds
        kwargs:
            Additional keyword arguments passed to fit method of keras sequential API
        Returns
        -------
        Keras.callbacks.History
            Keras History object
        """
        train_test_split_kwargs = train_test_split_kwargs or {}
        validation_x, validation_y = None, None
        x, y = self.x, self.y
        if validation_frac is not None:
            x, validation_x, y, validation_y = train_test_split(
                self.x, self.y, test_size=validation_frac, **train_test_split_kwargs
            )
        return self._fit(
            x=x,
            y=y,
            validation_x=validation_x,
            validation_y=validation_y,
            epochs=epochs,
            **kwargs
        )

    @check_model_init
    def plot_learning_curve(
        self,
        history: History or None = None,
        ax: Axes or None = None,
        figsize: tuple = (10, 10),
        plot_kwargs: dict or None = None,
        **fit_kwargs
    ):
        """
        This method will generate a learning curve using the History object generated
        from the fit method from the Keras sequential API.

        Parameters
        ----------
        history: History (optional)
            If not given, then the 'fit' method will be called and use the associated
            training data.
        ax: Matplotlib.Axes
        figsize: tuple (default=(10,10))
        plot_kwargs: dict (optional)
            Keyword arguments passed to Pandas.DataFrame.plot method
        fit_kwargs:
            Keyword arguments passed to fit method if 'history' is not given

        Returns
        -------
        Matplotlib.Axes
        """
        history = history or self.fit(**fit_kwargs)
        plot_kwargs = plot_kwargs or {}
        ax = ax or plt.subplots(figsize=figsize)[1]
        return pd.DataFrame(history.history).plot(ax=ax, **plot_kwargs)
