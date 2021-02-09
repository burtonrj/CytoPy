from ..build_models import build_keras_model
from .cell_classifier import CellClassifier, check_data_init, check_model_init
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import History
from matplotlib.pyplot import Axes
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


class KerasCellClassifier(CellClassifier):
    def __init__(self,
                 optimizer: str,
                 loss: str,
                 metrics: list,
                 **kwargs):
        self.optimizer = optimizer
        self.loss = loss
        self.metrics = metrics
        super().__init__(**kwargs)

    def build_model(self,
                    layers: list,
                    layer_params: list,
                    **compile_kwargs):
        self._model = build_keras_model(layers=layers,
                                        layer_params=layer_params,
                                        optimizer=self.optimizer,
                                        loss=self.loss,
                                        metrics=self.metrics,
                                        **compile_kwargs)

    @check_model_init
    def _predict(self,
                 x: pd.DataFrame,
                 threshold: float = 0.5):
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
        Numpy.Array, Numpy.Array
            Predicted labels, prediction probabilities
        """
        y_score = self.model.predict(x)
        if self.multi_label:
            y_pred = list(map(lambda yi: [int(i > threshold) for i in yi], y_score))
        else:
            y_pred = self.model.predict_classes(x)
        return y_pred, y_score

    def _fit(self,
             x: pd.DataFrame,
             y: np.ndarray,
             epochs: int = 100,
             validation_x: pd.DataFrame or None = None,
             validation_y: np.ndarray or None = None,
             **kwargs):
        """
        Overwrites the _fit method of CellClassifier to support Keras classifier.
        If a validation feature space and labels are provided, then these are passed
        to 'validation_data' of keras 'fit' method.

        Parameters
        ----------
        x: Pandas.DataFrame
            Training feature space to fit
        y: Numpy.Array
            Training labels
        epochs: int (default=100)
            Number of training rounds
        validation_x: Pandas.DataFrame
            Validation feature space
        validation_y: Numpy.Array
            Validation labels
        kwargs:
            Additional keyword arguments passed to fit method of Keras sequential API

        Returns
        -------
        Keras.callbacks.History
            Keras History object
        """
        if validation_x is not None:
            assert validation_y is not None, "validation_y cannot be None if validation_x given"
            return self.model.fit(x, to_categorical(y), epochs=epochs,
                                  validation_data=(validation_x, validation_y), **kwargs)
        return self.model.fit(x, to_categorical(y), epochs=epochs, **kwargs)

    @check_model_init
    @check_data_init
    def fit(self,
            validation_frac: float or None = 0.3,
            train_test_split_kwargs: dict or None = None,
            epochs: int = 100,
            **kwargs):
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
            x, validation_x, y, validation_y = train_test_split(self.x,
                                                                self.y,
                                                                test_size=validation_frac,
                                                                **train_test_split_kwargs)
        return self._fit(x=x, y=y, validation_x=validation_x, validation_y=validation_y, epochs=epochs, **kwargs)

    @check_model_init
    def plot_learning_curve(self,
                            history: History or None = None,
                            ax: Axes or None = None,
                            figsize: tuple = (10, 10),
                            plot_kwargs: dict or None = None,
                            **fit_kwargs):
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
