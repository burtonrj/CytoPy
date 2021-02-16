from ...data.experiment import Experiment
from ..build_models import build_sklearn_model
from .cell_classifier import CellClassifier, check_data_init, check_model_init
from . import utils
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, learning_curve
from matplotlib.pyplot import Axes
from inspect import signature
from warnings import warn
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import pickle


def _valid_multi_label(klass: str):
    """
    Checks if the specified Scikit-Learn class is valid for
    multi-label classification. If not, raises AssertionError.

    Parameters
    ----------
    klass: str

    Returns
    -------
    None
    """
    valid = ["DescisionTreeClassifier",
             "ExtraTreeClassifier",
             "ExtraTreesClassifier",
             "KNeighborsClassifier",
             "MLPClassifier",
             "RadiusNeighborsClassifier",
             "RandomForestClassifier",
             "RidgeClassifierCV"]
    err = f"Invalid Scikit-Learn class for multi-label classification, should be one of: {valid}"
    assert klass in valid, err


class SklearnCellClassifier(CellClassifier):
    def __init__(self,
                 klass: str,
                 params: dict,
                 multi_label: bool = False,
                 **kwargs):
        if multi_label:
            _valid_multi_label(klass)
        super().__init__(multi_label=multi_label, **kwargs)
        self.klass = klass
        self.params = params

    def inject_model(self, model: object):
        self._model = model

    def build_model(self):
        """
        Call prior to fit or predict. Initiates model and associates
        to self.model.

        Returns
        -------
        None
        """
        params = self.params or {}
        self._model = build_sklearn_model(klass=self.klass, **params)
        if self.class_weights:
            err = "Class weights defined yet the specified model does not support this"
            assert "sample_weight" in signature(self.model.fit).parameters.keys(), err
        return self

    @check_model_init
    def _predict(self,
                 x: pd.DataFrame,
                 threshold: float = 0.5):
        """
        Overrides CellClassifier._predict. Checks that the model has
        been initialised and calls relevant predict methods. If the model
        supports prediction probabilities directly, the 'predict_proba'
        method will be called, otherwise the probability of the positive
        class is determined using the 'decision_function' method.
        If multi_class is True, then positive association to a class is
        attributed to classes with a probability that exceeds the given
        threshold.

        Parameters
        ----------
        x: Pandas.DataFrame
            Feature space
        threshold: float (default=0.5)
            Threshold of positivity for multi-class prediction

        Returns
        -------
        Numpy.Array, Numpy.Array
            Predicted labels, prediction probabilities
        """
        if callable(getattr(self.model, "predict_proba", None)):
            y_score = self.model.predict_proba(x[self.features])
        else:
            y_score = self.model.decision_function(x[self.features])
        if self.multi_label:
            y_pred = list(map(lambda yi: [int(i > threshold) for i in yi], y_score))
        else:
            y_pred = self.model.predict(x[self.features])
        return y_pred, y_score

    @check_model_init
    def _fit(self, x: pd.DataFrame, y: np.ndarray, **kwargs):
        """
        Fits the model to loaded training data. If "load_data" is not
        called prior to this method, will raise AssertionError.
        If class_weights are defined the signature of the chosen model
        will be inspected. If sample_weight is supported, class weights
        will be imposed, otherwise a warning will be raised.

        Parameters
        ----------
        x: Pandas.DataFrame
            Feature space
        y: Numpy.Array
            Target labels
        kwargs
            Additional keyword arguments pass to "fit"

        Returns
        -------
        None
        """
        if self.class_weights is not None:
            if "sample_weight" in signature(self.model.fit).parameters.keys():
                sample_weight = np.array([self.class_weights.get(i) for i in y])
                self.model.fit(x, y, sample_weight=sample_weight, **kwargs)
            else:
                warn("Class weights defined yet the specified model does not support this.")
                self.model.fit(x, y, **kwargs)
        else:
            self.model.fit(x, y, **kwargs)

    @check_model_init
    @check_data_init
    def hyperparameter_tuning(self,
                              param_grid: dict,
                              method: str = "grid_search",
                              **kwargs):
        """
        Perform hyperparameter tuning using either exhaustive grid search
        or randomised grid search.

        Parameters
        ----------
        param_grid: dict
            Search space
        method: str (default="grid_search")
            Should either be "grid_search" or "random"
        kwargs:
            Keyword arguments passed to grid search method

        Returns
        -------
        GridSearchCV or RandomizedSearchCV
        """
        assert method in ["grid_search", "random"], "Method should either be 'grid_search' for " \
                                                    "exhaustive search or 'random' for randomised " \
                                                    "grid search"
        if method == "grid_search":
            search_cv = GridSearchCV(estimator=self.model,
                                     param_grid=param_grid,
                                     **kwargs)
        else:
            search_cv = RandomizedSearchCV(estimator=self.model,
                                           param_distributions=param_grid,
                                           **kwargs)
        search_cv.fit(self.x, self.y)
        return search_cv

    @check_model_init
    def plot_learning_curve(self,
                            experiment: Experiment or None = None,
                            validation_id: str or None = None,
                            root_population: str or None = None,
                            ax: Axes or None = None,
                            x_label: str = "Training examples",
                            y_label: str = "Score",
                            train_sizes: np.array or None = None,
                            verbose: int = 1,
                            **kwargs):
        """
        This method will generate a learning curve using the Scikit-Learn utility function
        sklearn.model_selection.learning_curve.
        Either use the associated training data or a validation FileGroup by providing
        the Experiment object and the ID for the validation sample (validation_id).
        This validation sample should contain the same populations as the training data,
        which must be downstream of the 'root_population'.

        Parameters
        ----------
        experiment: Experiment (optional)
            If provided, should be the same Experiment training data was derived from
        validation_id: str (optional)
            Name of the sample to use for validation
        root_population: str (optional)
            If not given, will use the same root_population as training data
        ax: Matplotlib.Axes (optional)
            Axes object to use to draw plot
        x_label: str (default="Training examples")
            X-axis labels
        y_label: str (default="Score")
            Y-axis labels
        train_sizes: Numpy.Array (optional)
            Defaults to linear range between 0.1 and 1.0, with 10 steps
        kwargs:
            Additional keyword arguments passed to sklearn.model_selection.learning_curve

        Returns
        -------
        Matplotlib.Axes
        """
        x, y = self.x, self.y
        if validation_id is not None:
            assert all([x is not None for x in [experiment, root_population]]), \
                "For plotting learning curve for validation, must provide validation ID, experiment " \
                "object, and root population"
            x, y = self.load_validation(validation_id=validation_id,
                                        root_population=root_population,
                                        experiment=experiment)
        train_sizes = train_sizes or np.linspace(0.1, 1.0, 10)
        ax = ax or plt.subplots(figsize=(5, 5))[1]
        train_sizes, train_scores, test_scores = learning_curve(self.model,
                                                                x,
                                                                y,
                                                                verbose=verbose,
                                                                return_times=False,
                                                                train_sizes=train_sizes,
                                                                **kwargs)
        train_scores_mean = np.mean(train_scores, axis=1)
        train_scores_std = np.std(train_scores, axis=1)
        test_scores_mean = np.mean(test_scores, axis=1)
        test_scores_std = np.std(test_scores, axis=1)
        ax.grid()
        ax.fill_between(train_sizes, train_scores_mean - train_scores_std,
                        train_scores_mean + train_scores_std, alpha=0.1,
                        color="r")
        ax.fill_between(train_sizes, test_scores_mean - test_scores_std,
                        test_scores_mean + test_scores_std, alpha=0.1,
                        color="g")
        ax.plot(train_sizes, train_scores_mean, 'o-', color="r",
                label="Training score")
        ax.plot(train_sizes, test_scores_mean, 'o-', color="g",
                label="Cross-validation score")
        ax.legend(loc="best")
        ax.set_xlabel(xlabel=x_label)
        ax.set_ylabel(ylabel=y_label)
        return ax

    def plot_confusion_matrix(self,
                              cmap: str or None = None,
                              figsize: tuple = (10, 5),
                              x: pd.DataFrame or None = None,
                              y: np.ndarray or None = None,
                              **kwargs):
        """
        Wraps CytoPy.flow.supervised.confusion_matrix_plots (see for more details).
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
        y: Numpy.Array (optional)
            Target labels. If not given, will use associated training data. To use a validation
            dataset, use the 'load_validation' method to get relevant data.
        kwargs:
            Additional keyword arguments passed to CytoPy.flow.supervised.confusion_matrix_plots

        Returns
        -------

        """
        assert not sum([x is not None, y is not None]) == 1, "Cannot provide x without y and vice-versa"
        if x is None:
            x, y = self.x, self.y
        assert sum([i is None for i in [x, y]]) in [0, 2], \
            "If you provide 'x' you must provide 'y' and vice versa."
        return utils.confusion_matrix_plots(classifier=self.model,
                                            x=x,
                                            y=y,
                                            class_labels=["Unclassified"] + self.target_populations,
                                            cmap=cmap,
                                            figsize=figsize,
                                            **kwargs)

    def save_model(self, path: str, **kwargs):
        """
        Pickle the associated model and save to disk. WARNING: be aware of continuity issues.
        Compatibility with new releases of Scikit-Learn and CytoPy are not guaranteed.

        Parameters
        ----------
        path: str
            Where to save on disk
        kwargs:
            Additional keyword arguments passed to pickle.dump call

        Returns
        -------
        None
        """
        pickle.dump(self.model, open(path, "wb"), **kwargs)

    def load_model(self, path: str, **kwargs):
        """
        Load a pickled model from disk. WARNING: be aware of continuity issues.
        Compatibility with new releases of Scikit-Learn and CytoPy are not guaranteed.
        The loaded model must correspond to the expected method for this CellClassifier.

        Parameters
        ----------
        path: str
            Where to save on disk
        kwargs:
            Additional keyword arguments passed to pickle.dump call

        Returns
        -------
        None
        """
        model = pickle.load(open(path, "rb"), **kwargs)
        assert self.klass in str(type(model)), \
            f"Loaded model does not match Classifier, expected type: {self.klass}"
        self._model = model
