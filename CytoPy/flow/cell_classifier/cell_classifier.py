from ...feedback import progress_bar
from ...data.experiment import Experiment, FileGroup
from ...data.population import Population
from ...flow.transform import apply_transform, Scaler
from ...flow import sampling
from . import utils
from sklearn.model_selection import train_test_split, KFold, BaseCrossValidator
from imblearn.over_sampling import RandomOverSampler
from inspect import signature
import pandas as pd
import numpy as np
import logging

__author__ = "Ross Burton"
__copyright__ = "Copyright 2020, CytoPy"
__credits__ = ["Ross Burton", "Simone Cuff", "Andreas Artemiou", "Matthias Eberl"]
__license__ = "MIT"
__version__ = "1.0.0"
__maintainer__ = "Ross Burton"
__email__ = "burtonrj@cardiff.ac.uk"
__status__ = "Production"

DEFAULT_METRICS = ["balanced_accuracy_score", "f1_weighted", "roc_auc_score"]


def check_model_init(func):
    def wrapper(*args, **kwargs):
        assert args[0].model is not None, "Call 'build_model' prior to fit or predict"
        return func(*args, **kwargs)

    return wrapper


def check_data_init(func):
    def wrapper(*args, **kwargs):
        assert args[0].x is not None, "Call 'load_training_data' prior to fit"
        return func(*args, **kwargs)

    return wrapper


class CellClassifier:
    def __init__(self,
                 features: list,
                 target_populations: list,
                 multi_label: bool = False,
                 logging_level: int = logging.INFO,
                 log: str or None = None,
                 population_prefix: str = "CellClassifier_"):
        self._model = None
        self.features = features
        self.target_populations = target_populations
        self.multi_label = multi_label
        self.population_prefix = population_prefix
        self.x, self.y = None, None
        self.class_weights = None
        self.transformer = None
        self.scaler = None
        self._logging_level = logging_level
        self.logger = logging.getLogger("CellClassifier")
        self.logger.setLevel(logging_level)
        if log is not None:
            handler = logging.FileHandler(filename=log)
        else:
            handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
        self.logger.addHandler(handler)

    @property
    def model(self):
        return self._model

    @model.setter
    def model(self, _):
        raise ValueError("Model attribute is read-only. To load a pickled model use 'load_model' method")

    @check_model_init
    def set_params(self, **kwargs):
        _set_params = getattr(self.model, "set_params", None)
        if callable(_set_params):
            self.model.set_params(**kwargs)

    @property
    def logging_level(self):
        return self._logging_level

    @logging_level.setter
    def logging_level(self, value: int):
        self.logger.setLevel(value)
        self._logging_level = value

    def load_training_data(self,
                           experiment: Experiment,
                           reference: str,
                           root_population: str):
        ref = experiment.get_sample(reference)
        utils.assert_population_labels(ref=ref,
                                       expected_labels=self.target_populations)
        utils.check_downstream_populations(ref=ref,
                                           root_population=root_population,
                                           population_labels=self.target_populations)
        if self.multi_label:
            self.x, self.y = utils.multilabel(ref=ref,
                                              root_population=root_population,
                                              population_labels=self.target_populations,
                                              features=self.features)
        else:
            self.x, self.y = utils.singlelabel(ref=ref,
                                               root_population=root_population,
                                               population_labels=self.target_populations,
                                               features=self.features)
        return self

    @check_data_init
    def downsample(self,
                   method: str,
                   sample_size: int or float,
                   **kwargs):
        x = self.x.copy()
        x["y"] = self.y
        if method == "uniform":
            x = sampling.uniform_downsampling(data=x, sample_size=sample_size, **kwargs)
        elif method == "density":
            x = sampling.density_dependent_downsampling(data=x, sample_size=sample_size, **kwargs)
        elif method == "faithful":
            x = sampling.faithful_downsampling(data=x, **kwargs)
        else:
            valid = ['uniform', 'density', 'faithful']
            raise ValueError(f"Invalid method, must be one of {valid}")
        self.x, self.y = x[self.features], x["y"].values
        return self

    @check_data_init
    def transform(self,
                  method: str,
                  **kwargs):
        self.x, self.transformer = apply_transform(data=self.x,
                                                   features=self.features,
                                                   method=method,
                                                   return_transformer=True,
                                                   **kwargs)
        return self

    @check_data_init
    def oversample(self):
        self.x, self.y = RandomOverSampler(random_state=42).fit_resample(self.x, self.y)
        return self

    @check_data_init
    def scale(self,
              method: str = "standard",
              **kwargs):
        self.scaler = Scaler(method=method, **kwargs)
        self.x = self.scaler(data=self.x, features=self.features)
        return self

    @check_data_init
    def compute_class_weights(self):
        assert not self.multi_label, "Class weights not supported for multi-class classifiers"
        if self.model is not None:
            err = "Chosen model does not support use of class weights, perhaps try 'oversample' instead?"
            assert "sample_weight" in signature(self.model.fit).parameters.keys(), err
        self.class_weights = utils.auto_weights(y=self.y)
        return self

    @check_data_init
    @check_model_init
    def _fit(self,
             x: pd.DataFrame,
             y: np.ndarray,
             **kwargs):
        if self.class_weights is not None:
            self.model.fit(x, y, class_weight=self.class_weights, **kwargs)
        self.model.fit(x, y, **kwargs)

    @check_model_init
    def _predict(self, x: pd.DataFrame, threshold: float = 0.5):
        predict_proba = getattr(self.model, "predict_proba", None)
        if callable(predict_proba):
            return self.model.predict(x), self.model.predict_proba(x)
        return self.model.predict(x), None

    @check_model_init
    @check_data_init
    def fit_train_test_split(self,
                             test_frac: float = 0.3,
                             metrics: list or None = None,
                             return_predictions: bool = True,
                             train_test_split_kwargs: dict or None = None,
                             **fit_kwargs):
        metrics = metrics or DEFAULT_METRICS
        train_test_split_kwargs = train_test_split_kwargs or {}
        self.logger.info("Generating training and testing data")
        x_train, x_test, y_train, y_test = train_test_split(self.x,
                                                            self.y,
                                                            test_size=test_frac,
                                                            **train_test_split_kwargs)
        self.logger.info("Training model")
        self._fit(x=x_train, y=y_train, **fit_kwargs)
        results = dict()
        y_hat = dict()
        for key, (X, y) in zip(["train", "test"], [[x_train, y_train], [x_test, y_test]]):
            self.logger.info(f"Evaluating {key}ing performance....")
            y_pred, y_score = self._predict(X)
            y_hat[key] = {"y_pred": y_pred, "y_score": y_score}
            results[key] = utils.calc_metrics(metrics=metrics,
                                              y_pred=y_pred,
                                              y_score=y_score,
                                              y_true=y)
        if return_predictions:
            return results, y_hat
        return results, None

    @check_model_init
    @check_data_init
    def fit_cv(self,
               cross_validator: BaseCrossValidator or None = None,
               metrics: list or None = None,
               split_kwargs: dict or None = None,
               verbose: bool = True,
               **fit_kwargs):
        metrics = metrics or DEFAULT_METRICS
        split_kwargs = split_kwargs or {}
        cross_validator = cross_validator or KFold(n_splits=10, random_state=42, shuffle=True)
        training_results = list()
        testing_results = list()
        for train_idx, test_idx in progress_bar(cross_validator.split(X=self.x, y=self.y, **split_kwargs),
                                                total=cross_validator.n_splits,
                                                verbose=verbose):
            x_train, x_test = self.x.loc[train_idx], self.x.loc[test_idx]
            y_train, y_test = self.y[train_idx], self.y[test_idx]
            self._fit(x=x_train, y=y_train, **fit_kwargs)
            y_pred, y_score = self._predict(x=x_train)
            training_results.append(utils.calc_metrics(metrics=metrics,
                                                       y_pred=y_pred,
                                                       y_score=y_score,
                                                       y_true=y_train))
            y_pred, y_score = self._predict(x_test)
            testing_results.append(utils.calc_metrics(metrics=metrics,
                                                      y_pred=y_pred,
                                                      y_score=y_score,
                                                      y_true=y_test))
        return training_results, testing_results

    @check_model_init
    @check_data_init
    def fit(self, **kwargs):
        self._fit(x=self.x, y=self.y, **kwargs)
        return self

    def _add_unclassified_population(self,
                                     x: pd.DataFrame,
                                     y_pred: np.ndarray,
                                     root_population: str,
                                     target: FileGroup):
        idx = x.index.values[np.where(y_pred == 0)[0]]
        target.add_population(Population(population_name=f"{self.population_prefix}_Unclassified",
                                         source="classifier",
                                         index=idx,
                                         n=len(idx),
                                         parent=root_population,
                                         warnings=["supervised_classification"]))

    @check_model_init
    def predict(self,
                experiment: Experiment,
                sample_id: str,
                root_population: str,
                threshold: float = 0.5,
                return_predictions: bool = True):
        target = experiment.get_sample(sample_id)
        x = target.load_population_df(population=root_population,
                                      transform=None)[self.features]
        if self.transformer is not None:
            x = self.transformer.scale(data=x, features=self.features)
        if self.scaler is not None:
            x = self.scaler(data=x, features=self.features)
        y_pred, y_score = self._predict(x=x, threshold=threshold)
        if not self.multi_label:
            self._add_unclassified_population(x=x,
                                              y_pred=y_pred,
                                              root_population=root_population,
                                              target=target)
        for i, pop in enumerate(self.target_populations):
            if self.multi_label:
                idx = x.index.values[np.where(y_pred[:, i + 1] == 1)[0]]
            else:
                idx = x.index.values[np.where(y_pred == i + 1)[0]]
            target.add_population(Population(population_name=f"{self.population_prefix}_{pop}",
                                             source="classifier",
                                             index=idx,
                                             n=len(idx),
                                             parent=root_population,
                                             warnings=["supervised_classification"]))
        if return_predictions:
            return target, {"y_pred": y_pred, "y_score": y_score}
        return target, None

    def load_validation(self,
                        experiment: Experiment,
                        validation_id: str,
                        root_population: str):
        val = experiment.get_sample(validation_id)
        assert all([x in val.tree.keys() for x in self.target_populations]), \
            f"Validation sample should contain the following populations: {self.target_populations}"
        if self.multi_label:
            x, y = utils.multilabel(ref=val,
                                    root_population=root_population,
                                    population_labels=self.target_populations,
                                    features=self.features)
        else:
            x, y = utils.singlelabel(ref=val,
                                     root_population=root_population,
                                     population_labels=self.target_populations,
                                     features=self.features)
        if self.transformer is not None:
            x = self.transformer.scale(data=x, features=self.features)
        if self.scaler is not None:
            x = self.scaler(data=x, features=self.features)
        return x, y

    @check_model_init
    def validate_classifier(self,
                            experiment: Experiment,
                            validation_id: str,
                            root_population: str,
                            threshold: float = 0.5,
                            metrics: list or None = None,
                            return_predictions: bool = True):
        metrics = metrics or DEFAULT_METRICS
        x, y = self.load_validation(experiment=experiment,
                                    validation_id=validation_id,
                                    root_population=root_population)
        y_pred, y_score = self._predict(x, threshold=threshold)
        results = utils.calc_metrics(metrics=metrics,
                                     y_true=y,
                                     y_pred=y_pred,
                                     y_score=y_score)
        if return_predictions:
            return results, {"y_pred": y_pred, "y_score": y_score}
        return results, None
