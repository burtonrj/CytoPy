from ..feedback import vprint, progress_bar
from ..flow.transforms import scaler
from ..flow import supervised
from ..flow import sampling
from .experiment import Experiment, FileGroup
from .population import Population, create_signature
from imblearn.over_sampling import RandomOverSampler
from sklearn.model_selection import train_test_split, KFold, learning_curve, \
    BaseCrossValidator, GridSearchCV, RandomizedSearchCV
from inspect import signature
from matplotlib.axes import Axes
from warnings import warn
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import mongoengine
import pickle

DEFAULT_METRICS = ["balanced_accuracy_score", "f1_weighted", "roc_auc_score"]


class CellClassifier(mongoengine.Document):
    name = mongoengine.StringField(required=True, unique=True)
    features = mongoengine.ListField(required=True)
    multi_class = mongoengine.BooleanField(default=False)
    target_populations = mongoengine.ListField(required=True)
    transform = mongoengine.StringField(default="logicle")
    scale = mongoengine.StringField(choices=["standard", "norm"])
    scale_kwargs = mongoengine.DictField()
    oversample = mongoengine.BooleanField(default=False)
    class_weights = mongoengine.DictField()
    downsample = mongoengine.StringField(choices=['uniform', 'density', 'faithful'])
    downsample_kwargs = mongoengine.DictField()
    population_prefix = mongoengine.StringField(default="sml")

    meta = {"allow_inheritance": True}

    def __init__(self, *args, **values):
        self.verbose = values.pop("verbose", True)
        self.print = vprint(self.verbose)
        self._model = None
        self.x, self.y = None, None
        super().__init__(*args, **values)

    @property
    def model(self):
        return self._model

    @model.setter
    def model(self, _):
        raise ValueError("Model attribute is read-only. To use a different Scikit-Learn class "
                         "define a new CellClassifier. To load a pickled model use 'load_model' method")

    def check_model_init(self):
        assert self.model is not None, "Call 'build_model' prior to fit or predict"

    def check_data_init(self):
        assert self.x is not None, "Call 'load_training_data' prior to fit"

    def load_training_data(self,
                           experiment: Experiment,
                           reference: str,
                           root_population: str):
        self.print("Loading reference sample...")
        ref = experiment.get_sample(reference)
        self.print("Checking population labels...")
        supervised.assert_population_labels(ref=ref, expected_labels=self.target_populations)
        supervised.check_downstream_populations(ref=ref,
                                                root_population=root_population,
                                                population_labels=self.target_populations)
        self.print("Creating training data...")
        if self.multi_class:
            self.x, self.y = supervised.multilabel(ref=ref,
                                                   root_population=root_population,
                                                   population_labels=self.target_populations,
                                                   features=self.features,
                                                   transform=self.transform)
        else:
            self.x, self.y = supervised.singlelabel(ref=ref,
                                                    root_population=root_population,
                                                    population_labels=self.target_populations,
                                                    features=self.features,
                                                    transform=self.transform)
        if self.scale:
            self.print("Scaling data...")
            self.x = self._scale_data(data=self.x)
        else:
            warn("For the majority of classifiers it is recommended to scale the data (exception being tree-based "
                 "algorithms)")
        if self.downsample:
            self.print("Down-sampling...")
            self._downsample(x=self.x, y=self.y)
        if self.oversample:
            self.x, self.y = RandomOverSampler(random_state=42).fit_resample(self.x, self.y)
        self.print('Ready for training!')

    def _scale_data(self, data: pd.DataFrame):
        kwargs = self.scale_kwargs or {}
        data[self.features] = scaler(data,
                                     return_scaler=False,
                                     scale_method=self.scale,
                                     **kwargs)
        return data

    def auto_class_weights(self):
        self.check_data_init()
        assert not self.multi_class, "Class weights not supported for multi-class classifiers"
        self.class_weights = supervised.auto_weights(y=self.y)

    def _downsample(self,
                    x: pd.DataFrame,
                    y: np.ndarray):
        x["y"] = y
        valid = ['uniform', 'density', 'faithful']
        kwargs = self.downsample_kwargs or {}
        assert self.downsample in valid, f"Downsample should have a value of: {valid}"
        if self.downsample == "uniform":
            if "sample_size" not in kwargs.keys():
                warn("No sample_size given, defaulting to 0.5 (will half data)")
            x = sampling.uniform_downsampling(data=x, sample_size=kwargs.get("sample_size", 0.5))
        elif self.downsample == "density":
            x = sampling.density_dependent_downsampling(data=x, **kwargs)
        else:
            if "h" not in kwargs.keys():
                warn("Parameter 'h' not given for faithful downsampling, defaulting to 0.01")
            x = sampling.faithful_downsampling(data=x, h=kwargs.get("h", 0.01))
        y = x["y"].values
        x.drop("y", inplace=True)
        return x, y

    def _fit(self, x: pd.DataFrame, y: np.ndarray, **kwargs):
        self.check_model_init()
        if self.class_weights is not None:
            self.model.fit(x, y, class_weight=self.class_weights, **kwargs)
        self.model.fit(x, y, **kwargs)

    def _predict(self, x: pd.DataFrame, threshold: float):
        self.check_model_init()
        return self.model.predict(x), self.model.predict_proba(x)

    def fit_train_test_split(self,
                             test_frac: float = 0.3,
                             metrics: list or None = None,
                             return_predictions: bool = True,
                             threshold: float = 0.5,
                             train_test_split_kwargs: dict or None = None,
                             fit_kwargs: dict or None = None):
        self.check_data_init()
        self.print("==========================================")
        metrics = metrics or DEFAULT_METRICS
        train_test_split_kwargs = train_test_split_kwargs or {}
        fit_kwargs = fit_kwargs or {}
        self.print("Spliting data into training and testing sets....")
        x_train, x_test, y_train, y_test = train_test_split(self.x,
                                                            self.y,
                                                            test_size=test_frac,
                                                            **train_test_split_kwargs)
        self.print("Fitting model....")
        self._fit(x=x_train, y=y_train, **fit_kwargs)
        results = dict()
        y_hat = dict()
        for key, (X, y) in zip(["train", "test"], [[x_train, y_train], [x_test, y_test]]):
            self.print(f"Evaluating {key}ing performance....")
            y_pred, y_score = self._predict(X, threshold=threshold)
            y_hat[key] = {"y_pred": y_pred, "y_score": y_score}
            results[key] = supervised.calc_metrics(metrics=metrics,
                                                   y_pred=y_pred,
                                                   y_score=y_score,
                                                   y_true=y)
        self.print("==========================================")
        if return_predictions:
            return results, y_hat
        return results

    def fit_cv(self,
               cross_validator: BaseCrossValidator or None = None,
               metrics: list or None = None,
               threshold: float = 0.5,
               split_kwargs: dict or None = None,
               fit_kwargs: dict or None = None):
        metrics = metrics or DEFAULT_METRICS
        split_kwargs = split_kwargs or {}
        fit_kwargs = fit_kwargs or {}
        cross_validator = cross_validator or KFold(n_splits=10, random_state=42, shuffle=True)
        training_results = list()
        testing_results = list()
        for train_idx, test_idx in progress_bar(cross_validator.split(self.x, **split_kwargs),
                                                verbose=self.verbose):
            x_train, x_test = self.x.values[train_idx], self.x.values[test_idx]
            y_train, y_test = self.y[train_idx], self.y[test_idx]
            self._fit(x=x_train, y=y_train, **fit_kwargs)
            y_pred, y_score = self._predict(x=x_train, threshold=threshold)
            training_results.append(supervised.calc_metrics(metrics=metrics,
                                                            y_pred=y_pred,
                                                            y_score=y_score,
                                                            y_true=y_train))
            y_pred, y_score = self._predict(x_test, threshold=threshold)
            testing_results.append(supervised.calc_metrics(metrics=metrics,
                                                           y_pred=y_pred,
                                                           y_score=y_score,
                                                           y_true=y_test))
        return training_results, testing_results

    def fit(self, **kwargs):
        self._fit(x=self.x, y=self.y, **kwargs)

    def _add_unclassified_population(self,
                                     x: pd.DataFrame,
                                     y_pred: np.ndarray,
                                     root_population: str,
                                     target: FileGroup):
        idx = x.index.values[np.where(y_pred == 0)[0]]
        target.add_population(Population(population_name=f"{self.population_prefix}_Unclassified",
                                         index=idx,
                                         n=len(idx),
                                         parent=root_population,
                                         warnings=["supervised_classification"],
                                         signature=create_signature(data=x, idx=idx)))

    def predict(self,
                experiment: Experiment,
                sample_id: str,
                root_population: str,
                threshold: float = 0.5,
                return_predictions: bool = True):
        self.check_model_init()
        target = experiment.get_sample(sample_id)
        x = target.load_population_df(population=root_population,
                                      transform=self.transform)[self.features]
        y_pred, y_score = self._predict(x=x, threshold=threshold)
        if not self.multi_class:
            self._add_unclassified_population(x=x,
                                              y_pred=y_pred,
                                              root_population=root_population,
                                              target=target)
        for i, pop in enumerate(self.target_populations):
            if self.multi_class:
                idx = x.index.values[np.where(y_pred[:, i + 1] == 1)[0]]
            else:
                idx = x.index.values[np.where(y_pred == i + 1)[0]]
            target.add_population(Population(population_name=f"{self.population_prefix}_{pop}",
                                             index=idx,
                                             n=len(idx),
                                             parent=root_population,
                                             warnings=["supervised_classification"],
                                             signature=create_signature(data=x, idx=idx)))
        if return_predictions:
            return target, {"y_pred": y_pred, "y_score": y_score}
        return target

    def load_validation(self,
                        experiment: Experiment,
                        validation_id: str,
                        root_population: str):
        val = experiment.get_sample(validation_id)
        assert all([x in val.tree.keys() for x in self.target_populations]), \
            f"Validation sample should contain the following populations: {self.target_populations}"
        if self.multi_class:
            x, y = supervised.multilabel(ref=val,
                                         root_population=root_population,
                                         population_labels=self.target_populations,
                                         transform=self.transform,
                                         features=self.features)
        else:
            x, y = supervised.singlelabel(ref=val,
                                          root_population=root_population,
                                          population_labels=self.target_populations,
                                          transform=self.transform,
                                          features=self.features)
        if self.scale:
            x = self._scale_data(data=x)
        return x, y

    def validate_classifier(self,
                            experiment: Experiment,
                            validation_id: str,
                            root_population: str,
                            threshold: float = 0.5,
                            metrics: list or None = None,
                            return_predictions: bool = True):
        self.check_model_init()
        metrics = metrics or DEFAULT_METRICS
        x, y = self.load_validation(experiment=experiment,
                                    validation_id=validation_id,
                                    root_population=root_population)
        y_pred, y_score = self._predict(x, threshold=threshold)
        results = supervised.calc_metrics(metrics=metrics,
                                          y_true=y,
                                          y_pred=y_pred,
                                          y_score=y_score)
        if return_predictions:
            return results, {"y_pred": y_pred, "y_score": y_score}
        return results


def _valid_multi_class(klass: str):
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
    klass = mongoengine.StringField(required=True)
    params = mongoengine.DictField()

    def __init__(self, *args, **values):
        assert "klass" in values.keys(), "klass is required"
        multi_class = values.get("multi_class", False)
        if multi_class:
            _valid_multi_class(values.get("klass"))
        super().__init__(*args, **values)

    def build_model(self):
        params = self.params or {}
        self._model = supervised.build_sklearn_model(klass=self.klass, **params)
        if self.class_weights:
            err = "Class weights defined yet the specified model does not support this"
            assert "sample_weight" in signature(self.model.fit).parameters.keys(), err

    def _predict(self,
                 x: pd.DataFrame,
                 threshold: float = 0.5):
        self.check_model_init()
        if "predict_proba" in dir(self.model):
            y_score = self.model.predict_proba(x[self.features])
        else:
            y_score = self.model.decision_function(x[self.features])
        if self.multi_class:
            y_pred = list(map(lambda yi: [int(i > threshold) for i in yi], y_score))
        else:
            y_pred = self.model.predict(x[self.features])
        return y_pred, y_score

    def _fit(self, x: pd.DataFrame, y: np.ndarray, **kwargs):
        self.check_model_init()
        if self.class_weights is not None:
            err = "Class weights defined yet the specified model does not support this"
            assert "sample_weight" in signature(self.model.fit).parameters.keys(), err
            sample_weight = np.array([self.class_weights.get(i) for i in y])
            self.model.fit(x, y, sample_weight=sample_weight, **kwargs)
        else:
            self.model.fit(x, y, **kwargs)

    def hyperparameter_tuning(self,
                              param_grid: dict,
                              method: str = "grid_search",
                              **kwargs):
        self.check_model_init()
        self.check_data_init()
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

    def plot_learning_curve(self,
                            experiment: Experiment or None = None,
                            validation_id: str or None = None,
                            root_population: str or None = None,
                            ax: Axes or None = None,
                            x_label: str = "Training examples",
                            y_label: str = "Score",
                            train_sizes: np.array or None = None,
                            **kwargs):
        self.check_model_init()
        x, y = self.x, self.y
        if validation_id is not None:
            assert all([x is not None for x in [experiment, root_population]]), \
                "For plotting learning curve for validation, must provide validaiton ID, expeirment " \
                "object, and root population"
            x, y = self.load_validation(validation_id=validation_id,
                                        root_population=root_population,
                                        experiment=experiment)
        train_sizes = train_sizes or np.linspace(0.1, 1.0, 10)
        verbose = int(self.verbose)
        ax = ax or plt.subplots(figsize=(5, 5))[1]
        train_sizes, train_scores, test_scores, _, _ = learning_curve(self.model, x, y, verbose=verbose,
                                                                      train_sizes=train_sizes, **kwargs)
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
        assert sum([i is None for i in [x, y]]) in [0, 2], \
            "If you provide 'x' you must provide 'y' and vice versa."
        return supervised.confusion_matrix_plots(classifier=self.model,
                                                 x=x or self.x,
                                                 y=y or self.y,
                                                 class_labels=self.target_populations,
                                                 cmap=cmap,
                                                 figsize=figsize,
                                                 **kwargs)

    def save_model(self, path: str, **kwargs):
        pickle.dump(self.model, open(path, "w"), **kwargs)

    def load_model(self, path: str, **kwargs):
        model = pickle.load(open(path, "rb"), **kwargs)
        assert self.klass in str(type(model)), \
            f"Loaded model does not match Classifier, expected type: {self.klass}"
        self._model = model


class Layer(mongoengine.EmbeddedDocument):
    klass = mongoengine.StringField()
    kwargs = mongoengine.DictField()


class KerasCellClassifier(CellClassifier):
    model_params = mongoengine.StringField()
    input_layer = mongoengine.EmbeddedDocumentField(Layer)
    layers = mongoengine.EmbeddedDocumentListField(Layer)
    optimizer = mongoengine.StringField()
    loss = mongoengine.StringField()
    metrics = mongoengine.ListField()
    epochs = mongoengine.IntField()
    compile_kwargs = mongoengine.DictField()

    def build_model(self):
        kwargs = self.compile_kwargs or {}
        self._model = supervised.build_keras_model(layers=self.layers,
                                                   optimizer=self.optimizer,
                                                   loss=self.loss,
                                                   metrics=self.metrics,
                                                   **kwargs)

    def _predict(self,
                 x: pd.DataFrame,
                 threshold: float = 0.5):
        self.check_model_init()
        y_score = self.model.predict(x)
        if self.multi_class:
            y_pred = list(map(lambda yi: [int(i > threshold) for i in yi], y_score))
        else:
            y_pred = self.model.predict_classes(x)
        return y_pred, y_score
