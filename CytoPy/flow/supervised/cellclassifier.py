from ...data.supervised_classifier import SklearnClassifier, KerasClassifier
from ...data.experiments import Experiment
from ...data.fcs import Population, PopulationGeometry
from ...feedback import vprint, progress_bar
from ..sampling import density_dependent_downsampling, faithful_downsampling
from ..gating_tools import Gating, check_population_tree
from ..transforms import scaler
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import train_test_split, KFold, learning_curve
from sklearn import metrics as skmetrics
from keras.models import Sequential, load_model
from imblearn.over_sampling import RandomOverSampler
from anytree import Node
from warnings import warn
from matplotlib.axes import Axes
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import inspect
import pickle


def _build_sklearn_model(classifier: SklearnClassifier):
    assert classifier.klass in globals().keys(), f"Module {classifier.klass} not found, have you imported it into " \
                                                 f"the working environment?"
    kwargs = {k: v for k, v in classifier.params}
    return globals()[classifier.klass](**kwargs)


def _build_keras_model(classifier: KerasClassifier):
    required_imports = [x.klass for x in classifier.layers]
    for i in required_imports:
        assert i in globals().keys(), f"Module {i} not found, have you imported it into " \
                                      f"the working environment?"
    model = Sequential()
    for layer in classifier.layers:
        keras_klass = globals()[layer.klass]({k: v for k, v in layer.kwargs})
        model.add(keras_klass)
    model.compile(optimizer=classifier.optimizer,
                  loss=classifier.loss,
                  metrics=classifier.metrics,
                  **{k: v for k, v in classifier.compile_kwargs})
    return model


def _metrics(metrics: list,
             y_true: np.array,
             y_pred: np.array or None = None,
             y_score: np.array or None = None):
    results = dict()
    for m in metrics:
        if "f1" in m:
            avg = m.split("_")[1]
            f = getattr(skmetrics, "f1_score")
            assert y_pred is not None, "For F1 score predictions must be provided;`y_pred` is None"
            results[m] = f(y_true=y_true, y_pred=y_pred, average=avg)
        else:
            f = getattr(skmetrics, m)
            if "y_score" in inspect.signature(f).parameters.keys():
                assert y_score is not None, f"Metric required {m} requires probabilities of positive class but " \
                                            f"y_score not provided; y_score is None."
                results[m] = f(y_true=y_true, y_score=y_score)
            elif "y_pred" in inspect.signature(f).parameters.keys():
                results[m] = f(y_true=y_true, y_pred=y_pred)
            else:
                raise ValueError("Unexpected metric. Signature should contain either 'y_score' or 'y_pred'")
    return results


class CellClassifier:
    def __init__(self,
                 classifier: SklearnClassifier or KerasClassifier,
                 experiment: Experiment,
                 ref_sample: str,
                 population_labels: list or None = None,
                 verbose: bool = True):
        if type(classifier) == SklearnClassifier:
            self.model = _build_sklearn_model(classifier=classifier)
            self.model_type = "sklearn"
        elif type(classifier) == KerasClassifier:
            self.model = _build_keras_model(classifier=classifier)
            self.model_type = "keras"
        else:
            raise ValueError("Classifier must be either SklearnClassifier or KerasClassifier")
        self.classifier = classifier
        self.experiment = experiment
        self.verbose = verbose
        self.print = vprint(verbose)
        self.class_weights = None
        self.print("----- Building CellClassifier -----")
        assert ref_sample in experiment.list_samples(), "Invalid reference sample, could not be found in given " \
                                                        "experiment"
        self.print("Loading reference sample...")
        ref = Gating(experiment=experiment,
                     sample_id=ref_sample,
                     include_controls=False)
        self.population_labels = ref.valid_populations(populations=population_labels)
        assert len(self.population_labels) > 2, "Reference sample does not contain any gated populations"
        if len(self.population_labels) != len(population_labels):
            warn("One or more given population labels does not tie up with the populations in the reference "
                 f"sample, defaulting to the following valid labels: {self.population_labels}")
        check_population_tree(gating=ref, populations=self.population_labels)
        features = [x for x in ref.data.get("primary").columns if x in classifier.features]
        if len(features) != len(classifier.features):
            warn(f"One or more features missing from reference sample, "
                 f"proceeding with the following features: {features}")
        self.classifier.features = features
        downstream = ref.list_downstream_populations(self.population_labels[0])
        assert all([x in downstream for x in self.population_labels[1:]]), \
            "The first population in population_labels should be the 'root' population, with all further populations " \
            "being downstream from this 'root'. The given population_labels has one or more populations that is not " \
            "downstream from the given root."
        self.print("Preparing training and testing data...")
        if classifier.multi_label:
            self.X, self.y = self._multilabel(ref=ref)
        else:
            self.X, self.y = self._singlelabel(ref=ref)
        if classifier.scale:
            self.print("Scaling data...")
            if classifier.scale not in ["standard", "norm", "robust", "power"]:
                warn('Scale method must be one of: "standard", "norm", "robust", "power", defaulting to standard '
                     'scaling (unit variance)')
                classifier.scale = "standard"
            kwargs = classifier.scale_kwargs or {}
            self.X[self.classifier.features] = scaler(self.X,
                                                      return_scaler=False,
                                                      scale_method=classifier.scale,
                                                      **kwargs)
        else:
            warn("For the majority of classifiers it is recommended to scale the data (exception being tree-based "
                 "algorithms)")
        if classifier.balance:
            self._balance()
        if classifier.downsample:
            self._downsample()
        self.print('Ready for training!')

    def _multilabel(self,
                    ref: Gating):
        root = ref.get_population_df(population_name=self.population_labels[0],
                                     transform=self.classifier.transform)
        for pop in self.population_labels[1:]:
            root[pop] = 0
            root.loc[ref.populations.get(pop).index, pop] = 1
        return root[self.classifier.features], root[self.population_labels[1:]]

    def _singlelabel(self,
                     ref: Gating):
        root = ref.get_population_df(population_name=self.population_labels[0],
                                     transform=self.classifier.transform)
        y = np.zeros(root.shape[0])
        for i, pop in enumerate(self.population_labels[1:]):
            pop_idx = ref.populations[pop].index
            np.put(y, pop_idx, i + 1)
        return root[self.classifier.features], pd.DataFrame(y, columns=self.population_labels[1:])

    def _balance(self):
        if self.classifier.balance == "oversample":
            ros = RandomOverSampler(random_state=42)
            self.X, self.y = ros.fit_resample(self.X, self.y)
        elif self.classifier.balance == "auto-weights":
            assert not self.classifier.multi_label, "Class weights not supported for multi-label classifiers"
            classes = np.arange(0, len(self.population_labels) - 1)
            weights = compute_class_weight('balanced',
                                           classes=classes,
                                           y=self.y)
            self.class_weights = {k: w for k, w in zip(classes, weights)}
        elif self.classifier.balance_dict:
            assert not self.classifier.multi_label, "Class weights not supported for multi-label classifiers"
            class_weights = {k: w for k, w in self.classifier.balance_dict}
            self.class_weights = {i: class_weights.get(p) for i, p in enumerate(self.population_labels[1:])}
        else:
            raise ValueError("Balance should have a value 'oversample' or 'auto-weights', alternatively, "
                             "populate balance_dict with (label, weight) pairs")

    def _downsample(self):
        kwargs = {k: v for k, v in self.classifier.downsample_kwargs}
        if self.classifier.downsample == "uniform":
            frac = kwargs.get("frac", 0.5)
            self.X = self.X.sample(frac=frac)
            self.y = self.y[self.y.index.isin(self.X.index)]
        elif self.classifier.downsample == "density":
            self.X = density_dependent_downsampling(data=self.X,
                                                    features=self.classifier.features,
                                                    **kwargs)
            self.y = self.y[self.y.index.isin(self.X.index)]
        elif self.classifier.downsample == "faithful":
            self.X = faithful_downsampling(data=self.X, **kwargs)
            self.y = self.y[self.y.index.isin(self.X.index)]
        raise ValueError("Downsample should have a value of: 'uniform', 'density', or 'faithful'")

    def _predict(self,
                 X: np.array,
                 threshold: float = 0.5):
        if self.model_type == "sklearn":
            predict = getattr(self.model, "predict")
            predict_proba = getattr(self.model, "predict_proba")
        elif self.model_type == "keras":
            predict = getattr(self.model, "predict_classes")
            predict_proba = getattr(self.model, "predict")
        else:
            raise ValueError("Invalid model type")

        y_proba = predict_proba(X)
        if self.classifier.multi_label:
            y_pred = list(map(lambda x: [int(i > threshold) for i in x], y_proba))
        else:
            y_pred = predict(X)
        y_score = np.array([x[np.argmax(x)] for x in y_proba])
        return y_pred, y_proba, y_score

    def fit_train_test_split(self,
                             threshold: float = 0.5,
                             test_frac: float = 0.2,
                             metrics: list or None = None,
                             return_predictions: bool = True,
                             train_test_split_kwargs: dict or None = None,
                             fit_kwargs: dict or None = None):
        self.print("==========================================")
        metrics = metrics or ["balanced_accuracy_score", "f1_weighted", "roc_auc_score"]
        train_test_split_kwargs = train_test_split_kwargs or {}
        fit_kwargs = fit_kwargs or {}
        self.print("Spliting data into training and testing sets....")
        X_train, X_test, y_train, y_test = train_test_split(self.X,
                                                            self.y,
                                                            test_size=test_frac,
                                                            **train_test_split_kwargs)
        self.print("Fitting model....")
        if self.class_weights is not None:
            if self.model_type == "sklearn" and "class_weight" in self.model.get_params().keys():
                self.model.fit(X_train, y_train, class_weight=self.class_weights, **fit_kwargs)
            elif self.model_type == "keras":
                self.model.fit(X_train, y_train, class_weight=self.class_weights, **fit_kwargs)
            else:
                warn("Class weights not supported, continuing without weighting classes")
                self.model.fit(X_train, y_train, **fit_kwargs)
        else:
            self.model.fit(X_train, y_train, **fit_kwargs)
        results = dict()
        y_hat = dict()
        for key, (X, y) in zip(["train", "test"], [[X_train, y_train], [X_test, y_test]]):
            self.print(f"Evaluating {key}ing performance....")
            y_pred, y_proba, y_score = self._predict(X, threshold=threshold)
            y_hat[results] = {"y_pred": y_pred, "y_proba": y_proba, "y_score": y_score}
            results[key] = _metrics(metrics=metrics, y_pred=y_pred, y_score=y_score, y_true=y)
        if return_predictions:
            return results, return_predictions
        return results

    def fit_kfold(self,
                  threshold: float = 0.5,
                  n_splits: int = 10,
                  metrics: list or None = None,
                  shuffle: bool = True,
                  random_state: int = 42):
        metrics = metrics or ["balanced_accuracy_score", "f1_weighted", "roc_auc_score"]
        kf = KFold(n_splits=n_splits,
                   random_state=random_state,
                   shuffle=shuffle)
        training_results = list()
        testing_results = list()
        for train_idx, test_idx in progress_bar(kf.split(self.X), verbose=self.verbose):
            X_train, X_test = self.X.values[train_idx], self.X.values[test_idx]
            y_train, y_test = self.y[train_idx], self.y[test_idx]
            self.model.fit(X_train, y_train)
            y_pred, y_proba, y_score = self._predict(X_train, threshold=threshold)
            training_results.append(_metrics(metrics=metrics, y_pred=y_pred, y_score=y_score, y_true=y_train))
            y_pred, y_proba, y_score = self._predict(X_test, threshold=threshold)
            testing_results.append(_metrics(metrics=metrics, y_pred=y_pred, y_score=y_score, y_true=y_test))
        return training_results, testing_results

    def predict(self,
                sample_id: str,
                threshold: float = 0.5,
                return_predictions: bool = True,
                **kwargs):
        g = Gating(experiment=self.experiment,
                   sample_id=sample_id,
                   include_controls=False)
        assert self.population_labels[0] in g.populations.keys(), f"Root population {self.population_labels[0]} " \
                                                                  f"missing from {sample_id}"
        X = g.get_population_df(population_name=self.population_labels[0],
                                transform=self.classifier.transform,
                                transform_features=self.classifier.features)[self.classifier.features]
        self.model.fit(X, **kwargs)
        y_pred, y_proba, y_score = self._predict(X, threshold=threshold)
        if not self.classifier.multi_label:
            idx = np.where(y_pred == 0)[0]
            g.populations["Unclassified"] = Population(population_name="Unclassified",
                                                       index=idx,
                                                       n=len(idx),
                                                       parent=self.population_labels[0],
                                                       warnings=["supervised_classification"])
            g.tree["Unclassified"] = Node(name="Unclassified", parent=g.tree[self.population_labels[0]])
        for i, pop in enumerate(self.population_labels[1:]):
            if self.classifier.multi_label:
                idx = np.where(y_pred[:, i] == 1)[0]
            else:
                idx = np.where(y_pred == i)[0]
            g.populations[pop] = Population(population_name=pop,
                                            index=idx,
                                            n=len(idx),
                                            parent=self.population_labels[0],
                                            warnings=["supervised_classification"])
            g.tree[pop] = Node(name=pop, parent=g.tree[self.population_labels[0]])
        if return_predictions:
            return g, {"y_pred": y_pred, "y_proba": y_proba, "y_score": y_score}
        return g

    def load_validation(self,
                        validation_id: str):
        g = Gating(experiment=self.experiment,
                   sample_id=validation_id,
                   include_controls=False)
        assert all([x in g.populations.keys() for x in self.population_labels]), "Validation sample should contain " \
                                                                                 "the following populations: " \
                                                                                 f"{self.population_labels}"
        if self.classifier.multi_label:
            X, y = self._multilabel(ref=g)
        else:
            X, y = self._singlelabel(ref=g)
        return X, y

    def plot_learning_curve(self,
                            ax: Axes or None = None,
                            validation_id: str or None = None,
                            x_label: str = "Training examples",
                            y_label: str = "Score",
                            train_sizes: np.array or None = None,
                            **kwargs):
        X, y = self.X, self.y
        if validation_id is not None:
            X, y = self.load_validation(validation_id=validation_id)

        train_sizes = train_sizes or np.linspace(0.1, 1.0, 10)
        verbose = 0
        if self.verbose:
            verbose = 1
        ax = ax or plt.subplots(figsize=(5, 5))[1]
        train_sizes, train_scores, test_scores, _, _ = learning_curve(self.model, X, y, verbose=verbose,
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

    def validate(self,
                 validation_id: str,
                 threshold: float = 0.5,
                 metrics: list or None = None,
                 return_predictions: bool = True):
        metrics = metrics or ["balanced_accuracy_score", "f1_weighted"]
        X, y = self.load_validation(validation_id=validation_id)
        y_pred, y_proba, y_score = self._predict(X, threshold=threshold)
        results = _metrics(metrics=metrics, y_true=y, y_pred=y_pred, y_score=y_score)
        if return_predictions:
            return results, {"y_pred": y_pred, "y_proba": y_proba, "y_score": y_score}
        return results

    def save_model(self,
                   path: str,
                   **kwargs):
        if self.model_type == "sklearn":
            pickle.dump(self.model, open(path, "wb"), **kwargs)
        else:
            self.model.save(filepath=path, **kwargs)

    def load_model(self, path: str, **kwargs):
        if self.model_type == "sklearn":
            model = pickle.load(open(path, "rb"), **kwargs)
            assert isinstance(model, type(self.model)), f"Loaded model does not match Classifier, expected type: " \
                                                        f"{type(self.model)}"
        else:
            self.model = load_model(filepath=path, **kwargs)

