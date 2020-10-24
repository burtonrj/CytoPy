from CytoPy.data.experiment import Experiment
from CytoPy.data.population import Population, create_signature
from CytoPy.data.fcs import FileGroup
from CytoPy.feedback import vprint, progress_bar
from CytoPy.flow.transforms import scaler
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import train_test_split, KFold, learning_curve
from sklearn import metrics as skmetrics
from sklearn.discriminant_analysis import *
from sklearn.neighbors import *
from sklearn.ensemble import *
from sklearn.svm import *
from keras.models import Sequential, load_model
from imblearn.over_sampling import RandomOverSampler
from warnings import warn
from matplotlib.axes import Axes
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import inspect
import pickle


def build_sklearn_model(klass: str,
                        **params):
    """
    Initiate a SklearnClassifier object using Classes in the global environment

    Parameters
    ----------
    klass: str

    Returns
    -------
    object
    """
    assert klass in globals().keys(), \
        f"Module {klass} not found, have you imported it into the working environment?"
    return globals()[klass](**params)


def build_keras_model(classifier: KerasClassifier):
    """
    Create and compile a Keras Sequential model using the given KerasClassifier object

    Parameters
    ----------
    classifier: KerasClassifier

    Returns
    -------
    object
    """
    required_imports = [x.klass for x in classifier.layers]
    for i in required_imports:
        assert i in globals().keys(), f"Module {i} not found, have you imported it into " \
                                      f"the working environment?"
    model = Sequential()
    for layer in classifier.layers:
        keras_klass = globals()[layer.klass](**layer.kwargs)
        model.add(keras_klass)
    model.compile(optimizer=classifier.optimizer,
                  loss=classifier.loss,
                  metrics=classifier.metrics,
                  **classifier.compile_kwargs)
    return model


def calc_metrics(metrics: list,
                 y_true: np.array,
                 y_pred: np.array or None = None,
                 y_score: np.array or None = None) -> dict:
    """
    Given a list of Scikit-Learn supported metrics (https://scikit-learn.org/stable/modules/model_evaluation.html)
    return a dictionary of results after checking that the required inputs are provided.

    Parameters
    ----------
    metrics: list
        List of string values; names of required metrics
    y_true: Numpy.Array
        True labels or binary label indicators. The binary and multiclass cases expect labels
        with shape (n_samples,) while the multilabel case expects binary label indicators with
        shape (n_samples, n_classes).
    y_pred: Numpy.Array
        Estimated targets as returned by a classifier
    y_score: Numpy.Array
        Target scores. In the binary and multilabel cases, these can be either probability
        estimates or non-thresholded decision values (as returned by decision_function on
        some classifiers). In the multiclass case, these must be probability estimates which
         sum to 1. The binary case expects a shape (n_samples,), and the scores must be the
         scores of the class with the greater label. The multiclass and multilabel cases expect
         a shape (n_samples, n_classes). In the multiclass case, the order of the class scores must
         correspond to the order of labels, if provided, or else to the numerical or
         lexicographical order of the labels in y_true.
    Returns
    -------

    """
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
                assert y_score is not None, f"Metric requested ({m}) requires probabilities of positive class but " \
                                            f"y_score not provided; y_score is None."
                results[m] = f(y_true=y_true, y_score=y_score)
            elif "y_pred" in inspect.signature(f).parameters.keys():
                results[m] = f(y_true=y_true, y_pred=y_pred)
            else:
                raise ValueError("Unexpected metric. Signature should contain either 'y_score' or 'y_pred'")
    return results

def confusion_matrix_plots(classifier,
                           x: pd.DataFrame,
                           y: np.ndarray,
                           class_labels: list,
                           cmap: str or None = None,
                           figsize: tuple = (10, 5),
                           **kwargs):
    cmap = cmap or plt.cm.Blues
    fig, axes = plt.subplots(2, figsize=figsize)
    titles = ["Confusion matrix, without normalisation", "Confusion matrix; normalised"]
    for i, (title, norm) in enumerate(zip(titles, [False, True])):
        ax = skmetrics.plot_confusion_matrix(estimator=classifier,
                                             X=x,
                                             y=y,
                                             display_labels=class_labels,
                                             cmap=cmap,
                                             normalize=norm,
                                             ax=axes[i],
                                             **kwargs)
        axes[i].set_title(title)
    return fig



def assert_population_labels(ref: FileGroup,
                            expected_labels: list):
    """
    Given some reference FileGroup and the expected population labels, check the
    validity of the labels and return list of valid populations only.

    Parameters
    ----------
    ref: FileGroup
    expected_labels: list

    Returns
    -------
    List
    """
    assert len(ref.populations) >= 2, "Reference sample does not contain any gated populations"
    for x in expected_labels:
        assert x in ref.tree.keys(), f"Ref FileGroup missing expected population {x}"


def check_downstream_populations(ref: FileGroup,
                                 root_population: str,
                                 population_labels: list) -> None:
    """
    Check that in the ordered list of population labels, all populaitons are downstream 
    of the given 'root' population.

    Parameters
    ----------
    ref: FileGroup
    root_population: str
    population_labels: list

    Returns
    -------
    None
    """
    downstream = ref.list_downstream_populations(root_population)
    assert all([x in downstream for x in population_labels]), \
        "The first population in population_labels should be the 'root' population, with all further populations " \
        "being downstream from this 'root'. The given population_labels has one or more populations that is not " \
        "downstream from the given root."


def multilabel(ref: FileGroup,
               root_population: str,
                population_labels: list,
                transform: str,
                features: list) -> (pd.DataFrame, pd.DataFrame):
    """
    Load the root population DataFrame from the reference FileGroup (assumed to be the first
    population in 'population_labels'). Then iterate over the remaining population creating a
    dummy matrix of population affiliations for each row of the root population.

    Parameters
    ----------
    ref: FileGroup
    population_labels: list
    transform: str
    features: list

    Returns
    -------
    (Pandas.DataFrame, Pandas.DataFrame)
        Root population flourescent intensity values, population affiliations (dummy matrix)
    """
    root = ref.load_population_df(population=root_population,
                                  transform=transform)
    for pop in population_labels:
        root[pop] = 0
        root.loc[ref.get_population(pop).index, pop] = 1
    return root[features], root[population_labels]


def singlelabel(ref: FileGroup,
                root_population: str,
                 population_labels: list,
                 transform: str,
                 features: list) -> (pd.DataFrame, np.ndarray):
    """
    Load the root population DataFrame from the reference FileGroup (assumed to be the first
    population in 'population_labels'). Then iterate over the remaining population creating a
    Array of population affiliations; each cell (row) is associated to their terminal leaf node
    in the FileGroup population tree.

    Parameters
    ----------
    ref: FileGroup
    population_labels: list
    transform: str
    features: list

    Returns
    -------
    (Pandas.DataFrame, Numpy.Array)
        Root population flourescent intensity values, labels
    """
    root = ref.load_population_df(population=root_population,
                                  transform=transform)
    y = np.zeros(root.shape[0])
    for i, pop in enumerate(population_labels):
        pop_idx = ref.get_population(population_name=pop).index
        np.put(y, pop_idx, i + 1)
    return root[features], y

def auto_weights(y: np.ndarray,
                 population_labels: list):
    classes = np.arange(0, len(population_labels) - 1)
    return compute_class_weight('balanced',
                                classes=classes,
                                y=y)


class CellClassifier:
    """
    Create a supervised classifier for the annotation of cytometry data. This class
    takes a template for a Scikit-Learn or Keras implementation of a supervised learning
    algorithm and provides the necessary apparatus to load cytometry data, train a model
    and then predict the populations of subsequent biological specimens. The predicted
    populations follow the same design as gated populations; they are represented by
    a Population object associated to the FileGroup.

    Attributes
    ===========
    classifier: SklearnClassifier or KerasClassifier
        A definition for wither a Scikit-Learn supervised classifier or Keras classifier,
        see CytoPy.data.supervised_classifier to read how to define
    experiment: Experiment
        Experiment to load data from and classify
    ref_sample: str
        Name of the sample that will be loaded from the given experiment and used as
        training data
    population_labels: list
        Populations the user wishes to predict in subsequent unlabelled samples
    verbose: bool (default=True)
        Whether to print feedback to stdout
    population_prefix: str (default="sml")
        Prefix assigned to predicted populations
    """

    def __init__(self,
                 classifier: SklearnClassifier or KerasClassifier,
                 experiment: Experiment,
                 ref_sample: str,
                 population_labels: list or None = None,
                 verbose: bool = True,
                 population_prefix: str = "sml"):
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
        self.population_prefix = population_prefix
        assert ref_sample in experiment.list_samples(), \
            "Invalid reference sample, could not be found in given experiment"
        self.ref = ref_sample
        self.population_labels = population_labels
        self.X, self.y = None, None

    def _check_init(self):
        assert self.X is not None, "Call 'load_training_data prior to fit'"

    def load_training_data(self):
        self.print("Loading reference sample...")
        ref = self.experiment.get_sample(self.ref)
        self.print("Checking population labels...")
        _load_population_labels(ref=ref, expected_labels=self.population_labels)
        _check_downstream_populations(ref=ref, population_labels=self.population_labels)
        self.print("Creating training data...")
        if self.classifier.multi_label:
            self.X, self.y = _multilabel(ref=ref,
                                         population_labels=self.population_labels,
                                         features=self.classifier.features,
                                         transform=self.classifier.transform)
        else:
            self.X, self.y = _singlelabel(ref=ref,
                                          population_labels=self.population_labels,
                                          features=self.classifier.features,
                                          transform=self.classifier.transform)
        if self.classifier.scale:
            self.print("Scaling data...")
            self._scale_data()
        else:
            warn("For the majority of classifiers it is recommended to scale the data (exception being tree-based "
                 "algorithms)")
        self._balance()
        self._downsample()
        self.print('Ready for training!')

    def _scale_data(self):
        kwargs = self.classifier.scale_kwargs or {}
        self.X[self.classifier.features] = scaler(self.X,
                                                  return_scaler=False,
                                                  scale_method=self.classifier.scale,
                                                  **kwargs)

    def _balance(self):
        if self.classifier.balance == "oversample":
            self.print("Correcting imbalance with oversampling...")
            self.X, self.y = RandomOverSampler(random_state=42).fit_resample(self.X, self.y)
        elif self.classifier.balance:
            self.print("Setting up class weights...")
            self.class_weights = _class_weights(y=self.y,
                                                classifier=self.classifier,
                                                population_labels=self.population_labels)

    def _downsample(self):
        if self.classifier.downsample:
            self.print("Downsampling data...")
            kwargs = self.classifier.downsample_kwargs or {}
            self.X, self.y = _downsample(X=self.X,
                                         y=self.y,
                                         features=self.classifier.features,
                                         method=self.classifier.downsample,
                                         **kwargs)

    def _predict(self,
                 X: pd.DataFrame,
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
        self._check_init()
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
            results[key] = calc_metrics(metrics=metrics, y_pred=y_pred, y_score=y_score, y_true=y)
        if return_predictions:
            return results, return_predictions
        return results

    def hyperparameter_tuning(self,
                              params: dict
                              method: str = "grid",):

    def fit_kfold(self,
                  threshold: float = 0.5,
                  n_splits: int = 10,
                  metrics: list or None = None,
                  shuffle: bool = True,
                  random_state: int = 42):
        self._check_init()
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
            training_results.append(calc_metrics(metrics=metrics, y_pred=y_pred, y_score=y_score, y_true=y_train))
            y_pred, y_proba, y_score = self._predict(X_test, threshold=threshold)
            testing_results.append(calc_metrics(metrics=metrics, y_pred=y_pred, y_score=y_score, y_true=y_test))
        return training_results, testing_results

    def predict(self,
                sample_id: str,
                parent_population: str,
                threshold: float = 0.5,
                return_predictions: bool = True,
                **kwargs):
        target = self.experiment.get_sample(sample_id)
        assert parent_population in target.tree.keys(), \
            f"Parent population {parent_population} missing from {sample_id}"
        X = target.load_population_df(population=self.population_labels[0],
                                      transform=self.classifier.transform)[self.classifier.features]
        self.model.fit(X, **kwargs)
        y_pred, y_proba, y_score = self._predict(X, threshold=threshold)
        if not self.classifier.multi_label:
            idx = X.index.values[np.where(y_pred == 0)[0]]
            target.add_population(Population(population_name=f"{self.population_prefix}_Unclassified",
                                             index=idx,
                                             n=len(idx),
                                             parent=parent_population,
                                             warnings=["supervised_classification"],
                                             signature=create_signature(data=X, idx=idx)))
        for i, pop in enumerate(self.population_labels[1:]):
            if self.classifier.multi_label:
                idx = X.index.values[np.where(y_pred[:, i] == 1)[0]]
            else:
                idx = X.index.values[np.where(y_pred == i)[0]]
            target.add_population(Population(population_name=f"{self.population_prefix}_{pop}",
                                             index=idx,
                                             n=len(idx),
                                             parent=parent_population,
                                             warnings=["supervised_classification"],
                                             signature=create_signature(data=X, idx=idx)))
        if return_predictions:
            return target, {"y_pred": y_pred, "y_proba": y_proba, "y_score": y_score}
        return target

    def load_validation(self,
                        validation_id: str,
                        parent_population: str):
        val = self.experiment.get_sample(validation_id)
        populations = [parent_population] + self.population_labels[:1]
        assert all([x in val.tree.keys() for x in populations]), \
            f"Validation sample should contain the following populations: {populations}"
        if self.classifier.multi_label:
            X, y = _multilabel(ref=val,
                               population_labels=populations,
                               transform=self.classifier.transform,
                               features=self.classifier.features)
        else:
            X, y = _singlelabel(ref=val,
                                population_labels=populations,
                                transform=self.classifier.transform,
                                features=self.classifier.features)
        return X, y

    def plot_learning_curve(self,
                            ax: Axes or None = None,
                            validation_id: str or None = None,
                            parent_population: str or None = None,
                            x_label: str = "Training examples",
                            y_label: str = "Score",
                            train_sizes: np.array or None = None,
                            **kwargs):
        self._check_init()
        X, y = self.X, self.y
        if validation_id is not None:
            assert parent_population is not None, "Must provide a parent population"
            X, y = self.load_validation(validation_id=validation_id,
                                        parent_population=parent_population)

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
                 parent_population: str,
                 threshold: float = 0.5,
                 metrics: list or None = None,
                 return_predictions: bool = True):
        metrics = metrics or ["balanced_accuracy_score", "f1_weighted"]
        X, y = self.load_validation(validation_id=validation_id, parent_population=parent_population)
        y_pred, y_proba, y_score = self._predict(X, threshold=threshold)
        results = calc_metrics(metrics=metrics, y_true=y, y_pred=y_pred, y_score=y_score)
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


class SklearnCellClassifier(CellClassifier):

    def __init__(self,
                 classifier: SklearnClassifier,
                 *args,
                 **kwargs):
        assert isinstance(classifier, SklearnClassifier), "Expected classifier of type SklearnClassifier"
        super().__init__(*args, **kwargs)


class KerasCellClassifier(CellClassifier):
    pass