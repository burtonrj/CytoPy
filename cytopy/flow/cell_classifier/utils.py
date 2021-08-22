import inspect
import logging
from typing import Iterable
from typing import List
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from sklearn import metrics as skmetrics
from sklearn.base import ClassifierMixin
from sklearn.model_selection import learning_curve
from sklearn.utils.class_weight import compute_class_weight


logger = logging.getLogger(__name__)


def plot_learning_curve(
    model: ClassifierMixin,
    x: pl.DataFrame,
    y: np.ndarray,
    ax: Optional[plt.Axes] = None,
    x_label: str = "Training examples",
    y_label: str = "Score",
    train_sizes: Optional[np.array] = None,
    verbose: int = 1,
    **kwargs,
):
    train_sizes = train_sizes or np.linspace(0.1, 1.0, 10)
    ax = ax or plt.subplots(figsize=(5, 5))[1]
    train_sizes, train_scores, test_scores = learning_curve(
        model,
        x.to_numpy(),
        y,
        verbose=verbose,
        return_times=False,
        train_sizes=train_sizes,
        **kwargs,
    )
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    ax.grid()
    ax.fill_between(
        train_sizes,
        train_scores_mean - train_scores_std,
        train_scores_mean + train_scores_std,
        alpha=0.1,
        color="r",
    )
    ax.fill_between(
        train_sizes,
        test_scores_mean - test_scores_std,
        test_scores_mean + test_scores_std,
        alpha=0.1,
        color="g",
    )
    ax.plot(train_sizes, train_scores_mean, "o-", color="r", label="Training score")
    ax.plot(
        train_sizes,
        test_scores_mean,
        "o-",
        color="g",
        label="Cross-validation score",
    )
    ax.legend(loc="best")
    ax.set_xlabel(xlabel=x_label)
    ax.set_ylabel(ylabel=y_label)
    return ax


def calc_metrics(
    metrics: list,
    y_true: np.array,
    y_pred: np.array or None = None,
    y_score: np.array or None = None,
) -> dict:
    """
    Given a list of Scikit-Learn supported metrics (https://scikit-learn.org/stable/modules/model_evaluation.html)
    or callable functions with signature 'y_true', 'y_pred' and 'y_score', return a dictionary of results after
    checking that the required inputs are provided.

    Parameters
    ----------
    metrics: list
        List of string values; names of required metrics
    y_true: numpy.ndarray
        True labels or binary label indicators. The binary and multiclass cases expect labels
        with shape (n_samples,) while the multilabel case expects binary label indicators with
        shape (n_samples, n_classes).
    y_pred: numpy.ndarray
        Estimated targets as returned by a classifier
    y_score: numpy.ndarray
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
    dict
        Dictionary of performance metrics

    Raises
    ------
    AssertionError
        F1 score requested yet y_pred is missing

    AttributeError
        Requested metric requires probability scores and y_score is None

    ValueError
        Invalid metric provided; possibly missing signatures: 'y_true', 'y_score' or 'y_pred'
    """
    results = dict()
    i = 1
    for m in metrics:
        try:
            if callable(m):
                results[f"custom_metric_{i}"] = m(y_true=y_true, y_pred=y_pred, y_score=y_score)
                i += 1
                continue
            if "f1" in m:
                avg = m.split("_")
                if len(avg) == 2:
                    avg = m.split("_")[1]
                else:
                    avg = None
                f = getattr(skmetrics, "f1_score")
                assert y_pred is not None, "For F1 score predictions must be provided;`y_pred` is None"
                results[m] = f(y_true=y_true, y_pred=y_pred, average=avg)
            elif m == "roc_auc_score":
                f = getattr(skmetrics, m)
                results[m] = f(y_true=y_true, y_score=y_score, multi_class="ovr", average="weighted")
            else:
                f = getattr(skmetrics, m)
                if "y_score" in inspect.signature(f).parameters.keys():
                    if y_score is None:
                        raise AttributeError(
                            f"Metric requested ({m}) requires probabilities of positive class but "
                            f"y_score not provided; y_score is None."
                        )
                    results[m] = f(y_true=y_true, y_score=y_score)
                elif "y_pred" in inspect.signature(f).parameters.keys():
                    results[m] = f(y_true=y_true, y_pred=y_pred)
                else:
                    raise ValueError("Unexpected metric. Signature should contain either 'y_score' or 'y_pred'")
        except ValueError as e:
            logger.error(f"Unable to compute {m}: {e}")

    return results


def confusion_matrix_plots(
    classifier,
    x: pl.DataFrame,
    y: np.ndarray,
    class_labels: list,
    cmap: str or None = None,
    figsize: tuple = (8, 20),
    **kwargs,
):
    """
    Generate a figure of two heatmaps showing a confusion matrix, one normalised
    by support one showing raw values, displaying a classifiers performance.
    Returns Matplotlib.Figure object.

    Parameters
    ----------
    classifier: object
        Scikit-Learn classifier
    x: Pandas.DataFrame
        Feature space
    y: numpy.ndarray
        Labels
    class_labels: list
        Class labels (as they should be displayed on the axis)
    cmap: str
        Colour scheme, defaults to Matplotlib Blues
    figsize: tuple (default=(10,5))
        Size of the figure
    kwargs:
        Additional keyword arguments passed to sklearn.metrics.plot_confusion_matrix

    Returns
    -------
    Matplotlib.Figure
    """
    cmap = cmap or plt.get_cmap("Blues")
    fig, axes = plt.subplots(2, 1, figsize=figsize)
    titles = ["Confusion matrix, without normalisation", "Confusion matrix; normalised"]
    for i, (title, norm) in enumerate(zip(titles, [None, "true"])):
        skmetrics.plot_confusion_matrix(
            classifier,
            x.to_numpy(),
            y,
            display_labels=class_labels,
            cmap=cmap,
            normalize=norm,
            ax=axes[i],
            **kwargs,
        )
        axes[i].set_title(title)
        axes[i].set_xticklabels(axes[i].get_xticklabels(), rotation=90)
        axes[i].grid(False)
    fig.tight_layout()
    return fig


def assert_population_labels(ref, expected_labels: list):
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

    Raises
    -------
    AssertionError
        Ref missing expected populations
    """
    assert len(ref.populations) > 1, "Reference sample does not contain any gated populations"
    for x in expected_labels:
        assert x in ref.tree.keys(), f"Ref FileGroup missing expected population {x}"


def check_downstream_populations(ref, root_population: str, population_labels: list) -> None:
    """
    Check that in the ordered list of population labels, all populations are downstream
    of the given 'root' population.

    Parameters
    ----------
    ref: FileGroup
    root_population: str
    population_labels: list

    Returns
    -------
    None

    Raises
    ------
    AssertionError
        One or more populations not downstream of root
    """
    downstream = ref.list_downstream_populations(root_population)
    assert all([x in downstream for x in population_labels]), (
        "The first population in population_labels should be the 'root' population, with all further populations "
        "being downstream from this 'root'. The given population_labels has one or more populations that is not "
        "downstream from the given root."
    )


def multilabel(
    ref,
    root_population: str,
    population_labels: list,
    features: list,
    idx: Optional[Iterable[int]] = None,
) -> (pl.DataFrame, pl.DataFrame):
    """
    Load the root population DataFrame from the reference FileGroup (assumed to be the first
    population in 'population_labels'). Then iterate over the remaining population creating a
    dummy matrix of population affiliations for each row of the root population.

    Parameters
    ----------
    ref: FileGroup
    root_population: str
    population_labels: list
    features: list

    Returns
    -------
    (Pandas.DataFrame, Pandas.DataFrame)
        Root population flourescent intensity values, population affiliations (dummy matrix)
    """
    root = ref.load_population_df(population=root_population, transform=None)
    if idx is not None:
        root = root[root.Index.is_in(idx), :]
    for pop in population_labels:
        pop_idx = [x for x in ref.get_population(population_name=pop).index if x in root.index]
        root[pop] = [0 for _ in range(root.shape[0])]
        root[pop_idx, pop] = 1
    return root[features], root[population_labels]


def singlelabel(
    ref,
    root_population: str,
    population_labels: list,
    features: list,
    idx: Optional[Iterable[int]] = None,
) -> (pl.DataFrame, np.ndarray):
    """
    Load the root population DataFrame from the reference FileGroup (assumed to be the first
    population in 'population_labels'). Then iterate over the remaining population creating a
    Array of population affiliations; each cell (row) is associated to their terminal leaf node
    in the FileGroup population tree.

    Parameters
    ----------
    root_population
    ref: FileGroup
    population_labels: list
    features: list

    Returns
    -------
    (Pandas.DataFrame, numpy.ndarray)
        Root population flourescent intensity values, labels
    """
    root = ref.load_population_df(population=root_population, transform=None)
    if idx is not None:
        root = root[root.Index.is_in(idx), :]
    root["label"] = 0
    for i, pop in enumerate(population_labels):
        pop_idx = [x for x in ref.get_population(population_name=pop).index if x in root.index]
        root[pop_idx, "label"] = i + 1
    y = root["label"].to_numpy()
    root = root.drop("label")
    return root[features], y


def auto_weights(y: np.ndarray):
    """
    Estimate optimal weights from a list of class labels.

    Parameters
    ----------
    y: numpy.ndarray

    Returns
    -------
    dict
        Dictionary of class weights {label: weight}
    """
    classes = np.unique(y)
    weights = compute_class_weight("balanced", classes=classes, y=y)
    return {i: w for i, w in enumerate(weights)}
