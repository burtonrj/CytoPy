from sklearn.utils.class_weight import compute_class_weight
from sklearn import metrics as skmetrics
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import inspect


def calc_metrics(metrics: list,
                 y_true: np.array,
                 y_pred: np.array or None = None,
                 y_score: np.array or None = None) -> dict:
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
            results[m] = f(y_true=y_true,
                           y_score=y_score,
                           multi_class="ovo",
                           average="macro")
        else:
            f = getattr(skmetrics, m)
            if "y_score" in inspect.signature(f).parameters.keys():
                if y_score is None:
                    raise AttributeError(f"Metric requested ({m}) requires probabilities of positive class but "
                                         f"y_score not provided; y_score is None.")
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
                           figsize: tuple = (8, 20),
                           **kwargs):
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
    for i, (title, norm) in enumerate(zip(titles, [None, 'true'])):
        skmetrics.plot_confusion_matrix(classifier,
                                        x,
                                        y,
                                        display_labels=class_labels,
                                        cmap=cmap,
                                        normalize=norm,
                                        ax=axes[i],
                                        **kwargs)
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
    assert len(ref.populations) >= 2, "Reference sample does not contain any gated populations"
    for x in expected_labels:
        assert x in ref.tree.keys(), f"Ref FileGroup missing expected population {x}"


def check_downstream_populations(ref,
                                 root_population: str,
                                 population_labels: list) -> None:
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
    assert all([x in downstream for x in population_labels]), \
        "The first population in population_labels should be the 'root' population, with all further populations " \
        "being downstream from this 'root'. The given population_labels has one or more populations that is not " \
        "downstream from the given root."


def multilabel(ref,
               root_population: str,
               population_labels: list,
               features: list) -> (pd.DataFrame, pd.DataFrame):
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
    root = ref.load_population_df(population=root_population,
                                  transform=None)
    for pop in population_labels:
        root[pop] = 0
        root.loc[ref.get_population(pop).index, pop] = 1
    return root[features], root[population_labels]


def singlelabel(ref,
                root_population: str,
                population_labels: list,
                features: list) -> (pd.DataFrame, np.ndarray):
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
    root = ref.load_population_df(population=root_population,
                                  transform=None)
    root["label"] = 0
    for i, pop in enumerate(population_labels):
        pop_idx = ref.get_population(population_name=pop).index
        root.loc[pop_idx, "label"] = i + 1
    y = root["label"].values
    root.drop("label", axis=1, inplace=True)
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
    weights = compute_class_weight('balanced',
                                   classes=classes,
                                   y=y)
    return {i: w for i, w in enumerate(weights)}
