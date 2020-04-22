from .utilities import predict_class
from sklearn.metrics import balanced_accuracy_score, precision_score, recall_score, f1_score, \
    label_ranking_average_precision_score, label_ranking_loss, classification_report, plot_confusion_matrix, \
    confusion_matrix
import matplotlib.pyplot as plt
from pprint import pprint
import pandas as pd
import numpy as np


def multi_label_performance(y: np.array,
                            y_probs: np.array):
    """
    Calculate ranking average precision and label ranking loss for a multi-label classifier

    Parameters
    ----------
    y: Numpy.array
        predicted labels
    y_probs: Numpy.array
        predicted probabilities

    Returns
    -------
    Pandas.DataFrame
    """
    return pd.DataFrame(dict(ranking_avg_precision=label_ranking_average_precision_score(y_true=y,
                                                                                         y_score=y_probs),
                             label_ranking_loss=label_ranking_loss(y_true=y, y_score=y_probs)))


def evaluate_model(classifier,
                   x: np.array,
                   y: np.array,
                   threshold: float or None = None):
    """
    Calculate performance of a given classifier, returning a DataFrame summarising the performance.
    Metrics include: weighted F1-score, balanced accuracy, precision, and recall

    Parameters
    ----------
    classifier: CellClassifier
    x: Numpy.array
        Feature space
    y: Numpy.array
        Labels
    threshold: float, optional

    Returns
    -------
    Pandas.DataFrame
    """
    y_probs = classifier.predict_proba(x)
    y_hat = predict_class(y_probs, threshold)
    return pd.DataFrame(dict(f1_score=[f1_score(y_true=y, y_pred=y_hat, average='weighted')],
                             ballanced_accuracy=[balanced_accuracy_score(y_true=y, y_pred=y_hat)],
                             precision=[precision_score(y_true=y, y_pred=y_hat, average='weighted')],
                             recall=[recall_score(y_true=y, y_pred=y_hat, average='weighted')]))


def report_card(classifier,
                x: np.array,
                y: np.array,
                mappings: dict = None,
                threshold: float = None,
                sample_weights: dict = None,
                include_confusion_matrix: bool = True,
                include_accuracies: bool = True):
    """
    Generate a 'report card'; details everything the 'evaluate_model' function does, but includes performance
    breakdown for each class and gives the option to plot a confusion matrix

    Parameters
    ----------
    classifier: CellClassifier
    x: Numpy.array
        Feature space
    y: Numpy.array
        Labels
    mappings: dict, optional
        Class mappings; if provided they act as a key for the plotted confusion matrix
    threshold: float, optional
        Threshold to pass to evaluate_model
    sample_weights: dict, optional
        Dictionary of sample weights; include if you wish for this to be accounted when calculating performance
    include_confusion_matrix: bool, (default=True)
        If True, plots a confusion matrix
    include_accuracies: bool, (default=True)
        If True, includes a per-class accuracy calculation

    Returns
    -------
    None
        Results printed to stdout
    """
    y_probs = classifier.predict_proba(x)
    y_hat = predict_class(y_probs, threshold)
    print('--------- CLASSIFICATION REPORT ---------')
    print(classification_report(y_true=y, y_pred=y_hat, sample_weight=sample_weights))
    print('\n')
    if include_accuracies:
        print('---------- Accuracy per class ----------')
        cm = confusion_matrix(y_true=y, y_pred=y_hat)
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        pprint(pd.DataFrame({'Accuracy': cm.diagonal()}).T)
    if mappings is not None:
        print('\n')
        print('..........Class Mappings............')
        pprint(mappings)
    print('-----------------------------------------')
    if include_confusion_matrix:
        if 'keras' in str(type(classifier)):
            print('Confusion matrix plot currently not supported for Keras deep learning model')
            return
        plot_cm(classifier, x, y).show()


def plot_cm(classifier,
            x: np.array,
            y: np.array,
            normalize: bool = True,
            **kwargs):
    """
    Generate a confusion matrix plot using
    https://scikit-learn.org/stable/modules/generated/sklearn.metrics.plot_confusion_matrix.html

    Parameters
    ----------
    classifier: CellClassifier
    x: Numpy.array
        Feature space
    y: Numpy.array
        Labels
    normalize: bool, (default=True)
        If True, values are normalised over the true values
    kwargs:
        Additional keyword arguments to be passed to plot_confusion_matrix call

    Returns
    -------
    Matplotlib.Figure
    """
    if not normalize:
        return plot_confusion_matrix(classifier, x, y, **kwargs)
    else:
        fig, ax = plt.subplots(ncols=2,  figsize=(20, 12))
        plot_confusion_matrix(classifier, x, y, ax=ax[0], cmap=plt.cm.Blues, **kwargs)
        plot_confusion_matrix(classifier, x, y, ax=ax[1], cmap=plt.cm.Blues, normalize='true', **kwargs)
        ax[0].set_title('Confusion Matrix')
        ax[1].set_title('Confusion Matrix (Normalised by true condition)')
        return fig
