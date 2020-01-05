from immunova.flow.supervised.utilities import predict_class
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, \
    label_ranking_average_precision_score, label_ranking_loss, classification_report, plot_confusion_matrix
import matplotlib.pyplot as plt
import pandas as pd
import pprint


def multi_label_performance(y, y_probs):
    return pd.DataFrame(dict(ranking_avg_precision=label_ranking_average_precision_score(y_true=y,
                                                                                         y_score=y_probs),
                             label_ranking_loss=label_ranking_loss(y_true=y, y_score=y_probs)))


def evaluate_model(classifier, x, y, threshold=None):
    y_probs = classifier.predict_proba(x)
    y_hat = predict_class(y_probs, threshold)
    return pd.DataFrame(dict(f1_score=[f1_score(y_true=y, y_pred=y_hat, average='weighted')],
                             accuracy=[accuracy_score(y_true=y, y_pred=y_hat)],
                             precision=[precision_score(y_true=y, y_pred=y_hat, average='weighted')],
                             recall=[recall_score(y_true=y, y_pred=y_hat, average='weighted')]))


def report_card(classifier, x, y, mappings=None, threshold=None, sample_weights=None, include_confusion_matrix=True):
    y_probs = classifier.predict_proba(x)
    y_hat = predict_class(y_probs, threshold)
    print('--------- CLASSIFICATION REPORT ---------')
    print(classification_report(y_true=y, y_pred=y_hat, sample_weight=sample_weights))
    if mappings is not None:
        print('\n')
        print('..........Class Mappings............')
        pprint.pprint(mappings)
    print('-----------------------------------------')
    if include_confusion_matrix:
        return confusion_matrix(classifier, x, y)


def confusion_matrix(classifier, x, y, normalize=True, **kwargs):
    if not normalize:
        return plot_confusion_matrix(classifier, x, y, **kwargs)
    else:
        fig, ax = plt.subplots(ncols=2,  figsize=(20,12))
        plot_confusion_matrix(classifier, x, y, ax=ax[0], cmap=plt.cm.Blues, **kwargs)
        plot_confusion_matrix(classifier, x, y, ax=ax[1], cmap=plt.cm.Blues, normalize='true', **kwargs)
        ax[0].set_title('Confusion Matrix')
        ax[1].set_title('Confusion Matrix (Normalised by true condition)')
        return fig
