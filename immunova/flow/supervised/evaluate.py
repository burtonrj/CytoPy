from immunova.flow.supervised.utilities import predict_class
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, \
    label_ranking_average_precision_score, label_ranking_loss, classification_report
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


def report_card(classifier, x, y, mappings=None, threshold=None, sample_weights=None):
    y_probs = classifier.predict_proba(x)
    y_hat = predict_class(y_probs, threshold)
    print('--------- CLASSIFICATION REPORT ---------')
    print(classification_report(y, y_hat, sample_weight=sample_weights))
    if mappings is not None:
        print('\n')
        print('Class Mappings...')
        pprint.pprint(mappings)
    print('-----------------------------------------')

