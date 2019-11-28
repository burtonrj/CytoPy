from immunova.flow.supervised_algo.utilities import  predict_class
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, multilabel_confusion_matrix, \
    roc_auc_score, label_ranking_average_precision_score, label_ranking_loss
import pandas as pd


def multi_label_performance(y, y_probs):
    return pd.DataFrame(dict(ranking_avg_precision=label_ranking_average_precision_score(y_true=y,
                                                                                         y_score=y_probs),
                             label_ranking_loss=label_ranking_loss(y_true=y, y_score=y_probs)))


def evaluate_model(classifier, x, y, multi_label, threshold=None):
    y_probs = classifier.predict(x)
    y_hat = predict_class(y_probs, threshold)
    if multi_label == 'one hot encoding':
        return multi_label_performance(y, y_hat)
    return pd.DataFrame(dict(f1_score=f1_score(y_true=y, y_pred=y_hat, average='weighted'),
                             accuracy=accuracy_score(y_true=y, y_pred=y_hat),
                             precision=precision_score(y_true=y, y_pred=y_hat, average='weighted'),
                             recall=recall_score(y_true=y, y_pred=y_hat, average='weighted'),
                             auc_score=roc_auc_score(y_true=y, y_score=[x[i] for i, x in zip(y_hat, y_probs)])))
