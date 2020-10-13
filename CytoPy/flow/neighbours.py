from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import balanced_accuracy_score
import pandas as pd
import numpy as np


def calculate_optimal_neighbours(x: pd.DataFrame,
                                 y: np.array,
                                 scoring: str,
                                 **kwargs):
    """
    Calculate the opitmal n_neighbours parameter for KNeighborsClassifier using GridSearchCV.
    Returns optimal n and highest score

    Parameters
    ----------
    x: Pandas.DataFrame
    y: np.array
    scoring: str
    kwargs: dict

    Returns
    -------
    int, float
    """
    n = np.arange(int(x.shape[0] * 0.01),
                  int(x.shape[0] * 0.05),
                  int(x.shape[0] * 0.01) / 2, dtype=np.int)
    knn = KNeighborsClassifier(**kwargs)
    grid_cv = GridSearchCV(knn, {"n_neighbors": n}, scoring=scoring, n_jobs=-1, cv=10)
    grid_cv.fit(x, y)
    return grid_cv.best_params_.get("n_neighbors"), grid_cv.best_score_


def knn(data: pd.DataFrame,
        labels: np.array,
        features: list,
        n_neighbours: int,
        holdout_size: float = 0.2,
        random_state: int = 42,
        return_model: bool = False,
        **kwargs):
    """
    Train a nearest neighbours classifier (scikit-learn implementation) and return
    the balanced accuracy score for both training and validation.

    Parameters
    ----------
    data: Pandas.DataFrame
    labels: Numpy.Array
    features: list
    n_neighbours: int
    holdout_size: float (default=0.2)
    random_state: int (default=42)
    return_model: bool (default=False)
    kwargs: dict
        Keyword arguments passed to KNeighborsClassifier initialisation

    Returns
    -------
    (float, float) or (float, float, object)
        Training balanced accuracy score, Validation balanced accuracy score,
        Classifier (if return_model is True)
    """
    X_train, X_test, y_train, y_test = train_test_split(data[features].values,
                                                        labels,
                                                        test_size=holdout_size,
                                                        random_state=random_state)
    knn = KNeighborsClassifier(n_neighbors=n_neighbours, **kwargs)
    knn.fit(X_train, y_train)
    train_acc = balanced_accuracy_score(y_pred=knn.predict(X_train), y_true=y_train)
    val_acc = balanced_accuracy_score(y_pred=knn.predict(X_test), y_true=y_test)
    if return_model:
        return train_acc, val_acc, knn
    return train_acc, val_acc