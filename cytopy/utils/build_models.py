from typing import Type

from sklearn.discriminant_analysis import *
from sklearn.ensemble import *
from sklearn.linear_model import *
from sklearn.neighbors import *
from sklearn.svm import *
from tensorflow.keras.layers import *
from tensorflow.keras.models import Sequential
from xgboost import XGBClassifier


def build_sklearn_model(klass: str, **params) -> Type:
    """
    Initiate a SklearnClassifier object using Classes in the global environment

    Parameters
    ----------
    klass: str

    Returns
    -------
    Type
    """
    assert klass in globals().keys(), (
        f"Module {klass} not found, is this a Scikit-Learn (or like) classifier? It might "
        f"not currently be supported. See the docs for details."
    )
    return globals()[klass](**params)
