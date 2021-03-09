from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import *
from xgboost import XGBClassifier
from sklearn.linear_model import *
from sklearn.discriminant_analysis import *
from sklearn.neighbors import *
from sklearn.ensemble import *
from sklearn.svm import *


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
        f"Module {klass} not found, is this a Scikit-Learn (or like) classifier? It might " \
        f"not currently be supported. See the docs for details."
    return globals()[klass](**params)


def build_keras_model(layers: list,
                      layer_params: list,
                      optimizer: str,
                      loss: str,
                      metrics: list,
                      input_shape: tuple,
                      **kwargs):
    """
    Create and compile a Keras Sequential model using the given KerasClassifier object

    Parameters
    ----------
    metrics: list
        See https://keras.io/api/metrics/
    loss: str
        See https://keras.io/api/losses/
    optimizer: str
        See https://keras.io/api/optimizers/
    layers: list
        List of Layer objects (see https://keras.io/api/layers/)
    layer_params: list
    input_shape: tuple

    Returns
    -------
    Sequential
    """
    for layer_klass in layers:
        e = f"{layer_klass} is not a valid Keras Layer or is not currently supported by CytoPy"
        assert layer_klass in globals().keys(), e
    model = Sequential()
    input_layer = globals()[layers[0]](shape=input_shape, **layer_params[0])
    model.add(input_layer)
    for layer_klass, lkwargs in zip(layers[1:], layer_params[1:]):
        layer_klass = globals()[layer_klass](**lkwargs)
        model.add(layer_klass)
    model.compile(optimizer=optimizer,
                  loss=loss,
                  metrics=metrics,
                  **kwargs)
    return model
