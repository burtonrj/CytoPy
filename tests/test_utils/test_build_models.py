import pytest
from tensorflow.keras import Sequential

from cytopy.utils import build_models


def test_build_sklearn_model_error():
    with pytest.raises(AssertionError):
        build_models.build_sklearn_model(klass="DUMMY", **{"dummy": "dummy"})


def test_build_sklearn_model():
    params = dict(n_jobs=-1, normalize=True)
    model = build_models.build_sklearn_model("LinearRegression", **params)
    assert model.get_params().get("n_jobs") == -1
    assert model.get_params().get("normalize")
