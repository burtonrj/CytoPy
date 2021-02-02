from .conftest import create_lognormal_data, create_linear_data
from ..flow import transform
import pytest


def test_remove_negative_values_warnings():
    data = create_lognormal_data()
    for f in ["x", "y"]:
        warning = f"Feature {f} contains negative values. Chosen Transformer requires values " \
                  f">=0, all values <=0 will be forced to the minimum valid values in {f}"
        with pytest.warns(UserWarning) as warn:
            transform.remove_negative_values(data, [f])
        assert str(warn.list[0].message) == warning


def test_remove_negative_values():
    data = create_lognormal_data()
    valid = transform.remove_negative_values(data, ["x", "y"])
    assert data.shape == valid.shape
    assert (valid.x > 0).all()
    assert (valid.y > 0).all()

