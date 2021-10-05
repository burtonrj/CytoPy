import gc
import logging
import time

import numpy as np
import pandas as pd
import pytest
from flowutils.transforms import logicle

from ..utils import transform
from .conftest import create_lognormal_data

logger = logging.getLogger(__name__)


@pytest.fixture(scope="module", autouse=True)
def populate_cache():
    data = np.round(np.random.lognormal(mean=5, sigma=3, size=int(10e6)), decimals=4)
    transformed = logicle(data.reshape(-1, 1), channel_indices=[0]).reshape(-1)
    cache = dict(zip(data, transformed))
    transform.CACHE.append(cache)
    transform.CACHE_KWARGS.append(dict(t=262144, m=4.5, w=0.5, a=0))
    yield
    transform.CACHE = []
    transform.CACHE_KWARGS = []
    gc.collect()


def test_access_cache():
    assert isinstance(transform.CACHE, list)
    assert isinstance(transform.CACHE[0], dict)
    assert len(transform.CACHE) == 1
    assert len(transform.CACHE[0]) == int(10e6)
    assert isinstance(transform.CACHE_KWARGS, list)
    assert isinstance(transform.CACHE_KWARGS[0], dict)
    assert len(transform.CACHE_KWARGS) == 1
    assert len(transform.CACHE_KWARGS[0]) == 4


@pytest.fixture
def dummy_data():
    data = pd.DataFrame({"x": [1, 2, 3, 4, 5], "y": [1, 2, 3, 4, 5], "z": [1, 2, 3, 4, 5]})
    cached = pd.DataFrame({"x": [None, 2, None, 4, 5], "y": [1, None, 3, 4, None], "z": [1, 2, 3, None, 5]})
    missing_mask = cached.isnull()
    return data, cached, missing_mask


@pytest.fixture
def dummy_log_normal():
    return pd.DataFrame(
        {
            "x": np.random.lognormal(mean=5, sigma=2.9, size=int(1e5)),
            "y": np.random.lognormal(mean=4.9, sigma=3, size=int(1e5)),
            "z": np.random.lognormal(mean=5, sigma=3, size=int(1e5)),
        }
    )


def test_cache_lookup(dummy_log_normal):
    cached, missing_mask = transform.cache_lookup(
        cache=transform.CACHE[0], data=dummy_log_normal.round(4), inverse=False
    )
    assert isinstance(cached, pd.DataFrame)
    assert isinstance(missing_mask, pd.DataFrame)
    for c in ["x", "y", "z"]:
        assert missing_mask.dtypes[c] == bool
    assert missing_mask.sum().sum() < (dummy_log_normal.shape[0] * dummy_log_normal.shape[1])


def test_flatten_and_index(dummy_data):
    data, cached, missing_mask = dummy_data
    to_transform = transform.flatten_and_index(data=data, missing_mask=missing_mask)
    assert isinstance(to_transform, pd.DataFrame)
    assert set(to_transform.columns) == {"column", "index", "data"}
    assert to_transform.shape[0] == missing_mask.sum().sum()


def test_reconstruct(dummy_data):
    data, cached, missing_mask = dummy_data
    to_transform = transform.flatten_and_index(data=data, missing_mask=missing_mask)
    transform.reconstruct(transformed=to_transform, cached=cached)
    for c in cached.columns:
        assert np.array_equal(np.array([1, 2, 3, 4, 5]), cached[c].values)


def test_update_cache():
    transform.update_cache(original=np.array([99, 999, 9999]), transformed=np.array([11, 111, 1111]), cache_idx=0)
    assert transform.CACHE[0][99] == 11
    assert transform.CACHE[0][999] == 111
    assert transform.CACHE[0][9999] == 1111


def test_transform_with_cache(dummy_log_normal):
    transformer = transform.LogicleTransformer()
    start = time.perf_counter()
    for _ in range(3):
        logger.info(f"Starting transform {_ + 1}: {time.perf_counter() - start}")
        transformed = transform.transform_with_cache(data=dummy_log_normal, scaler=transformer)
        x = transform.logicle_wrapper(
            dummy_log_normal.round(5)[["x"]].values, func=transformer.transform, **transformer.kwargs
        )
        assert np.array_equal(transformed["x"].values, x)
        logger.info(f"Complete: {time.perf_counter() - start}")


def test_transform_with_cache_inverse(dummy_log_normal):
    transformer = transform.LogicleTransformer()
    start = time.perf_counter()
    for _ in range(3):
        logger.info(f"Starting inverse transform {_ + 1}: {time.perf_counter() - start}")
        transformed = transform.transform_with_cache(data=dummy_log_normal, scaler=transformer)
        inverse_transformed = transform.transform_with_cache(data=transformed, scaler=transformer, inverse=True)
        assert np.array_equal(dummy_log_normal.round(5)["x"].values, inverse_transformed["x"].values)
        logger.info(f"Complete: {time.perf_counter() - start}")


def test_remove_negative_values_warnings():
    data = create_lognormal_data()
    for f in ["x", "y"]:
        warning = (
            f"Feature {f} contains negative values. Chosen Transformer requires values "
            f">=0, all values <=0 will be forced to the minimum valid values in {f}"
        )
        with pytest.warns(UserWarning) as warn:
            transform.remove_negative_values(data, [f])
        assert str(warn.list[0].message) == warning


def test_remove_negative_values():
    data = create_lognormal_data()
    valid = transform.remove_negative_values(data, ["x", "y"])
    assert data.shape == valid.shape
    assert (valid.x > 0).all()
    assert (valid.y > 0).all()
