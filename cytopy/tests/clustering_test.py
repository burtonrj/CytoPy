import numpy as np
import pandas as pd
import pytest

from ..flow.clustering import clustering


def test_remove_null_features():
    nulls = pd.DataFrame(
        {"x": [1, 2, np.nan, 4], "y": [1, 2, 3, 4], "z": [np.nan, np.nan, np.nan, np.nan], "w": [1, 2, 3, 4]}
    )
    assert {"y", "w"} == set(clustering.remove_null_features(nulls))
    assert {"w"} == set(clustering.remove_null_features(nulls, features=["x", "z", "w"]))


@pytest.mark.parametrize("scale,method", [(None, "median"), ("minmax", "mean"), ("standard", "median")])
def test_summarise_clusters(small_blobs, scale, method):
    x, _ = small_blobs
    data = clustering.summarise_clusters(data=x, features=["f1", "f2", "f3"], scale=scale, summary_method=method)
    assert isinstance(data, pd.DataFrame)
