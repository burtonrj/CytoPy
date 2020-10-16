from ...flow.batch_effects import covar_euclidean_norm, scale_data
from string import ascii_uppercase
import pandas as pd
import numpy as np


def example_data_dict():
    data = dict()
    for i, label in enumerate(ascii_uppercase[0:9]):
        data[label] = pd.DataFrame({column: np.random.normal(i + 1, 1.5, 1000)
                                    for column in ascii_uppercase[0: 10]})
    return data


def test_covar_euclidean_norm():
    data = example_data_dict()
    ref = covar_euclidean_norm(data=data)
    assert ref == "E"


def test_scale_data():
    data = example_data_dict()
    scaled = scale_data(data=data)
    assert isinstance(scaled, dict)
    for df in scaled.values():
        assert all(list(map(lambda x: x > 1, df.values)))
        assert all(list(map(lambda x: x < 0, df.values)))

