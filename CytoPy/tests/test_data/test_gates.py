from ...data.gates import create_signature
import pandas as pd
import numpy as np
import pytest


def test_create_signature():
    example = pd.DataFrame({"x": [15, 22, 80, 32],
                            "y": [55, 32, 10, 11],
                            "z": [42, 87, 91, 10]})
    x = create_signature(example)
    assert isinstance(x, dict)
    assert x.get("x") == np.median([15, 22, 80, 32])
    assert x.get("y") == np.median([55, 32, 10, 11])
    assert x.get("z") == np.median([42, 87, 91, 10])
    x = create_signature(example, summary_method=np.mean)
    assert x.get("x") == np.mean([15, 22, 80, 32])
    assert x.get("y") == np.mean([55, 32, 10, 11])
    assert x.get("z") == np.mean([42, 87, 91, 10])
    x = create_signature(example, idx=[1, 2], summary_method=np.mean)
    assert x.get("x") == np.mean([22, 80])
    assert x.get("y") == np.mean([32, 10])
    assert x.get("z") == np.mean([87, 91])

