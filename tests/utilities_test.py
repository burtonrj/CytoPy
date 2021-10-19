import pandas as pd
from shapely.geometry import Polygon

from cytopy.utils.geometry import inside_polygon


def test_inside_polygon():
    poly = Polygon(
        [
            [1.0, 1.0],
            [1.0, 2.5],
            [1.5, 3.4],
            [2.5, 3.4],
            [2.5, 1.0],
            [1.6, 1.4],
            [1.0, 1.0],
        ]
    )
    df = pd.DataFrame(
        {
            "x": [1.2, 2.2, 6.4, 0.23, 1.5, 1.2, 2.0, 2.1, 5.0],
            "y": [1.5, 3.5, 3.5, 2.0, 1.5, 0.5, 2.0, 2.7, 2.0],
        }
    )
    df = inside_polygon(data=df, x="x", y="y", poly=poly)
    assert df.shape[0] == 4
    assert list(df.index.values) == [0, 4, 6, 7]
