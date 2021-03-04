from ..data.geometry import inside_polygon
from shapely.geometry import Polygon
import pandas as pd


def test_inside_polygon():
    poly = Polygon([[1., 1.], [1., 2.5], [1.5, 3.4],
                    [2.5, 3.4], [2.5, 1.], [1.6, 1.4],
                    [1., 1.]])
    df = pd.DataFrame({"x": [1.2, 2.2, 6.4, 0.23, 1.5, 1.2, 2., 2.1, 5.],
                       "y": [1.5, 3.5, 3.5, 2.0, 1.5, 0.5, 2., 2.7, 2.]})
    df = inside_polygon(df=df, x="x", y="y", poly=poly)
    assert df.shape[0] == 4
    assert list(df.index.values) == [0, 4, 6, 7]
