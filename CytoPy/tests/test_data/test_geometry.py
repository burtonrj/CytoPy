from ...data.geometry import PopulationGeometry, ThresholdGeom, PolygonGeom, create_polygon, polygon_overlap
from shapely.geometry import Polygon
import numpy as np
import pytest


def test_create_geom():
    kwargs = dict(x="X",
                  y="Y",
                  transform_x="logicle",
                  transform_y="logicle")
    test = PopulationGeometry(**kwargs)
    for k, v in kwargs.items():
        assert test[k] == v


def test_create_threshold():
    kwargs = dict(x="X",
                  y="Y",
                  transform_x="logicle",
                  transform_y="logicle",
                  x_threshold=4.344,
                  y_threshold=2.435)
    test = ThresholdGeom(**kwargs)
    for k, v in kwargs.items():
        assert test[k] == v


def test_create_polygongeom():
    kwargs = dict(x="X",
                  y="Y",
                  transform_x="logicle",
                  transform_y="logicle",
                  x_values=list(np.random.normal(0, 0.5, 1000)),
                  y_values=list(np.random.normal(0, 0.5, 1000)))
    test = PolygonGeom(**kwargs)
    for k, v in kwargs.items():
        assert test[k] == v


def test_create_polygon():
    x = [2, 6, 9, 10, 2]
    y = [5, 19, 18, 10, 5]
    poly = create_polygon(x, y)
    assert isinstance(poly, Polygon)
    assert np.array_equal(poly.exterior.xy[0], np.array(x))
    assert np.array_equal(poly.exterior.xy[1], np.array(y))


@pytest.mark.parametrize("poly1,poly2,expected",
                         [(np.array([[0, 4.], [10, 4.], [10, 8.2], [10, 8.2], [0, 8.2], [0, 4.]]),
                           np.array([[0, 4.], [5, 4.], [5, 8.2], [5, 8.2], [0, 8.2], [0, 4.]]),
                           0.5),
                          (np.array([[0, 4.], [10, 4.], [10, 8.2], [10, 8.2], [0, 4.]]),
                           np.array([[12, 4.], [15, 4.], [15, 8.2], [15, 8.2], [12, 4.]]),
                           0.0)])
def test_polygon_overlap(poly1, poly2, expected):
    poly1, poly2 = Polygon(poly1), Polygon(poly2)
    assert polygon_overlap(poly1, poly2) == expected
    assert polygon_overlap(poly1, poly2, threshold=0.6) == 0.

