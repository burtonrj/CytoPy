import numpy as np
import pytest
from shapely.geometry import Polygon as Poly

from cytopy.data import population
from cytopy.data.population import PolygonGeom
from cytopy.data.population import ThresholdGeom


def generate_polygons():
    poly1 = PolygonGeom(x_values=[0, 0, 5, 5, 0], y_values=[0, 5, 5, 0, 0])
    poly2 = PolygonGeom(x_values=[2.5, 2.5, 5, 5, 2.5], y_values=[0, 5, 5, 0, 0])
    poly3 = PolygonGeom(x_values=[6, 6, 10, 10, 6], y_values=[0, 5, 5, 0, 0])
    return poly1, poly2, poly3


def norm(x):
    return list(map(lambda i: (i - min(x)) / (max(x) - min(x)), x))


def test_polygon_shape():
    poly = PolygonGeom(x_values=[0, 0, 5, 5, 0], y_values=[0, 5, 5, 0, 0])
    assert isinstance(poly.shape, Poly)
    assert np.array_equal(poly.shape.exterior.xy[0], np.array([0, 0, 5, 5, 0]))
    assert np.array_equal(poly.shape.exterior.xy[1], np.array([0, 5, 5, 0, 0]))


def test_population_init():
    x = population.Population(population_name="test", parent="test_parent")
    x.index = np.array([0, 1, 2, 3, 4, 5])
    assert np.array_equal(np.array([0, 1, 2, 3, 4, 5]), x.index)
    assert x.n == 6


def test_check_overlap_invalid_shape():
    geom = ThresholdGeom()
    x = population.Population(population_name="test", parent="test_parent", geom=geom)
    y = population.Population(population_name="test", parent="test_parent", geom=geom)
    with pytest.raises(AssertionError) as exp:
        population._check_overlap(x, y, error=True)
    assert str(exp.value) == "Only Polygon geometries can be checked for overlap"


def test_check_overlap_error():
    poly1, _, poly2 = generate_polygons()
    x = population.Population(population_name="test", parent="test_parent", geom=poly1)
    y = population.Population(population_name="test", parent="test_parent", geom=poly2)
    with pytest.raises(AssertionError) as exp:
        population._check_overlap(x, y, error=True)
    assert str(exp.value) == "Invalid: non-overlapping populations"


def test_check_overlap():
    poly1, poly2, _ = generate_polygons()
    x = population.Population(population_name="test", parent="test_parent", geom=poly1)
    y = population.Population(population_name="test", parent="test_parent", geom=poly2)
    assert population._check_overlap(x, y, error=False) is True
    poly1, _, poly2 = generate_polygons()
    x = population.Population(population_name="test", parent="test_parent", geom=poly1)
    y = population.Population(population_name="test", parent="test_parent", geom=poly2)
    assert population._check_overlap(x, y, error=False) is False


def test_merge_index():
    x = population.Population(population_name="test", parent="test_parent")
    y = population.Population(population_name="test", parent="test_parent")
    x.index = np.array([0, 1, 2, 3, 4, 5, 11, 13])
    y.index = np.array([0, 1, 3, 8, 11, 15, 19])
    idx = population.merge_index(x, y)
    assert np.array_equal(idx, np.array([0, 1, 2, 3, 4, 5, 8, 11, 13, 15, 19]))


def test_merge_signatures():
    x = population.Population(population_name="test")
    x.signature = dict(x=10.0, y=10.0, z=20.0)
    y = population.Population(population_name="test")
    y.signature = dict(x=20.0, y=50.0, z=5.0)
    sig = population._merge_signatures(x, y)
    assert sig.get("x") == 15.0
    assert sig.get("y") == 30.0
    assert sig.get("z") == 12.5


def create_threshold_pops():
    left = population.Population(
        population_name="left",
        parent="test",
        geom=ThresholdGeom(x_threshold=0.5, y_threshold=1.5),
        index=np.array([0, 1, 2, 3, 4, 5]),
        definition="++",
        signature=dict(x=5, y=5),
    )
    right = population.Population(
        population_name="right",
        parent="test",
        geom=ThresholdGeom(x_threshold=0.5, y_threshold=1.5),
        index=np.array([0, 1, 2, 3, 8, 11]),
        definition="+-",
        signature=dict(x=15, y=15),
    )
    return left, right


def test_merge_thresholds():
    left, right = create_threshold_pops()
    merged = population._merge_thresholds(left, right, "merged")
    assert merged.population_name == "merged"
    assert isinstance(merged.geom, ThresholdGeom)
    assert merged.geom.x_threshold == 0.5
    assert merged.geom.y_threshold == 1.5
    assert np.array_equal(merged.index, np.array([0, 1, 2, 3, 4, 5, 8, 11]))
    assert merged.definition == "++,+-"
    assert merged.signature.get("x") == 10
    assert merged.signature.get("y") == 10
    assert merged.parent == "test"


def create_poly_pops():
    poly1, poly2, _ = generate_polygons()
    left = population.Population(
        population_name="left",
        parent="test",
        geom=poly1,
        index=np.array([0, 1, 2, 3, 4, 5]),
        definition="++",
        signature=dict(x=5, y=5),
    )
    right = population.Population(
        population_name="right",
        parent="test",
        geom=poly2,
        index=np.array([0, 1, 2, 3, 8, 11]),
        definition="+-",
        signature=dict(x=15, y=15),
    )
    return left, right


def test_merge_polygons():
    left, right = create_poly_pops()
    merged = population._merge_polygons(left, right, "merged")
    assert merged.population_name == "merged"
    assert isinstance(merged.geom, PolygonGeom)
    assert merged.geom.x_values == [0.0, 0.0, 2.5, 5.0, 5.0, 2.5, 0.0]
    assert merged.geom.y_values == [0.0, 5.0, 5.0, 5.0, 0.0, 0.0, 0.0]
    assert np.array_equal(merged.index, np.array([0, 1, 2, 3, 4, 5, 8, 11]))
    assert merged.signature.get("x") == 10
    assert merged.signature.get("y") == 10
    assert merged.parent == "test"
