from CytoPy.data.populations import Cluster, Population, Polygon, Threshold, _check_overlap, _merge_index, \
    _merge_signatures, _merge_thresholds, _merge_polygons
from shapely.geometry import Polygon as Poly
import numpy as np
import pytest


def generate_polygons():
    poly1 = Polygon(x_values=[0, 0, 5, 5, 0],
                    y_values=[0, 5, 5, 0, 0])
    poly2 = Polygon(x_values=[2.5, 2.5, 5, 5, 2.5],
                    y_values=[0, 5, 5, 0, 0])
    poly3 = Polygon(x_values=[6, 6, 10, 10, 6],
                    y_values=[0, 5, 5, 0, 0])
    return poly1, poly2, poly3


def test_cluster_init():
    x = Cluster(cluster_id="test",
                n=1000)
    x.index = [0, 1, 2, 3, 4]
    assert np.array_equal(np.array([0, 1, 2, 3, 4]), x.index)
    x = Cluster(cluster_id="test",
                n=1000,
                index=[0, 1, 2, 3, 4])
    assert np.array_equal(np.array([0, 1, 2, 3, 4]), x.index)


def test_polygon_shape():
    poly, _, _ = Polygon(x_values=[0, 0, 5, 5, 0],
                         y_values=[0, 5, 5, 0, 0])
    assert isinstance(poly.shape, Poly)
    assert np.array_equal(poly.shape.exterior.xy[0], np.array([0, 0, 5, 5, 0]))
    assert np.array_equal(poly.shape.exterior.xy[1], np.array([0, 5, 5, 0, 0]))


def test_overlap():
    poly1, poly2, _ = generate_polygons()
    assert poly1.overlap(poly2.shape) == 0.5
    assert poly2.overlap(poly1.shape) == 1.0
    assert poly1.overlap(poly2.shape, 0.6) == 0.0
    assert poly2.overlap(poly1.shape, 0.6) == 1.0


def test_population_init():
    x = Population(population_name="test",
                   parent="test_parent")
    x.index = [0, 1, 2, 3, 4, 5]
    assert np.array_equal(np.array([0, 1, 2, 3, 4, 5]), x.index)
    assert x.n == 6
    x.set_ctrl_index(x=np.array([0, 1, 2, 3, 4, 5]),
                     y=np.array([4, 5, 6, 7, 8, 9]))
    assert np.array_equal(x.ctrl_index["x"], np.array([0, 1, 2, 3, 4, 5]))


@pytest.mark.parametrize("ctrl_idx,err", [(("x", "x"), "ctrl_idx should be type numpy.array"),
                                          (("x", [0, 1, 2, 3, 4]), "ctrl_idx should be type numpy.array")])
def test_population_ctrl_idx_error(ctrl_idx, err):
    x = Population(population_name="test",
                   parent="test_parent")
    with pytest.raises(AssertionError) as exp:
        x.set_ctrl_index(**{ctrl_idx[0]: ctrl_idx[1]})
    assert str(exp.value) == err


def test_check_overlap_invalid_shape():
    geom = Threshold()
    x = Population(population_name="test",
                   parent="test_parent",
                   geom=geom)
    y = Population(population_name="test",
                   parent="test_parent",
                   geom=geom)
    with pytest.raises(AssertionError) as exp:
        _check_overlap(x, y, error=True)
    assert str(exp.value) == "Only Polygon geometries can be checked for overlap"


def test_check_overlap_error():
    poly1, _, poly2 = generate_polygons()
    x = Population(population_name="test",
                   parent="test_parent",
                   geom=poly1)
    y = Population(population_name="test",
                   parent="test_parent",
                   geom=poly2)
    with pytest.raises(AssertionError) as exp:
        _check_overlap(x, y, error=True)
    assert str(exp.value) == "Invalid: non-overlapping populations"


def test_check_overlap():
    poly1, poly2, _ = generate_polygons()
    x = Population(population_name="test",
                   parent="test_parent",
                   geom=poly1)
    y = Population(population_name="test",
                   parent="test_parent",
                   geom=poly2)
    assert _check_overlap(x, y, error=False) is True
    poly1, _, poly2 = generate_polygons()
    x = Population(population_name="test",
                   parent="test_parent",
                   geom=poly1)
    y = Population(population_name="test",
                   parent="test_parent",
                   geom=poly2)
    assert _check_overlap(x, y, error=False) is False


def test_merge_index():
    x = Population(population_name="test",
                   parent="test_parent")
    y = Population(population_name="test",
                   parent="test_parent")
    x.index = np.array([0, 1, 2, 3, 4, 5, 11, 13])
    y.index = np.array([0, 1, 3, 8, 11, 15, 19])
    idx = _merge_index(x, y)
    assert np.array_equal(idx, np.array([0, 1, 2, 3, 4, 5, 8, 11, 13, 15, 19]))


def test_merge_signatures():
    x = Population(population_name="test")
    x.signature = dict(x=10., y=10., z=20.)
    y = Population(population_name="test")
    y.signature = dict(x=20., y=50., z=5.)
    sig = _merge_signatures(x, y)
    assert sig.get("x") == 15.
    assert sig.get("y") == 30.
    assert sig.get("z") == 12.5


def create_threshold_pops():
    left = Population(population_name="left",
                      parent="test",
                      geom=Threshold(x_threshold=0.5,
                                     y_threshold=1.5),
                      index=np.array([0, 1, 2, 3, 4, 5]),
                      definition="++",
                      signature=dict(x=5, y=5))
    right = Population(population_name="right",
                       parent="test",
                       geom=Threshold(x_threshold=0.5,
                                      y_threshold=1.5),
                       index=np.array([0, 1, 2, 3, 8, 11]),
                       definition="+-",
                       signature=dict(x=15, y=15))
    return left, right


def test_merge_thresholds():
    left, right = create_threshold_pops()
    merged = _merge_thresholds(left, right, "merged")
    assert merged.population_name == "merged"
    assert isinstance(merged.geom, Threshold)
    assert merged.geom.x_threshold == 0.5
    assert merged.geom.y_threshold == 1.5
    assert np.array_equal(merged.index, np.array([0, 1, 2, 3, 4, 5, 8, 11]))
    assert merged.definition == "++,+-"
    assert merged.signature.get("x") == 10
    assert merged.signature.get("y") == 10
    assert merged.parent == "test"


def create_poly_pops():
    poly1, poly2, _ = generate_polygons()
    left = Population(population_name="left",
                      parent="test",
                      geom=poly1,
                      index=np.array([0, 1, 2, 3, 4, 5]),
                      definition="++",
                      signature=dict(x=5, y=5))
    right = Population(population_name="right",
                       parent="test",
                       geom=poly2,
                       index=np.array([0, 1, 2, 3, 8, 11]),
                       definition="+-",
                       signature=dict(x=15, y=15))
    return left, right


def test_merge_polygons():
    left, right = create_poly_pops()
    merged = _merge_polygons(left, right, "merged")
    assert merged.population_name == "merged"
    assert isinstance(merged.geom, Polygon)
    assert merged.geom.x_values == [0.0, 0.0, 2.5, 5.0, 5.0, 2.5, 0.0]
    assert merged.geom.y_values == [0.0, 5.0, 5.0, 5.0, 0.0, 0.0, 0.0]
    assert np.array_equal(merged.index, np.array([0, 1, 2, 3, 4, 5, 8, 11]))
    assert merged.signature.get("x") == 10
    assert merged.signature.get("y") == 10
    assert merged.parent == "test"
