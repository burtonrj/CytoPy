from CytoPy.data import population
from CytoPy.data.geometry import ThresholdGeom, PolygonGeom
from shapely.geometry import Polygon as Poly
import pandas as pd
import numpy as np
import pytest


def generate_polygons():
    poly1 = PolygonGeom(x_values=[0, 0, 5, 5, 0],
                        y_values=[0, 5, 5, 0, 0])
    poly2 = PolygonGeom(x_values=[2.5, 2.5, 5, 5, 2.5],
                        y_values=[0, 5, 5, 0, 0])
    poly3 = PolygonGeom(x_values=[6, 6, 10, 10, 6],
                        y_values=[0, 5, 5, 0, 0])
    return poly1, poly2, poly3


def norm(x):
    return list(map(lambda i: (i - min(x)) / (max(x) - min(x)), x))


def test_create_signature():
    d = {"x": [15, 22, 80, 32],
         "y": [55, 32, 10, 11],
         "z": [42, 87, 91, 10]}
    d_norm = {k: norm(x) for k, x in d.items()}
    example = pd.DataFrame(d)
    x = population.create_signature(example)
    y = population.create_signature(example, summary_method=np.mean)
    z = population.create_signature(example, idx=[1, 2], summary_method=np.mean)
    assert isinstance(x, dict)
    assert isinstance(y, dict)
    assert isinstance(z, dict)
    for i in ["x", "y", "z"]:
        assert pytest.approx(x.get(i), 0.001) == np.median(d_norm.get(i))
        assert pytest.approx(y.get(i), 0.001) == np.mean(d_norm.get(i))
        assert pytest.approx(z.get(i), 0.001) == np.mean(np.array(d_norm.get(i))[[1, 2]])


def test_cluster_init():
    x = population.Cluster(cluster_id="test",
                           n=1000)
    x.index = [0, 1, 2, 3, 4]
    assert np.array_equal(np.array([0, 1, 2, 3, 4]), x.index)
    x = population.Cluster(cluster_id="test",
                           n=1000,
                           index=[0, 1, 2, 3, 4])
    assert np.array_equal(np.array([0, 1, 2, 3, 4]), x.index)


def test_polygon_shape():
    poly = PolygonGeom(x_values=[0, 0, 5, 5, 0],
                       y_values=[0, 5, 5, 0, 0])
    assert isinstance(poly.shape, Poly)
    assert np.array_equal(poly.shape.exterior.xy[0], np.array([0, 0, 5, 5, 0]))
    assert np.array_equal(poly.shape.exterior.xy[1], np.array([0, 5, 5, 0, 0]))


def test_population_init():
    x = population.Population(population_name="test",
                              parent="test_parent")
    x.index = np.array([0, 1, 2, 3, 4, 5])
    assert np.array_equal(np.array([0, 1, 2, 3, 4, 5]), x.index)
    assert x.n == 6
    x.set_ctrl_index(x=np.array([0, 1, 2, 3, 4, 5]),
                     y=np.array([4, 5, 6, 7, 8, 9]))
    assert np.array_equal(x.ctrl_index["x"], np.array([0, 1, 2, 3, 4, 5]))


def add_example_clusters(pop: population.Population):
    for i in range(10):
        pop.add_cluster(population.Cluster(cluster_id=f"cluster{i}",
                                           meta_label=f"meta{i}",
                                           tag="tag1"))
    for i in range(10):
        pop.add_cluster(population.Cluster(cluster_id=f"cluster{i}",
                                           meta_label=f"meta{i}",
                                           tag="tag2"))


def test_population_add_cluster():
    x = population.Population(population_name="test",
                              parent="test_parent")
    add_example_clusters(x)
    assert len(x.clusters) == 20


@pytest.mark.parametrize("params,n", [({"tag": "tag1"}, 10),
                                      ({"cluster_ids": ["cluster1"],
                                        "tag": "tag1"}, 19),
                                      ({"cluster_ids": ["cluster1", "cluster2"],
                                        "tag": "tag1"}, 18),
                                      ({"cluster_ids": ["cluster1"],
                                        "tag": "tag2",
                                        "meta_labels": ["meta4"]}, 20),
                                      ({"tag": "tag2",
                                        "meta_labels": ["meta1", "meta2", "meta3", "meta10"]}, 17)])
def test_population_delete_cluster(params, n):
    x = population.Population(population_name="test",
                              parent="test_parent")
    add_example_clusters(x)
    x.delete_cluster(**params)
    assert len(x.clusters) == n


@pytest.mark.parametrize("params,n", [({"tag": "tag1"}, 10),
                                      ({"cluster_ids": ["cluster1"],
                                        "tag": "tag1"}, 1),
                                      ({"cluster_ids": ["cluster1", "cluster2"],
                                        "tag": "tag1"}, 2),
                                      ({"cluster_ids": ["cluster1"],
                                        "tag": "tag2",
                                        "meta_labels": ["meta4"]}, 0),
                                      ({"tag": "tag2",
                                        "meta_labels": ["meta1", "meta2", "meta3", "meta10"]}, 3)])
def test_population_get_clusters(params, n):
    x = population.Population(population_name="test",
                              parent="test_parent")
    add_example_clusters(x)
    assert len(x.get_clusters(**params)) == n


@pytest.mark.parametrize("ctrl_idx,err", [(("x", "x"), "ctrl_idx should be type numpy.array"),
                                          (("x", [0, 1, 2, 3, 4]), "ctrl_idx should be type numpy.array")])
def test_population_ctrl_idx_error(ctrl_idx, err):
    x = population.Population(population_name="test",
                              parent="test_parent")
    with pytest.raises(AssertionError) as exp:
        x.set_ctrl_index(**{ctrl_idx[0]: ctrl_idx[1]})
    assert str(exp.value) == err


def test_check_overlap_invalid_shape():
    geom = ThresholdGeom()
    x = population.Population(population_name="test",
                              parent="test_parent",
                              geom=geom)
    y = population.Population(population_name="test",
                              parent="test_parent",
                              geom=geom)
    with pytest.raises(AssertionError) as exp:
        population._check_overlap(x, y, error=True)
    assert str(exp.value) == "Only Polygon geometries can be checked for overlap"


def test_check_overlap_error():
    poly1, _, poly2 = generate_polygons()
    x = population.Population(population_name="test",
                              parent="test_parent",
                              geom=poly1)
    y = population.Population(population_name="test",
                              parent="test_parent",
                              geom=poly2)
    with pytest.raises(AssertionError) as exp:
        population._check_overlap(x, y, error=True)
    assert str(exp.value) == "Invalid: non-overlapping populations"


def test_check_overlap():
    poly1, poly2, _ = generate_polygons()
    x = population.Population(population_name="test",
                              parent="test_parent",
                              geom=poly1)
    y = population.Population(population_name="test",
                              parent="test_parent",
                              geom=poly2)
    assert population._check_overlap(x, y, error=False) is True
    poly1, _, poly2 = generate_polygons()
    x = population.Population(population_name="test",
                              parent="test_parent",
                              geom=poly1)
    y = population.Population(population_name="test",
                              parent="test_parent",
                              geom=poly2)
    assert population._check_overlap(x, y, error=False) is False


def test_merge_index():
    x = population.Population(population_name="test",
                              parent="test_parent")
    y = population.Population(population_name="test",
                              parent="test_parent")
    x.index = np.array([0, 1, 2, 3, 4, 5, 11, 13])
    y.index = np.array([0, 1, 3, 8, 11, 15, 19])
    idx = population._merge_index(x, y)
    assert np.array_equal(idx, np.array([0, 1, 2, 3, 4, 5, 8, 11, 13, 15, 19]))


def test_merge_signatures():
    x = population.Population(population_name="test")
    x.signature = dict(x=10., y=10., z=20.)
    y = population.Population(population_name="test")
    y.signature = dict(x=20., y=50., z=5.)
    sig = population._merge_signatures(x, y)
    assert sig.get("x") == 15.
    assert sig.get("y") == 30.
    assert sig.get("z") == 12.5


def create_threshold_pops():
    left = population.Population(population_name="left",
                                 parent="test",
                                 geom=ThresholdGeom(x_threshold=0.5,
                                                    y_threshold=1.5),
                                 index=np.array([0, 1, 2, 3, 4, 5]),
                                 definition="++",
                                 signature=dict(x=5, y=5))
    right = population.Population(population_name="right",
                                  parent="test",
                                  geom=ThresholdGeom(x_threshold=0.5,
                                                     y_threshold=1.5),
                                  index=np.array([0, 1, 2, 3, 8, 11]),
                                  definition="+-",
                                  signature=dict(x=15, y=15))
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
    left = population.Population(population_name="left",
                                 parent="test",
                                 geom=poly1,
                                 index=np.array([0, 1, 2, 3, 4, 5]),
                                 definition="++",
                                 signature=dict(x=5, y=5))
    right = population.Population(population_name="right",
                                  parent="test",
                                  geom=poly2,
                                  index=np.array([0, 1, 2, 3, 8, 11]),
                                  definition="+-",
                                  signature=dict(x=15, y=15))
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
