from ..data.populations import Cluster, PopulationGeometry
from shapely.geometry import Polygon
import numpy as np
import pytest


@pytest.mark.parametrize("h5path,err_msg", [(None, "Cluster has not been previously defined, therefore you must provide a h5path"),
                                            ("does_not_exist.hdf5", "Invalid path, does_not_exist.hdf does not exist")])
def test_init_cluster_error(h5path, err_msg):
    with pytest.raises(AssertionError) as exp:
        Cluster(cluster_id="wrong",
                n_events=1000,
                prop_of_root=0.5,
                h5path=h5path)
    assert str(exp.value) == err_msg


def test_init_cluster():
    x = Cluster(cluster_id="test",
                n_events=1000,
                prop_of_root=0.5,
                h5path="test.hdf5")
    assert x.index is None
    x = Cluster(cluster_id="test",
                n_events=1000,
                prop_of_root=0.5,
                h5path="test.hdf5",
                index=[0, 1, 2, 3, 4, 5])
    assert x.index == [0, 1, 2, 3, 4, 5]
    # Cluster shouldn't write to disk
    x = Cluster(cluster_id="test",
                n_events=1000,
                prop_of_root=0.5,
                h5path="test.hdf5")
    assert x.index is None


def test_population_geometry_shape():
    poly = PopulationGeometry(x_values=[0, 0, 5, 5, 0],
                              y_values=[0, 5, 5, 0, 0])
    assert isinstance(poly.shape, Polygon)
    assert np.array_equal(poly.shape.exterior.xy[0], np.array([0, 0, 5, 5, 0]))
    assert np.array_equal(poly.shape.exterior.xy[1], np.array([0, 5, 5, 0, 0]))
    circle = PopulationGeometry(width=5,
                                height=5,
                                center=(10, 10),
                                angle=0)
    assert isinstance(circle.shape, Polygon)
    assert circle.shape.area == pytest.approx(np.pi * (circle.width**2), 1.)
    threshold = PopulationGeometry(x_threshold=2.5,
                                   y_threshold=2.5)
    assert threshold.shape is None
