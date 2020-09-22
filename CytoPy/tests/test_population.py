from ..data.populations import Cluster
import pytest
import h5py


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



