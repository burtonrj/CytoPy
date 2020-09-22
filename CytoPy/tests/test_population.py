from ..data.populations import Cluster
import pytest


@pytest.mark.parametrize("h5path,err_msg", [(None, "Cluster has not been previously defined, therefore you must provide a h5path"),
                                            "does_not_exist.hdf", f"Invalid path, does_not_exist.hdf does not exist"])
def test_init_cluster_error(h5path, err_msg):
    with pytest.raises(AssertionError) as exp:
        Cluster(cluster_id="wrong",
                n_events=1000,
                prop_of_root=0.5,
                h5path=h5path)
    assert str(exp.value) == err_msg
