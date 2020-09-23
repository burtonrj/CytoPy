from ..data.fcs import  _column_names, FileGroup
from ..data.populations import Cluster, Population
from ..data.mappings import ChannelMap
import pandas as pd
import numpy as np
import pytest
import h5py
import os


def create_example_data(file_id: str):
    with h5py.File(f"{os.getcwd()}/test_data/{file_id}.hdf5") as f:
        f.create_group("index")
        f.create_group("index/test_pop")
        f.create_dataset("index/test_pop/primary", data=np.array([1, 2, 3, 4, 5]))
        f.create_dataset("index/test_pop/ctrl1", data=np.array([1, 2, 3, 4, 5]))
        f.create_group("clusters")
        f.create_group("clusters/test_pop")
        f.create_dataset("clusters/test_pop/cluster1", data=np.array([1, 2, 3, 4, 5]))


def delete_example_data(file_id: str):
    os.remove(f"{os.getcwd()}/test_data/{file_id}.hdf5")


def test_column_names():
    test = pd.DataFrame({"x": [0, 1, 2, 3, 4, 5],
                         "y": [0, 1, 2, 3, 4, 5],
                         "z": [0, 1, 2, 3, 4, 5]})
    test_mappings = [ChannelMap(marker="Marker1",
                                channel="Channel1"),
                     ChannelMap(marker="Marker2"),
                     ChannelMap(channel="Channel2")]
    x = _column_names(test, test_mappings, "marker")
    assert x.columns.tolist() == ["Marker1", "Marker2", "Channel2"]
    x = _column_names(test, test_mappings, "channel")
    assert x.columns.tolist() == ["Channel1", "Marker2", "Channel2"]


def test_filegroup_init():
    x = FileGroup(primary_id="test", data_directory=f"{os.getcwd()}/test_data")
    assert x.h5path == f"{os.getcwd()}/test_data/{x.id}.hdf5"


def test_filegroup_load_populations_error():
    x = FileGroup(primary_id="test", data_directory=f"{os.getcwd()}/test_data")
    with pytest.raises(AssertionError) as exp:
        x._load_populations()
    assert str(exp.value) == f"Could not locate FileGroup HDF5 record {x.h5path}"


def test_filegroup_load_populations():
    x = FileGroup(primary_id="test", data_directory=f"{os.getcwd()}/test_data")
    create_example_data(x.id)
    pop = Population(population_name="test_pop")
    pop.clusters.append(Cluster(cluster_id="cluster1"))
    x.populations.append(pop)
    x._load_populations()
    assert np.array_equal(x.populations[0].index, np.array([1, 2, 3, 4, 5]))
    assert isinstance(x.populations[0].ctrl_index, dict)
    assert "ctrl1" in x.populations[0].ctrl_index.keys()
    assert np.array_equal(x.populations[0].ctrl_index.get("ctrl1"), np.array([1, 2, 3, 4, 5]))
    assert np.array_equal(x.populations[0].clusters[0].index, np.array([1, 2, 3, 4, 5]))
    delete_example_data(x.id)
