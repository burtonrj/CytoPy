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
    for x in ["primary", "ctrl1", "ctrl2"]:
        pd.DataFrame(np.random.rand(100, 5)).to_hdf(f"{os.getcwd()}/test_data/{file_id}.hdf5", key=x)


def create_example_filegroup():
    x = FileGroup(primary_id="test", data_directory=f"{os.getcwd()}/test_data", controls=["ctrl1", "ctrl2"])
    x.channel_mappings = [ChannelMap(marker=f"Marker{i+1}", channel=f"Channel{i+1}") for i in range(5)]
    return x


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
    x = create_example_filegroup()
    with pytest.raises(AssertionError) as exp:
        x._load_populations()
    assert str(exp.value) == f"Could not locate FileGroup HDF5 record {x.h5path}"


def test_filegroup_load_populations():
    x = create_example_filegroup()
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


@pytest.mark.parametrize("sample_size,include_ctrls,columns,keys",
                         [(None, False, "marker", ["primary"]),
                          (50, False, "marker", ["primary"]),
                          (50, True, "marker", ["primary", "controls"]),
                          (80, True, "channel", ["primary", "controls"])])
def test_filegroup_load(sample_size, include_ctrls, columns, keys):
    x = create_example_filegroup()
    create_example_data(x.id)
    df = x.load(sample_size=sample_size,
                include_controls=include_ctrls,
                columns=columns)
    sample_size = sample_size or 100
    assert isinstance(df, dict)
    assert all(x in df.keys() for x in keys)
    assert df["primary"].columns.tolist() == [f"{columns.capitalize()}{i+1}" for i in range(5)]
    assert df["primary"].shape[0] == sample_size
    if "controls" in df.keys():
        assert all(x in df["controls"].keys() for x in ["ctrl1", "ctrl2"])
        assert df["controls"]["ctrl1"].shape[0] == sample_size
    delete_example_data(x.id)


def test_add_file():
    pass


def test_valid_mappings():
    pass


def test_delete_pop():
    pass


def test_update_pop():
    pass


def test_get_pop():
    pass


def test_get_pop_by_parent():
    pass


def test_write_pop():
    pass


def test_create_pop_grps():
    pass


def test_reset_pop_data():
    pass


def test_delete():
    pass

