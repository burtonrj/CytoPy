from CytoPy.data.fcs import _column_names, FileGroup
from CytoPy.data.populations import Cluster, Population
from CytoPy.data.mappings import ChannelMap
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


def create_example_filegroup(include_controls: bool = True,
                             include_mappings: bool = True):
    x = FileGroup(primary_id="test", data_directory=f"{os.getcwd()}/test_data")
    if include_mappings:
        x.channel_mappings = [ChannelMap(marker=f"Marker{i+1}", channel=f"Channel{i+1}") for i in range(5)]
    if include_controls:
        x.controls = ["ctrl1", "ctrl2"]
    return x


def create_example_populations(x: FileGroup):
    x.populations = [Population(population_name="root", n=10000, index=np.random.rand(10000)),
                     Population(population_name="p2", n=500,
                                index=np.random.rand(500), parent="root",
                                ctrl_index={"ctrl1": np.random.rand(500)}),
                     Population(population_name="p3", n=500,
                                index=np.random.rand(500), parent="root",
                                ctrl_index={"ctrl1": np.random.rand(500)}),
                     Population(population_name="p4", n=500, index=np.random.rand(500), parent="root"),
                     Population(population_name="p5", n=500, index=np.random.rand(500), parent="root",
                                clusters=[Cluster(cluster_id="c1",
                                                  index=np.random.rand(500),
                                                  tag="test",
                                                  n=500)])]
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


def test_filegroup_init_empty():
    with pytest.warns(UserWarning) as warn:
        x = FileGroup(primary_id="test", data_directory=f"{os.getcwd()}/test_data")
        assert x.h5path == f"{os.getcwd()}/test_data/{x.id}.hdf5"
    assert str(warn.list[0].message) == "FileGroup is empty!"


@pytest.mark.parametrize("sample,columns,include_controls",
                         [(None, "marker", False),
                          (None, "marker", True),
                          (None, "columns", False),
                          (0.5, "columns", False),
                          (0.5, "marker", True)])
def test_filegroup_load(sample, columns, include_controls):
    x = FileGroup(primary_id="test",
                  data_directory=f"{os.getcwd()}/test_data",
                  channel_mappings=[ChannelMap(channel="X", marker="CD4"),
                                    ChannelMap(channel="Y", marker="CD8")])
    assert x.h5path == f"{os.getcwd()}/test_data/{x.id}.hdf5"
    pd.DataFrame({"X": np.random.normal(1, 0.5, 1000),
                  "Y": np.random.normal(5, 1.5, 1000)}).to_hdf(key="primary",
                                                               path_or_buf=x.h5path)
    pd.DataFrame({"X": np.random.normal(1.5, 0.5, 500),
                  "Y": np.random.normal(5.5, 1.5, 500)}).to_hdf(key="ctrl1",
                                                                path_or_buf=x.h5path)
    pd.DataFrame({"X": np.random.normal(0.5, 0.5, 500),
                  "Y": np.random.normal(4.5, 1.5, 500)}).to_hdf(key="ctrl2",
                                                                path_or_buf=x.h5path)
    data = x.load(sample_size=sample,
                  columns=columns,
                  include_controls=include_controls)
    assert isinstance(data, dict)
    assert "primary" in data.keys()
    assert isinstance(data.get("primary"), pd.DataFrame)
    if columns == "columns":
        assert set(data.get("primary").columns.tolist()) == {"X", "Y"}
    else:
        assert set(data.get("primary").columns.tolist()) == {"CD4", "CD8"}
    if sample is not None:
        assert data.get("primary").shape[0] == 500
    if include_controls:
        assert "controls" in data.keys()
        assert isinstance(data.get("controls"), dict)
        for c in ["ctrl1", "ctrl2"]:
            assert c in data.get("controls").keys()
            if columns == "columns":
                assert set(data.get(c).columns.tolist()) == {"X", "Y"}
            else:
                assert set(data.get(c).columns.tolist()) == {"CD4", "CD8"}
            if sample is not None:
                assert data.get(c).shape[0] == 250


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


@pytest.mark.parametrize("ctrl_id,err", [(None, "There can only be one primary file associated to each file group"),
                                         ("ctrl1", "Control file with ID ctrl1 already exists"),
                                         ("ctrl2", "Control file with ID ctrl2 already exists")])
def test_add_file_err_duplicate(ctrl_id, err):
    x = create_example_filegroup()
    create_example_data(x.id)
    control = ctrl_id is not None
    with pytest.raises(AssertionError) as exp:
        x.add_file(data=np.random.rand(1000, 5),
                   channel_mappings=[dict(channel=f"Channel{i + 1}", marker=f"Marker{i + 1}") for i in range(5)],
                   control=control,
                   ctrl_id=ctrl_id)
    assert str(exp.value) == err


def test_add_file():
    x = create_example_filegroup(include_controls=False)
    x.add_file(data=np.random.rand(1000, 5),
               channel_mappings=[dict(channel=f"Channel{i + 1}", marker=f"Marker{i + 1}") for i in range(5)],
               control=False)
    with h5py.File(x.h5path, "r") as f:
        assert "primary" in f.keys()
        assert f["primary"]["axis0"].shape[0] == 5
        assert f["primary"]["axis1"].shape[0] == 1000
    x.add_file(data=np.random.rand(1000, 5),
               channel_mappings=[dict(channel=f"Channel{i + 1}", marker=f"Marker{i + 1}") for i in range(5)],
               control=True,
               ctrl_id="ctrl1")
    with h5py.File(x.h5path, "r") as f:
        assert "ctrl1" in f.keys()
        assert f["ctrl1"]["axis0"].shape[0] == 5
        assert f["ctrl1"]["axis1"].shape[0] == 1000


@pytest.mark.parametrize("pop,err", [("cat", "Provide a list of population names for removal"),
                                     (["root"], "Cannot delete root population")])
def test_delete_pop_err(pop, err):
    x = create_example_filegroup()
    with pytest.raises(AssertionError) as exp:
        x.delete_populations(pop)
    assert str(exp.value) == err


def test_delete_pop():
    x = create_example_filegroup()
    x = create_example_populations(x)
    create_example_data(x.id)
    x.save()
    x.delete_populations(["p3"])
    assert "p3" not in list(x.list_populations())
    x.delete_populations(["p2", "p5"])
    assert {"root", "p4"} == set([p.population_name for p in x.populations])
    delete_example_data(x.id)


def test_update_pop():
    x = create_example_filegroup()
    x = create_example_populations(x)
    create_example_data(x.id)
    x.update_population(population_name="p3", new_population=Population(population_name="new",
                                                                        index=np.random.rand(1000),
                                                                        n=1000))
    assert "p3" not in list(x.list_populations())
    assert "new" in list(x.list_populations())
    delete_example_data(x.id)


def test_get_pop_by_parent():
    x = create_example_filegroup()
    x = create_example_populations(x)
    assert len(list(x.get_population_by_parent("root"))) == 5


def test_write_pop():
    x = create_example_filegroup()
    x = create_example_populations(x)
    x._write_populations()
    p = x.get_population("p3")
    assert p.n == 500
    assert p.prop_of_parent == 500/10000
    assert p.prop_of_total == 500 / 10000
    with h5py.File(x.h5path, "r") as f:
        assert all(x in f["index"].keys() for x in ["root", "p2", "p3", "p4", "p5"])
        assert "ctrl1" in f["index/p2"].keys()
        assert "ctrl1" in f["index/p3"].keys()
        assert f["index/p5/primary"][:].shape[0] == 500
        assert f["clusters/p5/c1"][:].shape[0] == 500
        assert f["index/p2/ctrl1"][:].shape[0] == 500


def test_reset_pop_data():
    x = create_example_filegroup()
    create_example_data(x.id)
    x.populations.append(Population(population_name="test_pop",
                                    ctrl_index=dict(ctrl1=np.random.rand(5))))
    x._hdf_reset_population_data()
    with h5py.File(x.h5path, "r") as f:
        assert "primary" not in f["index/test_pop"].keys()
        assert "ctrl1" not in f["index/test_pop"].keys()
        assert "test_pop" not in f["clusters"].keys()
