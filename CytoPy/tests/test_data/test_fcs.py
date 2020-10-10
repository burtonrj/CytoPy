from ...data.fcs import _column_names, FileGroup
from ...data.project import Project
from ...data.population import Cluster, Population
from ...data.mapping import ChannelMap
import pandas as pd
import numpy as np
import pytest
import h5py
import os


@pytest.fixture(scope="module", autouse=True)
def example_data_setup():
    test_project = Project(project_id="test")
    test_project.add_experiment(experiment_id="test experiment",
                                data_directory=f"{os.getcwd()}/test_data",
                                panel_definition=f"{os.getcwd()}/CytoPy/tests/assets/test_panel.xlsx")


def load_exp():
    test_project = Project.objects(project_id="test").get()
    return test_project.load_experiment("test experiment")


def test_create_fcs_file():
    test_exp = load_exp()
    fg_id = test_exp.add_new_sample(sample_id="test sample",
                                    primary_path=f"{os.getcwd()}/CytoPy/tests/assets/test.FCS",
                                    controls_path={"test_ctrl": f"{os.getcwd()}/CytoPy/tests/assets/test.FCS"},
                                    compensate=False)
    assert os.path.isfile(f"{os.getcwd()}/test_data/{fg_id}.hdf5")
    fg = test_exp.get_sample("test sample")
    assert len(fg.populations) == 1
    assert fg.populations[0].population_name == "root"


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
    x = FileGroup(primary_id="test", data_directory=f"{os.getcwd()}/test_data")
    assert x.h5path == f"{os.getcwd()}/test_data/{x.id}.hdf5"
    assert len(x.populations) == 0
    x.save()
    x = FileGroup.objects(primary_id="test").get()
    assert len(x.populations) == 1


@pytest.mark.parametrize("columns", ["marker", "channel"])
def test_filegroup_init(columns):
    fg = create_example()
    # Test primary and control data loaded
    assert isinstance(fg, FileGroup)
    assert "primary" in fg.data.keys()
    assert isinstance(fg.data.get("primary"), pd.DataFrame)
    assert "controls" in fg.data.keys()
    assert isinstance(fg.data.get("controls"), dict)
    assert all([x in fg.data.get("controls").keys() for x in ["ctrl1", "ctrl2", "ctrl3"]])
    assert all([isinstance(x, pd.DataFrame) for x in fg.data.get("controls").values()])
    # Test column mappings
    if columns == "columns":
        assert set(fg.data.get("primary").columns.tolist()) == {"X", "Y"}
    else:
        assert set(fg.data.get("primary").columns.tolist()) == {"CD4", "CD8"}
    # Test populations constructed
    assert len(fg.populations) == 3
    assert len(fg.tree.keys()) == 3
    expected_population_names = {"root", "test_pop1", "test_pop2"}
    assert set([p.population_name for p in fg.populations]) == expected_population_names
    assert set(fg.tree.keys()) == expected_population_names

    delete_example_data(fg.id)


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
    assert df["primary"].columns.tolist() == [f"{columns.capitalize()}{i + 1}" for i in range(5)]
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
    assert p.prop_of_parent == 500 / 10000
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
