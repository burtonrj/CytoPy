from ...data.fcs import _column_names, FileGroup
from ...data.project import Project
from ...data.population import Cluster, Population
from ...data.mapping import ChannelMap
import pandas as pd
import numpy as np
import pytest
import h5py
import os


@pytest.fixture
def example_filegroup():
    test_project = Project(project_id="test")
    exp = test_project.add_experiment(experiment_id="test experiment",
                                      data_directory=f"{os.getcwd()}/test_data",
                                      panel_definition=f"{os.getcwd()}/CytoPy/tests/assets/test_panel.xlsx")
    yield exp.add_new_sample(sample_id="test sample",
                             primary_path=f"{os.getcwd()}/CytoPy/tests/assets/test.FCS",
                             controls_path={"test_ctrl": f"{os.getcwd()}/CytoPy/tests/assets/test.FCS"},
                             compensate=False)
    test_project.delete()


def test_create_fcs_file(example_filegroup):
    fg = example_filegroup
    assert os.path.isfile(f"{os.getcwd()}/test_data/{fg.id}.hdf5")
    experiment = Project.objects(project_id="test").get().load_experiment("test experiment")
    fg = experiment.get_sample("test sample")
    primary_data = pd.read_hdf(path_or_buf=fg.h5path, key="primary")
    ctrl_data = pd.read_hdf(path_or_buf=fg.h5path, key="test_ctrl")
    assert len(fg.populations) == 1
    assert fg.populations[0].population_name == "root"
    assert len(fg.tree.keys()) == 1
    assert list(fg.tree.keys())[0] == "root"
    root = fg.populations[0]
    assert root.parent == "root"
    assert root.population_name == "root"
    assert np.array_equal(primary_data.index.values, root.index)
    assert np.array_equal(ctrl_data.index.values, root.ctrl_index.get("test_ctrl"))
    assert primary_data.shape == (30000, 7)
    assert ctrl_data.shape == (30000, 7)


@pytest.mark.parametrize("sample_size,include_controls,columns",
                         [(None, False, "marker"),
                          (None, True, "channel"),
                          (0.5, True, "marker"),
                          (500, False, "channel")])
def test_filegroup_load(example_filegroup, sample_size, include_controls, columns):
    fg = example_filegroup
    data = fg.load(sample_size=sample_size,
                   include_controls=include_controls,
                   columns=columns)
    assert isinstance(data, dict)
    assert "primary" in data.keys()
    assert isinstance(data.get("primary"), pd.DataFrame)
    if include_controls:
        assert "controls" in data.keys()
        assert "test_ctrl" in data.get("controls").keys()
        assert isinstance(data.get("controls").get("test_ctrl"), pd.DataFrame)
    if isinstance(sample_size, int):
        assert data.get("primary").shape[0] == sample_size
    if isinstance(sample_size, float):
        assert data.get("primary").shape[0] == 15000
    expected_columns = [cm[columns] for cm in fg.channel_mappings]
    assert all([x in data.get("primary").columns for x in expected_columns])


def test_add_population():
    pass


def test_get_population():
    pass


def test_get_population_by_parent():
    pass


def test_list_populations():
    pass


def test_print_population_tree():
    pass


def test_delete_one_population():
    pass


def test_delete_many_populations():
    pass


def test_delete_all_populations():
    pass


def test_delete_clusters():
    pass


def test_update_population():
    pass


def test_delete():
    pass


def delete_example_data(file_id: str):
    os.remove(f"{os.getcwd()}/test_data/{file_id}.hdf5")
