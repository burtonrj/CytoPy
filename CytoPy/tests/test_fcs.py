from CytoPy.data.fcs import *
from CytoPy.data.project import Project
from CytoPy.data.population import Population
from .conftest import reload_filegroup, create_example_populations
import pandas as pd
import numpy as np
import pytest
import h5py
import os


def create_test_h5file(path: str,
                       empty: bool = False):
    """
    Create a H5 test file

    Parameters
    ----------
    path: str
        Where to create the file
    empty: bool (default=False)
        If True, fill with example data

    Returns
    -------
    None
    """
    with h5py.File(path, "w") as f:
        f.create_group("index")
        if empty:
            return
        f.create_group("index/test_pop")
        f.create_group("clusters/test_pop")
        f.create_dataset("index/test_pop/primary", data=np.random.random_integers(1000, size=1000))
        f.create_dataset("index/test_pop/test_ctrl1", data=np.random.random_integers(1000, size=1000))
        f.create_dataset("index/test_pop/test_ctrl2", data=np.random.random_integers(1000, size=1000))


def add_dummy_ctrl(fg: FileGroup, ctrl_id: str):
    """
    Add dummy control data to the given FileGroup

    Parameters
    ----------
    fg: FileGroup
    ctrl_id: str

    Returns
    -------
    None
    """
    data = pd.DataFrame([np.random.random(size=1000) for _ in range(6)]).T
    fg.add_ctrl_file(ctrl_id=ctrl_id,
                     data=data,
                     channels=[f"channel{i + 1}" for i in range(6)],
                     markers=[f"marker{i + 1}" for i in range(6)])
    fg.save()


def test_h5_read_population_primary_index():
    path = f"{os.getcwd()}/test_data/test.h5"
    create_test_h5file(path=path, empty=True)
    with h5py.File(path, "r") as f:
        assert h5_read_population_primary_index("test_pop", f) is None
    create_test_h5file(path=path, empty=False)
    with h5py.File(path, "r") as f:
        x = h5_read_population_primary_index("test_pop", f)
        assert x is not None
        assert x.shape[0] == 1000


def test_set_column_names():
    channels = [None, None, None, "channel1", "channel2", "channel3"]
    markers = [f"marker{i + 1}" for i in range(6)]
    data = pd.DataFrame([np.random.random(size=1000) for _ in range(6)]).T
    x = set_column_names(df=data,
                         channels=channels,
                         markers=markers,
                         preference="markers")
    assert np.array_equal(x.columns.values, markers)
    cols = ["marker1", "marker2", "marker3", "channel1", "channel2", "channel3"]
    x = set_column_names(df=data,
                         channels=channels,
                         markers=markers,
                         preference="channels")
    assert np.array_equal(x.columns.values, cols)


def test_init_new_fcs_file(example_populated_experiment):
    fg = example_populated_experiment.get_sample("test sample")
    assert os.path.isfile(f"{os.getcwd()}/test_data/{fg.id}.hdf5")
    experiment = Project.objects(project_id="test").get().get_experiment("test experiment")
    fg = experiment.get_sample("test sample")
    primary_data = fg.data("primary")
    ctrl_data = fg.data("test_ctrl")
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


@pytest.mark.parametrize("source,sample_size", [("primary", None),
                                                ("primary", 1000),
                                                ("test_ctrl", None),
                                                ("test_ctrl", 1000)])
def test_access_data(example_populated_experiment, source, sample_size):
    fg = example_populated_experiment.get_sample("test sample")
    df = fg.data(source, sample_size)
    assert isinstance(df, pd.DataFrame)
    if sample_size is not None:
        assert df.shape == (1000, 7)
    else:
        assert df.shape == (30000, 7)


def test_add_ctrl_file_already_exists_error(example_populated_experiment):
    fg = example_populated_experiment.get_sample("test sample")
    data = pd.DataFrame([np.random.random(size=1000) for _ in range(6)]).T
    with pytest.raises(AssertionError) as err:
        fg.add_ctrl_file(ctrl_id="test_ctrl",
                         data=data,
                         channels=[f"channel{i+1}" for i in range(6)],
                         markers=[f"marker{i+1}" for i in range(6)])
    assert str(err.value) == "Entry for test_ctrl already exists"


def test_add_ctrl_file(example_populated_experiment):
    fg = example_populated_experiment.get_sample("test sample")
    add_dummy_ctrl(fg, "test_ctrl2")
    assert "test_ctrl2" in fg.controls
    with h5py.File(fg.h5path, "r") as f:
        assert "test_ctrl2" in f["index/root"].keys()
        assert f["index/root/test_ctrl2"][:].shape[0] == 1000


def test_load_population_indexes(example_populated_experiment):
    fg = example_populated_experiment.get_sample("test sample")
    add_dummy_ctrl(fg, "test_ctrl2")
    fg._load_population_indexes()
    assert fg.get_population("root").index.shape[0] == 30000
    assert fg.get_population("root").ctrl_index.get("test_ctrl2").shape[0] == 1000


def test_add_population(example_populated_experiment):
    create_example_populations(example_populated_experiment.get_sample("test sample")).save()
    fg = reload_filegroup(project_id="test",
                          exp_id="test experiment",
                          sample_id="test sample")
    # Check population objects
    assert len(fg.populations) == 4
    assert all([x in [p.population_name for p in fg.populations]
                for x in ["root", "pop1", "pop2", "pop3"]])
    # Check indexes
    pop_idx = {p.population_name: p.index for p in fg.populations}
    pop_ctrl_idx = {p.population_name: p.ctrl_index.get("test_ctrl") for p in fg.populations}
    for data_dict in [pop_idx, pop_ctrl_idx]:
        for name, expected_n in zip(["root", "pop1", "pop2", "pop3"],
                                    [30000, 24000, 12000, 6000]):
            assert len(data_dict.get(name)) == expected_n
    # Check trees
    assert all([x in fg.tree.keys() for x in ["root", "pop1", "pop2", "pop3"]])
    assert not fg.tree.get("root").parent
    assert fg.tree.get("pop1").parent == fg.tree.get("root")
    assert fg.tree.get("pop2").parent == fg.tree.get("pop1")
    assert fg.tree.get("pop3").parent == fg.tree.get("pop2")


@pytest.mark.parametrize("pop_name,n", [("pop1", 24000), ("pop2", 12000), ("pop3", 6000)])
def test_load_population_df(example_populated_experiment, pop_name, n):
    create_example_populations(example_populated_experiment.get_sample("test sample")).save()
    fg = reload_filegroup(project_id="test",
                          exp_id="test experiment",
                          sample_id="test sample")
    df = fg.load_population_df(population=pop_name, transform="logicle")
    assert isinstance(df, pd.DataFrame)
    assert df.shape == (n, 7)


@pytest.mark.parametrize("pop_name,n", [("pop1", 24000), ("pop2", 12000), ("pop3", 6000)])
def test_load_ctrl_population_df(example_populated_experiment, pop_name, n):
    create_example_populations(example_populated_experiment.get_sample("test sample")).save()
    fg = reload_filegroup(project_id="test",
                          exp_id="test experiment",
                          sample_id="test sample")
    df = fg.load_ctrl_population_df(ctrl="test_ctrl", population=pop_name, transform="logicle")
    assert isinstance(df, pd.DataFrame)
    assert df.shape == (n, 7)


@pytest.mark.parametrize("pop_name,n", [("pop1", 24000), ("pop2", 12000), ("pop3", 6000)])
def test_estimate_ctrl_population(example_populated_experiment, pop_name, n):
    create_example_populations(example_populated_experiment.get_sample("test sample")).save()
    fg = reload_filegroup(project_id="test",
                          exp_id="test experiment",
                          sample_id="test sample")
    pop = fg.get_population(pop_name)
    pop.ctrl_index.pop("test_ctrl")
    transforms = {"FS Lin": None, "SS Log": None}
    mappings = {x: {"transformations": transforms, "features": ["FS Lin", "SS Log"]}
                for x in ["pop1", "pop2", "pop3"]}
    fg.estimate_ctrl_population(ctrl="test_ctrl",
                                population=pop_name,
                                downsample=0.9,
                                population_mappings=mappings)
    pop = fg.get_population(pop_name)
    assert "test_ctrl" in pop.ctrl_index.keys()
    assert isinstance(pop.ctrl_index.get("test_ctrl"), np.ndarray)
    assert pop.ctrl_index.get("test_ctrl").shape[0] > 0


def assert_correct_label(labels: pd.Series, expected_label: str):
    assert len(labels.unique()) == 1
    assert labels.unique()[0] == expected_label


def test_label_downstream_populations(example_populated_experiment):
    fg = create_example_populations(example_populated_experiment.get_sample("test sample"))
    data = fg.load_population_df(population="root", label_downstream_affiliations=True)
    assert "population_label" in data.columns
    pop3_labels = data.loc[fg.get_population("pop3").index]["population_label"]
    assert_correct_label(pop3_labels, "pop3")
    pop2_idx = np.array([x for x in fg.get_population("pop2").index if x not in fg.get_population("pop3").index])
    pop2_labels = data.loc[pop2_idx]["population_label"]
    assert_correct_label(pop2_labels, "pop2")
    pop1_idx = np.array([x for x in fg.get_population("pop1").index
                         if x not in fg.get_population("pop2").index and
                         x not in fg.get_population("pop3").index])
    pop1_labels = data.loc[pop1_idx]["population_label"]
    assert_correct_label(pop1_labels, "pop1")


def test_list_populations(example_populated_experiment):
    fg = create_example_populations(example_populated_experiment.get_sample("test sample"))
    assert set(fg.list_populations()) == {"root", "pop1", "pop2", "pop3"}


@pytest.mark.parametrize("pop_name", ["root", "pop1", "pop2", "pop3"])
def test_get_population(example_populated_experiment, pop_name):
    fg = create_example_populations(example_populated_experiment.get_sample("test sample"))
    pop = fg.get_population(population_name=pop_name)
    assert pop.population_name == pop_name


@pytest.mark.parametrize("parent,children",
                         [("root", ["pop1"]),
                          ("pop1", ["pop2"]),
                          ("pop2", ["pop3"]),
                          ("pop3", [])])
def test_get_population_by_parent(example_populated_experiment, parent, children):
    fg = create_example_populations(example_populated_experiment.get_sample("test sample"))
    assert [p.population_name for p in fg.get_population_by_parent(parent=parent)] == children


@pytest.mark.parametrize("population,downstream_populations",
                         [("root", ["pop1", "pop2", "pop3"]),
                          ("pop1", ["pop2", "pop3"]),
                          ("pop2", ["pop3"]),
                          ("pop3", [])])
def test_list_downstream_populations(example_populated_experiment, population, downstream_populations):
    fg = create_example_populations(example_populated_experiment.get_sample("test sample"))
    assert fg.list_downstream_populations(population=population) == downstream_populations


def test_print_population_tree(example_populated_experiment):
    fg = create_example_populations(example_populated_experiment.get_sample("test sample"))
    fg.print_population_tree()


def test_delete_population_error_root(example_populated_experiment):
    fg = create_example_populations(example_populated_experiment.get_sample("test sample"))
    with pytest.raises(AssertionError) as err:
        fg.delete_populations(populations=["root"])
    assert str(err.value) == "Cannot delete root population"


def test_delete_population_error_not_list(example_populated_experiment):
    fg = create_example_populations(example_populated_experiment.get_sample("test sample"))
    with pytest.raises(AssertionError) as err:
        fg.delete_populations(populations="pop1")
    assert str(err.value) == "Provide a list of population names for removal"


def assert_population_tree(fg: FileGroup,
                           expected_pops: list):
    assert list(fg.tree.keys()) == list(fg.list_populations())
    assert list(fg.tree.keys()) == expected_pops
    assert list(fg.list_populations()) == expected_pops


def test_delete_population(example_populated_experiment):
    fg = create_example_populations(example_populated_experiment.get_sample("test sample"))
    fg.delete_populations(populations=["pop3"])
    fg.save()
    assert_population_tree(fg, ["root", "pop1", "pop2"])


def test_delete_population_downstream_effects(example_populated_experiment):
    fg = create_example_populations(example_populated_experiment.get_sample("test sample"))
    downstream_pops = set(fg.list_downstream_populations("pop1"))
    with pytest.warns(UserWarning) as warn:
        fg.delete_populations(populations=["pop1"])
    assert str(warn.list[0].message) == "The following populations are downstream of one or more of the " \
                                        "populations listed for deletion and will therefore be deleted: " \
                                        f"{downstream_pops}"
    assert_population_tree(fg, ["root"])


def test_delete_many_populations(example_populated_experiment):
    fg = create_example_populations(example_populated_experiment.get_sample("test sample"))
    fg.add_population(Population(population_name="pop4",
                                 parent="pop2",
                                 index=np.array([])))
    fg.delete_populations(populations=["pop2", "pop3", "pop4"])
    fg.save()
    assert_population_tree(fg, ["root", "pop1"])


def test_delete_all_populations(example_populated_experiment):
    fg = create_example_populations(example_populated_experiment.get_sample("test sample"))
    fg.delete_populations(populations="all")
    fg.save()
    assert_population_tree(fg, ["root"])


@pytest.mark.parametrize("pop_name, expected_pops",
                         [("pop3", []),
                          ("pop2", ["pop3"]),
                          ("pop1", ["pop2", "pop3"]),
                          ("root", ["pop1", "pop2", "pop3"])])
def test_list_downstream_pops(example_populated_experiment, pop_name, expected_pops):
    fg = create_example_populations(example_populated_experiment.get_sample("test sample"))
    assert set(fg.list_downstream_populations(pop_name)) == set(expected_pops)


def test_subtract_populations(example_populated_experiment):
    fg = create_example_populations(example_populated_experiment.get_sample("test sample"))
    fg.get_population("pop2").geom = PolygonGeom(x="FS Lin", y="SS Log", transform_x=None, transform_y=None)
    fg.get_population("pop3").geom = PolygonGeom(x="FS Lin", y="SS Log", transform_x=None, transform_y=None)
    fg.subtract_populations(left=fg.get_population("pop2"),
                            right=fg.get_population("pop3"),
                            new_population_name="pop4")
    assert "pop4" in list(fg.list_populations())
    assert fg.get_population("pop4").n == (fg.get_population("pop2").n - fg.get_population("pop3").n)


def test_delete(example_populated_experiment):
    fg = create_example_populations(example_populated_experiment.get_sample("test sample"))
    fg.save()
    path = fg.h5path
    fg.delete()
    assert not os.path.isfile(path=path)
    with pytest.raises(AssertionError) as err:
        reload_filegroup(project_id="test", exp_id="test experiment", sample_id="test sample")
    assert str(err.value) == f"Invalid sample: test sample not associated with this experiment"
