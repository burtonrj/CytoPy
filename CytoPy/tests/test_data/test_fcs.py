from ...data.fcs import FileGroup
from ...data.project import Project
from ...data.population import Cluster, Population
import pandas as pd
import numpy as np
import pytest
import os


@pytest.fixture
def example_filegroup():
    test_project = Project(project_id="test")
    exp = test_project.add_experiment(experiment_id="test experiment",
                                      data_directory=f"{os.getcwd()}/test_data",
                                      panel_definition=f"{os.getcwd()}/CytoPy/tests/assets/test_panel.xlsx")
    exp.add_new_sample(sample_id="test sample",
                       primary_path=f"{os.getcwd()}/CytoPy/tests/assets/test.FCS",
                       controls_path={"test_ctrl": f"{os.getcwd()}/CytoPy/tests/assets/test.FCS"},
                       compensate=False)
    yield exp.get_sample(sample_id="test sample")
    test_project.delete()


def create_populations(filegroup: FileGroup):
    p1data = filegroup.data("primary", sample_size=0.8)
    p1ctrldata = filegroup.data("test_ctrl", sample_size=0.8)
    p2data = p1data.sample(frac=0.5)
    p2ctrldata = p1ctrldata.sample(frac=0.5)
    p3data = p2data.sample(frac=0.5)
    p3ctrldata = p2ctrldata.sample(frac=0.5)
    populations = list()
    for pname, parent, data, ctrldata in zip(["pop1", "pop2", "pop3"],
                                             ["root", "pop1", "pop2"],
                                             [p1data, p2data, p3data],
                                             [p1ctrldata, p2ctrldata, p3ctrldata]):
        p = Population(population_name=pname,
                       n=data.shape[0],
                       parent=parent,
                       index=data.index.values)
        p.set_ctrl_index(test_ctrl=ctrldata.index.values)
        cluster_idx = data.sample(frac=0.25).index.values
        p.add_cluster(Cluster(cluster_id="test cluster",
                              index=cluster_idx,
                              n=len(cluster_idx),
                              prop_of_events=len(cluster_idx) / 30000,
                              tag="testing"))
        populations.append(p)
    for p in populations:
        filegroup.add_population(population=p)
    return filegroup, populations


def reload_file():
    fg = (Project.objects(project_id="test").
          get()
          .load_experiment("test experiment")
          .get_sample("test sample"))
    return fg


def test_init_new_fcs_file(example_filegroup):
    fg = example_filegroup
    assert os.path.isfile(f"{os.getcwd()}/test_data/{fg.id}.hdf5")
    experiment = Project.objects(project_id="test").get().load_experiment("test experiment")
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
def test_access_data(example_filegroup, source, sample_size):
    fg = example_filegroup
    df = fg.data(source, sample_size)
    assert isinstance(df, pd.DataFrame)
    if sample_size is not None:
        assert df.shape == (1000, 7)
    else:
        assert df.shape == (30000, 7)


def test_add_population(example_filegroup):
    fg, populations = create_populations(filegroup=example_filegroup)
    fg.save()
    experiment = Project.objects(project_id="test").get().load_experiment("test experiment")
    fg = experiment.get_sample("test sample")
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
    # Check cluster idx
    clustered_pops = [p for p in fg.populations if p.population_name != "root"]
    cluster_idx = {p.population_name: p.clusters[0].index for p in clustered_pops}
    for name, n in zip(["pop1", "pop2", "pop3"],
                       [24000, 12000, 6000]):
        assert len(cluster_idx.get(name)) == n * 0.25
    # Check trees
    assert all([x in fg.tree.keys() for x in ["root", "pop1", "pop2", "pop3"]])
    assert not fg.tree.get("root").parent
    assert fg.tree.get("pop1").parent == fg.tree.get("root")
    assert fg.tree.get("pop2").parent == fg.tree.get("pop1")
    assert fg.tree.get("pop3").parent == fg.tree.get("pop2")


@pytest.mark.parametrize("pop_name,n", [("pop1", 24000), ("pop2", 12000), ("pop3", 6000)])
def test_load_population_df(example_filegroup, pop_name, n):
    fg, populations = create_populations(filegroup=example_filegroup)
    fg.save()
    df = fg.load_population_df(population=pop_name, transform="logicle")
    assert isinstance(df, pd.DataFrame)
    assert df.shape == (n, 7)


@pytest.mark.parametrize("pop_name,n", [("pop1", 24000), ("pop2", 12000), ("pop3", 6000)])
def test_load_ctrl_population_df(example_filegroup, pop_name, n):
    fg, populations = create_populations(filegroup=example_filegroup)
    fg.save()
    df = fg.load_ctrl_population_df(ctrl="test_ctrl", population=pop_name, transform="logicle")
    assert isinstance(df, pd.DataFrame)
    assert df.shape == (n, 7)


@pytest.mark.parametrize("pop_name", ["root", "pop1", "pop2", "pop3"])
def test_get_population(example_filegroup, pop_name):
    fg, populations = create_populations(filegroup=example_filegroup)
    pop = fg.get_population(population_name=pop_name)
    assert pop.population_name == pop_name


@pytest.mark.parametrize("parent,children",
                         [("root", ["pop1"]),
                          ("pop1", ["pop2"]),
                          ("pop2", ["pop3"]),
                          ("pop3", [])])
def test_get_population_by_parent(example_filegroup, parent, children):
    fg, populations = create_populations(filegroup=example_filegroup)
    assert [p.population_name for p in fg.get_population_by_parent(parent=parent)] == children


@pytest.mark.parametrize("population,downstream_populations",
                         [("root", ["pop1", "pop2", "pop3"]),
                          ("pop1", ["pop2", "pop3"]),
                          ("pop2", ["pop3"]),
                          ("pop3", [])])
def test_list_downstream_populations(example_filegroup, population, downstream_populations):
    fg, populations = create_populations(filegroup=example_filegroup)
    assert fg.list_downstream_populations(population=population) == downstream_populations


def test_list_populations(example_filegroup):
    fg, populations = create_populations(filegroup=example_filegroup)
    assert list(fg.list_populations()) == ["root", "pop1", "pop2", "pop3"]


def test_print_population_tree(example_filegroup):
    fg, populations = create_populations(filegroup=example_filegroup)
    fg.print_population_tree()


def test_delete_population_error_root(example_filegroup):
    fg, populations = create_populations(filegroup=example_filegroup)
    with pytest.raises(AssertionError) as err:
        fg.delete_populations(populations=["root"])
    assert str(err.value) == "Cannot delete root population"


def test_delete_population_error_not_list(example_filegroup):
    fg, populations = create_populations(filegroup=example_filegroup)
    with pytest.raises(AssertionError) as err:
        fg.delete_populations(populations="pop1")
    assert str(err.value) == "Provide a list of population names for removal"


def assert_population_tree(fg: FileGroup,
                           expected_pops: list):
    assert list(fg.tree.keys()) == list(fg.list_populations())
    assert list(fg.tree.keys()) == expected_pops
    assert list(fg.list_populations()) == expected_pops


def test_delete_population(example_filegroup):
    fg, populations = create_populations(filegroup=example_filegroup)
    fg.delete_populations(populations=["pop3"])
    fg.save()
    assert_population_tree(reload_file(), ["root", "pop1", "pop2"])


def test_delete_population_downstream_effects(example_filegroup):
    fg, populations = create_populations(filegroup=example_filegroup)
    downstream_pops = set(fg.list_downstream_populations("pop1"))
    with pytest.warns(UserWarning) as warn:
        fg.delete_populations(populations=["pop1"])
    assert str(warn.list[0].message) == "The following populations are downstream of one or more of the " \
                                        "populations listed for deletion and will therefore be deleted: " \
                                        f"{downstream_pops}"
    assert_population_tree(reload_file(), ["root"])


def test_delete_many_populations(example_filegroup):
    fg, populations = create_populations(filegroup=example_filegroup)
    fg.add_population(Population(population_name="pop4",
                                 parent="pop2",
                                 index=np.array([])))
    fg.delete_populations(populations=["pop2", "pop3", "pop4"])
    fg.save()
    assert_population_tree(reload_file(), ["root", "pop1"])


def test_delete_all_populations(example_filegroup):
    fg, populations = create_populations(filegroup=example_filegroup)
    fg.delete_populations(populations="all")
    fg.save()
    assert_population_tree(reload_file(), ["root"])


@pytest.mark.parametrize("drop_all,drop_tag,drop_metalabel,expected_clusters",
                         [(False, "testing", None, ["test cluster 0", "test cluster 1", "test cluster 2"]),
                          (False, "testing 2", None, ["test cluster 2"]),
                          (False, None, "test meta 1", ["test cluster 2"]),
                          (False, None, "test meta 2", ["test cluster 0", "test cluster 1"]),
                          (True, None, None, [])])
def test_delete_clusters(example_filegroup, drop_all, drop_tag, drop_metalabel, expected_clusters):
    fg, populations = create_populations(filegroup=example_filegroup)
    for name in ["pop4", "pop5", "pop6"]:
        p = Population(population_name=name,
                       n=1000,
                       parent="pop2",
                       index=np.arange(0, 1000))
        p.set_ctrl_index(test_ctrl=np.arange(0, 1000))
        cluster_idx = np.arange(0, 1000)
        for i, (tag, metalabel) in enumerate(zip(["testing 2", "testing 2", "testing 3"],
                                             ["test meta 1", "test meta 1", "test meta 2"])):
            p.add_cluster(Cluster(cluster_id=f"test cluster {i}",
                                  index=cluster_idx,
                                  n=len(cluster_idx),
                                  prop_of_events=len(cluster_idx) / 30000,
                                  tag=tag,
                                  meta_label=metalabel))
        fg.add_population(population=p)
    fg.save()
    fg = reload_file()
    fg.delete_clusters(tag=drop_tag, meta_label=drop_metalabel, drop_all=drop_all)
    fg.save()
    fg = reload_file()
    for pop_name in ["pop4", "pop5", "pop6"]:
        p = fg.get_population(population_name=pop_name)
        assert p.list_clusters() == expected_clusters


def test_delete(example_filegroup):
    fg, populations = create_populations(filegroup=example_filegroup)
    fg.save()
    path = fg.h5path
    fg.delete()
    assert not os.path.isfile(path=path)
    with pytest.raises(AssertionError) as err:
        reload_file()
    assert str(err.value) == f"Invalid sample: test sample not associated with this experiment"
