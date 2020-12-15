from CytoPy.data.fcs import *
from CytoPy.data.project import Project
from CytoPy.data.population import Cluster, Population
import pandas as pd
import numpy as np
import pytest
import h5py
import os


def create_example_population_indexes(filegroup: FileGroup,
                                      initial_population_prop: float = 0.8,
                                      downstream_population_prop: float = 0.5,
                                      cluster_frac: float = 0.25,
                                      n_populations: int = 3):
    """
    Create example index data for a specified number of example populations.

    Parameters
    ----------
    filegroup: FileGroup
    initial_population_prop: float (default=0.8)
        Fraction of events to sample for the first population
    downstream_population_prop: float (default=0.5)
        Fraction of events to sample from n-1 population to form downstream population
    cluster_frac: float (default=0.25)
        Fraction of events to sample from primary to use as example Cluster
    n_populations: int (default=3)
        Total number of populations to generate (must be at least 2)

    Returns
    -------
    List
        List of dictionary objects with keys 'primary', 'cluster' and 'ctrl' corresponding to events for
        primary data and "test_ctrl"
    """
    assert n_populations > 1, "n_populations must be equal to or greater than 2"
    primary = filegroup.data("primary", sample_size=initial_population_prop)
    populations = [{"primary": primary,
                    "ctrl": filegroup.data("test_ctrl", sample_size=initial_population_prop),
                    "cluster": primary.sample(frac=cluster_frac)}]
    for i in range(n_populations - 1):
        primary = populations[i + 1].get("primary").sample(frac=downstream_population_prop)
        populations.append({"primary": primary,
                            "ctrl": populations[i + 1].get("ctrl").sample(frac=downstream_population_prop),
                            "cluster": primary.sample(frac=cluster_frac)})
    return list(map(lambda x: {"primary": x["primary"].index.values,
                               "ctrl": x["ctrl"].index.values,
                               "cluster": x["cluster"].index.values},
                    populations))


def create_example_populations(filegroup: FileGroup,
                               initial_population_prop: float = 0.8,
                               downstream_population_prop: float = 0.5,
                               cluster_frac: float = 0.25,
                               n_populations: int = 3):
    """
    Given a FileGroup add the given number of example populations.

    Parameters
    ----------
    filegroup: FileGroup
    initial_population_prop: float (default=0.8)
        Fraction of events to sample for the first population
    downstream_population_prop: float (default=0.5)
        Fraction of events to sample from n-1 population to form downstream population
    cluster_frac: float (default=0.25)
        Fraction of events to sample from primary to use as example Cluster
    n_populations: int (default=3)
        Total number of populations to generate (must be at least 2)

    Returns
    -------
    FileGroup
    """
    pop_idx = create_example_population_indexes(filegroup=filegroup,
                                                initial_population_prop=initial_population_prop,
                                                downstream_population_prop=downstream_population_prop,
                                                cluster_frac=cluster_frac,
                                                n_populations=n_populations)
    for pname, parent, idx in zip([f"pop{i + 1}" for i in range(n_populations)],
                                  ["root"] + [f"pop{i + 1}" for i in range(n_populations - 1)],
                                  pop_idx):
        p = Population(population_name=pname,
                       n=len(idx.get("primary")),
                       parent=parent,
                       index=idx.get("primary"))
        p.set_ctrl_index(test_ctrl=idx.get("ctrl"))
        p.add_cluster(Cluster(cluster_id="test cluster",
                              index=idx.get("cluster"),
                              n=len(idx.get("cluster")),
                              prop_of_events=len(idx.get("cluster")) / 30000,
                              tag="testing"))
        filegroup.add_population(population=p)
    return filegroup


def create_test_h5file(path: str,
                       empty: bool = False):
    with h5py.File(path, "w") as f:
        f.create_group("index")
        f.create_group("clusters")
        if empty:
            return
        f.create_group("index/test_pop")
        f.create_group("clusters/test_pop")
        f.create_dataset("index/test_pop/primary", data=np.random.random_integers(1000, size=1000))
        f.create_dataset("index/test_pop/test_ctrl1", data=np.random.random_integers(1000, size=1000))
        f.create_dataset("index/test_pop/test_ctrl2", data=np.random.random_integers(1000, size=1000))
        f.create_dataset("clusters/test_pop/cluster1_tag1", data=np.random.random_integers(1000, size=1000))
        f.create_dataset("clusters/test_pop/cluster2_tag2", data=np.random.random_integers(1000, size=1000))


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


def test_h5_read_population_ctrl_index():
    path = f"{os.getcwd()}/test_data/test.h5"
    create_test_h5file(path=path, empty=True)
    with h5py.File(path, "r") as f:
        assert h5_read_population_ctrl_index("test_pop", f) is None
    create_test_h5file(path=path, empty=False)
    with h5py.File(path, "r") as f:
        x = h5_read_population_ctrl_index("test_pop", f)
        assert isinstance(x, dict)
        assert len(x) == 2
        assert x.get("test_ctrl1").shape[0] == 1000
        assert x.get("test_ctrl2").shape[0] == 1000


def test_h5_read_population_clusters():
    path = f"{os.getcwd()}/test_data/test.h5"
    create_test_h5file(path=path, empty=True)
    with h5py.File(path, "r") as f:
        assert h5_read_population_clusters("test_pop", f) is None
    create_test_h5file(path=path, empty=False)
    with h5py.File(path, "r") as f:
        x = h5_read_population_clusters("test_pop", f)
        assert isinstance(x, dict)
        assert len(x) == 2
        assert x.get("cluster1_tag1").shape[0] == 1000
        assert x.get("cluster2_tag2").shape[0] == 1000


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
