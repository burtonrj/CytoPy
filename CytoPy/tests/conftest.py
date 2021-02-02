from CytoPy.tests import assets
from ..data.population import Population
from ..data.project import Project
from ..data.experiment import FileGroup
from mongoengine.connection import connect, disconnect
import pytest
import shutil
import sys
import os


@pytest.fixture(scope='session', autouse=True)
def setup():
    """
    Setup testing database

    Yields
    -------
    None
    """
    sys.path.append("/home/ross/CytoPy")
    os.mkdir(f"{os.getcwd()}/test_data")
    connect("test", host="mongomock://localhost", alias="core")
    yield
    shutil.rmtree(f"{os.getcwd()}/test_data", ignore_errors=True)
    disconnect(alias="core")


@pytest.fixture
def example_populated_experiment():
    """
    Generate an example Experiment populated with a single FileGroup "test sample"

    Yields
    -------
    Experiment
    """
    test_project = Project(project_id="test")
    exp = test_project.add_experiment(experiment_id="test experiment",
                                      data_directory=f"{os.getcwd()}/test_data",
                                      panel_definition=f"{assets.__path__._path[0]}/test_panel.xlsx")
    exp.add_fcs_files(sample_id="test sample",
                      primary=f"{assets.__path__._path[0]}/test.FCS",
                      controls={"test_ctrl": f"{assets.__path__._path[0]}/test.FCS"},
                      compensate=False)
    yield exp
    test_project.reload()
    test_project.delete()


def reload_filegroup(project_id: str,
                     exp_id: str,
                     sample_id: str):
    """
    Reload a FileGroup

    Parameters
    ----------
    project_id: str
    exp_id: str
    sample_id: str

    Returns
    -------
    FileGroup
    """
    fg = (Project.objects(project_id=project_id)
          .get()
          .load_experiment(exp_id)
          .get_sample(sample_id))
    return fg


def create_example_population_indexes(filegroup: FileGroup,
                                      initial_population_prop: float = 0.8,
                                      downstream_population_prop: float = 0.5,
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
                    "ctrl": filegroup.data("test_ctrl", sample_size=initial_population_prop)}]
    for i in range(n_populations - 1):
        primary = populations[i].get("primary").sample(frac=downstream_population_prop)
        populations.append({"primary": primary,
                            "ctrl": populations[i].get("ctrl").sample(frac=downstream_population_prop)})
    return list(map(lambda x: {"primary": x["primary"].index.values,
                               "ctrl": x["ctrl"].index.values},
                    populations))


def create_example_populations(filegroup: FileGroup,
                               initial_population_prop: float = 0.8,
                               downstream_population_prop: float = 0.5,
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
                                                n_populations=n_populations)
    for pname, parent, idx in zip([f"pop{i + 1}" for i in range(n_populations)],
                                  ["root"] + [f"pop{i + 1}" for i in range(n_populations - 1)],
                                  pop_idx):
        p = Population(population_name=pname,
                       n=len(idx.get("primary")),
                       parent=parent,
                       index=idx.get("primary"),
                       source="gate")
        p.set_ctrl_index(test_ctrl=idx.get("ctrl"))
        filegroup.add_population(population=p)
    filegroup.save()
    return filegroup
