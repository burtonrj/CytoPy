from cytopy.tests import assets
from ..data.population import Population
from ..data.project import Project
from ..data.experiment import FileGroup
from mongoengine.connection import connect, disconnect
import pandas as pd
import numpy as np
import inspect
import pytest
import shutil
import sys
import os
ASSET_PATH = inspect.getmodule(assets).__path__[0]


@pytest.fixture(scope='session', autouse=True)
def setup():
    """
    Setup testing database

    Yields
    -------
    None
    """

    os.mkdir(f"{ASSET_PATH}/test_data")
    connect("test", host="mongomock://localhost", alias="core")
    yield
    shutil.rmtree(f"{ASSET_PATH}/test_data", ignore_errors=True)
    disconnect(alias="core")


@pytest.fixture
def example_populated_experiment():
    """
    Generate an example Experiment populated with a single FileGroup "test sample"

    Yields
    -------
    Experiment
    """
    test_project = Project(project_id="test", data_directory=f"{os.getcwd()}/test_data")
    exp = test_project.add_experiment(experiment_id="test experiment",
                                      panel_definition=f"{ASSET_PATH}/test_panel.xlsx")
    exp.add_fcs_files(sample_id="test sample",
                      primary=f"{ASSET_PATH}/test.fcs",
                      controls={"test_ctrl": f"{ASSET_PATH}/test.fcs"},
                      compensate=False)
    yield exp
    test_project.reload()
    test_project.delete()
    os.mkdir(f"{ASSET_PATH}/test_data")


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
          .get_experiment(exp_id)
          .get_sample(sample_id))
    return fg


def create_example_populations(filegroup: FileGroup,
                               n_populations: int = 3):
    """
    Given a FileGroup add the given number of example populations.

    Parameters
    ----------
    filegroup: FileGroup
    n_populations: int (default=3)
        Total number of populations to generate (must be at least 2)

    Returns
    -------
    FileGroup
    """
    for pname, parent in zip([f"pop{i + 1}" for i in range(n_populations)],
                             ["root"] + [f"pop{i + 1}" for i in range(n_populations - 1)]):
        parent_df = filegroup.load_population_df(population=parent,
                                                 transform="logicle")
        x = parent_df["FS Lin"].median()
        idx = parent_df[parent_df["FS Lin"] >= x].index.values
        p = Population(population_name=pname,
                       n=len(idx),
                       parent=parent,
                       index=idx,
                       source="gate")
        filegroup.add_population(population=p)
    filegroup.save()
    return filegroup


def create_logicle_like(u: list, s: list, size: list):
    assert len(u) == len(s), "s and u should be equal length"
    lognormal = [np.random.lognormal(mean=u[i], sigma=s[i], size=int(size[i]))
                 for i in range(len(u))]
    return np.concatenate(lognormal)


def create_linear_data():
    x = np.concatenate([np.random.normal(loc=3.2, scale=0.8, size=100000),
                       np.random.normal(loc=0.95, scale=1.1, size=100000)])
    y = np.concatenate([np.random.normal(loc=3.1, scale=0.85, size=100000),
                       np.random.normal(loc=0.5, scale=1.4, size=100000)])
    return pd.DataFrame({"x": x, "y": y})


def create_lognormal_data():
    x = np.concatenate([np.random.normal(loc=4.2, scale=0.8, size=50000),
                        np.random.lognormal(mean=4.2, sigma=0.8, size=50000),
                        np.random.lognormal(mean=7.2, sigma=0.8, size=50000),
                       np.random.lognormal(mean=0.8, sigma=0.95, size=50000)])
    y = np.concatenate([np.random.normal(loc=3.2, scale=0.8, size=50000),
                        np.random.lognormal(mean=4.1, sigma=0.8, size=50000),
                        np.random.lognormal(mean=6.2, sigma=0.8, size=50000),
                       np.random.lognormal(mean=1.4, sigma=0.7, size=50000)])
    return pd.DataFrame({"x": x, "y": y})
