import inspect
import logging
import os
import random
import shutil
from logging.config import dictConfig

import numpy as np
import pandas as pd
import pytest
from mongoengine.connection import connect
from mongoengine.connection import disconnect

from ..data.experiment import FileGroup
from ..data.population import Population
from ..data.project import Project
from ..data.setup import Config
from cytopy.tests import assets

ASSET_PATH = inspect.getmodule(assets).__path__[0]
config = Config()
dictConfig(config.logging_config)
logger = logging.getLogger(__name__)


@pytest.fixture(scope="session", autouse=True)
def setup():
    """
    Setup testing database

    Yields
    -------
    None
    """
    logger.info("Setting up testing server")
    # Setup local paths
    temp_data_path = os.path.join(ASSET_PATH, "data")
    os.mkdir(temp_data_path)

    # Connect and create project
    logger.info("Creating mock database 'test' and Project 'test_project'")
    connect("test", host="mongomock://localhost", alias="core")
    project = Project(project_id="test_project")
    project.save()

    # Add some fake subjects
    for i in range(12):
        subject_id = f"subject_{str(i+1).zfill(3)}"
        logger.info(f"Creating mock subject {subject_id}")
        project.add_subject(
            subject_id=subject_id, age=random.randint(18, 99), gender=["male", "female"][random.randint(0, 1)]
        )

    # Populate with GVHD flow data
    logger.info("Creating mock experiment 'test_exp'")
    test_exp = project.add_experiment(
        experiment_id="test_exp", panel_definition=os.path.join(ASSET_PATH, "test_panel.xlsx")
    )
    for i in range(12):
        file_id = str(i + 1).zfill(3)
        logger.info(f"Adding test FCS data {file_id}")
        path = os.path.join(ASSET_PATH, "gvhd_fcs", f"{file_id}.fcs")
        test_exp.add_fcs_files(sample_id=file_id, subject_id=f"subject_{file_id}", primary_data=path, compensate=False)

    # Yield to tests
    yield

    # Destroy local temp data and disconnect
    logger.info("Destroying test data and disconnecting")
    shutil.rmtree(temp_data_path, ignore_errors=True)
    disconnect(alias="core")


@pytest.fixture
def add_populations():
    logger.info("Adding test populations")
    project = Project.objects(project_id="test_project").get()
    exp = project.get_experiment(experiment_id="test_exp")
    for fg in exp.fcs_files:
        logger.info(f"Adding populations to {fg.primary_id}")
        labels = pd.read_csv(os.path.join(ASSET_PATH, "gvhd_labels", f"{fg.primary_id}.csv"))
        for pop_i in labels.V1.unique():
            idx = labels[labels.V1 == pop_i].index.values
            pop = Population(population_name=f"population_{pop_i}", n=len(idx), index=idx, parent="root")
            fg.add_population(population=pop)
        fg.save()


def reload_filegroup(project_id: str, exp_id: str, sample_id: str):
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
    fg = Project.objects(project_id=project_id).get().get_experiment(exp_id).get_sample(sample_id)
    return fg


def create_example_populations(filegroup: FileGroup, n_populations: int = 3):
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
    for pname, parent in zip(
        [f"pop{i + 1}" for i in range(n_populations)],
        ["root"] + [f"pop{i + 1}" for i in range(n_populations - 1)],
    ):
        parent_df = filegroup.load_population_df(population=parent, transform="logicle")
        x = parent_df["FS Lin"].median()
        idx = parent_df[parent_df["FS Lin"] >= x].index.values
        p = Population(population_name=pname, n=len(idx), parent=parent, index=idx, source="gate")
        filegroup.add_population(population=p)
    filegroup.save()
    return filegroup


def create_logicle_like(u: list, s: list, size: list):
    assert len(u) == len(s), "s and u should be equal length"
    lognormal = [np.random.lognormal(mean=u[i], sigma=s[i], size=int(size[i])) for i in range(len(u))]
    return np.concatenate(lognormal)


def create_linear_data():
    x = np.concatenate(
        [
            np.random.normal(loc=3.2, scale=0.8, size=100000),
            np.random.normal(loc=0.95, scale=1.1, size=100000),
        ]
    )
    y = np.concatenate(
        [
            np.random.normal(loc=3.1, scale=0.85, size=100000),
            np.random.normal(loc=0.5, scale=1.4, size=100000),
        ]
    )
    return pd.DataFrame({"x": x, "y": y})


def create_lognormal_data():
    x = np.concatenate(
        [
            np.random.normal(loc=4.2, scale=0.8, size=50000),
            np.random.lognormal(mean=4.2, sigma=0.8, size=50000),
            np.random.lognormal(mean=7.2, sigma=0.8, size=50000),
            np.random.lognormal(mean=0.8, sigma=0.95, size=50000),
        ]
    )
    y = np.concatenate(
        [
            np.random.normal(loc=3.2, scale=0.8, size=50000),
            np.random.lognormal(mean=4.1, sigma=0.8, size=50000),
            np.random.lognormal(mean=6.2, sigma=0.8, size=50000),
            np.random.lognormal(mean=1.4, sigma=0.7, size=50000),
        ]
    )
    return pd.DataFrame({"x": x, "y": y})
