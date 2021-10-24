import inspect
import logging
import os
import random
from logging.config import dictConfig
from typing import List
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from mongoengine.connection import connect
from mongoengine.connection import disconnect
from sklearn.datasets import make_blobs

from cytopy.data.experiment import FileGroup
from cytopy.data.population import Population
from cytopy.data.project import Project
from cytopy.data.setup import Config
from tests import assets

ASSET_PATH = inspect.getmodule(assets).__path__[0]
config = Config()
dictConfig(config.logging_config)
logger = logging.getLogger(__name__)


@pytest.fixture(scope="session")
def setup():
    """
    Setup testing database

    Yields
    -------
    None
    """
    logger.info("Setting up testing server")
    # Obtain local paths
    temp_data_path = os.path.join(ASSET_PATH, "data")

    # Connect and create project
    logger.info("Creating mock database 'test' and Project 'test_project'")
    connect("test", host="mongomock://localhost", alias="core")
    project = Project(project_id="test_project", data_directory=temp_data_path)
    project.save()

    # Add some fake subjects
    logger.info("Creating mock subjects")
    for i in range(12):
        subject_id = f"subject_{str(i + 1).zfill(3)}"
        logger.debug(f"Creating mock subject {subject_id}")
        project.add_subject(
            subject_id=subject_id, age=random.randint(18, 99), gender=["male", "female"][random.randint(0, 1)]
        )

    # Populate with GVHD utils data
    logger.info("Creating mock experiment 'test_exp'")
    test_exp = project.add_experiment(experiment_id="test_exp")
    test_exp.generate_panel(panel_definition=os.path.join(ASSET_PATH, "test_panel.xlsx"))

    logger.info("Adding test FCS data")
    for i in range(12):
        file_id = str(i + 1).zfill(3)
        logger.debug(f"Adding test FCS data {file_id}")
        path = os.path.join(ASSET_PATH, "gvhd_fcs", f"{file_id}.fcs")
        test_exp.add_filegroup(
            sample_id=file_id, subject_id=f"subject_{file_id}", paths=dict(primary=path), compensate=False
        )

    # Yield to tests
    yield

    # Destroy local temp data and disconnect
    logger.info("Destroying test data and disconnecting")
    disconnect(alias="core")


def savefig(figure: plt.Figure, filename: str):
    output_dir = config["test_config"]["figure_output_path"]
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)
    output_path = os.path.join(output_dir, filename)
    figure.savefig(output_path, bbox_inches="tight")


def add_populations(filegroups: Optional[List[str]] = None):
    logger.info("Adding test populations")
    project = Project.objects(project_id="test_project").get()
    exp = project.get_experiment(experiment_id="test_exp")
    filegroups = filegroups or exp.list_samples()
    for _id in filegroups:
        fg = exp.get_sample(sample_id=_id)
        logger.info(f"Adding populations to {fg.primary_id}")
        labels = pd.read_csv(os.path.join(ASSET_PATH, "gvhd_labels", f"{fg.primary_id}.csv"))
        for pop_i in labels.V1.unique():
            idx = labels[labels.V1 == pop_i].index.values
            pop = Population(
                population_name=f"population_{pop_i}", n=len(idx), index=idx, parent="root", source="cluster"
            )
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
        parent_df = filegroup.load_population_df(population=parent, transform="asinh")
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


@pytest.fixture
def big_blobs():
    x, y = make_blobs(n_samples=4000000, n_features=15, random_state=42, centers=8)
    return pd.DataFrame(x, columns=[f"f{i + 1}" for i in range(15)]), y


@pytest.fixture
def small_blobs():
    x, y = make_blobs(n_samples=1000, n_features=3, random_state=42, centers=3)
    return pd.DataFrame(x, columns=[f"f{i + 1}" for i in range(3)]), y


@pytest.fixture
def small_high_dim_dataframe():
    data = list()
    column_names = [f"f{i + 1}" for i in range(10)]
    for i, k in enumerate([5, 5, 6, 6, 5]):
        x = pd.DataFrame(
            make_blobs(n_samples=1000, n_features=10, cluster_std=2.5, random_state=i, centers=k)[0],
            columns=column_names,
        )
        x["sample_id"] = f"sample_{i}"
        data.append(x)
    return pd.concat(data).reset_index().rename({"index": "original_index"}, axis=1)
