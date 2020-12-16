from ..data.project import Project
from ..data.experiment import Experiment, FileGroup
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
                                      panel_definition=f"{os.getcwd()}/tests/assets/test_panel.xlsx")
    exp.add_fcs_files(sample_id="test sample",
                      primary_path=f"{os.getcwd()}/tests/assets/test.FCS",
                      controls_path={"test_ctrl": f"{os.getcwd()}/tests/assets/test.FCS"},
                      compensate=False)
    yield exp
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









