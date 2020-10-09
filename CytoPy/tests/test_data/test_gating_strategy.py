from ...data.project import Project
from ...data.experiment import Experiment
import pytest
import os


@pytest.fixture(scope="module", autouse=True)
def example_data_setup():
    test_project = Project(project_name="test")
    test_exp = test_project.add_experiment(experiment_id="test experiment",
                                           data_directory=f"{os.getcwd()}/test_data",
                                           panel_definition=f"{os.getcwd()}/tests/assets/test_panel.xlsx")
    test_exp.add_new_sample(sample_id="test sample",
                            primary_path=f"{os.getcwd()}/tests/assets/test.FCS")


