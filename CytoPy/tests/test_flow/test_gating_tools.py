from ...data.experiments import Experiment
from ...tests import assets
import pytest
import os


@pytest.fixture(autouse=True)
def create_example():
    test_exp = Experiment(experiment_id="test",
                          data_directory=f"{os.getcwd()}/test_data",
                          panel_definition=f"{assets.__path__._path[0]}/test_panel.xlsx")
