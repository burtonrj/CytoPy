from ...data.project import Project
from ...data.experiment import Experiment
import pytest
import os


@pytest.fixture
def example_experiment():
    test_project = Project(project_name="test")
    test_exp = test_project.add_experiment(experiment_id="test experiment",
                                           data_directory=f"{os.getcwd()}/test_data",
                                           panel_definition=f"{os.getcwd()}/tests/assets/test_panel.xlsx")
    test_exp.add_new_sample(sample_id="test sample",
                            primary_path=f"{os.getcwd()}/tests/assets/test.FCS",
                            controls_path={"ctrl1": f"{os.getcwd()}/tests/assets/test.FCS"},
                            compensate=False)
    yield test_exp
    test_project.delete()


def test_load_data(example_experiment):
    pass


def test_preview_gate(example_experiment):
    pass


def test_apply_gate(example_experiment):
    pass


def test_apply_all(example_experiment):
    pass


def test_delete_gate(example_experiment):
    pass


def test_edit_gate(example_experiment):
    pass


def test_delete_population(example_experiment):
    pass


def test_delete_action(example_experiment):
    pass


def test_plot_gate(example_experiment):
    pass


def test_plot_backgate(example_experiment):
    pass


def test_plot_population(example_experiment):
    pass


def test_print_population_tree(example_experiment):
    pass


def test_population_stats(example_experiment):
    pass


def test_estimate_ctrl_population(example_experiment):
    pass


def test_merge_populations(example_experiment):
    pass


def test_subtract_populations(example_experiment):
    pass


def test_save(example_experiment):
    pass


def test_delete():
    pass
