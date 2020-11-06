from CytoPy.data.project import Project
from CytoPy.tests import assets
import pytest
import os


@pytest.fixture()
def create_project():
    p = Project(project_id="test")
    p.save()
    yield p
    p.delete()


def test_add_experiment(create_project):
    p = create_project
    p.add_experiment(experiment_id="test1",
                     data_directory=f"{os.getcwd()}/test_data",
                     panel_definition=f"{assets.__path__._path[0]}/test_panel.xlsx")
    assert len(p.experiments) == 1
    p.add_experiment(experiment_id="test2",
                     data_directory=f"{os.getcwd()}/test_data",
                     panel_name="test1_panel")
    assert len(p.experiments) == 2


def test_add_experiment_duplicate_err(create_project):
    p = create_project
    p.add_experiment(experiment_id="test1",
                     data_directory=f"{os.getcwd()}/test_data",
                     panel_definition=f"{assets.__path__._path[0]}/test_panel.xlsx")
    with pytest.raises(AssertionError) as err:
        p.add_experiment(experiment_id="test1",
                         data_directory=f"{os.getcwd()}/test_data",
                         panel_definition=f"{assets.__path__._path[0]}/test_panel.xlsx")
    assert str(err.value) == 'Error: Experiment with id test1 already exists!'


def test_list_experiments(create_project):
    p = create_project
    p.add_experiment(experiment_id="test1",
                     data_directory=f"{os.getcwd()}/test_data",
                     panel_definition=f"{assets.__path__._path[0]}/test_panel.xlsx")
    assert list(p.list_experiments()) == ["test1"]


def test_load_experiment(create_project):
    p = create_project
    p.add_experiment(experiment_id="test1",
                     data_directory=f"{os.getcwd()}/test_data",
                     panel_definition=f"{assets.__path__._path[0]}/test_panel.xlsx")
    e = p.load_experiment(experiment_id="test1")
    assert e.experiment_id == "test1"


def test_add_subject(create_project):
    p = create_project
    p.add_subject(subject_id="test_subject")
    assert len(p.subjects) == 1


def test_list_subjects(create_project):
    p = create_project
    p.add_subject(subject_id="test_subject")
    assert list(p.list_subjects()) == ["test_subject"]


def test_get_subject(create_project):
    p = create_project
    p.add_subject(subject_id="test_subject")
    s = p.get_subject(subject_id="test_subject")
    assert s.subject_id == "test_subject"
