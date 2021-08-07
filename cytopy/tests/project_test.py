import os

import pytest

from cytopy.data.errors import *
from cytopy.data.project import Project
from cytopy.tests import assets


@pytest.fixture()
def create_project():
    os.mkdir(f"{os.getcwd()}/test_data2")
    p = Project(project_id="test", data_directory=f"{os.getcwd()}/test_data2")
    p.save()
    yield p
    p.delete()


def test_create_project_warn_directory():
    x = f"{os.getcwd()}/test_data2"
    with pytest.warns(UserWarning) as warn_:
        Project(project_id="test", data_directory=x)
    warning = (
        f"Could not locate data directory at path {x}, all further operations "
        f"will likely resolve in errors as single cell data will not be attainable. Update the "
        f"data directory before continuing using the 'update_data_directory' method."
    )
    assert str(warn_.list[0].message) == warning


def test_add_experiment(create_project):
    p = create_project
    p.add_experiment(
        experiment_id="test1",
        panel_definition=f"{assets.__path__._path[0]}/test_panel.xlsx",
    )
    assert len(p.experiments) == 1
    p.add_experiment(
        experiment_id="test2",
        panel_definition=f"{assets.__path__._path[0]}/test_panel.xlsx",
    )
    assert len(p.experiments) == 2


def test_add_experiment_duplicate_err(create_project):
    p = create_project
    p.add_experiment(
        experiment_id="test1",
        panel_definition=f"{assets.__path__._path[0]}/test_panel.xlsx",
    )
    with pytest.raises(DuplicateExperimentError) as err:
        p.add_experiment(
            experiment_id="test1",
            panel_definition=f"{assets.__path__._path[0]}/test_panel.xlsx",
        )
    assert str(err.value) == "Experiment with id test1 already exists!"


def test_list_experiments(create_project):
    p = create_project
    p.add_experiment(
        experiment_id="test1",
        panel_definition=f"{assets.__path__._path[0]}/test_panel.xlsx",
    )
    assert list(p.list_experiments()) == ["test1"]


def test_load_experiment(create_project):
    p = create_project
    p.add_experiment(
        experiment_id="test1",
        panel_definition=f"{assets.__path__._path[0]}/test_panel.xlsx",
    )
    e = p.get_experiment(experiment_id="test1")
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
