from ...data.project import Project
import pytest


@pytest.fixture()
def create_project():
    p = Project(project_id="test")
    p.save()
    yield p
    p.delete()


def test_add_experiment():
    pass


def test_list_experiments():
    pass


def test_load_experiment():
    pass


def test_add_subject():
    pass


def test_list_subjects():
    pass


def test_get_subject():
    pass


def test_delete_subject():
    pass