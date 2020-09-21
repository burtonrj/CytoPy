from mongoengine.connection import disconnect, _get_db
from ..data.setup import global_init
import pytest


@pytest.fixture(scope='session', autouse=True)
def mongo_setup():
    global_init(database_name="test", alias="core")
    yield
    db = _get_db(alias="core")
    db.drop_database("test")
    disconnect(alias="core")


