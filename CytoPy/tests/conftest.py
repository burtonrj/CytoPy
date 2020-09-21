from mongoengine.connection import disconnect, _get_db
from ..data.setup import global_init
import pytest
import sys


@pytest.fixture(scope='session', autouse=True)
def mongo_setup():
    sys.path.append("/home/rossco/CytoPy")
    global_init(database_name="test", alias="core")
    yield
    db = _get_db(alias="core")
    try:
        db.drop_database()
    except TypeError:
        pass
    disconnect(alias="core")


