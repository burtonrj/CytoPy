from mongoengine.connection import disconnect, _get_db
from ..data.setup import global_init
import pytest
import h5py
import sys
import os


@pytest.fixture(scope='session', autouse=True)
def test_setup():
    sys.path.append("/home/rossco/CytoPy")
    global_init(database_name="test", alias="core")
    f = h5py.File('test.hdf5', 'w')
    f.close()
    yield
    db = _get_db(alias="core")
    try:
        db.drop_database()
    except TypeError:
        pass
    os.remove("test.hdf5")
    disconnect(alias="core")


