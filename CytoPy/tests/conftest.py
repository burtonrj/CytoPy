from mongoengine.connection import connect, disconnect, _get_db
import pytest
import shutil
import sys
import os


@pytest.fixture(scope='session', autouse=True)
def setup():
    sys.path.append("/home/ross/CytoPy")
    os.mkdir(f"{os.getcwd()}/test_data")
    connect("test", host="mongomock://localhost", alias="core")
    yield
    shutil.rmtree(f"{os.getcwd()}/test_data", ignore_errors=True)
    disconnect(alias="core")


