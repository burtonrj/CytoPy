from mongoengine.connection import connect, disconnect
import numpy as np
import pytest
import h5py
import sys
import os


@pytest.fixture(scope='session', autouse=True)
def test_setup():
    sys.path.append("/home/rossco/CytoPy")
    os.mkdir(f"{os.getcwd()}/test_data")
    connect("test", host="mongomock://localhost", alias="core")
    yield
    os.rmdir(f"{os.getcwd()}/test_data")
    disconnect(alias="core")

