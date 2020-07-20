import unittest
from mongoengine import connect, disconnect


class TestSettings(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        connect('testing', host='mongomock://localhost')

    @classmethod
    def tearDownClass(cls):
       disconnect()

    def test_init(self):
        try:
            settings = Settings(data_directory="/media/ross/extdrive/test",
                                user="Ross",
                                compression="GZIP")
            passed = True
        except ValueError as e:
            passed = False
        self.assertTrue(passed)
        try:
            settings = Settings(data_directory="/media/ross/extdrive/test/not_real",
                                user="Ross",
                                compression="GZIP")
            passed = False
        except ValueError as e:
            passed = True
        self.assertTrue(passed)
        try:
            settings = Settings(data_directory="/media/ross/extdrive/test",
                                user="Ross",
                                compression="INVALID")
            passed = False
        except ValueError as e:
            passed = True
        self.assertTrue(passed)



