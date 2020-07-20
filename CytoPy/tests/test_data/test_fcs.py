import sys
sys.path.append('/home/ross/CytoPy')

from CytoPy.data.project import Project
from CytoPy.data.mongo_setup import global_init
from CytoPy.tests import setup_with_dummy_data
from mongoengine import connect
import unittest

db = connect('test')
db.drop_database('test')
global_init('test')


class TestFCS(unittest.TestCase):
    def test_FileGroup(self):
        test_project = Project.objects(project_id='test').get()
        test_exp = test_project.load_experiment('test_experiment_dummy')
        # Testing data retrieval
        test_grp = test_exp.get_sample('dummy_test')
        test_file = test_grp.files[0]
        self.assertEqual(test_file.get().geom, (100, 3))
        self.assertEqual(test_file.get(sample=10).geom, (10, 3))
        data = test_exp.get_sample_data('dummy_test', include_controls=True)
        primary = [d for d in data if d.get('typ') == 'complete'][0]
        ctrl = [d for d in data if d.get('typ') == 'control'][0]
        self.assertEqual(primary.get('data').geom, (100, 3))
        self.assertEqual(ctrl.get('data').geom, (100, 3))
        self.assertListEqual(ctrl.get('data').columns.tolist(), ['feature0', 'feature1', 'blobID'])
        data = test_exp.get_sample_data('dummy_test', include_controls=True, columns_default='channel')
        primary = [d for d in data if d.get('typ') == 'complete'][0]
        self.assertListEqual(primary.get('data').columns.tolist(), ['var0', 'var1', 'var2'])


if __name__ == '__main__':
    setup_with_dummy_data()
    unittest.main()
