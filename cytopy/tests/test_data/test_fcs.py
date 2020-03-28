import sys
sys.path.append('/home/ross/CytoPy')

from cytopy.data.project import Project
from cytopy.data.mongo_setup import global_init
from cytopy.data.fcs import File, FileGroup, ChannelMap
from cytopy.tests.utilities import make_example_date, basic_setup
from mongoengine import connect
import unittest

db = connect('test')
db.drop_database('test')
global_init('test')


class TestFCS(unittest.TestCase):
    def test_FileGroup(self):
        # Create example data
        example_data = make_example_date(n_samples=100, centers=3, n_features=2)
        # Create dummy channel mappings
        mappings = [ChannelMap(channel='var0', marker='feature0'),
                    ChannelMap(channel='var1', marker='feature1'),
                    ChannelMap(channel='var2', marker='blobID')]
        # Populate data
        test_project = Project.objects(project_id='test').get()
        test_exp = test_project.load_experiment('test_experiment_dummy')
        test_grp = FileGroup(primary_id='dummy_test',
                             flags='dummy')
        test_file = File(file_id='dummy_file', channel_mappings=mappings)
        test_ctrl = File(file_id='dummy_ctrl', channel_mappings=mappings, file_type='control')
        test_file.put(example_data.values)
        test_ctrl.put(example_data.values)
        test_grp.files = [test_file, test_ctrl]
        test_grp.save()
        test_exp.fcs_files.append(test_grp)
        test_exp.save()

        # Testing data retrieval
        test_grp = test_exp.pull_sample('dummy_test')
        test_file = test_grp.files[0]
        self.assertEqual(test_file.pull().shape, (100, 3))
        self.assertEqual(test_file.pull(sample=10).shape, (10, 3))
        data = test_exp.pull_sample_data('dummy_test', include_controls=True)
        primary = [d for d in data if d.get('typ') == 'complete'][0]
        ctrl = [d for d in data if d.get('typ') == 'control'][0]
        self.assertEqual(primary.get('data').shape, (100, 3))
        self.assertEqual(ctrl.get('data').shape, (100, 3))
        self.assertListEqual(ctrl.get('data').columns.tolist(), ['feature0', 'feature1', 'blobID'])
        data = test_exp.pull_sample_data('dummy_test', include_controls=True, columns_default='channel')
        primary = [d for d in data if d.get('typ') == 'complete'][0]
        self.assertListEqual(primary.get('data').columns.tolist(), ['var0', 'var1', 'var2'])


if __name__ == '__main__':
    basic_setup()
    unittest.main()
