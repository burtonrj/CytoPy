import sys
sys.path.append('/home/ross/CytoPy')

from data.project import Project
from data.mongo_setup import global_init
from tests.utilities import basic_setup
from mongoengine import connect
import unittest

db = connect('test')
db.drop_database('test')
global_init('test')


class TextFCSExperiment(unittest.TestCase):

    def test_FCSExperiment(self):
        test_project = Project.objects(project_id='test').get()
        test_exp = test_project.load_experiment('test_experiment_aml')
        self.assertEqual(test_exp.experiment_id, 'test_experiment_aml')
        self.assertEqual(test_exp.panel.panel_name, 'test')
        test_exp.add_new_sample(sample_id='test_sample',
                                file_path='../data/test.FCS',
                                controls=[{'path': '../data/test.FCS', 'control_id': 'test_control'}],
                                subject_id='test_subject',
                                compensate=False)
        self.assertEqual(test_exp.list_samples(), ['test_sample'])
        self.assertTrue(test_exp.sample_exists('test_sample'))
        self.assertEqual(test_exp.pull_sample('test_sample').primary_id, 'test_sample')
        correct_mappings = {'FS Lin': 'FS Lin',
                            'SS Log': 'SS Log',
                            'FL1 Log': 'IgG1-FITC',
                            'FL2 Log': 'IgG1-PE',
                            'FL3 Log': 'CD45-ECD',
                            'FL4 Log': 'IgG1-PC5',
                            'FL5 Log': 'IgG1-PC7'}
        correct_mappings = [{'channel': k, 'marker': v} for k, v in correct_mappings.items()]
        self.assertListEqual(test_exp.pull_sample_mappings('test_sample').get('test_sample'), correct_mappings)
        data = test_exp.pull_sample_data('test_sample', include_controls=True)
        primary = [d for d in data if d.get('typ') == 'complete'][0]
        ctrl = [d for d in data if d.get('typ') == 'control'][0]
        self.assertEqual(type(data), list)
        self.assertEqual(len(data), 2)
        self.assertEqual(type(primary), dict)
        self.assertEqual(type(ctrl), dict)
        self.assertEqual(primary.get('data').shape, (30000, 7))
        self.assertListEqual(primary.get('data').columns.tolist(),
                             ['FS Lin', 'SS Log', 'IgG1-FITC', 'IgG1-PE', 'CD45-ECD', 'IgG1-PC5', 'IgG1-PC7'])
        data = test_exp.pull_sample_data('test_sample', include_controls=False)
        self.assertEqual(len(data), 1)
        data = test_exp.pull_sample_data('test_sample', sample_size=5000, include_controls=False)
        self.assertEqual(data[0].get('data').shape, (5000, 7))


if __name__ == '__main__':
    basic_setup()
    unittest.main()
