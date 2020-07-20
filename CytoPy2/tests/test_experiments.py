from CytoPy2.data.experiments import Experiment, Panel
from CytoPy2.data.subjects import Subject
from mongoengine import connect, disconnect
import unittest


class TextExperiment(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        connect('testing', host='mongomock://localhost')

    @classmethod
    def tearDownClass(cls):
        disconnect()

    def test_init(self):
        exp = Experiment(experiment_id="test1",
                         panel_definition="test_data/test_panel.xlsx",
                         panel_name="testing_panel",
                         data_directory="/media/ross/extdrive2/CytoPy_data/test")
        self.assertEqual(Panel.objects(panel_name="testing_panel").get(),
                         exp.panel)
        exp = Experiment(experiment_id="test2",
                         panel_name="testing_panel",
                         data_directory="/media/ross/extdrive2/CytoPy_data/test")
        self.assertEqual(Panel.objects(panel_name="testing_panel").get(),
                         exp.panel)

    def test_sample_creation(self):
        exp = Experiment(experiment_id="test1",
                         panel_definition="test_data/test_panel.xlsx",
                         panel_name="testing_panel",
                         data_directory="/media/ross/extdrive2/CytoPy_data/test")
        exp.add_new_sample(sample_id='test_sample',
                           file_path='../data/test.FCS',
                           controls=[{'path': '../data/test.FCS', 'control_id': 'test_control'}],
                           subject_id='test_subject',
                           compensate=False)
        self.assertEqual(exp.list_samples(), ['test_sample'])
        self.assertTrue(exp.sample_exists('test_sample'))
        self.assertEqual(exp.get_sample('test_sample').primary_id, 'test_sample')
        correct_mappings = {'FS Lin': 'FS Lin',
                            'SS Log': 'SS Log',
                            'FL1 Log': 'IgG1-FITC',
                            'FL2 Log': 'IgG1-PE',
                            'FL3 Log': 'CD45-ECD',
                            'FL4 Log': 'IgG1-PC5',
                            'FL5 Log': 'IgG1-PC7'}
        correct_mappings = [{'channel': k, 'marker': v} for k, v in correct_mappings.items()]
        self.assertListEqual(exp.pull_sample_mappings('test_sample').get('test_sample'), correct_mappings)
        data = exp.get_sample_data('test_sample', include_controls=True)
        primary = data.get("primary")
        ctrl = data.get("controls").get("test_control")
        self.assertEqual(primary.shape, (30000, 7))
        self.assertEqual(ctrl.shape, (30000, 7))
        self.assertListEqual(primary.columns.tolist(),
                             ['FS Lin', 'SS Log', 'IgG1-FITC', 'IgG1-PE', 'CD45-ECD', 'IgG1-PC5', 'IgG1-PC7'])
        data = exp.get_sample_data('test_sample', include_controls=False)
        self.assertIsNone(data.get("controls"))
        data = exp.get_sample_data('test_sample', sample_size=5000, include_controls=False)
        self.assertEqual(data.get('primary').geom, (5000, 7))


if __name__ == '__main__':
    unittest.main()
