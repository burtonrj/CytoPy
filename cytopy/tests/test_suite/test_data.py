from warnings import filterwarnings

filterwarnings('ignore')
from cytopy.data.project import Project
from cytopy.data.mongo_setup import global_init
from cytopy.data.panel import NormalisedName, Panel, create_regex
from cytopy.data.fcs import File, FileGroup, ChannelMap
from .utilities import make_example_date
import unittest
import sys

unittest.TestLoader.sortTestMethodsUsing = None
sys.path.append('/home/ross/CytoPy')
global_init('test')


class TestPanel(unittest.TestCase):
    def testCreateRegex(self):
        self.assertEqual('<*\s*PE[\s.-]+Cy7\s*>*', create_regex('PE-Cy7', initials=False))
        self.assertEqual('<*\s*A(lexa)*[\s.-]+F(luor)*[\s.-]+488[\s.-]+A\s*>*', create_regex('Alexa-Fluor-488-A'))

    def testChannelMap(self):
        test = ChannelMap(channel='test_channel', marker='test_marker')
        self.assertEqual(test.check_matched_pair('test_channel', 'test_marker'), True)
        self.assertEqual(test.check_matched_pair('test_channel', 'dummy'), False)
        self.assertEqual(test.check_matched_pair('dummy', 'test_marker'), False)
        self.assertEqual(test.check_matched_pair('dummy', 'dummy'), False)
        self.assertEqual(type(test.to_python()), dict)
        self.assertTrue(test.to_python().get('marker') == 'test_marker')
        self.assertTrue(test.to_python().get('channel') == 'test_channel')

    def testNormalisedName(self):
        test = NormalisedName(standard='testing',
                              regex_str='<*\s*test[\s.-]+[0-9]+\s*>*',
                              permutations=['test', 'testing', 'allTheTESTS'],
                              case_sensitive=False)
        self.assertEqual(test.query('test'), 'testing')
        self.assertEqual(test.query('testing'), 'testing')
        self.assertEqual(test.query('allTheTESTS'), 'testing')
        self.assertEqual(test.query('< test-55 >'), 'testing')
        self.assertEqual(test.query('<----NOTVALID---->'), None)

    def testPanel(self):
        test = Panel(panel_name='test')
        test.create_from_excel('test_data/test_panel.xlsx')
        channels = [nn.standard for nn in test.channels]
        markers = [nn.standard for nn in test.markers]
        self.assertTrue(
            all([c in channels for c in ['FS Lin', 'SS Log', 'FL1 Log', 'FL2 Log', 'FL3 Log', 'FL4 Log', 'FL5 Log']]))
        self.assertTrue(all(
            [m in markers for m in ['FS Lin', 'SS Log', 'IgG1-FITC', 'IgG1-PE', 'CD45-ECD', 'IgG1-PC5', 'IgG1-PC7']]))
        correct_mappings = {'FS Lin': '<FS-Lin>',
                            'SS Log': '<SS-Log>',
                            'FL1 Log': '<IgG1-FITC>',
                            'FL2 Log': '<IgG1-PE>',
                            'FL3 Log': '<CD45-ECD>',
                            'FL4 Log': '<IgG1-PC5>',
                            'FL5 Log': '<IgG1-PC7>'}
        correct_mappings = [{'channel': k, 'marker': v} for k, v in correct_mappings.items()]
        invalid_mappings = {'FS Lin': '<--------->',
                            'SS Log': '<SS-Log>',
                            'FL1 Log': '<--------->',
                            'FL2 Log': '<IgG1-PE>',
                            'FL3 Log': '<CD45-ECD>',
                            'FL4 Log': '<----------->',
                            'FL5 Log': '<IgG1-PC7>'}
        invalid_mappings = [{'channel': k, 'marker': v} for k, v in invalid_mappings.items()]
        mappings, err = test.standardise_names(correct_mappings)
        self.assertFalse(err)
        mappings, err = test.standardise_names(invalid_mappings)
        self.assertTrue(err)
        test.save()


class TestCreate(unittest.TestCase):
    """
    Test the creation of projects, experiments, and subjects, and populating
    them with data
    """

    def testProject(self):
        test_project = Project(project_id='test', owner='test')
        test_project.add_experiment('test_experiment_aml', panel_name='test')
        test_project.add_experiment('test_experiment_dummy', panel_name='test')
        test_project.add_subject('test_subject', testing=True)
        self.assertEqual(test_project.list_subjects(), ['test_subject'])
        s = test_project.pull_subject('test_subject')
        self.assertEqual(s.testing, True)
        self.assertEqual(s.subject_id, 'test_subject')
        self.assertEqual(test_project.list_fcs_experiments(), ['test_experiment_aml'])
        test_project.save()

    def testFCSExperiment(self):
        test_project = Project.objects(project_id='test').get()
        test_exp = test_project.load_experiment('test_experiment_aml')
        self.assertEqual(test_exp.experiment_id, 'test_experiment_aml')
        self.assertEqual(test_exp.panel.panel_name, 'test')
        test_exp.add_new_sample(sample_id='test_sample',
                                file_path='../test_data/test.FCS',
                                controls={'path': 'test_data/test.FCS', 'control_id': 'test_control'},
                                subject_id='test_subject')
        self.assertEqual(test_exp.list_samples(), ['test_sample'])
        self.assertTrue(test_exp.sample_exists('test_sample'))
        self.assertEqual(test_exp.pull_sample('test_sample').sample_id, 'test_sample')
        correct_mappings = {'FS Lin': 'FS-Lin',
                            'SS Log': 'SS-Log',
                            'FL1 Log': 'IgG1-FITC',
                            'FL2 Log': 'IgG1-PE',
                            'FL3 Log': 'CD45-ECD',
                            'FL4 Log': 'IgG1-PC5',
                            'FL5 Log': 'IgG1-PC7'}
        self.assertDictEqual(test_exp.pull_sample_mappings('test_sample'), correct_mappings)
        data = test_exp.pull_sample('test_sample', include_controls=True)
        primary = [d for d in data if d.get('typ') == 'complete'][0]
        ctrl = [d for d in data if d.get('typ') == 'control'][0]
        self.assertEqual(type(data), list)
        self.assertEqual(len(data), 2)
        self.assertEqual(type(primary), dict)
        self.assertEqual(type(ctrl), dict)
        self.assertEqual(primary.get('data').shape, (30000, 7))
        self.assertListEqual(primary.get('data').columns.tolist(),
                             ['FS Lin', 'SS Log', 'IgG1-FITC', 'IgG1-PE', 'CD45-ECD', 'IgG1-PC5', 'IgG1-PC7'])
        data = test_exp.pull_sample('test_sample', include_controls=False)
        self.assertEqual(len(data), 1)
        data = test_exp.pull_sample('test_sample', sample_size=5000, include_controls=False)
        self.assertEqual(data[0].get('data').shape, (5000, 7))

    def testFileGroup(self):
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
        test_file.put(example_data)
        test_ctrl.put(example_data)
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
