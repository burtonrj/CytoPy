from immunova.data.fcs_experiments import NormalisedName, Panel, FCSExperiment
from immunova.data.utilities import filter_fcs_files, get_fcs_file_paths
from immunova.data.fcs import File, FileGroup, ChannelMap
from immunova.data.mongo_setup import test_init
from datetime import datetime
import pandas as pd
import numpy as np
import unittest
test_init()
log = open('performance_log.txt', 'a')
log.write('----------------------------------------------------------\n')
log.write(f"Testing data creation and access. {datetime.now().strftime('%Y/%m/%d %H:%M')}\n")
log.write('----------------------------------------------------------\n')


def measure_performance(name):
    def decorator(func):
        def wrapper(*args, **kwargs):
            start = datetime.now()
            func(*args, **kwargs)
            end = datetime.now()
            log.write(f"{name}: {(end - start).__str__()}")
        return wrapper
    return decorator


class TestFile(unittest.TestCase):
    @measure_performance('file.as_dataframe')
    def test_as_dataframe(self):
        test_data = np.random.rand(4, 4)
        mappings = [ChannelMap(channel='PE-Cy7', marker='CD4'),
                    ChannelMap(channel='PE-Cy5', marker='CD3'),
                    ChannelMap(channel='APC-H7', marker='CD15'),
                    ChannelMap(channel='7AAD', marker='CD11b')]
        test_file = File(file_id='test', channel_mappings=mappings)
        t1 = test_file.as_dataframe(test_data, columns_default='marker')
        t2 = test_file.as_dataframe(test_data, columns_default='marker')
        self.assertEqual(t1.columns.to_list(), ['CD4', 'CD3', 'CD15', 'CD11b'])
        self.assertEqual(t2.columns.to_list(), ['PE-Cy7', 'PE-Cy5', 'APC-H7', '7AAD'])


class TestPanel(unittest.TestCase):
    @measure_performance('NormalisedName.query')
    def test_normalised_name(self):
        start = datetime.now()
        t1 = NormalisedName(standard='correct', regex_str='Alexa\s*-*Fluor\s*-*488\s*-*A',
                            case_sensitive=False)
        t2 = NormalisedName(standard='correct', regex_str='Alexa\s*-*Fluor\s*-*488\s*-*A',
                            case_sensitive=True)
        t3 = NormalisedName(standard='correct', regex_str='Alexa\s*-*Fluor\s*-*488\s*-*A',
                            case_sensitive=True, permutations='Alexa-FLour-488A,AF488A')
        self.assertEqual(t1.query('Alexa-Fluor-488-A'), 'correct')
        self.assertEqual(t1.query('alexa-fluor-488-a'), 'correct')
        self.assertIsNone(t1.query('Pflour-488-A'))
        self.assertIsNone(t2.query('alexa-fluor-488-a'))
        self.assertEqual(t2.query('Alexa-Fluor-488-A'), 'correct')
        self.assertEqual(t3.query('Alexa-Fluor-488-A'), 'correct')
        self.assertEqual(t3.query('Alexa-FLour-488A'), 'correct')
        self.assertIsNone(t3.query('Alexa-Flour-488-A'))
        self.assertEqual(t3.query('AF488A'), 'correct')
        end = datetime.now()

    @measure_performance('Panel.check_excel_template')
    def test_check_excel(self):
        panel = Panel()
        panel.panel_name = 'test'
        self.assertIsNotNone(panel.check_excel_template('test_data/panels/valid.xlsx'))
        self.assertIsNone(panel.check_excel_template('test_data/panels/invalid_headings1.xlsx'))
        self.assertIsNone(panel.check_excel_template('test_data/panels/invalid_headings2.xlsx'))
        self.assertIsNone(panel.check_excel_template('test_data/panels/invalid_missing.xlsx'))
        self.assertIsNone(panel.check_excel_template('test_data/panels/invalid_sheet.xlsx'))
        self.assertIsNone(panel.check_excel_template('test_data/panels/invalid_dup.xlsx'))

    @measure_performance('Create a panel')
    def test_create(self):
        def test_cases(t, p):
            c1 = p.channels[0]
            t.assertEqual(c1.standard, 'FSC-A')
            t.assertEqual(c1.regex, 'FSC\s*-*A$')
            t.assertEqual(c1.case_sensitive, False)
            t.assertEqual(c1.permutations, '')
            c2 = p.channels[1]
            t.assertEqual(c2.standard, 'Alexa Fluor 488-A')
            t.assertEqual(c2.regex, 'Alexa\s*-*Fluor\s*-*488\s*-*A$')
            t.assertEqual(c2.case_sensitive, True)
            t.assertEqual(c2.permutations, 'AF 488-A, AF488A')
            m = p.markers[0]
            t.assertEqual(m.standard, 'CD57')
            t.assertEqual(m.regex, 'CD\s*-*57$')
            t.assertEqual(m.case_sensitive, False)
            t.assertEqual(m.permutations, '')
            mp1 = p.mappings[0]
            mp2 = p.mappings[1]
            t.assertEqual(mp1.channel, 'FSC-A')
            t.assertEqual(mp1.marker, None)
            t.assertEqual(mp2.channel, 'Alexa Fluor 488-A')
            t.assertEqual(mp2.marker, 'CD57')

        panel = Panel()
        panel.panel_name = 'test'
        x = panel.create_from_excel('test_data/panels/valid.xlsx')
        self.assertEqual(x, True)
        test_cases(self, panel)

        pd = dict(channels=[dict(standard='Alexa Fluor 488-A',
                                 regex='Alexa\s*-*Fluor\s*-*488\s*-*A$',
                                 case=1,
                                 permutations='AF 488-A, AF488A'),
                            dict(standard='FSC-A',
                                 regex='FSC\s*-*A$',
                                 case=0,
                                 permutations='')],
                  markers=[dict(standard='CD57',
                                regex='CD\s*-*57$',
                                case=0,
                                permutations='')],
                  mappings=[('FSC-A', None), ('Alexa Fluor 488-A', 'CD57')])
        panel = Panel()
        panel.panel_name = 'test'
        x = panel.create_from_dict(pd)
        self.assertEqual(x, True)
        test_cases(self, panel)


class TestCreateExperiment(unittest.TestCase):
    @measure_performance('Create experiment')
    def test_create(self):
        # Create exp1
        test_exp1 = FCSExperiment()
        test_exp1.experiment_id = 'test_exp1'
        test_panel = Panel()
        test_panel.panel_name = 'test_panel1'
        test_panel.create_from_excel('test_data/panels/valid.xlsx')
        test_panel.save()
        test_exp1.panel = test_panel
        test_exp1.save()
        # Create exp2
        test_exp2 = FCSExperiment()
        test_exp2.experiment_id = 'test_exp2'
        test_panel = Panel()
        test_panel.panel_name = 'test_panel2'
        test_panel.create_from_excel('test_data/panels/test.xlsx')
        test_panel.save()
        test_exp2.panel = test_panel
        test_exp2.save()
        self.assertIsNotNone(FCSExperiment.objects(experiment_id='test_exp1'))
        self.assertIsNotNone(FCSExperiment.objects(experiment_id='test_exp2'))

    @measure_performance('Add samples to experiment')
    def test_add(self):
        test_exp1 = FCSExperiment.objects(experiment_id='test_exp1').get()
        test_exp2 = FCSExperiment.objects(experiment_id='test_exp1').get()
        # Test case 1
        hc_root = 'test_data/fcs/hc1/day1/t1/'
        hc_path = f'{hc_root}healthy control 001 SD_1 LT1 whole panel_001.fcs'
        hc_controls = [f'{hc_root}{x}' for x in ['healthy control 001 SD_LT1-2 FMO CD57_002.fcs',
                                                 'healthy control 001 SD_LT1-3 FMO CCR7_003.fcs',
                                                 'healthy control 001 SD_LT1-4 FMO CD45RA_004.fcs',
                                                 'healthy control 001 SD_LT1-5 FMO CD27_005.fcs']]
        test_exp1.add_new_sample('hc1', hc_path, controls=hc_controls, feedback=False)
        test_exp2.add_new_sample('hc1', hc_path, controls=hc_controls, feedback=False,
                                 catch_standardisation_errors=True)
        self.assertEqual(len(test_exp1.fcs_files), 1)
        self.assertEqual(len(test_exp2.fcs_files), 0)
        f = test_exp1.fcs_files[0]
        self.assertEqual(f.primary_id, 'hc1')
        self.assertEqual(len(f.files), 5)
        test_exp2.add_new_sample('hc1', hc_path, controls=hc_controls, feedback=False,
                                 catch_standardisation_errors=False)

    @measure_performance('Pull data from experiment')
    def test_fetch(self):
        test_exp1 = FCSExperiment.objects(experiment_id='test_exp1').get()
        mappings = test_exp1.pull_sample_mappings('hc1')
        correct_mappings = pd.read_excel('test_data/panels/valid.xlsx',
                                         sheet_name='mapping')
        correct_channels = correct_mappings.channel.values
        correct_markers = correct_mappings.marker.values
        for i, x in enumerate(mappings):
            self.assertEqual(x.channel, correct_channels[i])
            self.assertEqual(x.marker, correct_markers[i])
        data = test_exp1.pull_sample_data('hc1')
        self.assertIsInstance(data, list)
        for x in data:
            self.assertIsInstance(x, dict)
            self.assertIsInstance(x['data'], pd.DataFrame)
            self.assertEqual(x['data'].columns.to_list()[0:5], correct_channels[0:5])
            self.assertEqual(x['data'].columns.to_list()[5:], correct_markers[5:])

    @measure_performance('Remove samples from experiment')
    def test_remove(self):
        for x in ['test_exp1', 'test_exp2']:
            t = FCSExperiment.objects(experiment_id='test_exp1').get()
            f_id = t.pull_sample('hc1').id.__str__()
            x = t.remove_sample('hc1')
            self.assertEqual(x, True)
            self.assertIsNone(t.pull_sample('hc1'))
            self.assertIsNone(FileGroup.objects(id=f_id))


class TestUtilities(unittest.TestCase):
    def filter_fcs_files(self):
        search_result1 = filter_fcs_files('test_data/search_test')
        expected = {'test.fcs', 'test2.fcs', 'test3.fcs',
                    'test3_FMO_BA.fcs', 'test3_FMO_BB.fcs',
                    'test3_FMO_BC.fcs'}
        self.assertEqual(set(search_result1), expected)
        search_result2 = filter_fcs_files('test_data/search_test', exclude_comps=False)
        expected = {'test.fcs', 'test2.fcs', 'test3.fcs',
                    'test3_FMO_BA.fcs', 'test3_FMO_BB.fcs',
                    'test3_FMO_BC.fcs', 'test3_Comp.fcs',
                    'test_comp.fcs'}
        self.assertEqual(set(search_result2), expected)
        search_result3 = filter_fcs_files('test_data/search_test/x')
        expected = {'test2.fcs', 'test_comp.fcs'}
        self.assertEqual(set(search_result3), expected)
        search_result4 = filter_fcs_files('test_data/search_test/x/z')
        expected = set()
        self.assertEqual(set(search_result4), expected)




    def get_fcs_file_paths(self):
        search_results1 = get_fcs_file_paths('test_data/search_test/y',
                                             control_names=['BA', 'BB', 'BC'],
                                             ctrl_id='FMO')
        search_results2 = get_fcs_file_paths('test_data/search_test/y',
                                             control_names=['BA', 'BB', 'BC'],
                                             ctrl_id='FMO', ignore_comp=False)
        search_results3 = get_fcs_file_paths('test_data/search_test/y',
                                             control_names=['BA', 'BB'],
                                             ctrl_id='FMO', ignore_comp=True)