from immunova.data.fcs_experiments import NormalisedName, Panel, FCSExperiment
from immunova.data.fcs import File, FileGroup, ChannelMap
from immunova.data.mongo_setup import test_init
import pandas as pd
import numpy as np
import unittest
test_init()


class TestFile(unittest.TestCase):
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

    def test_normalised_name(self):
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

    def test_check_excel(self):
        panel = Panel()
        panel.panel_name = 'test'
        self.assertIsNotNone(panel.check_excel_template('test_data/panels/valid.xlsx'))
        self.assertIsNone(panel.check_excel_template('test_data/panels/invalid_headings1.xlsx'))
        self.assertIsNone(panel.check_excel_template('test_data/panels/invalid_headings2.xlsx'))
        self.assertIsNone(panel.check_excel_template('test_data/panels/invalid_missing.xlsx'))
        self.assertIsNone(panel.check_excel_template('test_data/panels/invalid_sheet.xlsx'))
        self.assertIsNone(panel.check_excel_template('test_data/panels/invalid_dup.xlsx'))

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
    def test_create_and_add(self):
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
        # Test case 1
        hc_root = 'test_data/fcs/hc1/day1/t1/'
        hc_path = f'{hc_root}healthy control 001 SD_1 LT1 whole panel_001.fcs'
        hc_controls = [f'{hc_root}{x}' for x in ['healthy control 001 SD_LT1-2 FMO CD57_002.fcs',
                                                 'healthy control 001 SD_LT1-3 FMO CCR7_003.fcs',
                                                 'healthy control 001 SD_LT1-4 FMO CD45RA_004.fcs',
                                                 'healthy control 001 SD_LT1-5 FMO CD27_005.fcs']]
        test_exp1.add_new_sample('hc1', hc_path, controls=hc_controls, feedback=False)
        # # test file created and validate file
        self.assertEqual(len(test_exp1.fcs_files), 1)
        f = test_exp1.fcs_files[0]
        self.assertEqual(f.primary_id, 'hc1')
        self.assertEqual(len(f.files), 5)



    def test_add_samples(self):
        pass

