import sys
sys.path.append('/home/ross/CytoPy')

from CytoPy.data.panel import NormalisedName, Panel, create_regex
from CytoPy.data.fcs import ChannelMap
from CytoPy.data.mongo_setup import global_init
from mongoengine import connect
import unittest

db = connect('test')
db.drop_database('test')
global_init('test')


class TestPanel(unittest.TestCase):
    def test_CreateRegex(self):
        self.assertEqual('<*\s*PE[\s.-]+Cy7\s*>*', create_regex('PE-Cy7', initials=False))
        self.assertEqual('<*\s*A(lexa)*[\s.-]+F(luor)*[\s.-]+488[\s.-]+A\s*>*', create_regex('Alexa-Fluor-488-A'))

    def test_ChannelMap(self):
        test = ChannelMap(channel='test_channel', marker='test_marker')
        self.assertEqual(test.check_matched_pair('test_channel', 'test_marker'), True)
        self.assertEqual(test.check_matched_pair('test_channel', 'dummy'), False)
        self.assertEqual(test.check_matched_pair('dummy', 'test_marker'), False)
        self.assertEqual(test.check_matched_pair('dummy', 'dummy'), False)
        self.assertEqual(type(test.to_dict()), dict)
        self.assertTrue(test.to_dict().get('marker') == 'test_marker')
        self.assertTrue(test.to_dict().get('channel') == 'test_channel')

    def test_NormalisedName(self):
        test = NormalisedName(standard='testing',
                              regex_str='<*\s*test[\s.-]+[0-9]+\s*>*',
                              permutations='test,testing,allTheTESTS',
                              case_sensitive=False)
        self.assertEqual(test.query('test'), 'testing')
        self.assertEqual(test.query('testing'), 'testing')
        self.assertEqual(test.query('allTheTESTS'), 'testing')
        self.assertEqual(test.query('< test-55 >'), 'testing')
        self.assertEqual(test.query('<----NOTVALID---->'), None)

    def test_Panel(self):
        test = Panel(panel_name='test')
        test.create_from_excel('../data/test_panel.xlsx')
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
        correct_mappings = [[k, v] for k, v in correct_mappings.items()]
        invalid_mappings = {'FS Lin': '<--------->',
                            'SS Log': '<SS-Log>',
                            'FL1 Log': '<--------->',
                            'FL2 Log': '<IgG1-PE>',
                            'FL3 Log': '<CD45-ECD>',
                            'FL4 Log': '<----------->',
                            'FL5 Log': '<IgG1-PC7>'}
        invalid_mappings = [[k, v] for k, v in invalid_mappings.items()]
        mappings, err = test.standardise_names(correct_mappings)
        self.assertFalse(err)
        mappings, err = test.standardise_names(invalid_mappings)
        self.assertTrue(err)
        test.save()


if __name__ == '__main__':
    unittest.main()