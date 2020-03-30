import sys
sys.path.append('/home/rossco/CytoPy')
# Data imports
from cytopy.data.mongo_setup import global_init
from cytopy.flow.gating.defaults import Geom, ChildPopulation, ChildPopulationCollection, ChildConstructError
import unittest

global_init('test')


class TestGeom(unittest.TestCase):
    def test_shape(self):
        err = False
        try:
            test = Geom(shape='NONSENSE')
        except AssertionError:
            err = True
        self.assertTrue(err)

    def test_keys(self):
        test = Geom(shape='ellipse', x='x', y='y', other='test')
        self.assertEqual(test.get('other'), 'test')

    def test_as_dict(self):
        test = Geom(shape='ellipse', x='x', y='y', other='test')
        self.assertDictEqual(test.as_dict(), dict(shape='ellipse', x='x', y='y', other='test'))


class TestChildPopulation(unittest.TestCase):
    def test_create(self):
        invalid1 = dict(gate_type='threshold_1d')
        invalid2 = dict(gate_type='threshold_1d', definition='+')
        invalid3 = dict(gate_type='geom', definition='NONSENSE', name='test')
        invalid4 = dict(gate_type='threshold_2d')
        invalid5 = dict(gate_type='threshold_2d', definition=[], )

        valid1 = dict(gate_type='geom', definition='+', name='test')
        valid2 = dict(gate_type='threshold_1d', definition='+', name='test')


class TestChildPopulationColleciton(unittest.TestCase):

    def testChildPopulationCollection(self):
        test1d = ChildPopulationCollection(gate_type='threshold_1d')
        test1d.add_population('positive', definition='+')
        test1d.add_population('negative', definition='-')
        self.assertEqual(test1d.fetch_by_definition('+'), 'positive')
        self.assertEqual(test1d.fetch_by_definition('++'), None)
        test1d.remove_population('positive')
        self.assertListEqual(list(test1d.populations.keys()), ['negative'])

        test2d = ChildPopulationCollection(gate_type='threshold_2d')
        test2d.add_population('pospos', definition='++')
        test2d.add_population('other', definition=['--', '-+', '+-'])
        self.assertEqual(test1d.fetch_by_definition('+'), None)
        self.assertEqual(test1d.fetch_by_definition('++'), 'pospos')
        self.assertEqual(test1d.fetch_by_definition('--'), 'other')