import sys
sys.path.append('/home/ross/CytoPy')
# Data imports
from cytopy.data.mongo_setup import global_init
from cytopy.flow.gating.defaults import Geom, ChildPopulation, ChildPopulationCollection, _validate_input
import numpy as np
import unittest

global_init('test')


class TestGeom(unittest.TestCase):
    def test_shape(self):
        err = False
        try:
            test = Geom(shape='NONSENSE', x='x', y='y')
        except AssertionError:
            err = True
        self.assertTrue(err)

    def test_keys(self):
        test = Geom(shape='ellipse', x='x', y='y', other='test')
        self.assertEqual(test.get('other'), 'test')

    def test_as_dict(self):
        test = Geom(shape='ellipse', x='x', y='y', other='test')
        self.assertDictEqual(test.as_dict(), dict(shape='ellipse', x='x', y='y', other='test'))


class TestValidateInput(unittest.TestCase):
    def _is_error(self, i, expected_outcome):
        err = False
        try:
            _validate_input(**i)
        except AssertionError:
            err = True
        self.assertEqual(err, expected_outcome)

    def test1(self):
        self._is_error(dict(gate_type='threshold_1d'), expected_outcome=True)

    def test2(self):
        self._is_error(dict(gate_type='threshold_2d', name='test'), expected_outcome=True)

    def test3(self):
        self._is_error(dict(gate_type='threshold_1d', definition='+'), expected_outcome=True)

    def test4(self):
        self._is_error(dict(gate_type='geom', definition='NONSENSE', name='test'), expected_outcome=True)

    def test5(self):
        self._is_error(dict(gate_type='threshold_1d', definition='NONSENSE', name='test'), expected_outcome=True)

    def test6(self):
        self._is_error(dict(gate_type='threshold_1d', definition=111, name='test'), expected_outcome=True)

    def test7(self):
        self._is_error(dict(gate_type='threshold_2d'), expected_outcome=True)

    def test8(self):
        self._is_error(dict(gate_type='threshold_2d', definition=[], name='test'), expected_outcome=True)

    def test9(self):
        self._is_error(dict(gate_type='threshold_2d', definition=['+'], name='test'), expected_outcome=True)

    def test10(self):
        self._is_error(dict(gate_type='threshold_2d', definition='+', name='test'), expected_outcome=True)

    def test11(self):
        self._is_error(dict(gate_type='cluster', target=[0, 1]), expected_outcome = True)

    def test12(self):
        self._is_error(dict(gate_type='cluster', target=[0, 1], weight='NONSENSE', name='test'),
                       expected_outcome = True)

    def test13(self):
        self._is_error(dict(gate_type='cluster', target=['x', 1], weight='NONSENSE', name='test'),
                       expected_outcome = True)

    def test14(self):
        self._is_error(dict(gate_type='cluster', target=[0, 1, 2], weight='NONSENSE', name='test'),
                       expected_outcome = True)

    def test15(self):
        self._is_error(dict(gate_type='threshold_1d', definition='+', name='test'), expected_outcome = False)

    def test16(self):
        self._is_error(dict(gate_type='geom', definition='+', name='test'), expected_outcome = False)

    def test17(self):
        self._is_error(dict(gate_type='threshold_2d', definition='++', name='test'), expected_outcome = False)

    def test18(self):
        self._is_error(dict(gate_type='threshold_2d', definition=['++', '+-'], name='test'), expected_outcome = False)

    def test19(self):
        self._is_error(dict(gate_type='cluster', weight=2, name='test', target=[0, 1]), expected_outcome = False)

    def test20(self):
        self._is_error(dict(gate_type='cluster', weight=2, name='test', target=[0.5, 1.5]), expected_outcome = False)


class TestChildPopulation(unittest.TestCase):
    def test_create(self):
        test = ChildPopulation(gate_type='geom', definition='+', name='test')
        self.assertEqual(type(test), ChildPopulation)
        self.assertDictEqual(test.properties, dict(name='test', definition='+'))

    def test_update_index(self):
        test = ChildPopulation(gate_type='geom', definition='+', name='test')
        test.update_index(np.array([0, 1, 2, 3]))
        self.assertListEqual([0, 1, 2, 3], list(test.index))
        test.update_index(np.array([2, 3, 4, 5]), merge_options='overwrite')
        self.assertListEqual([2, 3, 4, 5], list(test.index))
        test.update_index(np.array([4, 5, 6, 7]), merge_options='merge')
        self.assertListEqual([2, 3, 4, 5, 6, 7], list(test.index))


class TestChildPopulationCollectiecitonon(unittest.TestCase):
    def test_create(self):
        test = ChildPopulationCollection(gate_type='threshold_1d')
        self.assertEqual(test.gate_type, 'threshold_1d')

    def test_create_from_dict(self):
        test = dict(gate_type='threshold_1d', populations=[dict(name='pos', definition='+'),
                                                           dict(name='neg', definition='-')])
        test = ChildPopulationCollection(json_dict=test)
        self.assertListEqual(['pos', 'neg'], list(test.populations.keys()))
        self.assertEqual(test.populations.get('pos').properties.get('definition'), '+')
        self.assertEqual(test.populations.get('neg').properties.get('definition'), '-')

    def test_serialise(self):
        test = ChildPopulationCollection(gate_type='threshold_1d')
        test.add_population('pos', definition='+')
        test.add_population('neg', definition='-')
        test = test.serialise()
        self.assertDictEqual(test, dict(gate_type='threshold_1d',
                                        populations=[dict(name='pos', definition='+'),
                                                     dict(name='neg', definition='-')]))

    def test_add_population(self):
        test = ChildPopulationCollection(gate_type='threshold_1d')
        test.add_population('pos', definition='+')
        test.add_population('neg', definition='-')
        self.assertListEqual(['pos', 'neg'], list(test.populations.keys()))
        self.assertEqual(test.populations.get('pos').properties.get('definition'), '+')
        self.assertEqual(test.populations.get('neg').properties.get('definition'), '-')

    def test_remove_population(self):
        test = ChildPopulationCollection(gate_type='threshold_1d')
        test.add_population('pos', definition='+')
        test.add_population('neg', definition='-')
        test.remove_population('pos')
        self.assertListEqual(['neg'], list(test.populations.keys()))

    def test_fetch_by_def(self):
        test2d = ChildPopulationCollection(gate_type='threshold_2d')
        test2d.add_population('pospos', definition='++')
        test2d.add_population('other', definition=['--', '-+', '+-'])
        self.assertEqual(test2d.fetch_by_definition('++'), 'pospos')
        self.assertEqual(test2d.fetch_by_definition('--'), 'other')


if __name__ == '__main__':
    unittest.main()

