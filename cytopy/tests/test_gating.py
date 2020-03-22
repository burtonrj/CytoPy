from warnings import filterwarnings
filterwarnings('ignore')
# Data imports
from cytopy.data.project import Project
from cytopy.data.mongo_setup import global_init
# Gating imports
from ..flow.gating.actions import Gating, ChildPopulationCollection
from ..flow.gating.base import Gate
from ..flow.gating import utilities
# Other tools
from .utilities import make_example_date
from sklearn.neighbors import KernelDensity
from scipy.signal import find_peaks
import numpy as np
import pandas as pd
import unittest
import sys

unittest.TestLoader.sortTestMethodsUsing = None
sys.path.append('/home/rossc/CytoPy')
global_init('test')


class TestUtilities(unittest.TestCase):
    def test_check_peak(self):
        probs = np.array([0, 0, 0, 0.01, 0, 0, 2, 0, 0, 3, 0, 0, 0.01])
        peaks = np.where(np.array(probs) > 0)
        self.assertEqual(len(utilities.check_peak(peaks, probs, t=0.05)), 2)
        self.assertEqual(len(utilities.check_peak(peaks, probs, t=0.5)), 4)

    def test_find_local_minima(self):
        data = make_example_date()
        data = pd.concat([data[data.blobID != 2],
                          data[data.blobID == 2].sample(frac=0.25)])
        d = data['feature0'].values
        density = KernelDensity(bandwidth=0.5, kernel='gaussian')
        density.fit(d[:, None])
        x_d = np.linspace(min(d), max(d), 1000)
        prob = np.exp(density.score_samples(x_d[:, None]))
        peaks = find_peaks(prob)[0]
        self.assertAlmostEqual(utilities.find_local_minima(prob, x_d, peaks),
                               0.592, places=2)

    def test_inside_ellipse(self):
        data = make_example_date()
        mask = utilities.inside_ellipse(data[['feature0', 'feature1']].values,
                                        center=(4.5, 2.5),
                                        width=2.3,
                                        height=3,
                                        angle=0)
        correct = all(x == 1 for x in data.loc[mask].blobID.values)
        self.assertTrue(correct)

    def test_rectangular_filter(self):
        data = make_example_date()
        rect = dict(xmin=0, xmax=8, ymin=-2.5, ymax=6.0)
        self.assertTrue(all(x == 1 for x in utilities.rectangular_filter(data,
                                                                         x='feature0',
                                                                         y='feature1',
                                                                         definition=rect).blobID.values))

    def test_dds(self):
        def equal_ratio(d):
            from itertools import combinations
            ratios = [d[d.blobID == x[0]].shape[0] / d[d.blobID == x[1]].shape[0]
                      for x in combinations(samples.blobID.unique(), 2)]
            return combinations(ratios, 2)

        data = make_example_date(n_samples=10000)
        samples = utilities.density_dependent_downsample(data=data,
                                                         features=['feature0', 'feature1'],
                                                         mmd_sample_n=2000)
        for x, y in equal_ratio(samples):
            self.assertAlmostEqual(x, y, places=1)

    def test_get_params(self):
        class MakeshiftClass:
            def __init__(self, a, b, c, d='test', **kwargs):
                pass
        self.assertListEqual(utilities.get_params(MakeshiftClass),
                             ['a', 'b', 'c', 'd', 'kwargs'])
        self.assertListEqual(utilities.get_params(MakeshiftClass, required_only=True),
                             ['a', 'b', 'c', 'kwargs'])
        self.assertListEqual(utilities.get_params(MakeshiftClass,
                                                  required_only=True,
                                                  exclude_kwargs=True),
                             ['a', 'b', 'c'])

class TestGating(unittest.TestCase):

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

    def testGate(self):
        def _build(dimensions=1):
            example_data = make_example_date()
            if dimensions == 1:
                populations = ChildPopulationCollection()
                populations.add_population('positive', definition='+')
                populations.add_population('negative', definition='-')
                return (Gate(data=example_data,
                             x='feature0',
                             y='feature1',
                             child_populations=populations,
                             transform_x=None,
                             transform_y=None), None), example_data
            populations1 = ChildPopulationCollection()
            populations1.add_population('positive', definition='++')
            populations1.add_population('negative', definition=['--', '-+', '+-'])

            populations2 = ChildPopulationCollection()
            populations2.add_population('positive', definition=['++', '-+'])
            populations2.add_population('negative', definition=['--', '+-'])
            return (Gate(data=example_data,
                         x='feature0',
                         y='feature1',
                         child_populations=p,
                         transform_x=None,
                         transform_y=None) for p in [populations1, populations2]), example_data

        def _test(g, pos):
            neg = [i for i in data.index.values if i not in pos]
            self.assertListEqual(list(g.child_populations.populations['positive'].index), pos)
            self.assertListEqual(list(g.child_populations.populations['negative'].index), neg)

        # 1-D gate
        gate, _, data = _build()
        pos_idx = [1,  3,  5,  7,  8, 26, 31, 32, 38, 39, 42, 45, 49, 50, 51, 58, 61,
                   63, 66, 68, 69, 70, 74, 76, 78, 79, 81, 87, 89, 90, 91, 93, 97]
        gate.child_update_1d(threshold=0.3, method='test', merge_options='overwrite')
        _test(gate, pos_idx)

        gate.child_update_1d(threshold=4, method='test', merge_options='overwrite')
        pos_idx = [1,  3,  7,  8, 31, 32, 38, 39, 42, 51, 58, 66, 68, 69, 70, 74, 76,
                   78, 89, 90, 91, 93, 97]
        gate.child_update_1d(threshold=4, method='test', merge_options='overwrite')
        _test(gate, pos_idx)

        gate.child_update_1d(threshold=9, method='test', merge_options='merge')
        _test(gate, pos_idx)

        # 2-D gate
        gate1, gate2, data = _build(dimensions=2)
        pos_idx = [1,  3,  5,  7,  8, 26, 31, 32, 38, 39, 42, 45, 49, 50, 51, 58, 61,
                   63, 66, 68, 69, 70, 74, 76, 78, 79, 81, 87, 89, 90, 91, 93, 97]
        gate1.child_update_2d(x_threshold=2, y_threshold=-2.5, method='test')
        _test(gate1, pos_idx)

        pos_idx = [1,  2,  3,  5,  6,  7,  8,  9, 10, 13, 14, 17, 20, 23, 24, 25, 26,
                   31, 32, 34, 35, 36, 37, 38, 39, 41, 42, 43, 44, 45, 49, 50, 51, 52,
                   56, 57, 58, 59, 61, 63, 66, 68, 69, 70, 73, 74, 76, 78, 79, 80, 81,
                   82, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 99]
        gate2.child_update_2d(x_threshold=2, y_threshold=-2.5, method='test')
        _test(gate2, pos_idx)


    def testGating(self):
        # Initiate Gating object
        project = Project.objects(project_id='test').get()
        gate = Gating(experiment=project.load_experiment('test'),
                      sample_id='test_experiment_dummy')
        self.assertEqual(gate.data.shape, (100, 3))
        self.assertEqual(gate.ctrl.get('dummy_ctrl').shape, (100, 3))
        self.assertEqual(len(gate.populations), 1)

        # Density Gates
        test1d = ChildPopulationCollection(gate_type='threshold_1d')
        test1d.add_population('positive', definition='+')
        test1d.add_population('negative', definition='-')
        gate.create_gate('test1d', parent='root', class_='DensityThreshold', method='gate1d')

