import sys
sys.path.append('/home/ross/CytoPy')

from cytopy.data.mongo_setup import global_init
from cytopy.flow.gating import utilities
from cytopy.tests.utilities import make_example_date
from sklearn.neighbors import KernelDensity
from scipy.signal import find_peaks
from itertools import combinations
import numpy as np
import pandas as pd
import unittest

global_init('test')


class TestCheckPeak(unittest.TestCase):
    def test(self):
        probs = np.array([0, 0, 0, 0.05, 0, 0, 2, 0, 0, 3, 0, 0, 0.05])
        peaks = np.where(np.array(probs) > 0)[0]
        self.assertEqual(len(utilities.check_peak(peaks, probs, t=0.5)), 2)
        self.assertEqual(len(utilities.check_peak(peaks, probs, t=0.1)), 2)
        self.assertEqual(len(utilities.check_peak(peaks, probs, t=0.01)), 4)


class TestFindLocalMinima(unittest.TestCase):
    @staticmethod
    def _build():
        data = make_example_date()
        data = pd.concat([data[data.blobID != 2],
                          data[data.blobID == 2].sample(frac=0.25)])
        d = data['feature0'].values
        density = KernelDensity(bandwidth=0.5, kernel='gaussian')
        density.fit(d[:, None])
        x_d = np.linspace(min(d), max(d), 1000)
        prob = np.exp(density.score_samples(x_d[:, None]))
        peaks = find_peaks(prob)[0]
        return prob, peaks, x_d

    def test(self):
        prob, peaks, x_d = self._build()
        threshold = utilities.find_local_minima(prob, x_d, peaks)
        self.assertTrue(0.58 <= threshold <= 0.6)


class TestInsideEllipse(unittest.TestCase):
    @staticmethod
    def _build():
        data = make_example_date()
        mask = utilities.inside_ellipse(data[['feature0', 'feature1']].values,
                                        center=(4.5, 2.5),
                                        width=2.3,
                                        height=3,
                                        angle=0)
        return data, mask

    def test(self):
        data, mask = self._build()
        correct = all(x == 1 for x in data.loc[mask].blobID.values)
        self.assertTrue(correct)


class TestRectangularFilter(unittest.TestCase):
    def test(self):
        data = make_example_date()
        rect = dict(xmin=0, xmax=8, ymin=-2.5, ymax=6.0)
        self.assertTrue(all(x == 1 for x in utilities.rectangular_filter(data,
                                                                         x='feature0',
                                                                         y='feature1',
                                                                         definition=rect).blobID.values))


class TestDensityDependentDownsample(unittest.TestCase):
    @staticmethod
    def _equal_ratio(data, samples):
        ratios = [data[data.blobID == x[0]].shape[0] / data[data.blobID == x[1]].shape[0]
                  for x in combinations(samples.blobID.unique(), 2)]
        return combinations(ratios, 2)

    def test(self):
        data = make_example_date(n_samples=10000)
        samples = utilities.density_dependent_downsample(data=data,
                                                         features=['feature0', 'feature1'],
                                                         mmd_sample_n=2000)
        for x, y in self._equal_ratio(data, samples):
            self.assertAlmostEqual(x, y, places=1)


class TestGetParams(unittest.TestCase):
    class MakeshiftClass:
        def __init__(self, a, b, c, d='test', **kwargs):
            pass

    def test_basic(self):
        self.assertListEqual(utilities.get_params(self.MakeshiftClass),
                             ['a', 'b', 'c', 'd'])

    def include_kwargs(self):
        self.assertListEqual(utilities.get_params(self.MakeshiftClass, exclude_kwargs=False),
                             ['a', 'b', 'c', 'd', 'kwargs'])

    def test_requied_only(self):
        self.assertListEqual(utilities.get_params(self.MakeshiftClass, required_only=True),
                             ['a', 'b', 'c'])

    def test_required_only_exclude_kwargs(self):
        self.assertListEqual(utilities.get_params(self.MakeshiftClass,
                                                  required_only=True,
                                                  exclude_kwargs=True),
                             ['a', 'b', 'c'])


if __name__ == '__main__':
    unittest.main()