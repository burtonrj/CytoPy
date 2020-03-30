import sys
sys.path.append('/home/ross/CytoPy')

# Data imports
from cytopy.data.mongo_setup import global_init
from cytopy.flow.gating.defaults import ChildPopulationCollection
from cytopy.flow.gating import density
from cytopy.tests.utilities import make_example_date
from sklearn.neighbors import KernelDensity
from cytopy.flow.gating import quantile
from scipy.signal import find_peaks
import numpy as np
import unittest

global_init('test')


def _build_density_gate(dimensions: int or float = 1.,
                        return_data: bool = False,
                        quantile_gate: bool = False,
                        **kwargs):
    example_data = make_example_date(n_samples=1000)
    example_data['labels'] = example_data['blobID']

    if dimensions == 1:
        populations = ChildPopulationCollection(gate_type='threshold_1d')
        populations.add_population('positive', definition='+')
        populations.add_population('negative', definition='-')
    elif dimensions == 2:
        populations = ChildPopulationCollection(gate_type='threshold_2d')
        populations.add_population('positive', definition=['++', '-+'])
        populations.add_population('negative', definition=['--', '+-'])
    elif dimensions == 2.1:
        populations = ChildPopulationCollection(gate_type='threshold_2d')
        populations.add_population('positive', definition='++')
        populations.add_population('negative', definition=['--', '+-', '-+'])
    else:
        raise ValueError('Invalid dimensions')
    if quantile_gate:
        gate = quantile.Quantile(data=example_data,
                                 child_populations=populations,
                                 x='feature0',
                                 y='feature1',
                                 transform_x=None,
                                 transform_y=None,
                                 **kwargs)
    else:
        gate = density.DensityThreshold(data=example_data,
                                        child_populations=populations,
                                        x='feature0',
                                        y='feature1',
                                        transform_x=None,
                                        transform_y=None,
                                        **kwargs)
    if return_data:
        return gate, example_data
    return gate


class TestDensity(unittest.TestCase):

    @staticmethod
    def kde(data, x, bw: int or float = 1.0):
        dens = KernelDensity(bandwidth=bw, kernel='gaussian')
        d = data[x].values
        dens.fit(d[:, None])
        x_d = np.linspace(min(d), max(d), 1000)
        logprob = dens.score_samples(x_d[:, None])
        peaks = find_peaks(logprob)[0]
        return x_d, np.exp(logprob), peaks

    def test_eval_peaks(self):
        gate, data = _build_density_gate(dimensions=1, return_data=True)
        # 1 peak
        xx, probs, peaks = self.kde(data, 'feature0', bw=10)
        threshold, method = gate._evaluate_peaks(data=data,
                                                 peaks=peaks,
                                                 probs=probs,
                                                 xx=xx)
        self.assertEqual(method, 'Quantile')
        self.assertEqual(threshold, data['feature0'].quantile(0.95, interpolation='nearest'))

        # 2 peaks
        xx, probs, peaks = self.kde(data, 'feature0', bw=2)
        threshold, method = gate._evaluate_peaks(data=data,
                                                 peaks=peaks,
                                                 probs=probs,
                                                 xx=xx)
        self.assertEqual(method, 'Local minima between pair of highest peaks')
        self.assertAlmostEqual(threshold, 1.10, places=2)

        # >2 peaks
        xx, probs, peaks = self.kde(data, 'feature0', bw=0.5)
        threshold, method = gate._evaluate_peaks(data=data,
                                                 peaks=peaks,
                                                 probs=probs,
                                                 xx=xx)
        self.assertEqual(method, 'Local minima between pair of highest peaks')
        self.assertAlmostEqual(threshold, 1.32, places=2)

    def test_gate_1d(self):
        gate, data = _build_density_gate(dimensions=1,
                                         return_data=True,
                                         kde_bw=0.5)
        populations = gate.gate_1d()
        y = data[data.feature0 >= 1.32].index.values
        y_hat = populations.populations['positive'].index
        self.assertListEqual(list(y), list(y_hat))
        y = data[data.feature0 < 1.32].index.values
        y_hat = populations.populations['negative'].index
        self.assertListEqual(list(y), list(y_hat))

    def test_gate_2d(self):
        gate, data = _build_density_gate(dimensions=2.1,
                                         return_data=True,
                                         kde_bw=0.5)
        populations = gate.gate_2d()
        y = data[(data.feature0.round(decimals=2) >= 1.32) &
                 (data.feature1.round(decimals=2) >= -2.30)].index.values
        y_hat = populations.populations['positive'].index
        self.assertListEqual(list(y), list(y_hat))
        y = data[(data.feature0.round(decimals=2) < 1.32) |
                 (data.feature1.round(decimals=2) < -2.30)].index.values
        y_hat = populations.populations['negative'].index
        self.assertListEqual(list(y), list(y_hat))


class TestQuantile(unittest.TestCase):

    def test_gate_1d(self):
        gate, data = _build_density_gate(dimensions=1,
                                         return_data=True,
                                         q=0.95,
                                         quantile_gate=True)
        threshold = float(data['feature0'].quantile(0.95, interpolation='nearest'))
        y = list(data[data.feature0.round(2) >= round(threshold, 2)].index.values)
        y_hat = list(gate.gate_1d().populations['positive'].index)
        self.assertListEqual(y, y_hat)

    def test_gate_2d(self):
        gate, data = _build_density_gate(dimensions=2.1,
                                         return_data=True,
                                         q=0.95,
                                         quantile_gate=True)
        x_threshold = float(data['feature0'].quantile(0.95, interpolation='nearest'))
        y_threshold = float(data['feature1'].quantile(0.95, interpolation='nearest'))
        y = list(data[(data.feature0.round(2) >= round(x_threshold, 2)) &
                      (data.feature1.round(2) >= round(y_threshold, 2))].index.values)
        y_hat = list(gate.gate_2d().populations['positive'].index)
        self.assertListEqual(y, y_hat)


