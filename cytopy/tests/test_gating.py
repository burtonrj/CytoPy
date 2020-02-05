from cytopy.flow.gating.utilities import check_peak, find_local_minima, kde, inside_ellipse, rectangular_filter
from cytopy.data.mongo_setup import test_init
from sklearn.datasets import make_blobs
from sklearn.neighbors import KernelDensity
from scipy.signal import find_peaks
from datetime import datetime
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import unittest
import math
np.random.seed(42)
test_init()
log = open('performance_log.txt', 'a')
log.write('----------------------------------------------------------\n')
log.write(f"Testing gating functionality. {datetime.now().strftime('%Y/%m/%d %H:%M')}\n")
log.write('----------------------------------------------------------\n')


def generate_example_kde(u1, u2, s1, s2, bandwidth, plot=False):
    x1 = np.random.normal(loc=u1, scale=s1, size=500)
    x2 = np.random.normal(loc=u2, scale=s2, size=500)
    x = np.concatenate((x1, x2))
    xx = np.linspace(min(x), max(x), 1000)
    density = KernelDensity(bandwidth=bandwidth,
                            kernel='gaussian')
    density.fit(x[:, None])
    probs = np.exp(density.score_samples(xx[:, None]))
    if plot:
        plt.plot(xx, probs)
        plt.show()
    return xx, probs


def measure_performance(name):
    def decorator(func):
        def wrapper(*args, **kwargs):
            start = datetime.now()
            func(*args, **kwargs)
            end = datetime.now()
            log.write(f"{name}: {(end - start).__str__()}\n")
        return wrapper
    return decorator


class TestUtilities(unittest.TestCase):
    def test_checkpeaks(self):
        # Generate example data
        one_peak = generate_example_kde(4, 4, 1, 1, 0.5, False)
        two_peak = generate_example_kde(4, 8, 1, 1, 0.5, False)
        many_peak = generate_example_kde(4, 9, 1, 1, 0.1, False)

        peaks = check_peak(find_peaks(one_peak[1])[0],
                           one_peak[1])
        self.assertEqual(len(peaks), 1)
        peaks = check_peak(find_peaks(two_peak[1])[0],
                           two_peak[1])
        self.assertEqual(len(peaks), 2)
        peaks = check_peak(find_peaks(many_peak[1])[0],
                           many_peak[1], t=0.95)
        self.assertEqual(len(peaks), 2)

    def test_find_local_minima(self):
        two_peak = generate_example_kde(4, 8, 1, 1, 0.2, False)
        many_peak = generate_example_kde(4, 9, 1, 1, 0.1, False)
        peaks = check_peak(find_peaks(two_peak[1])[0],
                           two_peak[1])
        local_min = find_local_minima(two_peak[1],
                                      two_peak[0],
                                      peaks)
        self.assertTrue(5 <= local_min <= 7)

        peaks = check_peak(find_peaks(many_peak[1])[0],
                           many_peak[1])
        local_min = find_local_minima(many_peak[1],
                                      many_peak[0],
                                      peaks)
        self.assertTrue(5 <= local_min <= 7)

    def test_inside_ellipse(self):
        X, _ = make_blobs(n_samples=30, centers=1, n_features=2, random_state=42)
        e = dict(center=(-4, 8), width=2.2, height=0.8,
                 angle=-60)
        data = pd.DataFrame(X, columns=['x', 'y'])
        data['truth'] = 0
        data['truth'] = data['truth'].mask((data['x'] < -3.3) & (data['y'] < 8.6), 1)
        data['predicted'] = inside_ellipse(data[['x', 'y']].values,
                                           **e)
        self.assertTrue(all(data['truth'] == data['predicted']))

    def test_rectangular_filter(self):
        X, _ = make_blobs(n_samples=30, centers=1, n_features=2, random_state=42)
        r = dict(xmin=-1.7, ymin=7, xmax=0, ymax=11)
        data = pd.DataFrame(X, columns=['x', 'y'])
        data['truth'] = 0
        mask = (data['x'] < 0) & (data['x'] > -1.7) & (data['y'] < 11) & (data['y'] > 7)
        data['truth'] = data['truth'].mask(mask, 1)
        predicted = rectangular_filter(data, 'x', 'y', r)
        data['predicted'] = 0
        data['predicted'] = data['predicted'].mask(data.index.isin(predicted.index), 1)
        self.assertTrue(all(data['truth'] == data['predicted']))
