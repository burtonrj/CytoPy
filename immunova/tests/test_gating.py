import unittest
from sklearn.datasets.samples_generator import make_blobs
from immunova.flow.gating.dbscan import dbscan_gate
from immunova.flow.gating.density import density_gate_1d
from immunova.flow.gating.fmo import density_1d_fmo, density_2d_fmo
from immunova.flow.gating.mixturemodel import mm_gate
from immunova.flow.gating.quantile import quantile_gate
from immunova.flow.gating.static import rect_gate
from immunova.flow.gating.utilities import find_local_minima, check_peak, kde, inside_ellipse, \
    density_dependent_downsample, rectangular_filter
from functools import partial
import numpy as np
import pandas as pd


class TestGatingFunctions(unittest.TestCase):

    def __init__(self, *args, **kwargs):
        super(TestGatingFunctions, self).__init__(*args, **kwargs)
        self.examples2D = self.gen_data()
        self.examples1D = self.gen_data(n_features=1)

    @staticmethod
    def gen_data(n_features=2):
        # make_blobs generates a two dimensional matrix with 3 centers by default
        X, y = make_blobs(n_samples=5000, random_state=42, n_features=n_features)
        X_uneven = np.vstack((X[y == 0][:1000], X[y == 1][:500], X[y == 2][:100]))
        transformation = [[0.60834549, -0.63667341], [-0.40887718, 0.85253229]]
        X_anisotropic = np.dot(X, transformation)
        X_varied, _ = make_blobs(n_samples=5000, random_state=42, cluster_std=[1.0, 2.5, 0.5], n_features=n_features)
        column_names = ['x1', 'x2']
        return dict(uneven=pd.DataFrame(X_uneven, columns=column_names),
                    anisotropic=pd.DataFrame(X_anisotropic, columns=column_names),
                    varied=pd.DataFrame(X_varied, columns=column_names))

    def test_dbscan(self):
        f = partial(dbscan_gate, x='x1', y='x2', min_pop_size=50, distance_nn=5, core_only=False)

        output_uneven = f(self.examples2D['uneven'], expected_population=[dict(id='pop1', target=(-7.5, -8)),
                                                                          dict(id='pop2', target=(5.0, 1)),
                                                                          dict(id='pop3', target=(-1.5, 10))])


if __name__ == '__main__':
    unittest.main()
