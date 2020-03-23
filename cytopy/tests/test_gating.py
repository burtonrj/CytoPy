from warnings import filterwarnings
filterwarnings('ignore')
# Data imports
from ..data.project import Project
from ..data.fcs import Population
from ..data.gating import Gate as DataGate
from ..data.mongo_setup import global_init
# Gating imports
from ..flow.gating.actions import Gating, ChildPopulationCollection
from ..flow.gating.base import Gate
from ..flow.gating import dbscan
from ..flow.gating import density
from ..flow.gating import utilities
from ..flow.gating import mixturemodel
from ..flow.gating import quantile
from ..flow.gating import static
# Other tools
from .utilities import make_example_date
from sklearn.neighbors import KernelDensity
from scipy.signal import find_peaks
import numpy as np
import pandas as pd
import unittest
import sys

unittest.TestLoader.sortTestMethodsUsing = None
sys.path.append('/home/rossco/CytoPy')
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


class TestGate(unittest.TestCase):

    @staticmethod
    def _build_gate(dimensions=1):
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

    def _test_index_updates(self, g, pos, data):
        neg = [i for i in data.index.values if i not in pos]
        self.assertListEqual(list(g.child_populations.populations['positive'].index), pos)
        self.assertListEqual(list(g.child_populations.populations['negative'].index), neg)

    def test_1d_update(self):
        # 1-D gate
        gate, _, data = self._build_gate()
        pos_idx = [1,  3,  5,  7,  8, 26, 31, 32, 38, 39, 42, 45, 49, 50, 51, 58, 61,
                   63, 66, 68, 69, 70, 74, 76, 78, 79, 81, 87, 89, 90, 91, 93, 97]
        gate.child_update_1d(threshold=0.3, method='test', merge_options='overwrite')
        self._test_index_updates(gate, pos_idx, data)

        gate.child_update_1d(threshold=4, method='test', merge_options='overwrite')
        pos_idx = [1,  3,  7,  8, 31, 32, 38, 39, 42, 51, 58, 66, 68, 69, 70, 74, 76,
                   78, 89, 90, 91, 93, 97]
        gate.child_update_1d(threshold=4, method='test', merge_options='overwrite')
        self._test_index_updates(gate, pos_idx, data)

        gate.child_update_1d(threshold=9, method='test', merge_options='merge')
        self._test_index_updates(gate, pos_idx, data)

    def test_2d_update(self):
        # 2-D gate
        gate1, gate2, data = self._build_gate(dimensions=2)
        pos_idx = [1,  3,  5,  7,  8, 26, 31, 32, 38, 39, 42, 45, 49, 50, 51, 58, 61,
                   63, 66, 68, 69, 70, 74, 76, 78, 79, 81, 87, 89, 90, 91, 93, 97]
        gate1.child_update_2d(x_threshold=2, y_threshold=-2.5, method='test')
        self._test_index_updates(gate1, pos_idx, data)

        pos_idx = [1,  2,  3,  5,  6,  7,  8,  9, 10, 13, 14, 17, 20, 23, 24, 25, 26,
                   31, 32, 34, 35, 36, 37, 38, 39, 41, 42, 43, 44, 45, 49, 50, 51, 52,
                   56, 57, 58, 59, 61, 63, 66, 68, 69, 70, 73, 74, 76, 78, 79, 80, 81,
                   82, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 99]
        gate2.child_update_2d(x_threshold=2, y_threshold=-2.5, method='test')
        self._test_index_updates(gate2, pos_idx, data)

    def test_generate_chunks(self):
        example_data = make_example_date(n_samples=100)
        populations = ChildPopulationCollection()
        populations.add_population('positive', definition='+')
        populations.add_population('negative', definition='-')
        gate = Gate(data=example_data,
                    x='feature0',
                    y='feature1',
                    child_populations=populations,
                    transform_x=None,
                    transform_y=None)
        chunks = gate.generate_chunks(chunksize=10)
        self.assertTrue(len(chunks) == 10)
        self.assertTrue(all(x.shape[0] == 10 for x in chunks))

    def test_generate_poly(self):
        example_data = make_example_date(n_samples=100)
        example_data['labels'] = example_data['bloblID']
        populations = ChildPopulationCollection()
        populations.add_population('positive', definition='+')
        populations.add_population('negative', definition='-')
        gate = Gate(data=example_data,
                    x='feature0',
                    y='feature1',
                    child_populations=populations,
                    transform_x=None,
                    transform_y=None)
        self.assertTrue(len(gate.generate_polygons()) == 3)


class TestDBSCAN(unittest.TestCase):

    def _build(self, return_data: bool = False):
        example_data = make_example_date(n_samples=10000)
        example_data['labels'] = example_data['blobID']
        populations = ChildPopulationCollection(gate_type='cluster')
        populations.add_population('blob1', target=(-2.5, 10))
        populations.add_population('blob2', target=(5, 1))
        populations.add_population('blob3', target=(-7.5, -7.5))

        gate = dbscan.DensityBasedClustering(data=example_data,
                                             child_populations=populations,
                                             x='feature0',
                                             y='feature1',
                                             transform_x=None,
                                             transform_y=None,
                                             min_pop_size=10)
        if return_data:
            return gate, example_data
        return gate

    def test_meta_assignmetn(self):
        test_df = pd.DataFrame({'A': [0, 1, 2, 3],
                                'labels': [0, 1, 2, 3],
                                'chunk_idx': [0, 0, 0, 0]})
        ref_df = pd.DataFrame({'chunk_idx': [0, 0, 1, 1],
                               'cluster': [0, 1, 0, 1],
                               'meta_cluster': ['0', '1', '0', '1']})
        modified_df = dbscan.meta_assignment(test_df, ref_df)
        self.assertListEqual(list(modified_df.labels.values), ['0', '1'])

    def test_meta_clustering(self):
        gate = self._build()
        data = gate.data.copy()
        data['chunk_idx'] = 0
        cluster_centroids = gate._meta_clustering(clustered_chunks=[data])
        self.assertEqual(cluster_centroids.shape[0], 3)

    def test_post_cluster_checks(self):
        gate = self._build()
        data = gate.data.copy()
        data = data[data.labels == 1]
        gate._post_cluster_checks(data)
        self.assertEqual(gate.warnings[0],
                         'Failed to identify any distinct populations')
        data = gate.data.copy()
        data = data[data.labels.isin(0., 1.0)]
        gate._post_cluster_checks(data)
        self.assertEqual(gate.warnings[0],
                         'Expected 3 populations, identified 2')

    def test_dbscan_knn(self):
        gate = self._build()
        gate._dbscan_knn(distance_nn=0.5, core_only=False)
        self.assertEqual(len(gate.data.labels.unique()), 3)

    def test_match_pop_to_cluster(self):
        gate = self._build()
        cluster_polygons = gate.generate_polygons()
        self.assertEqual(gate._match_pop_to_cluster(target_population='blob1',
                                                    cluster_polygons=cluster_polygons), 0.0)
        self.assertEqual(gate._match_pop_to_cluster(target_population='blob2',
                                                    cluster_polygons=cluster_polygons), 1.0)
        self.assertEqual(gate._match_pop_to_cluster(target_population='blob3',
                                                    cluster_polygons=cluster_polygons), 2.0)

    def _clustering(self, method='dbscan'):
        gate, data = self._build(return_data=True)
        blob1_idx = data[data.blobID == 0].index.values
        blob2_idx = data[data.blobID == 1].index.values
        blob3_idx = data[data.blobID == 2].index.values
        if method == 'dbscan':
            populations = gate.dbscan(distance_nn=0.25)
        else:
            populations = gate.hdbscan()
        for p, idx in zip(['blob1', 'blob2', 'blob3'], [blob1_idx, blob2_idx, blob3_idx]):
            self.assertListEqual(list(populations.populations[p].index), list(idx))

    def test_dbscan(self):
        self._clustering()

    def test_hdbscan(self):
        self._clustering(method='hdbscan')


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
        self.assertEqual(method, 'Local minima between pair of highest peaks')
        self.assertAlmostEqual(threshold, 1.32, places=2)

    def test_gate_1d(self):
        gate, data = _build_density_gate(dimensions=1,
                                         return_data=True,
                                         kde_bw=0.5)
        populations = gate.gate_1d()
        y = data[data.feature0 >= 1.32].index.values
        y_hat = populations.populations['positive'].index.values
        self.assertListEqual(list(y), list(y_hat))
        y = data[data.feature0 < 1.32].index.values
        y_hat = populations.populations['negative'].index.values
        self.assertListEqual(list(y), list(y_hat))

    def test_gate_2d(self):
        gate, data = _build_density_gate(dimensions=2,
                                         return_data=True,
                                         kde_bw=0.5)
        populations = gate.gate_2d()
        y = data[(data.feature0.round(decimals=2) >= 1.32) &
                 (data.feature1.round(decimals=2) >= -2.30)]
        y_hat = populations.populations['positive'].index.values
        self.assertListEqual(list(y), list(y_hat))
        y = data[(data.feature0.round(decimals=2) < 1.32) &
                 (data.feature1.round(decimals=2) < -2.30)]
        y_hat = populations.populations['negative'].index.values
        self.assertListEqual(list(y), list(y_hat))


class TestMixtureModel(unittest.TestCase):

    @staticmethod
    def _build(return_data: bool = False,
               blobs=3,
               **kwargs):
        example_data = make_example_date(n_samples=1000, centers=blobs)
        example_data['labels'] = example_data['blobID']

        populations = ChildPopulationCollection(gate_type='geom')
        populations.add_population('positive', definition='+')
        populations.add_population('negative', definition='-')

        gate = mixturemodel.MixtureModel(data=example_data,
                                         child_populations=populations,
                                         x='feature0',
                                         y='feature1',
                                         transform_x=None,
                                         transform_y=None,
                                         **kwargs)
        if return_data:
            return gate, example_data
        return gate

    def test_create_ellipse(self):
        gate, data = self._build(return_data=True,
                                 target=(-2.5, 10),
                                 k=3)
        pos_idx = data[data.blobID == 0].index.values
        populations = gate.gate()
        tp = [i for i in pos_idx if i in populations.populations['positive'].index]
        self.assertTrue(len(tp)/len(pos_idx) > 0.8)


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


class TestStatic(unittest.TestCase):

    @staticmethod
    def _build(populations: ChildPopulationCollection,
               return_data: bool = False,
               **kwargs):
        example_data = make_example_date(n_samples=1000, centers=3)

        gate = static.Static(data=example_data,
                             child_populations=populations,
                             x='feature0',
                             y='feature1',
                             transform_x=None,
                             transform_y=None,
                             **kwargs)
        if return_data:
            return gate, example_data
        return gate

    def test_rect_gate(self):
        populations = ChildPopulationCollection(gate_type='geom')
        populations.add_population('positive', definition='+')
        populations.add_population('negative', definition='-')
        gate, data = self._build(populations=populations,
                                 return_data=True)
        y = data[(data.feature0.round(2) >= 2.5) & (data.feature0.round(2) < 8) &
                 (data.feature1.round(2) >= -5) & (data.feature1.round(2) < 5)].index.values
        populations = gate.rect_gate(x_min=2.5, x_max=8, y_min=-5, y_max=5)
        y_hat = list(populations['positive'].index.values)
        self.assertListEqual(list(y), y_hat)

    def test_threshold_2d(self):
        populations = ChildPopulationCollection(gate_type='threshold_2d')
        populations.add_population('positive', definition='++')
        populations.add_population('negative', definition=['--', '+-', '-+'])
        gate, data = self._build(populations=populations,
                                 return_data=True)
        y = data[(data.feature0.round(2) >= 2.5) &
                 (data.feature1.round(2) >= -5)].index.values
        populations = gate.threshold_2d(threshold_x=2.5, threshold_y=-5)
        y_hat = list(populations['positive'].index.values)
        self.assertListEqual(list(y), y_hat)

    def test_ellipse(self):
        populations = ChildPopulationCollection(gate_type='geom')
        populations.add_population('positive', definition='+')
        populations.add_population('negative', definition='-')
        gate, data = self._build(populations=populations,
                                 return_data=True)
        y = data[data.blobID == 1.0].index.values
        y_hat = gate.ellipse_gate(centroid=(4., 1.),
                                  width=5,
                                  height=5,
                                  angle=0).populations['positive'].index.values
        self.assertListEqual(list(y), list(y_hat))


class TestGating(unittest.TestCase):

    def _build(self):
        # Initiate Gating object
        project = Project.objects(project_id='test').get()
        gate = Gating(experiment=project.load_experiment('test'),
                      sample_id='test_experiment_dummy')
        return gate

    def test_build(self):
        gate = self._build()
        self.assertEqual(gate.data.shape, (1000, 3))
        self.assertEqual(gate.ctrl.get('dummy_ctrl').shape, (100, 3))
        self.assertEqual(len(gate.populations), 1)

    @staticmethod
    def _dummy_pops():
        a = Population(population_name='a',
                       parent='root')
        b = Population(population_name='b',
                       parent='a')
        c = Population(population_name='c',
                       parent='a')
        d = Population(population_name='d',
                       parent='c')
        return a, b, c, d

    @staticmethod
    def _add_population(g):
        data = make_example_date(n_samples=100, centers=3, n_features=2)
        pos_idx = data[data.blobID == 0].index.values
        neg_idx = data[data.blobID != 0].index.values
        populations = ChildPopulationCollection(gate_type='geom')
        populations.add_population('positive', definition='+')
        populations.add_population('negative', definition='-')
        populations.populations['positive'].update_index(pos_idx)
        populations.populations['negative'].update_index(neg_idx)
        populations.populations['positive'].update_geom(shape='test',
                                                        x='feature0',
                                                        y='feature1')
        populations.populations['negative'].update_geom(shape='test',
                                                        x='feature0',
                                                        y='feature1')
        g.update_populations(output=populations,
                             parent_name='root')
        return g

    def test_construct_tree(self):
        def _test(pops):
            self.assertListEqual(list(pops.keys()),
                                 ['root', 'a', 'b', 'c', 'd'])
            self.assertEqual(pops.get('a').parent.name, 'root')
            self.assertEqual(pops.get('b').parent.name, 'a')
            self.assertEqual(pops.get('c').parent.name, 'a')
            self.assertEqual(pops.get('d').parent.name, 'c')

        gate = self._build()
        self.assertTrue(len(gate.populations) == 1)
        # Add dummy populations
        a, b, c, d = self._dummy_pops()
        gate.filegroup.populations = [gate.populations.get('root'), a, b, c, d]
        populations = gate._construct_tree(gate.filegroup)
        _test(populations)

        gate.filegroup.populations = [gate.populations.get('root'), d, c, b, a]
        populations = gate._construct_tree(gate.filegroup)
        _test(populations)

    def test_get_pop_df(self):
        gate = self._build()
        test = gate.get_population_df(population_name='root',
                                      transform=True,
                                      transform_method='logicle',
                                      transform_features='all',
                                      label=False,
                                      ctrl_id=None)
        self.assertEqual(test.shape, (100, 3))
        test = gate.get_population_df(population_name='root',
                                      transform=True,
                                      transform_method='logicle',
                                      transform_features='all',
                                      label=True,
                                      ctrl_id=None)
        self.assertEqual(test.shape, (100, 4))
        test = gate.get_population_df(population_name='root',
                                      transform=True,
                                      transform_method='logicle',
                                      transform_features='all',
                                      label=False,
                                      ctrl_id='dummy_ctrl')
        self.assertEqual(test.shape, (100, 3))

    def test_valid_pops(self):
        gate = self._build()
        # Add dummy populations
        a, b, c, d = self._dummy_pops()
        gate.filegroup.populations = [gate.populations.get('root'), a, b, c, d]
        gate.populations = gate._construct_tree(gate.filegroup)
        valid = gate.valid_populations(populations=['root', 'a', 'c', 'f', 'd'],
                                       verbose=False)
        self.assertListEqual(valid, ['root', 'a', 'c', 'd'])

    def test_search_ctrl_cache(self):
        gate = self._build()
        idx = gate.search_ctrl_cache(target_population='root',
                                     ctrl_id='dummy_ctrl')
        self.assertTrue(len(idx) == 100)
        df = gate.search_ctrl_cache(target_population='root',
                                    ctrl_id='dummy_ctrl',
                                    return_dataframe=True)
        self.assertEqual(df.shape, (100, 3))

    def test_update_populations(self):
        g = self._add_population(self._build())
        data = make_example_date(n_samples=100, centers=3, n_features=2)
        pos_idx = data[data.blobID == 0].index.values
        neg_idx = data[data.blobID != 0].index.values
        self.assertTrue('positive' in g.populations.keys())
        self.assertTrue('negative' in g.populations.keys())
        self.assertListEqual(list(g.populations.get('positive').index),
                             list(pos_idx))
        self.assertListEqual(list(g.populations.get('negative').index),
                             list(neg_idx))
        self.assertEqual(g.populations.get('positive').prop_of_parent,
                         len(pos_idx)/data.shape[0])
        self.assertEqual(g.populations.get('positive').prop_of_total,
                         len(pos_idx)/data.shape[0])
        self.assertEqual(g.populations.get('negative').prop_of_parent,
                         len(neg_idx)/data.shape[0])
        self.assertEqual(g.populations.get('negative').prop_of_total,
                         len(neg_idx)/data.shape[0])
        self.assertEqual(g.populations.get('positive').parent, 'root')
        self.assertEqual(g.populations.get('negative').parent, 'root')
        self.assertDictEqual(g.populations.get('positive').geom,
                             {'shape': 'test', 'x': 'feature0', 'y': 'feature1'})
        self.assertDictEqual(g.populations.get('negative').geom,
                             {'shape': 'test', 'x': 'feature0', 'y': 'feature1'})

    def test_predict_ctrl_pop(self):
        from sklearn.neighbors import KNeighborsClassifier
        model = KNeighborsClassifier(n_neighbors=5, n_jobs=-1)
        g = self._add_population(self._build())
        data = make_example_date(n_samples=100, centers=3, n_features=2)
        g._predict_ctrl_population(target_population='positive',
                                   ctrl_id='dummy_ctrl',
                                   model=model)
        y = data[data.blobID == 0].index.values
        y_hat = g.populations.get('positive').control_idx.get('dummy_ctrl')
        self.assertListEqual(list(y), list(y_hat))

    def test_check_class_args(self):
        class Dummy:
            def __init__(self):
                pass

        g = self._build()
        self.assertFalse(g._check_class_args(Dummy, method=''))
        self.assertFalse(g._check_class_args(dbscan.DensityBasedClustering,
                                             method='dbscan'))
        self.assertTrue(g._check_class_args(dbscan.DensityBasedClustering,
                                            method='dbscan',
                                            x='x',
                                            child_populations='',
                                            min_pop_size='',
                                            distance_nn=''))

    def test_merge(self):
        g = self._add_population(self._build())
        g.merge(population_left='positive',
                population_right='negative',
                new_population_name='merged')
        data = make_example_date(n_samples=100, centers=3, n_features=2)
        self.assertTrue('merged' in g.populations.keys())
        self.assertListEqual(list(g.populations.get('merged').index.values),
                             list(data.index.values))

    def test_subtraction(self):
        g = self._add_population(self._build())
        g.subtraction(target='positive',
                      parent='root',
                      new_population_name='subtraction')
        self.assertTrue('subtraction' in g.populations.keys())
        y = g.populations.get('negative').index.values
        y_hat = g.populations.get('subtraction').index.values
        self.assertListEqual(list(y), list(y_hat))

    @staticmethod
    def _dummy_gate(g, parent='root', children: list or None = None):
        populations = ChildPopulationCollection(gate_type='geom')
        if children:
            for pop in children:
                populations.add_population(pop.get('name'),
                                           definition=pop.get('definition'))
        else:
            populations.add_population('positive', definition='+')
            populations.add_population('negative', definition='-')

        g.create_gate(gate_name='test',
                      parent=parent,
                      class_='Static',
                      method='rect_gate',
                      kwargs=dict(x='feature0',
                                  y='feature1',
                                  transform_x=None,
                                  transform_y=None,
                                  x_min=1.5,
                                  x_max=8.0,
                                  y_min=-5,
                                  y_max=5.5),
                      child_populations=populations)
        return g

    def test_create_gate(self):
        g = self._dummy_gate(self._build())
        self.assertTrue('test' in g.gates.keys())

    def test_apply_checks(self):
        g = self._build()
        self.assertIsNone(g._apply_checks('not_a_gate'))
        g = self._dummy_gate(g, parent='not_a_population')
        self.assertIsNone(g._apply_checks('test'))
        g = self._dummy_gate(self._add_population(self._build()))
        self.assertIsNone(g._apply_checks('test'))
        g = self._dummy_gate(g, parent='root')
        self.assertIsNotNone(g._apply_checks('test'))

    def test_apply_gate(self):
        data = make_example_date(n_samples=100, centers=3, n_features=2)
        g = self._dummy_gate(self._build())
        self.assertTrue(g.apply('test', plot_output=False, feedback=False))
        y = data[data.blobID == 1].index.values
        y_hat = g.populations.get('positive').index.values
        self.assertListEqual(list(y), list(y_hat))


    def test_update_idx(self):
        pass

    def test_edit_gate(self):
        pass

    def test_nudge_threshold(self):
        pass

    def test_find_dependencies(self):
        pass

    def test_remove_pop(self):
        pass

    def test_print_tree(self):
        pass

    def test_pop_to_mongo(self):
        pass

    def test_save_ctrl_idx(self):
        pass

    def test_save(self):
        pass

    def test_cluster_idx(self):
        pass

    def test_reg_as_invalid(self):
        pass

    def test_check_downstream_overlaps(self):
        pass
