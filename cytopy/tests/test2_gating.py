# Data imports
from cytopy.data.project import Project
from cytopy.data.fcs import Population
from cytopy.data.mongo_setup import global_init
# Gating imports
from cytopy.flow.gating.actions import Gating, ChildPopulationCollection
from cytopy.flow.gating.base import Gate
from cytopy.flow.gating import dbscan
from cytopy.flow.gating import density
from cytopy.flow.gating import utilities
from cytopy.flow.gating import mixturemodel
from cytopy.flow.gating import quantile
from cytopy.flow.gating import static
# Other tools
from cytopy.tests.utilities import make_example_date
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

    def test_edit_gate(self):
        data = make_example_date(n_samples=100, centers=3, n_features=2)
        g = self._build()
        populations = ChildPopulationCollection(gate_type='geom')
        populations.add_population('positive', definition='++')
        populations.add_population('negative', definition=['--', '+-', '-+'])

        # Threshold 2D
        g.create_gate(gate_name='test',
                      parent='root',
                      class_='Static',
                      method='threshold_2d',
                      kwargs=dict(x='feature0',
                                  y='feature1',
                                  transform_x=None,
                                  transform_y=None,
                                  threshold_x=2.5,
                                  threshold_y=-5),
                      child_populations=populations)
        g.apply('test', plot_output=False, feedback=False)
        new_geom = {'positive': {'definition': '++',
                                 'x': 'feature0',
                                 'y': 'feature1',
                                 'threshold_x': -2.5,
                                 'threshold_y': 5},
                    'negative': {'definition': ['--', '+-', '-+'],
                                 'x': 'feature0',
                                 'y': 'feature1',
                                 'threshold_x': -2.5,
                                 'threshold_y': 5}
                    }
        g.edit_gate('test', updated_geom=new_geom)
        y = data[(data.feature0.round(2) >= -2.5) &
                 (data.feature1.round(2) >= 5)].index.values
        y_hat = g.populations.get('positive').index.values
        self.assertListEqual(list(y), list(y_hat))

        # Rectangular gate
        g.create_gate(gate_name='test',
                      parent='root',
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
        g.apply('test', plot_output=False, feedback=False)
        new_geom = {'positive': {'definition': '++',
                                 'x': 'feature0',
                                 'y': 'feature1',
                                 'x_min': -12,
                                 'x_max': -2.5,
                                 'y_min': -12,
                                 'y_max': 0},
                    'negative': {'definition': ['--', '+-', '-+'],
                                 'x': 'feature0',
                                 'y': 'feature1',
                                 'x_min': -12,
                                 'x_max': -2.5,
                                 'y_min': -12,
                                 'y_max': 0}
                    }
        g.edit_gate('test', updated_geom=new_geom)
        y = data[data.blobID == 2.0].index.values
        y_hat = g.populations.get('positive').index.values
        self.assertListEqual(list(y), list(y_hat))

        # Ellipse gate
        g.create_gate(gate_name='test',
                      parent='root',
                      class_='Static',
                      method='ellipse_gate',
                      kwargs=dict(x='feature0',
                                  y='feature1',
                                  transform_x=None,
                                  transform_y=None,
                                  centroid=(5.,2.5),
                                  width=5,
                                  height=5,
                                  angle=0),
                      child_populations=populations)
        g.apply('test', plot_output=False, feedback=False)
        new_geom = {'positive': {'definition': '++',
                                 'x': 'feature0',
                                 'y': 'feature1',
                                 'centroid': (-8., -7.5),
                                 'width': 5,
                                 'height': 5,
                                 'angle': 0},
                    'negative': {'definition': ['--', '+-', '-+'],
                                 'x': 'feature0',
                                 'y': 'feature1',
                                 'centroid': (-8., -7.5),
                                 'width': 5,
                                 'height': 5,
                                 'angle': 0}
                    }
        g.edit_gate('test', updated_geom=new_geom)
        y = data[data.blobID == 2.0].index.values
        y_hat = g.populations.get('positive').index.values
        self.assertListEqual(list(y), list(y_hat))

    def test_nudge_threshold(self):
        data = make_example_date(n_samples=100, centers=3, n_features=2)
        g = self._dummy_gate(self._build())
        g.nudge_threshold('test', new_x=-2.5, new_y=5)
        y = data[(data.feature0.round(2) >= -2.5) &
                 (data.feature1.round(2) >= 5)].index.values
        y_hat = g.populations.get('positive').index.values
        self.assertListEqual(list(y), list(y_hat))

    def test_find_dependencies(self):
        gate = self._build()
        # Add dummy populations
        a, b, c, d = self._dummy_pops()
        gate.filegroup.populations = [gate.populations.get('root'), a, b, c, d]
        gate.populations = gate._construct_tree(gate.filegroup)
        self.assertListEqual(gate.find_dependencies('a'),
                             ['a', 'b', 'c', 'd'])

    def test_remove_pop(self):
        gate = self._build()
        a, b, c, d = self._dummy_pops()
        gate.filegroup.populations = [gate.populations.get('root'), a, b, c, d]
        gate.populations = gate._construct_tree(gate.filegroup)
        gate.remove_population('c')
        self.assertListEqual(list(gate.populations.keys()),
                             ['root', 'a', 'b'])

    def test_pop_to_mongo(self):
        g = self._add_population(self._build())
        pop = g._population_to_mongo('positive')
        self.assertEqual(type(pop), Population)
        self.assertEqual('positive', pop.population_name)
        self.assertEqual('root', pop.parent)
        self.assertEqual(type(pop.geom), list)

    def test_save(self):
        g = self._build()
        g.save()
        g = self._build()
        self.assertEqual(len(g.filegroup.populations), 1)

    def test_reg_as_invalid(self):
        g = self._build()
        g.register_as_invalid()
        g = self._build()
        self.assertTrue('invalid' in g.filegroup.flags)

    def test_check_downstream_overlaps(self):
        gate = self._build()
        a, b, c, d = self._dummy_pops()
        gate.filegroup.populations = [gate.populations.get('root'), a, b, c, d]
        gate.populations = gate._construct_tree(gate.filegroup)
        self.assertTrue(gate.check_downstream_overlaps('d', population_labels=['c']))
        self.assertFalse(gate.check_downstream_overlaps('a', population_labels=['c']))
