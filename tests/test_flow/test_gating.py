import sys
sys.path.append('/home/ross/CytoPy')

# Data imports
from data.mongo_setup import global_init
from data.project import Project
from data.fcs import Population
from flow.gating.defaults import ChildPopulationCollection
from flow.gating.actions import Gating
from flow.gating import dbscan
from tests.utilities import make_example_date, setup_with_dummy_data
from mongoengine.connection import connect
from mongoengine.base import datastructures
import unittest


class TestGating(unittest.TestCase):
    @staticmethod
    def _build(dump=True):
        # Initiate Gating object
        if dump:
            db = connect('test')
            db.drop_database('test')
            global_init('test')
            setup_with_dummy_data()
        project = Project.objects(project_id='test').get()
        gate = Gating(experiment=project.load_experiment('test_experiment_dummy'),
                      sample_id='dummy_test')
        return gate

    def test_build(self):
        gate = self._build()
        self.assertEqual(gate.data.shape, (100, 3))
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
        populations = ChildPopulationCollection(gate_type='threshold_1d')
        populations.add_population('positive', definition='+')
        populations.add_population('negative', definition='-')
        populations.populations['positive'].update_index(pos_idx)
        populations.populations['negative'].update_index(neg_idx)
        populations.populations['positive'].update_geom(shape='threshold',
                                                        x='feature0',
                                                        y='feature1')
        populations.populations['negative'].update_geom(shape='threshold',
                                                        x='feature0',
                                                        y='feature1')
        g.update_populations(output=populations,
                             parent_name='root',
                             warnings=['this is a test'])
        return g

    def test_construct_tree(self):
        def _test(pops):
            self.assertEqual(pops.get('a').parent.name, 'root')
            self.assertEqual(pops.get('b').parent.name, 'a')
            self.assertEqual(pops.get('c').parent.name, 'a')
            self.assertEqual(pops.get('d').parent.name, 'c')

        gate = self._build()
        self.assertTrue(len(gate.populations) == 1)
        # Add dummy populations
        a, b, c, d = self._dummy_pops()
        gate.filegroup.populations = [Population(population_name='root'), a, b, c, d]
        populations = gate._construct_tree(gate.filegroup)
        _test(populations)

        gate.filegroup.populations = [Population(population_name='root'), d, c, b, a]
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
        gate.filegroup.populations = [Population(population_name='root'), a, b, c, d]
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
        self.assertEqual(g.populations.get('positive').parent.name, 'root')
        self.assertEqual(g.populations.get('negative').parent.name, 'root')
        self.assertDictEqual(g.populations.get('positive').geom,
                             {'shape': 'threshold', 'x': 'feature0', 'y': 'feature1'})
        self.assertDictEqual(g.populations.get('negative').geom,
                             {'shape': 'threshold', 'x': 'feature0', 'y': 'feature1'})

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
        self.assertFalse(g._check_class_args(dbscan.DensityClustering,
                                             method='dbscan'))
        self.assertTrue(g._check_class_args(dbscan.DensityClustering,
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
        self.assertListEqual(list(g.populations.get('merged').index),
                             list(data.index.values))

    def test_subtraction(self):
        g = self._add_population(self._build())
        g.subtraction(target=['positive'],
                      parent='root',
                      new_population_name='subtraction')
        self.assertTrue('subtraction' in g.populations.keys())
        y = g.populations.get('negative').index
        y_hat = g.populations.get('subtraction').index
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
        g = self._build()
        g = self._dummy_gate(g, parent='root')
        self.assertIsNotNone(g._apply_checks('test'))

    def test_apply_gate(self):
        data = make_example_date(n_samples=100, centers=3, n_features=2)
        g = self._dummy_gate(self._build())
        g.apply('test', plot_output=False, feedback=False)
        self.assertTrue(all([x in g.populations.keys() for x in ['positive', 'negative']]))
        y = data[data.blobID == 1].index.values
        y_hat = g.populations.get('positive').index
        self.assertListEqual(list(y), list(y_hat))

    def test_edit_gate(self):
        data = make_example_date(n_samples=100, centers=3, n_features=2)
        g = self._build()

        # Threshold 2D
        populations = ChildPopulationCollection(gate_type='threshold_2d')
        populations.add_population('positive', definition='++')
        populations.add_population('negative', definition=['--', '+-', '-+'])
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
                                 'threshold_y': 5,
                                 'shape': '2d_threshold',
                                 'transform_x': None,
                                 'transform_y': None},
                    'negative': {'definition': ['--', '+-', '-+'],
                                 'x': 'feature0',
                                 'y': 'feature1',
                                 'threshold_x': -2.5,
                                 'threshold_y': 5,
                                 'shape': '2d_threshold',
                                 'transform_x': None,
                                 'transform_y': None}
                    }
        g.edit_gate('test', updated_geom=new_geom)
        y = data[(data.feature0.round(2) >= -2.5) &
                 (data.feature1.round(2) >= 5)].index.values
        y_hat = g.populations.get('positive').index
        self.assertListEqual(list(y), list(y_hat))

        # Rectangular gate
        populations = ChildPopulationCollection(gate_type='geom')
        populations.add_population('positive', definition='+')
        populations.add_population('negative', definition='-')
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
        new_geom = {'positive': {'definition': '+',
                                 'x': 'feature0',
                                 'y': 'feature1',
                                 'x_min': -12,
                                 'x_max': -2.5,
                                 'y_min': -12,
                                 'y_max': 0,
                                 'shape': 'rect',
                                 'transform_x': None,
                                 'transform_y': None},
                    'negative': {'definition': '-',
                                 'x': 'feature0',
                                 'y': 'feature1',
                                 'x_min': -12,
                                 'x_max': -2.5,
                                 'y_min': -12,
                                 'y_max': 0,
                                 'shape': 'rect',
                                 'transform_x': None,
                                 'transform_y': None}
                    }
        g.edit_gate('test', updated_geom=new_geom)
        y = data[data.blobID == 2.0].index.values
        y_hat = g.populations.get('positive').index
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
                                  centroid=(5, 2.5),
                                  width=5,
                                  height=8,
                                  angle=0),
                      child_populations=populations)
        g.apply('test', plot_output=False, feedback=False)
        new_geom = {'positive': {'definition': '+',
                                 'x': 'feature0',
                                 'y': 'feature1',
                                 'centroid': (-7., -7),
                                 'width': 5,
                                 'height': 8,
                                 'angle': 0,
                                 'shape': 'ellipse',
                                 'transform_x': None,
                                 'transform_y': None},
                    'negative': {'definition': '-',
                                 'x': 'feature0',
                                 'y': 'feature1',
                                 'centroid': (-7., -7),
                                 'width': 5,
                                 'height': 8,
                                 'angle': 0,
                                 'shape': 'ellipse',
                                 'transform_x': None,
                                 'transform_y': None}
                    }
        g.edit_gate('test', updated_geom=new_geom)
        y = data[data.blobID == 2.0].index.values
        y_hat = g.populations.get('positive').index
        self.assertListEqual(list(y), list(y_hat))

    def test_nudge_threshold(self):
        # Make a dummy threshold gate
        g = self._build()
        populations = ChildPopulationCollection(gate_type='threshold_2d')
        populations.add_population('positive', definition='++')
        populations.add_population('negative', definition=['--', '+-', '-+'])
        g.create_gate(gate_name='test',
                      parent='root',
                      class_='Static',
                      method='threshold_2d',
                      kwargs=dict(x='feature0',
                                  y='feature1',
                                  transform_x=None,
                                  transform_y=None,
                                  threshold_x=1,
                                  threshold_y=-2.5),
                      child_populations=populations)
        g.apply('test')
        data = make_example_date(n_samples=100, centers=3, n_features=2)
        g.nudge_threshold('test', new_x=4, new_y=2.5)
        y = data[(data.feature0.round(2) >= 4) &
                 (data.feature1.round(2) >= 2.5)].index.values
        y_hat = g.populations.get('positive').index
        self.assertListEqual(list(y), list(y_hat))

    def test_find_dependencies(self):
        gate = self._build()
        # Add dummy populations
        a, b, c, d = self._dummy_pops()
        gate.filegroup.populations = [Population(population_name='root'), a, b, c, d]
        gate.populations = gate._construct_tree(gate.filegroup)
        self.assertListEqual(gate.find_dependencies('a'),
                             ['a', 'b', 'c', 'd'])

    def test_remove_pop(self):
        gate = self._build()
        a, b, c, d = self._dummy_pops()
        gate.filegroup.populations = [Population(population_name='root'), a, b, c, d]
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
        self.assertEqual(type(pop.geom), datastructures.BaseList)

    def test_save(self):
        g = self._build()
        g.save()
        g = self._build(dump=False)
        self.assertEqual(len(g.filegroup.populations), 1)

    def test_check_downstream_overlaps(self):
        gate = self._build()
        a, b, c, d = self._dummy_pops()
        gate.filegroup.populations = [Population(population_name='root'), a, b, c, d]
        gate.populations = gate._construct_tree(gate.filegroup)
        self.assertTrue(gate.check_downstream_overlaps('d', population_labels=['c']))
        self.assertFalse(gate.check_downstream_overlaps('a', population_labels=['c']))


if __name__ == '__main__':
    unittest.main()
