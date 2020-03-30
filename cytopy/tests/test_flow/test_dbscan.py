import sys
sys.path.append('/home/ross/CytoPy')

# Data imports
from cytopy.data.mongo_setup import global_init
from cytopy.tests.utilities import make_example_date
from cytopy.flow.gating.defaults import ChildPopulationCollection
from cytopy.flow.gating import dbscan
import pandas as pd
import unittest

global_init('test')


class TestDBSCAN(unittest.TestCase):

    def _build(self, return_data: bool = False, min_pop_size=2):
        example_data = make_example_date(n_samples=100)
        example_data['labels'] = example_data['blobID']
        populations = ChildPopulationCollection(gate_type='cluster')
        populations.add_population('blob1', target=(-2.5, 10), weight=1)
        populations.add_population('blob2', target=(5, 1), weight=1)
        populations.add_population('blob3', target=(-7.5, -7.5), weight=1)

        gate = dbscan.DensityBasedClustering(data=example_data,
                                             child_populations=populations,
                                             x='feature0',
                                             y='feature1',
                                             transform_x=None,
                                             transform_y=None,
                                             min_pop_size=min_pop_size)
        if return_data:
            return gate, example_data
        return gate

    def test_meta_assignment(self):
        test_df = pd.DataFrame({'A': [0, 1, 2, 3],
                                'labels': [0, 1, 2, 3],
                                'chunk_idx': [0, 0, 0, 0]})
        ref_df = pd.DataFrame({'chunk_idx': [0, 0, 1, 1],
                               'cluster': [0, 1, 0, 1],
                               'meta_cluster': ['0', '1', '0', '1']})
        err = False
        try:
            dbscan.meta_assignment(test_df, ref_df)
        except AssertionError:
            err = True
        self.assertTrue(err)

        test_df = pd.DataFrame({'A': [0, 1, 2, 3],
                                'labels': [0, 1, 0, 1],
                                'chunk_idx': [0, 0, 0, 0]})
        modified_df = dbscan.meta_assignment(test_df, ref_df)
        self.assertListEqual(list(modified_df.labels.values), ['0', '1', '0', '1'])

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
        gate = self._build()
        data = gate.data.copy()
        data = data[data.labels.isin([0., 1.0])]
        gate._post_cluster_checks(data)
        self.assertEqual(gate.warnings[0],
                         'Expected 3 populations, identified 2')

    def test_dbscan_knn(self):
        gate = self._build()
        gate._dbscan_knn(distance_nn=1.5, core_only=False)
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

    def _clustering(self, method='dbscan', eps=1.5, delta=2):
        gate, data = self._build(return_data=True, min_pop_size=delta)
        blob1_idx = data[data.blobID == 0].index.values
        blob2_idx = data[data.blobID == 1].index.values
        blob3_idx = data[data.blobID == 2].index.values
        if method == 'dbscan':
            populations = gate.dbscan(distance_nn=eps)
        else:
            populations = gate.hdbscan()
        for p, idx in zip(['blob1', 'blob2', 'blob3'], [blob1_idx, blob2_idx, blob3_idx]):
            self.assertListEqual(list(populations.populations[p].index), list(idx))

    def test_dbscan(self):
        self._clustering()

    def test_hdbscan(self):
        self._clustering(method='hdbscan', delta=5)


if __name__ == '__main__':
    unittest.main()
