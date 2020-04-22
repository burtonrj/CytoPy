import sys
sys.path.append('/home/ross/CytoPy')

from CytoPy.data.mongo_setup import global_init
from CytoPy.flow.clustering import main
from sklearn.cluster import AgglomerativeClustering, KMeans
import unittest

global_init('test')


class TestFilterDict(unittest.TestCase):
    def test(self):
        test_dict = {'x': [1, 2, 3],
                     'y': [4, 5, 6]}
        self.assertListEqual(list(main.filter_dict(test_dict, ['y']).keys()),
                             ['y'])


class TestFetchClusteringClass(unittest.TestCase):

    def test1(self):
        params = dict(cluster_class='kmeans',
                      x='x')
        cluster, params = main._fetch_clustering_class(params)
        self.assertEqual(type(cluster), KMeans)

    def test2(self):
        params = dict(cluster_class='agglomerative',
                      x='x')
        cluster, params = main._fetch_clustering_class(params)
        self.assertEqual(type(cluster), AgglomerativeClustering)
        self.assertListEqual(list(params.keys()), ['x'])


if __name__ == '__main__':
    unittest.main()
