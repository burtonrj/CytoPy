import sys
sys.path.append('/home/ross/CytoPy')

from data.subject import Subject
from data.mongo_setup import global_init
from flow.clustering import main
from tests.utilities import make_example_date
import unittest

global_init('test')


class TestExplore(unittest.TestCase):
    @staticmethod
    def _build():
        data = make_example_date(n_samples=100, centers=3, n_features=5)
        data['pt_id'] = 'test_pt'
        return main.Explorer(data=data)

    def test_drop(self):
        data = make_example_date(n_samples=100, centers=3, n_features=5)
        mask = data.blobID == 1
        e = self._build()
        e.drop_data(mask)
        y = data[~mask].index.values
        y_hat = e.data.index.values
        self.assertListEqual(list(y), list(y_hat))

    def test_load_meta(self):
        test_subject = Subject(subject_id='test_pt',
                               test=True)
        test_subject.save()
        e = self._build()
        e.load_meta('test')
        self.assertTrue('test' in e.data.columns)
        self.assertTrue(all(x is True for x in e.data.test))

    def test_dim_reduction(self):
        e = self._build()
        e.dimenionality_reduction(method='UMAP',
                                  features=[f'feature{i}' for i in range(5)])
        self.assertTrue('UMAP0' in e.data.columns)
        self.assertTrue('UMAP1' in e.data.columns)


if __name__ == '__main__':
    unittest.main()