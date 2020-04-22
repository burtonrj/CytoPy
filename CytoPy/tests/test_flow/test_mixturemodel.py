import sys
sys.path.append('/home/ross/CytoPy')

# Data imports
from CytoPy.data.mongo_setup import global_init
from CytoPy.flow import ChildPopulationCollection
from CytoPy.flow.gating import mixturemodel
from CytoPy.tests import make_example_date
import unittest

global_init('test')


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


if __name__ == '__main__':
    unittest.main()