import sys
sys.path.append('/home/ross/CytoPy')

# Data imports
from cytopy.data.mongo_setup import global_init
from cytopy.tests.utilities import make_example_date
from cytopy.flow.gating.base import Gate
from cytopy.flow.gating.defaults import ChildPopulationCollection
import numpy as np
import unittest

global_init('test')


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

    def _index_updates(self, g, pos, data):
        neg = [i for i in data.index.values if i not in pos]
        self.assertListEqual(list(g.child_populations.populations['positive'].index), pos)
        self.assertListEqual(list(g.child_populations.populations['negative'].index), neg)

    def test_1d_update(self):
        # 1-D gate
        (gate, _), data = self._build_gate()
        pos_idx = [1,  3,  5,  7,  8, 26, 31, 32, 38, 39, 42, 45, 49, 50, 51, 58, 61,
                   63, 66, 68, 69, 70, 74, 76, 78, 79, 81, 87, 89, 90, 91, 93, 97]
        gate.child_update_1d(threshold=0.3, method='test')
        self._index_updates(gate, pos_idx, data)

        gate.child_update_1d(threshold=4, method='test')
        pos_idx = [1,  3,  7,  8, 31, 32, 38, 39, 42, 51, 58, 66, 68, 69, 70, 74, 76,
                   78, 89, 90, 91, 93, 97]
        gate.child_update_1d(threshold=4, method='test')
        self._index_updates(gate, pos_idx, data)

        pos_idx = []
        gate.child_update_1d(threshold=9, method='test')
        self._index_updates(gate, pos_idx, data)

    def test_2d_update(self):
        # 2-D gate
        (gate1, gate2), data = self._build_gate(dimensions=2)
        pos_idx = [1,  3,  5,  7,  8, 26, 31, 32, 38, 39, 42, 45, 49, 50, 51, 58, 61,
                   63, 66, 68, 69, 70, 74, 76, 78, 79, 81, 87, 89, 90, 91, 93, 97]
        gate1.child_update_2d(x_threshold=2, y_threshold=-2.5, method='test')
        self._index_updates(gate1, pos_idx, data)

        pos_idx = [1, 3, 38, 58, 76, 78, 91, 93]
        gate1.child_update_2d(x_threshold=5, y_threshold=-2.5, method='test')
        self._index_updates(gate1, pos_idx, data)

        pos_idx = [1,  2,  3,  5,  6,  7,  8,  9, 10, 13, 14, 17, 20, 23, 24, 25, 26,
                   31, 32, 34, 35, 36, 37, 38, 39, 41, 42, 43, 44, 45, 49, 50, 51, 52,
                   56, 57, 58, 59, 61, 63, 66, 68, 69, 70, 73, 74, 76, 78, 79, 80, 81,
                   82, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 99]
        gate2.child_update_2d(x_threshold=2, y_threshold=-2.5, method='test')
        self._index_updates(gate2, pos_idx, data)

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
        example_data['labels'] = example_data['blobID']
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


if __name__ == '__main__':
    unittest.main()