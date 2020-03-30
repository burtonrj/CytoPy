import sys
sys.path.append('/home/ross/CytoPy')

# Data imports
from cytopy.data.mongo_setup import global_init
from cytopy.flow.gating.defaults import ChildPopulationCollection
from cytopy.flow.gating import static
from cytopy.tests.utilities import make_example_date
import unittest

global_init('test')


class TestStatic(unittest.TestCase):

    @staticmethod
    def _build(populations: ChildPopulationCollection,
               return_data: bool = False,
               n=1000,
               **kwargs):
        example_data = make_example_date(n_samples=n, centers=3)

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
        y_hat = list(populations.populations['positive'].index)
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
        y_hat = list(populations.populations['positive'].index)
        self.assertListEqual(list(y), y_hat)

    def test_ellipse(self):
        populations = ChildPopulationCollection(gate_type='geom')
        populations.add_population('positive', definition='+')
        populations.add_population('negative', definition='-')
        gate, data = self._build(populations=populations,
                                 return_data=True, n=100)
        y = data[data.blobID == 2.0].index.values
        y_hat = gate.ellipse_gate(centroid=(-7, -7),
                                  width=5,
                                  height=8,
                                  angle=0).populations['positive'].index
        self.assertListEqual(list(y), list(y_hat))


if __name__ == '__main__':
    unittest.main()
