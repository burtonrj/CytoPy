from immunova.flow.gating.defaults import ChildPopulationCollection
from immunova.flow.gating.base import Gate, GateError
import pandas as pd


class Quantile(Gate):
    def __init__(self, q: float = 0.95, **kwargs):
        """
        Perform either 1D or 2D quantile gating
        :param q: quantile for calculating threshold (float value between 0 and 1)
        :param kwargs: Gate constructor arguments (see immunova.flow.gating.base)
        """
        super().__init__(**kwargs)
        self.q = q

    def gate_1d(self):
        """
        Perform quantile gating in 1 dimensional space
        :return: Updated child populations
        """
        # If parent is empty just return the child populations with empty index array
        if self.empty_parent:
            return self.child_populations
        threshold = float(self.data[self.x].quantile(self.q, interpolation='nearest'))
        self.child_update_1d(threshold, 'Quantile', 'overwrite')
        return self.child_populations

    def gate_2d(self):
        """
        Perform quantile gating in 2 dimensional space
        :return: Updated child populations
        """
        # If parent is empty just return the child populations with empty index array
        if self.empty_parent:
            return self.child_populations
        if self.y is None:
            raise GateError('Value for `y` cannot be None if performing 2D gating')
        x_threshold = float(self.data[self.x].quantile(self.q, interpolation='nearest'))
        y_threshold = float(self.data[self.y].quantile(self.q, interpolation='nearest'))
        self.child_update_2d(x_threshold, y_threshold, method='Quantile')
        return self.child_populations
