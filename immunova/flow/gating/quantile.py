from immunova.flow.gating.defaults import ChildPopulationCollection
from immunova.flow.gating.base import Gate, GateError
import pandas as pd


class Quantile(Gate):
    def __init__(self, data: pd.DataFrame, x: str, child_populations: ChildPopulationCollection,
                 q: float = 0.95, y: str or None = None):
        """
        Perform either 1D or 2D quantile gating
        :param data: pandas dataframe of fcs data for gating
        :param x: name of X dimension
        :param y: name of Y dimension (optional)
        :param child_populations: ChildPopulationCollection (see docs)
        :param q: quantile for calculating threshold (float value between 0 and 1)
        """
        super().__init__(data=data, x=x, y=y, child_populations=child_populations,
                         frac=None, downsample_method='uniform',
                         density_downsample_kwargs=None)
        self.y = y
        self.q = q

    def gate_1d(self):
        """
        Perform quantile gating in 1 dimensional space
        :return: Updated child populations
        """
        # If parent is empty just return the child populations with empty index array
        if self.empty_parent:
            return self.child_populations
        threshold = self.data[self.x].quantile(self.q, interpolation='nearest')
        self.__child_update_1d(threshold, 'Quantile', 'overwrite')
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
        x_threshold = self.data[self.x].quantile(self.q, interpolation='nearest')
        y_threshold = self.data[self.y].quantile(self.q, interpolation='nearest')
        self.__child_update_2d(x_threshold, y_threshold, method='Quantile')
        return self.child_populations
