from .base import Gate, GateError


class Quantile(Gate):
    """
    Perform either 1D or 2D quantile gating

    Parameters
    -----------
    q: float
        quantile to act as threshold (float value between 0 and 1)
    kwargs:
        Gate constructor arguments (see cytopy.flow.gating.base)
    """
    def __init__(self,
                 q: float = 0.95,
                 **kwargs):
        super().__init__(**kwargs)
        self.q = q

    def gate_1d(self):
        """
        Perform quantile gating in 1 dimensional space

        Returns
        -------
        ChildPopulationCollection
            Updated child populations
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

        Returns
        --------
        ChildPopulationCollection
            Updated child populations
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
