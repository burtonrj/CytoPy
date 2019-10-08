from immunova.flow.gating.defaults import ChildPopulationCollection
import pandas as pd
import numpy as np


class Gate:
    """
    Base class for gate definition.

    Attributes:
        - data: pandas dataframe of fcs data for gating
        - x: name of X dimension
        - y: name of Y dimension (optional)
        - child populations: ChildPopulationCollection (see docs)
        - output: GateOutput object for standard gating output
    """
    def __init__(self, data: pd.DataFrame, x: str, y: str or None, child_populations: ChildPopulationCollection):
        """
        Constructor for Gate definition
        :param data: pandas dataframe of fcs data for gating
        :param x: name of X dimension
        :param y: name of Y dimension (optional)
        :param child_populations: ChildPopulationCollection (see docs)
        """
        self.data = data
        self.x = x
        self.y = y
        self.child_populations = child_populations
        self.error = False
        self.error_msg = None
        self.warnings = list()



