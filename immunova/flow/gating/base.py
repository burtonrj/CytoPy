from immunova.flow.gating.defaults import GateOutput, Geom
import pandas as pd


class Gate:
    """
    Base class for gate definition.

    Attributes:
        - data: pandas dataframe of fcs data for gating
        - x: name of X dimension
        - y: name of Y dimension (optional)
        - child populations: child_populations: dictionary of expected child populations (must conform to standards for
        child_population dictionary; see documentation for details ToDO link)
        - output: GateOutput object for standard gating output
    """
    def __init__(self, data: pd.DataFrame, x: str, y: str or None, child_populations: dict):
        """
        Constructor for Gate definition
        :param data: pandas dataframe of fcs data for gating
        :param x: name of X dimension
        :param y: name of Y dimension (optional)
        :param child_populations: dictionary of expected child populations (must conform to standards for
        child_population dictionary; see documentation for details ToDO link)
        """
        self.data = data
        self.x = x
        self.y = y
        self.child_populations = self.__validate_child_populations(child_populations)
        self.output = GateOutput()

