from flow.gating.utilities import boolean_gate
import pandas as pd


def rect_gate(data: pd.DataFrame, x: str, y: str,
              x_min: int or float, x_max: int or float,
              y_min: int or float, y_max: int or float,
              bool_gate: bool = False) -> dict:
    """
    Static rectangular gate
    :param data: parent population upon which the gate is applied
    :param x: name of the channel/marker for X dimension
    :param y: name of the channel/marker for Y dimension
    :param x_min: minimum value for x (x cooordinate for bottom left corner/top left corner)
    :param x_max: maximum value for x (x cooordinate for bottom right corner/top right corner)
    :param y_min: minimum value for y (y cooordinate for bottom left corner/bottom right corner)
    :param y_max: maximum value for y (y cooordinate for top right corner/top left corner)
    :param bool_gate: If True, return events NOT in gated population (return values outside of gate)
    :return: Gate output dictionary (see docs for gating conventions)
    """
    output = dict(pos_index=None, warnings=[], error=0, error_msg=None, geom=dict(shape='rect',
                                                                                  x_min=x_min,
                                                                                  x_max=x_max,
                                                                                  y_min=y_min,
                                                                                  y_max=y_max))

    pos_pop = data[(data[x] >= x_min) & (data[x] <= x_max)]
    pos_pop = boolean_gate(data, pos_pop[(pos_pop[y] >= y_min) & (pos_pop[y] <= y_max)], bool_gate)
    output['pos_index'] = pos_pop.index.values
    return output
