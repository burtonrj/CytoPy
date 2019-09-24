from immunova.flow.gating.defaults import GateOutput, Geom
import pandas as pd


def rect_gate(data: pd.DataFrame, x: str, y: str,
              x_min: int or float, x_max: int or float,
              y_min: int or float, y_max: int or float,
              child_populations: dict) -> GateOutput:
    """
    Static rectangular gate
    :param data: parent population upon which the gate is applied
    :param x: name of the channel/marker for X dimension
    :param y: name of the channel/marker for Y dimension
    :param child_name:
    :param x_min: minimum value for x (x cooordinate for bottom left corner/top left corner)
    :param x_max: maximum value for x (x cooordinate for bottom right corner/top right corner)
    :param y_min: minimum value for y (y cooordinate for bottom left corner/bottom right corner)
    :param y_max: maximum value for y (y cooordinate for top right corner/top left corner)
    :param bool_gate: If True, return events NOT in gated population (return values outside of gate)
    :return: dictionary of gating outputs (see documentation for internal standards)
    """
    output = GateOutput()
    geom = Geom(shape='rect', x=x, y=y, x_min=x_min, x_max=x_max, y_min=y_min, y_max=y_max)
    pos_pop = data[(data[x] >= x_min) & (data[x] <= x_max)]
    name = [name for name, x in child_populations.items() if x['definition'] == '+'][0]
    output.add_child(name=name, idx=pos_pop.index.values, geom=geom)
    name = [name for name, x in child_populations.items() if x['definition'] == '-']
    if name:
        neg_pop = data[~data.index.isin(pos_pop.index.values)]
        output.add_child(name=name[0], idx=neg_pop.index.values, geom=geom)
    return output
