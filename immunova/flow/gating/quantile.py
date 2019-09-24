import pandas as pd
from immunova.flow.gating.utilities import boolean_gate
from immunova.flow.gating.defaults import GateOutput, Geom


def quantile_gate(data: pd.DataFrame, x: str,
                  q: float or list, y=None) -> GateOutput:
    """
    Quantile gate
    :param data: parent population upon which the gate is applied
    :param x: x-axis dimension (string value for corresponding column)
    :param q: quantile to draw threshold
    :param y: y-axis dimension (string value for corresponding column) default = None
    :return: dictionary of gating outputs (see documentation for internal standards)
    """
    output = GateOutput()
    pos_pop = pd.DataFrame()
    qt = None
    if q > 1.0:
        q = q/100.0
    if not y:
        qt = data[x].quantile(q, interpolation='nearest')
        pos_pop = data[data[x] >= qt]
    if y:
        if type(q) != list:
            output.error = 1
            output.error_msg = 'If 2d gate, q must be of type length e.g. [0.95, 0.95]'
            return output
        qt1 = data[x].quantile(q[0], interpolation='nearest')
        qt2 = data[y].quantile(q[1], interpolation='nearest')
        pos_pop = data[(data[x] >= qt1) & (data[y] >= qt2)]
        qt = (qt1, qt2)
    if len(pos_pop) == 0 or len(pos_pop) == data.shape[0]:
        output.warnings.append('No events in gate')

    geom = Geom(shape='threshold', x=x, y=None, threshold=qt, method='quantile')
    output.add_child(name=f'{x}+', idx=pos_pop.index.values, geom=geom)
    neg_pop = data[~data.index.isin(pos_pop.index.values)]
    output.add_child(name=f'{x}-', idx=neg_pop.index.values, geom=geom)
    return output
