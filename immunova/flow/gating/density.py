from immunova.flow.gating.utilities import boolean_gate, kde, check_peak, find_local_minima
from immunova.flow.gating.defaults import GateOutput, Geom
from immunova.flow.gating.quantile import quantile_gate
from scipy.signal import find_peaks
import pandas as pd
import numpy as np


def density_gate_1d(data: pd.DataFrame, x: str, child_name: str,
                    bool_gate=False, q=0.95,
                    std=None, kde_bw=0.01, kde_sample_frac=0.25,
                    peak_threshold=None, ignore_double_pos=False) -> GateOutput:
    """
    1-Dimensional gating using density based methods
    :param data: pandas dataframe containing compensated and transformed flow cytometry data
    :param x: name of column to gate
    :param child_name:
    :param bool_gate: if False, the positive population is returned (>= threshold) else the negative population
    :param q: if only 1 peak is found, quantile gating is performed using this argument as the quantile
    :param std: alternative to quantile gating, the number of standard deviations from the mean can be used to
    determine the threshold
    :param kde_bw: bandwidth for gaussian kernel density smoothing
    :param kde_sample_frac: estimating the kernel density can be computationally expensive. By default this
    is estimated using a sample of the data. This parameter defines the fraction of data to use for kde estimation
    :param peak_threshold: if not None, then this value should be a float. This decimal value represents what the
    minimum height of a peak should be relevant to the highest peak found (e.g. if peak_threshold=0.05, then all peaks
    with a height < 0.05 of the heighest peak will be ignored)
    :param ignore_double_pos: if True, in the case that multiple peaks are detected, peaks to the right of
    the highest peak will be ignored in the local minima calculation
    :return: dictionary of gating outputs (see documentation for internal standards)
    """
    output = GateOutput()
    # Smooth the data with a kde
    probs, xx = kde(data, x, kde_bw, frac=kde_sample_frac)
    # Find peaks
    peaks = find_peaks(probs)[0]
    if peak_threshold:
        peaks = check_peak(peaks, probs, peak_threshold)
    if len(peaks) == 1:
        # If a quantile has been specified, then use quantile gate
        if q:
            return quantile_gate(data, child_name=child_name, x=x, q=q, bool_gate=bool_gate)
        if std:
            u = data[x].mean()
            s = data[x].std()
            threshold = u+(s*std)
            pos_pop = data[data[x] >= threshold]
            pos_pop = data[~data.index.isin(pos_pop.index)]
            geom = Geom(shape='threshold', x=x, y=None, method=f'>= {std} Standard Devs', threshold=threshold)
            output.add_child(name=child_name, idx=pos_pop.index.values, geom=geom)
            return output
    if len(peaks) > 1:
        if ignore_double_pos:
            probs_peaks = probs[peaks]
            highest_peak = np.where(probs_peaks == max(probs_peaks))[0][0]
            if highest_peak < len(peaks):
                if len(peaks[:highest_peak+1]) > 1:
                    peaks = peaks[:highest_peak+1]
        threshold = find_local_minima(probs, xx, peaks)
        geom = Geom(shape='threshold', x=x, y=None, threshold=threshold, method='Local minima; two highest peaks')
        pos_pop = data[data[x] >= threshold]
        pos_pop = boolean_gate(data, pos_pop, bool_gate)
        output.add_child(name=child_name, idx=pos_pop.index.values, geom=geom)
        return output
    output.error = 1
    output.error_msg = 'No peaks found!'
    return output


def density_gate_2d(data, x, y, child_populations: dict, kde_bw=0.05, q=0.99, peak_t=0.01):
        """
        2-dimensional density gating
        :param data:
        :param fmo_x:
        :param fmo_y:
        :param x:
        :param y:
        :param child_populations:
        :param kde_bw:
        :param q:
        :param peak_t:
        :return:
        """
        output = GateOutput()
        geom = Geom(shape='2d_threshold', x=x, y=y, threshold_x=None, threshold_y=None, method=None)
        result_x = density_gate_1d(data, child_name=x, x=x, kde_bw=kde_bw, q=q, peak_threshold=peak_t)
        result_y = density_gate_1d(data, child_name=y, x=y, kde_bw=kde_bw, q=q, peak_threshold=peak_t)
        # Check for errors
        for x in [result_x, result_y]:
            if x.error == 1:
                output.error = 1
                output.error_msg = x.error_msg
                return output

        # Update warnings
        output.warnings = result_x.warnings + result_y.warnings
        geom['threshold_x'] = result_x.child_populations[x]['geom']['threshold']
        geom['threshold_y'] = result_y.child_populations[y]['geom']['threshold']
        geom['method'] = result_x.child_populations[x]['geom']['method'] + ' ' + \
                         result_y.child_populations[y]['geom']['method']

        # Name populations
        x_idx = result_x.child_populations[x]['index']
        y_idx = result_y.child_populations[y]['index']
        for name, definition in child_populations.items():
            if definition == '++':
                pos_idx = np.intersect1d(x_idx, y_idx)
                output.add_child(name=name, idx=pos_idx, geom=geom)
            elif definition == '--':
                x_idx = data[~data.index.isin(x_idx)].index.values
                y_idx = data[~data.index.isin(y_idx)].index.values
                pos_idx = np.intersect1d(x_idx, y_idx)
            elif definition == '+-':
                y_idx = data[~data.index.isin(y_idx)].index.values
                pos_idx = np.intersect1d(x_idx, y_idx)
            elif definition == '-+':
                x_idx = data[~data.index.isin(x_idx)].index.values
                pos_idx = np.intersect1d(x_idx, y_idx)
            else:
                output.error = 1
                output.error_msg = f'Error: invalid child population definition for {name}, must be one of: ++, +-, -+, --'
                return output
            output.add_child(name=name, idx=pos_idx, geom=geom, merge_options='merge')
        return output
