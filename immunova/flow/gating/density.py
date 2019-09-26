from immunova.flow.gating.utilities import kde, check_peak, find_local_minima
from immunova.flow.gating.defaults import GateOutput, Geom
from scipy.signal import find_peaks
import pandas as pd
import numpy as np


def density_gate_1d(data: pd.DataFrame, x: str, child_populations: dict, q=0.95,
                    std=None, kde_bw=0.01, kde_sample_frac=0.25,
                    peak_threshold=None, ignore_double_pos=False) -> GateOutput:
    """
    1-Dimensional gating using density based methods
    :param data: pandas dataframe containing compensated and transformed flow cytometry data
    :param x: name of column to gate
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
    def add_pop(pop, definition):
        name = [name for name, x_ in child_populations.items() if x_['definition'] == definition][0]
        output.add_child(name=name, idx=pop.index.values, geom=geom.as_dict())

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
            threshold = data[x].quantile(q)
            pos_pop = data[data[x] >= threshold]
        elif std:
            u = data[x].mean()
            s = data[x].std()
            threshold = u+(s*std)
            pos_pop = data[data[x] >= threshold]
        else:
            output.error = 1
            output.error_msg = 'No quantile or standard deviation provided, unable to perform gating'
            return output
        geom = Geom(shape='threshold', x=x, y=None, method=f'>= {std} Standard Devs', threshold=np.float64(threshold))
        neg_pop = data[~data.index.isin(pos_pop.index.values)]
        add_pop(pos_pop, '+')
        add_pop(neg_pop, '-')
        return output
    if len(peaks) > 1:
        if ignore_double_pos:
            probs_peaks = probs[peaks]
            highest_peak = np.where(probs_peaks == max(probs_peaks))[0][0]
            if highest_peak < len(peaks):
                if len(peaks[:highest_peak+1]) > 1:
                    peaks = peaks[:highest_peak+1]
        threshold = find_local_minima(probs, xx, peaks)
        geom = Geom(shape='threshold', x=x, y=None, threshold=np.float64(threshold), method='Local minima; two highest peaks')
        pos_pop = data[data[x] >= threshold]
        neg_pop = data[data[x] < threshold]
        add_pop(pos_pop, '+')
        add_pop(neg_pop, '-')
        return output
    output.error = 1
    output.error_msg = 'No peaks found!'
    return output


def density_gate_2d(data, x, y, child_populations, kde_bw=0.05, q=0.99, peak_t=0.01):
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
        tc = {'neg': {'definition': '-'}, 'pos': {'definition': '+'}}
        result_x = density_gate_1d(data, x=x, kde_bw=kde_bw, q=q, peak_threshold=peak_t, child_populations=tc)
        result_y = density_gate_1d(data, x=y, kde_bw=kde_bw, q=q, peak_threshold=peak_t, child_populations=tc)
        # Check for errors
        for o in [result_x, result_y]:
            if o.error == 1:
                output.error = 1
                output.error_msg = x.error_msg
                return output

        # Update warnings
        output.warnings = result_x.warnings + result_y.warnings
        geom['threshold_x'] = result_x.child_populations['pos']['geom']['threshold']
        geom['threshold_y'] = result_y.child_populations['pos']['geom']['threshold']
        geom['method'] = result_x.child_populations['pos']['geom']['method'] + ' ' + \
                         result_y.child_populations['pos']['geom']['method']
        return multidem_density_output(child_populations, result_x, result_y, output, geom)


def multidem_density_output(child_populations, result_x, result_y, output, geom):
    # Name populations
    x_idx_pos = result_x.child_populations['pos']['index']
    y_idx_pos = result_y.child_populations['pos']['index']
    x_idx_neg = result_x.child_populations['neg']['index']
    y_idx_neg = result_y.child_populations['neg']['index']

    for name, d in child_populations.items():
        definition = d['definition']
        idx = []
        if definition == '++':
            idx = np.intersect1d(x_idx_pos, y_idx_pos)
        elif definition == '--':
            idx = np.intersect1d(x_idx_neg, y_idx_neg)
        elif definition == '+-':
            idx = np.intersect1d(x_idx_pos, y_idx_neg)
        elif definition == '-+':
            idx = np.intersect1d(x_idx_neg, y_idx_pos)
        output.add_child(name=name, idx=idx, geom=geom.as_dict(), merge_options='merge')
    return output
