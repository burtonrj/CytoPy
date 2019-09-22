from immunova.flow.gating.utilities import check_peak, boolean_gate, kde, find_local_minima
from immunova.flow.gating.defaults import GateOutput, Geom
from scipy.signal import find_peaks
from scipy.stats import norm
import pandas as pd
import numpy as np


def density_1d_fmo(data: pd.DataFrame, fmo: pd.DataFrame, child_name: str,
                   x: str, bool_gate=False, kde_bw=0.01,
                   kde_frac=0.5, q=0.99, peak_t=0.01, fmo_z=2):
    """
    FMO guided density gating
    :param data:
    :param fmo:
    :param child_name:
    :param x:
    :param bool_gate:
    :param kde_bw:
    :param kde_frac:
    :param q:
    :param peak_t:
    :param fmo_z:
    :return:
    """
    output = GateOutput()
    geom = Geom(shape='threshold', x=x, y=None, threshold=None, method=None)
    # KDE Smoothing and peak detection
    density = {}

    def kde_and_peaks(d, k):
        probs, xx_ = kde(d, x, kde_bw=kde_bw, kernel='gaussian', frac=kde_frac)
        peaks = find_peaks(probs)[0]
        peaks = check_peak(peaks, probs, peak_t)
        density[k] = dict(probs=probs, xx=xx_, peaks=peaks)

    kde_and_peaks(data, 'whole')
    if fmo.shape[0] > 0:
        kde_and_peaks(fmo, 'fmo')

    if 'fmo' in density.keys():
        # Find the FMO threshold
        if density['fmo']['peaks'].shape[0] == 1:
            fmo_threshold = fmo[x].quantile(q)
        elif density['fmo']['peaks'].shape[0] > 1:
            # Find local minima
            fmo_threshold = find_local_minima(**density['fmo'])
        else:
            output.error = 1
            output.error_msg = 'No peaks found'
            return output

        if density['whole']['peaks'].shape[0] == 1:
            # Use the FMO as a definitive cutoff
            geom['threshold'] = fmo_threshold
            geom['method'] = 'FMO threshold'
            pos_pop = data[data[x] >= fmo_threshold]
            pos_pop = boolean_gate(data, pos_pop, bool_gate)
            geom['threshold'] = fmo_threshold
            geom['method'] = 'FMO threshold (absolute)'
            output.add_child(name=child_name, idx=pos_pop.index.values, geom=geom)
            return output

        if density['whole']['peaks'].shape[0] > 1:
            # Find the region of minimum density between two highest peaks
            whole_threshold = find_local_minima(**density['whole'])
            # If FMO threshold z-score >3 flag to user
            p = norm.cdf(x=fmo_threshold, loc=whole_threshold, scale=0.1)
            z_score = norm.ppf(p)
            if abs(z_score) >= fmo_z:
                output.warnings.append("""FMO threshold z-score >2 (see documentation); the threshold
                as determined by the FMO is a significant distance from the region of minimum density between the
                two highest peaks see in the whole pane. Manual review of gating is advised.""")
                geom['threshold'] = whole_threshold
                geom['method'] = 'Local minima from primary data'
                pos_pop = data[data[x] >= whole_threshold]
            else:
                # Take the median value for the interval between the fmo and local minima
                xx = density['whole']['xx']
                if fmo_threshold > whole_threshold:
                    ot = np.median(xx[np.where(np.logical_and(xx > whole_threshold, xx < fmo_threshold))[0]])
                else:
                    ot = np.median(xx[np.where(np.logical_and(xx > fmo_threshold, xx < whole_threshold))[0]])
                geom['threshold'] = ot
                geom['method'] = 'Local minima; fmo guided'
                pos_pop = data[data[x] >= ot]
            pos_pop = boolean_gate(data, pos_pop, bool_gate)
            output.add_child(name=child_name, idx=pos_pop.index.values, geom=geom)
            return output
        else:
            output.error = 1
            output.error_msg = 'No peaks found. Is this dataset empty?'
            return output
    else:
        # No FMO data, calculate using primary data and add warning
        output.warnings.append(f'No FMO data provided for {x}, density gating calculated from primary data')
        if density['whole']['peaks'].shape[0] == 1:
            pos_pop = data[data[x] >= data[x].quantile(q)]
            pos_pop = boolean_gate(data, pos_pop, bool_gate)
            output.add_child(name=child_name, idx=pos_pop.index.values, geom=geom)
            return output
        if density['whole']['peaks'].shape[0] > 1:
            # Find the region of minimum density between two highest peaks
            whole_threshold = find_local_minima(**density['whole'])
            geom['threshold'] = whole_threshold
            geom['method'] = 'Local minima from primary data'
            pos_pop = data[data[x] >= whole_threshold]
            pos_pop = boolean_gate(data, pos_pop, bool_gate)
            output.add_child(name=child_name, idx=pos_pop.index.values, geom=geom)
            return output
        else:
            output.error = 1
            output.error_msg = 'No peaks found. Is this dataset empty?'
            return output


def density_2d_fmo(data, fmo_x, fmo_y, x, y, child_populations: dict,
                   kde_bw=0.05, q=0.99, peak_t=0.01):
    """
    FMO guided density based gating for two dimensional data
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
    fmo_result_x = density_1d_fmo(data, fmo_x, child_name=f'{x}_fmo', x=x, kde_bw=kde_bw, q=q, peak_t=peak_t)
    fmo_result_y = density_1d_fmo(data, fmo_y, child_name=f'{y}_fmo', x=y, kde_bw=kde_bw, q=q, peak_t=peak_t)
    # Check for errors
    for x in [fmo_result_x, fmo_result_y]:
        if x.error == 1:
            output.error = 1
            output.error_msg = x.error_msg
            return output

    # Update warnings
    output.warnings = fmo_result_x.warnings + fmo_result_y.warnings
    geom['threshold_x'] = fmo_result_x.child_populations[f'{x}_fmo']['geom']['threshold']
    geom['threshold_y'] = fmo_result_y.child_populations[f'{y}_fmo']['geom']['threshold']
    geom['method'] = fmo_result_x.child_populations[f'{x}_fmo']['geom']['method'] + ' ' + \
                     fmo_result_y.child_populations[f'{y}_fmo']['geom']['method']

    # Name populations
    x_fmo_idx = fmo_result_x.child_populations[f'{x}_fmo']['index']
    y_fmo_idx = fmo_result_y.child_populations[f'{y}_fmo']['index']
    for name, definition in child_populations.items():
        if definition == '++':
            pos_idx = np.intersect1d(x_fmo_idx, y_fmo_idx)
            output.add_child(name=name, idx=pos_idx, geom=geom)
        elif definition == '--':
            x_idx = data[~data.index.isin(x_fmo_idx)]
            y_idx = data[~data.index.isin(y_fmo_idx)]
            pos_idx = np.intersect1d(x_idx, y_idx)
        elif definition == '+-':
            y_idx = data[~data.index.isin(y_fmo_idx)]
            pos_idx = np.intersect1d(x_fmo_idx, y_idx)
        elif definition == '-+':
            x_idx = data[~data.index.isin(x_fmo_idx)]
            pos_idx = np.intersect1d(x_idx, y_fmo_idx)
        else:
            output.error = 1
            output.error_msg = f'Error: invalid child population definition for {name}, must be one of: ++, +-, -+, --'
            return output
        output.add_child(name=name, idx=pos_idx, geom=geom)
    return output

def density_2d_fmo(data, fmo_x, fmo_y, x, y, child_populations: dict,
                   kde_bw=0.05, q=0.99, peak_t=0.01):
    """
    FMO guided density based gating for two dimensional data
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
    fmo_result_x = density_1d_fmo(data, fmo_x, child_name=f'{x}_fmo', x=x, kde_bw=kde_bw, q=q, peak_t=peak_t)
    fmo_result_y = density_1d_fmo(data, fmo_y, child_name=f'{y}_fmo', x=y, kde_bw=kde_bw, q=q, peak_t=peak_t)
    # Check for errors
    for x in [fmo_result_x, fmo_result_y]:
        if x.error == 1:
            output.error = 1
            output.error_msg = x.error_msg
            return output

    # Update warnings
    output.warnings = fmo_result_x.warnings + fmo_result_y.warnings
    geom['threshold_x'] = fmo_result_x.child_populations[f'{x}_fmo']['geom']['threshold']
    geom['threshold_y'] = fmo_result_y.child_populations[f'{y}_fmo']['geom']['threshold']
    geom['method'] = fmo_result_x.child_populations[f'{x}_fmo']['geom']['method'] + ' ' + \
                     fmo_result_y.child_populations[f'{y}_fmo']['geom']['method']

    # Name populations
    x_fmo_idx = fmo_result_x.child_populations[f'{x}_fmo']['index']
    y_fmo_idx = fmo_result_y.child_populations[f'{y}_fmo']['index']
    for name, definition in child_populations.items():
        if definition == '++':
            pos_idx = np.intersect1d(x_fmo_idx, y_fmo_idx)
            output.add_child(name=name, idx=pos_idx, geom=geom)
        elif definition == '--':
            x_idx = data[~data.index.isin(x_fmo_idx)].index.values
            y_idx = data[~data.index.isin(y_fmo_idx)].index.values
            pos_idx = np.intersect1d(x_idx, y_idx)
        elif definition == '+-':
            y_idx = data[~data.index.isin(y_fmo_idx)].index.values
            pos_idx = np.intersect1d(x_fmo_idx, y_idx)
        elif definition == '-+':
            x_idx = data[~data.index.isin(x_fmo_idx)].index.values
            pos_idx = np.intersect1d(x_idx, y_fmo_idx)
        else:
            output.error = 1
            output.error_msg = f'Error: invalid child population definition for {name}, must be one of: ++, +-, -+, --'
            return output
        output.add_child(name=name, idx=pos_idx, geom=geom)
    return output
