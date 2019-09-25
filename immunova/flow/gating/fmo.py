from immunova.flow.gating.utilities import check_peak, kde, find_local_minima
from immunova.flow.gating.density import multidem_density_output
from immunova.flow.gating.defaults import GateOutput, Geom
from scipy.signal import find_peaks
from scipy.stats import norm
import pandas as pd
import numpy as np


def density_1d_fmo(data: pd.DataFrame, fmo_x: pd.DataFrame, child_populations: dict,
                   x: str, kde_bw=0.01, kde_frac=0.5, q=0.99,
                   peak_t=0.01, fmo_z=2):
    """
    FMO guided density gating
    :param data:
    :param fmo_x:
    :param x:
    :param kde_bw:
    :param kde_frac:
    :param q:
    :param peak_t:
    :param fmo_z:
    :return:
    """
    def add_pop(pop, definition):
        name = [name for name, x_ in child_populations.items() if x_['definition'] == definition][0]
        output.add_child(name=name, idx=pop.index.values, geom=geom.as_dict())

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
    if fmo_x.shape[0] > 0:
        kde_and_peaks(fmo_x, 'fmo')

    if 'fmo' in density.keys():
        # Find the FMO threshold
        if density['fmo']['peaks'].shape[0] == 1:
            fmo_threshold = fmo_x[x].quantile(q)
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
            geom['threshold'] = fmo_threshold
            geom['method'] = 'FMO threshold (absolute)'
            neg_pop = data[~data.index.isin(pos_pop.index.values)]
            add_pop(pos_pop, '+')
            add_pop(neg_pop, '-')
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
            neg_pop = data[~data.index.isin(pos_pop.index.values)]
            add_pop(pos_pop, '+')
            add_pop(neg_pop, '-')
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
            neg_pop = data[~data.index.isin(pos_pop.index.values)]
            add_pop(pos_pop, '+')
            add_pop(neg_pop, '-')
            return output
        if density['whole']['peaks'].shape[0] > 1:
            # Find the region of minimum density between two highest peaks
            whole_threshold = find_local_minima(**density['whole'])
            geom['threshold'] = whole_threshold
            geom['method'] = 'Local minima from primary data'
            pos_pop = data[data[x] >= whole_threshold]
            neg_pop = data[~data.index.isin(pos_pop.index.values)]
            add_pop(pos_pop, '+')
            add_pop(neg_pop, '-')
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
    tc = {'neg': {'definition': '-'}, 'pos': {'definition': '+'}}
    fmo_result_x = density_1d_fmo(data, fmo_x, x=x, kde_bw=kde_bw, q=q, peak_t=peak_t, child_populations=tc)
    fmo_result_y = density_1d_fmo(data, fmo_y, x=y, kde_bw=kde_bw, q=q, peak_t=peak_t, child_populations=tc)
    # Check for errors
    for o in [fmo_result_x, fmo_result_y]:
        if o.error == 1:
            output.error = 1
            output.error_msg = x.error_msg
            return output

    # Update warnings
    output.warnings = fmo_result_x.warnings + fmo_result_y.warnings
    geom['threshold_x'] = fmo_result_x.child_populations['pos']['geom']['threshold']
    geom['threshold_y'] = fmo_result_y.child_populations['pos']['geom']['threshold']
    geom['method'] = fmo_result_x.child_populations['pos']['geom']['method'] + ' ' + \
                     fmo_result_y.child_populations['pos']['geom']['method']
    return multidem_density_output(child_populations, fmo_result_x, fmo_result_y, output, geom)
