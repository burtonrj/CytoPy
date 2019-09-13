import numpy as np
import pandas as pd
from sklearn.neighbors import KernelDensity
from flowutils.transforms import logicle, hyperlog, log_transform, asinh


def __boolean_gate(data, pos_pop, boolean_gate):
    if boolean_gate:
        return data[~data.index.isin(pos_pop.index)]
    return pos_pop


def check_peak(peaks: np.array, probs: np.array, t=0.01) -> np.array:
    """Check peaks against largest peak in list,
    if peak < t*largest peak, then peak is removed
    :param peaks: array of indices for peaks
    :param probs: array of probability values of density estimate
    :param t: height threshold as a percentage of highest peak"""
    if peaks.shape[0] == 1:
        return peaks
    sorted_peaks = np.sort(probs[peaks])[::-1]
    real_peaks = list()
    real_peaks.append(np.where(probs == sorted_peaks[0])[0][0])
    for p in sorted_peaks[1:]:
        if p >= t*sorted_peaks[0]:
            real_peaks.append(np.where(probs == p)[0][0])
    return np.sort(np.array(real_peaks))


def find_local_minima(probs: np.array, xx: np.array, peaks: np.array) -> float:
    """
    Find local minima between the two highest peaks in the density distribution provided
    :param probs: probability for density estimate
    :param xx: x values for corresponding probabilities
    :param peaks: array of indices for identified peaks
    :return: local minima between highest peaks
    """
    sorted_peaks = np.sort(probs[peaks])[::-1]
    if sorted_peaks[0] == sorted_peaks[1]:
        p1_idx, p2_idx = np.where(probs == sorted_peaks[0])[0]
    else:
        p1_idx = np.where(probs == sorted_peaks[0])[0][0]
        p2_idx = np.where(probs == sorted_peaks[1])[0][0]
    if p1_idx < p2_idx:
        between_peaks = probs[p1_idx:p2_idx]
    else:
        between_peaks = probs[p2_idx:p1_idx]
    local_min = min(between_peaks)
    return xx[np.where(probs == local_min)[0][0]]


def kde(data: pd.DataFrame, x: str,
        kde_bw: float, kernel='gaussian', frac=0.25) -> np.array:
    """
    Generate a kernel density estimation using the scikit-learn implementation
    :param data: data for smoothing
    :param x: column name for density estimation
    :param kde_bw: bandwidth
    :param kernel: kernel to use for estimation (see scikit-learn documentation)
    :param frac: sample size as a fraction of total dataset to perform density estimation on
    :return: probability densities for array of 1000 x-axis values between min and max of data
    """
    density = KernelDensity(bandwidth=kde_bw, kernel=kernel)
    d = data[x].sample(frac=frac).values
    density.fit(d[:, None])
    x_d = np.linspace(min(d), max(d), 1000)
    logprob = density.score_samples(x_d[:, None])
    return np.exp(logprob), x_d


def apply_transform(data: pd.DataFrame, features_to_transform: list,
                    transform_method: str, prescale=1) -> pd.DataFrame:
    """
    Apply a transformation to the given dataset; valid transformation methods are:
    logicle, hyperlog, log_transform, or asinh
    :param data: pandas dataframe
    :param features_to_transform: a list of features to transform
    :param transform_method: string value indicating transformation method
    :param prescale: if using asinh transformaion this value is passed as the scalling argument
    :return: transformed pandas dataframe
    """
    if transform_method == 'logicle':
        return logicle(data, features_to_transform)
    if transform_method == 'hyperlog':
        return hyperlog(data, features_to_transform)
    if transform_method == 'log_transform':
        return log_transform(data, features_to_transform)
    if transform_method == 'asinh':
        return asinh(data, features_to_transform, prescale)
    print('Error: invalid transform_method, returning untransformed data')
    return data