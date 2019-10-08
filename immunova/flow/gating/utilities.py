import numpy as np
import pandas as pd
from sklearn.neighbors import KernelDensity
from flowutils.transforms import logicle, hyperlog, log_transform, asinh


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


def inside_ellipse(data: np.array, center: tuple,
                   width: int or float, height: int or float,
                   angle: int or float) -> object:
    """
    Return mask of two dimensional matrix specifying if a data point (row) falls
    within an ellipse
    :param data - two dimensional matrix (x,y)
    :param center - tuple of x,y coordinate corresponding to center of elipse
    :param width - semi-major axis of eplipse
    :param height - semi-minor axis of elipse
    :param angle - angle of ellipse
    :return numpy array of indices for values inside specified ellipse
    """
    cos_angle = np.cos(np.radians(180.-angle))
    sin_angle = np.sin(np.radians(180.-angle))

    x = data[:, 0]
    y = data[:, 1]

    xc = x - center[0]
    yc = y - center[1]

    xct = xc * cos_angle - yc * sin_angle
    yct = xc * sin_angle + yc * cos_angle

    rad_cc = (xct ** 2 / (width / 2.)**2) + (yct**2 / (height / 2.)**2)

    in_ellipse = []

    for r in rad_cc:
        if r <= 1.:
            # point in ellipse
            in_ellipse.append(True)
        else:
            # point not in ellipse
            in_ellipse.append(False)
    return in_ellipse


def rectangular_filter(data: pd.DataFrame, x: str, y: str, definition: dict) -> pd.DataFrame or str:
    """
    Given a pandas dataframe of fcs events data and a definition for a rectangular geom,
    filter the pandas dataframe and return only data contained within the rectangular geometric 2D plane
    :param data: pandas dataframe of fcs data to filter
    :param y: name of Y dimension (channel/marker name for column)
    :param x: name of X dimension (channel/marker name for column)
    :param definition: dictionary with keys: ['xmin', 'xmax', 'ymin', 'ymax'] each of integer/float value; see
    static.rect_gate for conventions
    :return: filtered pandas dataframe
    """
    if not all([x in ['xmin', 'xmax', 'ymin', 'ymax'] for x in definition.keys()]):
        return 'Error: invalid definition; must be dict with keys: xmin, xmax, ymin, ymax'
    data = data[(data[x] >= definition['xmin']) & (data[x] <= definition['xmax'])]
    data = data[(data[y] >= definition['ymin']) & (data[y] <= definition['ymax'])]
    return data





