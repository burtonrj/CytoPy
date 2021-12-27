from typing import Tuple
from typing import Union

import numpy as np
from detecta import detect_peaks
from KDEpy import FFTKDE


def kde_and_peak_finding(
    x: np.ndarray, kernel: str, bw: Union[str, float], min_peak_threshold: float, peak_boundary: float
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Fit a kernel density estimate and estimate peaks using detecta.detect_peaks

    Parameters
    ----------
    x: Numpy.Array
    kernel: Union[str, float]
        See KDEpy.FFTKDE for details
    bw: Union[str, float]
        See KDEpy.FFTKDE for details
    min_peak_threshold: float
        See detecta.detect_peaks
    peak_boundary: float
        See detecta.detect_peaks

    Returns
    -------
    Tuple[np.ndarray, np.ndarray, np.ndarray]
    """
    x_grid, p = FFTKDE(kernel=kernel, bw=bw).fit(x).evaluate()
    peaks = detect_peaks(p, mph=p[np.argmax(p)] * min_peak_threshold, mpd=len(p) * peak_boundary)
    return peaks, x_grid, p


def silverman(x):
    iqr = np.percentile(x, 75) - np.percentile(x, 25)
    return 0.9 * np.min([np.std(x), iqr / 1.34]) * len(x) ** (-1 / 5)
