from skfda.preprocessing.registration import landmark_registration
from skfda.representation.grid import FDataGrid
from multiprocessing import Pool, cpu_count
from functools import partial
from ..data.fcs import FileGroup
from detecta import detect_peaks
from KDEpy import FFTKDE
import pandas as pd
import numpy as np


def peaks(y: np.ndarray,
          x: np.ndarray,
          **kwargs):
    """
    Detect peaks of some function, y, in the grid space, x.

    Parameters
    ----------
    y: Numpy.Array
    x: Numpy.Array
    kwargs:
        Additional keyword arguments passed to detecta.detect_peaks function

    Returns
    -------
    List
    """
    p = detect_peaks(y, **kwargs)
    return [x[i] for i in p]


def match_landmarks(l1: np.ndarray, l2: np.ndarray):
    """
    Given the landmarks (peaks) of two functions L1 and L2, interate over L1 and match each landmark
    with it's nearest landmark in L2. Return an updated list of landmarks for L2 matching the order of
    their nearest landmark in L1.

    Parameters
    ----------
    l1: Numpy.Array
    l2: Numpy.Array

    Returns
    -------
    List
    """
    u2 = list()
    for i in l1:
        dist = [abs(i - ix) for ix in l2]
        u2.append(l2[np.argmin(dist)])
    return u2


def update_landmarks(landmarks: list):
    """
    Given an array of two elements (the landmarks - or peaks - of two functions), iterate over
    the landmarks of the function with the most landmarks, matching the landmarks to the
    nearest landmark in the opposing function. Result should be two lists of landmarks of
    equal length and in quasi-matching order.

    Parameters
    ----------
    landmarks: List

    Returns
    -------
    List
    """
    l1, l2 = landmarks
    if len(l1) < len(l2):
        u = match_landmarks(l1, l2)
        return [l1, u]
    u = match_landmarks(l2, l1)
    return [u, l2]


def find_nearest_x(value, y, grid):
    return grid[(np.abs(y - value)).argmin()]


def estimate_new_x(data: np.ndarray,
                   before: FDataGrid,
                   after: FDataGrid,
                   n_jobs: int = -1):
    if n_jobs <= 0:
        n_jobs = cpu_count()
    before = before.evaluate(data)[0].reshape(-1)
    y_after = after.data_matrix[0].reshape(-1)
    find_func = partial(find_nearest_x, y=y_after, grid=after.grid_points[0])
    with Pool(n_jobs) as pool:
        return np.array(list(pool.map(find_func, before)))


def align_data(data: pd.DataFrame,
               ref_data: pd.DataFrame,
               dims: list,
               n_jobs: int = -1,
               **kwargs):
    """
    Given some new data and reference data to align to, estimate the
    probability density function of each dimension (dims) within each
    of these dataframes. A peak finding algorithm (detecta.detect_peaks)
    is used to identify "landmarks" which are used to align each
    dimension in data with the equivalent in ref_data using landmark
    registration.

    Parameters
    ----------
    data: Pandas.DataFrame
    ref_data: Pandas.DataFrame
    dims: List
    kwargs:
        Additional keyword arguments passed to detecta.detect_peaks

    Returns
    -------

    """
    mph = kwargs.pop("mph", lambda y: 0.001 * np.max(y))
    data = data.copy()
    for d in dims:
        x = np.linspace(np.min([data[d].min(), ref_data[d].min()]) - 0.01,
                        np.max([data[d].max(), ref_data[d].max()]) + 0.01,
                        10000)
        y1 = (FFTKDE(kernel="gaussian",
                     bw="silverman")
              .fit(data[d].values)
              .evaluate(x))
        y2 = (FFTKDE(kernel="gaussian",
                     bw="silverman")
              .fit(ref_data[d].values)
              .evaluate(x))
        landmarks = update_landmarks([peaks(y, x, mph=mph(y), **kwargs) for y in [y1, y2]])
        fdata = FDataGrid([y1, y2], grid_points=x)
        shifted = landmark_registration(fdata, landmarks)
        data[d] = estimate_new_x(data=data[d].values,
                                 before=fdata,
                                 after=shifted,
                                 n_jobs=n_jobs)
    return data


def _load_dataframes(target: FileGroup,
                     ref: FileGroup,
                     population: str,
                     transform: dict or str or None = None,
                     ctrl: str or None = None):
    if ctrl is not None:
        return (target.load_ctrl_population_df(ctrl=ctrl,
                                               population=population,
                                               transform=transform),
                ref.load_ctrl_population_df(ctrl=ctrl,
                                            population=population,
                                            transform=transform))
    return (target.load_population_df(population=population,
                                      transform=transform),
            ref.load_population_df(population=population,
                                   transform=transform))


def normalise_data(target: FileGroup,
                   ref: FileGroup,
                   dims: list,
                   population: str,
                   transform: dict or str or None = "logicle",
                   ctrl: str or None = None,
                   **kwargs):
    target, ref = _load_dataframes(target, ref, population, transform, ctrl)
    return align_data(target, ref, dims, **kwargs)
