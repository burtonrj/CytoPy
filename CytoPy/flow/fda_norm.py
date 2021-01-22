from skfda.preprocessing.registration import landmark_registration
from skfda.representation.grid import FDataGrid
from ..data.fcs import FileGroup
from detecta import detect_peaks
from KDEpy import FFTKDE
import pandas as pd
import numpy as np


def load_reference_data(filegroup: FileGroup,
                        parent: str,
                        x: str or None,
                        y: str or None,
                        x_transform: str or None,
                        y_transform: str or None,
                        ctrl: str or None = None):
    """
    Load the parent population from the given FileGroup. This will be used
    as reference for landmark registration.

    Parameters
    ----------
    filegroup: FileGroup
        FileGroup containing cytometry data
    parent: str
        Parent population that we're interested in
    x: str or None
        Name of the x-axis dimension
    y: str or None
        Name of the x-axis dimension
    x_transform: str or None
        X-axis transformation
    y_transform: str or None
        Y-axis transformation
    ctrl: str or None (default=None)
        Name of control file to load. Loads population from primary data if None.

    Returns
    -------
    Pandas.DataFrame
    """
    if ctrl is not None:
        return filegroup.load_ctrl_population_df(ctrl=ctrl,
                                                 population=parent,
                                                 transform={x: x_transform,
                                                            y: y_transform})
    return filegroup.load_population_df(population=parent,
                                        transform={x: x_transform,
                                                   y: y_transform})


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
        dist = [abs(i-ix) for ix in l2]
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


def align_data(data: pd.DataFrame,
               ref_data: pd.DataFrame,
               dims: list,
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
    mph = kwargs.pop("mph", lambda y: 0.001*np.max(y))
    data = data.copy()
    for d in dims:
        x = np.linspace(np.min([data[d].min(), ref_data[d].min()]) - 0.01,
                        np.max([data[d].max(), ref_data[d].max()]) + 0.01,
                        100)
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
        data[d] = shifted.data_matrix[0]
    return data


def fda_norm(fit_predict):
    """
    Wrapper function that modifies the fit_predict call of a Gate
    object to normalise data prior to fit

    Parameters
    ----------
    fit_predict: Gate.fit_predict

    Returns
    -------
    callable
        Gate.fit_predict with data normalised using landmark registration
    """
    def normalise_data(gate, data: pd.DataFrame, ctrl_data: pd.DataFrame or None = None):
        if gate.fda_norm:
            assert gate.reference is not None, "No reference sample defined"
            kwargs = gate.fda_norm_kwargs or {}
            ref_data = load_reference_data(filegroup=gate.reference,
                                           parent=gate.parent,
                                           x=gate.x,
                                           y=gate.y,
                                           x_transform=gate.transformations.get("x"),
                                           y_transform=gate.transformations.get("y"))
            data = align_data(data=data,
                              ref_data=ref_data,
                              dims=[d for d in [gate.x, gate.y] if d is not None],
                              **kwargs)
            if ctrl_data is not None:
                ref_data = load_reference_data(filegroup=gate.reference,
                                               parent=gate.parent,
                                               x=gate.x,
                                               y=gate.y,
                                               x_transform=gate.transformations.get("x"),
                                               y_transform=gate.transformations.get("y"),
                                               ctrl=gate.ctrl)
                data = align_data(data=ctrl_data,
                                  ref_data=ref_data,
                                  dims=[d for d in [gate.x, gate.y] if d is not None],
                                  **kwargs)
        return fit_predict(data=data, ctrl_data=ctrl_data)
    return normalise_data
