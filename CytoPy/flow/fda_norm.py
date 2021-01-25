from skfda.preprocessing.registration import landmark_registration_warping, landmark_shift_deltas
from skfda.representation.grid import FDataGrid
from detecta import detect_peaks
from KDEpy import FFTKDE
import matplotlib.pyplot as plt
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
    if len(l1) == len(l2):
        return [l1, l2]
    if len(l1) < len(l2):
        u = match_landmarks(l1, l2)
        return [l1, u]
    u = match_landmarks(l2, l1)
    return [u, l2]


def estimate_pdfs(target: pd.DataFrame,
                  ref: pd.DataFrame,
                  var: str):
    """
    Given some target and reference DataFrame, estimate PDF for each using convolution based
    kernel density estimation (see KDEpy). 'var' is the variable of interest and should be a
    column in both ref and target

    Parameters
    ----------
    target: Pandas.DataFrame
    ref: Pandas.DataFrame
    var: str

    Returns
    -------
    (Numpy.Array, Numpy.Array, Numpy.Array)
        Target PDF, reference PDF, and grid space
    """
    min_ = np.min([target[var].min(), ref[var].min()])
    max_ = np.max([target[var].max(), ref[var].max()])
    x = np.linspace(min_ - 0.1,
                    max_ + 0.1,
                    100000)
    y1 = (FFTKDE(kernel="gaussian",
                 bw="silverman")
          .fit(target[var].values)
          .evaluate(x))
    y2 = (FFTKDE(kernel="gaussian",
                 bw="silverman")
          .fit(ref[var].values)
          .evaluate(x))
    return y1, y2, x


class LandmarkReg:
    def __init__(self,
                 target: pd.DataFrame,
                 ref: pd.DataFrame,
                 var: str,
                 mpt: float = 0.001,
                 **kwargs):
        y1, y2, x = estimate_pdfs(target, ref, var)
        self.landmarks = update_landmarks([peaks(y, x, mph=mpt * y.max(), **kwargs) for y in [y1, y2]])
        self.original_functions = FDataGrid([y1, y2], grid_points=x)
        self.warping_function = None
        self.adjusted_functions = None
        self.landmark_shift_deltas = None

    def __call__(self):
        self.warping_function = landmark_registration_warping(self.original_functions,
                                                              self.landmarks,
                                                              location=np.mean(self.landmarks, axis=0))
        self.adjusted_functions = self.original_functions.compose(self.warping_function)
        self.landmark_shift_deltas = landmark_shift_deltas(self.original_functions, self.landmarks)
        return self

    def plot_warping(self, ax: list or None = None):
        assert self.warping_function is not None, "Call object prior to plot"
        ax = ax or plt.subplots(1, 3, figsize=(15, 4))[1]
        assert len(ax) == 3, "Must provide exactly 3 axis objects"
        self.original_functions.plot(axes=ax[0])
        ax[0].set_title("Before")
        self.warping_function.plot(axes=ax[1])
        ax[1].set_title("Warping function")
        self.adjusted_functions.plot(axes=ax[2])
        ax[2].set_title("After")
        ax[0].legend(labels=["Target", "Reference"])
        return ax

    def shift_data(self,
                   x: np.ndarray):
        return self.warping_function.evaluate(x)[1].reshape(-1)

    def plot_shift(self,
                   x: np.ndarray,
                   ax: plt.Axes or None = None):
        ax = ax or plt.subplots(figsize=(5, 5))[1]
        shifted = self.shift_data(x)
        x = np.linspace(np.min(x) - 0.1,
                        np.max(x) + 0.1,
                        10000)
        y2 = (FFTKDE(kernel="gaussian",
                     bw="silverman")
              .fit(shifted)
              .evaluate(x))

        self.original_functions.plot(axes=ax)
        ax.plot(x, y2)
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        ax.legend(labels=["Before", "Ref", "After"])
        return ax
