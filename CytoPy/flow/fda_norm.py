#!/usr/bin.env/python
# -*- coding: utf-8 -*-
"""
CytoPy supports the following dimension reduction methods: UMAP, tSNE,
PCA, Kernel PCA, and PHATE. These are implemented through the dim_reduction
function. This takes a dataframe of single cell events and generates the
desired number of embeddings. These are returned as a matrix or
as appended columns to the given dataframe.

If you would like to contribute to CytoPy to expand the supported dimension
reduction methods, please contact us at burtonrj@cardiff.ac.uk

Copyright 2020 Ross Burton

Permission is hereby granted, free of charge, to any person
obtaining a copy of this software and associated documentation
files (the "Software"), to deal in the Software without restriction,
including without limitation the rights to use, copy, modify,
merge, publish, distribute, sublicense, and/or sell copies of the
Software, and to permit persons to whom the Software is furnished
to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included
in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""


from skfda.preprocessing.registration import landmark_registration_warping, landmark_shift_deltas
from skfda.representation.grid import FDataGrid
from sklearn.cluster import KMeans
from detecta import detect_peaks
from KDEpy import FFTKDE
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

__author__ = "Ross Burton"
__copyright__ = "Copyright 2020, CytoPy"
__credits__ = ["Ross Burton", "Simone Cuff", "Andreas Artemiou", "Matthias Eberl"]
__license__ = "MIT"
__version__ = "2.0.0"
__maintainer__ = "Ross Burton"
__email__ = "burtonrj@cardiff.ac.uk"
__status__ = "Production"


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


def filter_by_closest_centroid(x, labels, centroid):
    x, labels = np.array(x), np.array(labels)
    y1 = x[np.where(labels == 0)]
    y2 = x[np.where(labels == 1)]
    if len(y1) > 1:
        y1 = y1[np.abs(y1 - centroid).argmin()]
    else:
        y1 = y1[0]
    if len(y2) > 1:
        y2 = y2[np.abs(y2 - centroid).argmin()]
    else:
        y2 = y2[0]
    return y1, y2


def cluster_landmarks(p, plabels):
    km = KMeans(n_clusters=len(np.where(np.array(plabels)==0)[0]),
                random_state=42)
    km_labels = km.fit_predict(np.array(p).reshape(-1, 1))
    centroids = km.cluster_centers_.reshape(-1)
    return km_labels, centroids


def zero_entropy_clusters(km_labels, plabels, centroids):
    zero_entropy = list()
    for i in np.unique(km_labels):
        cluster_plabels = plabels[np.where(km_labels == i)]
        if len(np.unique(cluster_plabels)) == 1:
            zero_entropy.append(centroids[i])
    return zero_entropy


def unique_clusters_filter_nearest_centroid(p, plabels, km_labels, centroids):
    updated_peaks = list()
    updated_plabels = list()
    for i, centroid in enumerate(centroids):
        cluster_plabels = plabels[np.where(km_labels == i)]
        assert len(np.unique(cluster_plabels)) == 1, "Zero entropy cluster is not unique"
        updated_peaks.append(p[np.where(km_labels == i)])
        updated_plabels.append(np.unique(cluster_plabels)[0])
    return updated_peaks, updated_plabels


def match_landmarks(p: np.ndarray, plabels: np.ndarray):
    p, plabels = np.array(p), np.array(plabels)
    km_labels, centroids = cluster_landmarks(p, plabels)
    # Search for clusters with zero entropy
    zero_entropy = zero_entropy_clusters(km_labels, plabels, centroids)
    if len(zero_entropy) == len(centroids):
        # If all clusters have zero entropy
        # Keep peaks closest to centroid, update peaks and peak labels, and
        # perform recursive call
        match_landmarks(*unique_clusters_filter_nearest_centroid(p, plabels,
                                                                 km_labels,
                                                                 centroids))

    else:
        # Ignoring clusters with zero entropy, filter clusters to contain
        # exactly 1 peak from each class, keeping peaks closest to the centroid
        matching_peaks = list()
        for i, centroid in enumerate(centroids):
            if centroid not in zero_entropy:
                x = filter_by_closest_centroid(p[np.where(km_labels == i)],
                                               plabels[np.where(km_labels == i)],
                                               centroid)
                matching_peaks.append(x)
        matching_peaks = np.array(matching_peaks)
        matching_peaks = [np.sort(matching_peaks[:, 0]), np.sort(matching_peaks[:, 1])]
        return matching_peaks


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
        landmarks = [peaks(y, x, mph=mpt * y.max(), **kwargs) for y in [y1, y2]]
        plabels = np.concatenate([[0 for i in range(len(landmarks[0]))],
                                  [1 for i in range(len(landmarks[1]))]])
        landmarks = np.array([x for sl in landmarks for x in sl])
        self.landmarks = match_landmarks(landmarks, plabels)
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
        ax.legend(labels=["Before", "Ref", "After"])
        return ax
