#!/usr/bin.env/python
# -*- coding: utf-8 -*-
"""
This module provides normalisation methods using landmark registration, first described
with application to cytometry data by Hahne et al [1] with further expansion by Finak et al [2].
Landmark registration is implemented in the LandmarkReg class using Scikit-FDA.

[1] Hahne F, Khodabakhshi AH, Bashashati A, Wong CJ, Gascoyne RD,
Weng AP, Seyfert-Margolis V, Bourcier K, Asare A, Lumley T, Gentleman R,
Brinkman RR. Per-channel basis normalization methods for utils cytometry data.
Cytometry A. 2010 Feb;77(2):121-31. doi: 10.1002/cyto.a.20823. PMID: 19899135; PMCID: PMC3648208.

[2] Finak G, Jiang W, Krouse K, et al. High-throughput utils cytometry data normalization
for clinical trials. Cytometry A. 2014;85(3):277-286. doi:10.1002/cyto.a.22433

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
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from detecta import detect_peaks
from KDEpy import FFTKDE
from numba import jit
from skfda.preprocessing.registration import landmark_registration_warping
from skfda.preprocessing.registration import landmark_shift_deltas
from skfda.representation.grid import FDataGrid
from sklearn.cluster import KMeans


def peaks(y: np.ndarray, x: np.ndarray, **kwargs):
    """
    Detect peaks of some function, y, in the grid space, x.

    Parameters
    ----------
    y: numpy.ndarray
    x: numpy.ndarray
    kwargs:
        Additional keyword arguments passed to detecta.detect_peaks function

    Returns
    -------
    List
    """
    p = detect_peaks(y, **kwargs)
    return [x[i] for i in p]


@jit(nopython=True)
def filter_by_closest_centroid(x: np.ndarray, labels: np.ndarray, centroid: float):
    """
    Filter peaks ('x') to keep only those
    closest to their nearest centroid (centroid of clustered peaks).
    Labels indicate where the peak originated from; either target
    sample (0) or reference (1).

    Parameters
    ----------
    x: numpy.ndarray
    labels: numpy.ndarray
    centroid: float

    Returns
    -------
    float, float
        Peaks closest to centroid in cluster 1, Peaks closest to centroid in cluster 2
    """
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


def cluster_landmarks(p: np.ndarray, plabels: np.ndarray):
    """
    Cluster peaks (p). plabels indicate where the peak originated from; either target
    sample (0) or reference (1). The number of clusters, determined by KMeans clustering
    is equal to the number of peaks for the target sample.

    Parameters
    ----------
    p: numpy.ndarray
        Peaks
    plabels: numpy.ndarray
        Peak labels

    Returns
    -------
    numpy.ndarray, numpy.ndarray
        K Means labels for each peak, cluster centroids
    """
    km = KMeans(n_clusters=len(np.where(np.array(plabels) == 0)[0]), random_state=42)
    km_labels = km.fit_predict(np.array(p).reshape(-1, 1))
    centroids = km.cluster_centers_.reshape(-1)
    return km_labels, centroids


@jit(nopython=True)
def zero_entropy_clusters(km_labels: np.ndarray, plabels: np.ndarray, centroids: np.ndarray):
    """
    Determine which clusters (if any) have zero entropy (only contains
    peaks from a single sample; either target or reference)

    Parameters
    ----------
    km_labels: numpy.ndarray
        K means cluster labels
    plabels: numpy.ndarray
        Origin of the peak; either target (0) or reference (1)
    centroids: numpy.ndarray
        Cluster centroids

    Returns
    -------
    List
        List of centroids for clusters with zero entropy
    """
    zero_entropy = list()
    for i in np.unique(km_labels):
        cluster_plabels = plabels[np.where(km_labels == i)]
        if len(np.unique(cluster_plabels)) == 1:
            zero_entropy.append(centroids[i])
    return zero_entropy


@jit(nopython=True)
def unique_clusters_filter_nearest_centroid(
    p: np.ndarray, plabels: np.ndarray, km_labels: np.ndarray, centroids: np.ndarray
):
    """
    Under the assumption that clusters have zero entropy (that is, all
    peaks within a cluster originate from the same sample), filter
    peaks to keep only those nearest to the centroid.

    Parameters
    ----------
    p: numpy.ndarray
        Peaks
    plabels: numpy.ndarray
        Origin of the peak; either target (0) or reference (1)
    km_labels: numpy.ndarray
        Cluster label for each peak
    centroids: numpy.ndarray
        Cluster centroids

    Returns
    -------
    numpy.ndarray, numpy.ndarray
        Updated peaks and peak labels containing only those closest to cluster centroids

    Raises
    ------
    AssertionError
        If a supplied cluster entropy is not zero
    """
    updated_peaks = list()
    updated_plabels = list()
    for i, centroid in enumerate(centroids):
        cluster_plabels = plabels[np.where(km_labels == i)]
        assert len(np.unique(cluster_plabels)) == 1, "Zero entropy cluster is not unique"
        updated_peaks.append(p[np.where(km_labels == i)])
        updated_plabels.append(np.unique(cluster_plabels)[0])
    return updated_peaks, updated_plabels


def match_landmarks(p: np.ndarray, plabels: np.ndarray):
    """
    Given an array of peaks (p) labelled according to their origin (plabels; 0 being
    from target and 1 being from reference), match landmarks with each other, between samples,
    using K means clustering and a nearest centroid approach.

    Parameters
    ----------
    p: numpy.ndarray
    plabels: numpy.ndarray

    Returns
    -------
    numpy.ndarray
        (2, n) array, where n is the number of clusters. Order conserved between samples; first
        row is peaks from target, second row is peaks from reference.
    """
    p, plabels = np.array(p), np.array(plabels)
    km_labels, centroids = cluster_landmarks(p, plabels)
    # Search for clusters with zero entropy
    zero_entropy = zero_entropy_clusters(km_labels, plabels, centroids)
    if len(zero_entropy) == len(centroids):
        # If all clusters have zero entropy
        # Keep peaks closest to centroid, update peaks and peak labels, and
        # perform recursive call
        match_landmarks(*unique_clusters_filter_nearest_centroid(p, plabels, km_labels, centroids))

    else:
        # Ignoring clusters with zero entropy, filter clusters to contain
        # exactly 1 peak from each class, keeping peaks closest to the centroid
        matching_peaks = list()
        for i, centroid in enumerate(centroids):
            if centroid not in zero_entropy:
                x = filter_by_closest_centroid(
                    p[np.where(km_labels == i)],
                    plabels[np.where(km_labels == i)],
                    centroid,
                )
                matching_peaks.append(x)
        matching_peaks = np.array(matching_peaks)
        matching_peaks = [np.sort(matching_peaks[:, 0]), np.sort(matching_peaks[:, 1])]
        return matching_peaks


def estimate_pdfs(target: pd.DataFrame, ref: pd.DataFrame, var: str):
    """
    Given some target and reference DataFrame, estimate PDF for each using convolution based
    kernel density estimation (see KDEpy). 'var' is the variable of interest and should be a
    column in both ref and target

    Parameters
    ----------
    target: pandas.DataFrame
    ref: pandas.DataFrame
    var: str

    Returns
    -------
    (numpy.ndarray, numpy.ndarray, numpy.ndarray)
        Target PDF, reference PDF, and grid space
    """
    min_ = np.min([target[var].min(), ref[var].min()])
    max_ = np.max([target[var].max(), ref[var].max()])
    x = np.linspace(min_ - 0.1, max_ + 0.1, 100000)
    y1 = FFTKDE(kernel="gaussian", bw="silverman").fit(target[var].to_numpy()).evaluate(x)
    y2 = FFTKDE(kernel="gaussian", bw="silverman").fit(ref[var].to_numpy()).evaluate(x)
    return y1, y2, x


class LandmarkReg:
    """
    One technique for handling technical variation in cytometry data is local normalisation by
    aligning the probability density function of some data to a reference sample. This should
    be applied to a population immediately prior to applying a gate.

    The alignment algorithm is inspired by previous work [1, 2] and is performed as follows:
    1. The probability density function of some target data and a reference sample are estimated
    using a convolution based fast kernel density estimation algorithm (KDEpy.FFTKDE)
    2. Landmarks are identified in both samples as peaks of local maximal density.
    3. The peaks from both target and reference are combined and clustered using K means clustering; the
    number of clusters is chosen as the number of peaks identified in the target
    4. Unique pairings of peaks between samples, closest to the centroid of a cluster, are generated and
    used as landmarks.
    5. Landmark registration is performed using the Scikit-FDA package to generate a warping function, with
    the target location being the mean between paired peaks
    6. The warping function is applied to the target data, generating a new adjusted vector with high
    density regions matched to the reference sample

    [1] Hahne F, Khodabakhshi AH, Bashashati A, Wong CJ, Gascoyne RD,
    Weng AP, Seyfert-Margolis V, Bourcier K, Asare A, Lumley T, Gentleman R,
    Brinkman RR. Per-channel basis normalization methods for utils cytometry data.
    Cytometry A. 2010 Feb;77(2):121-31. doi: 10.1002/cyto.a.20823. PMID: 19899135; PMCID: PMC3648208.

    [2] Finak G, Jiang W, Krouse K, et al. High-throughput utils cytometry data normalization
    for clinical trials. Cytometry A. 2014;85(3):277-286. doi:10.1002/cyto.a.22433

    Parameters
    ----------
    target: pandas.DataFrame
        Target data to be transformed; must contain column corresponding to 'var'
    ref: pandas.DataFrame
        Reference data for computing alignment; must contain column corresponding to 'var'
    var: str
        Name of the target variable to align
    mpt: float (default=0.001)
        Minimum peak threshold; peaks that are less than the given percentage of the 'highest' peak
        (max density) will be ignored. Use this to remove small perturbations.
    kwargs:
        Additional keyword arguments passed to cytopy.utils.fda_norm.peaks call

    Attributes
    ----------
    landmarks: numpy.ndarray
        (2, n) array, where n is the number of clusters. Order conserved between samples; first
        row is peaks from target, second row is peaks from reference.
    original_functions: skfda.representation.grid.FDataGrid
        Original PDFs for target and reference
    warping_function: skfda.representation.grid.FDataGrid
        Warping function
    adjusted_functions: skfda.representation.grid.FDataGrid
        Registered curves following function compostion of original PDFs and warping function
    landmark_shift_deltas: numpy.ndarray
        Corresponding shifts to align the landmarks of the PDFs described in original_functions
    """

    def __init__(self, target: pd.DataFrame, ref: pd.DataFrame, var: str, mpt: float = 0.001, **kwargs):
        y1, y2, x = estimate_pdfs(target, ref, var)
        landmarks = [peaks(y, x, mph=mpt * y.max(), **kwargs) for y in [y1, y2]]
        plabels = np.concatenate(
            [
                [0 for _ in range(len(landmarks[0]))],
                [1 for _ in range(len(landmarks[1]))],
            ]
        )
        landmarks = np.array([x for sl in landmarks for x in sl])
        self.landmarks = match_landmarks(landmarks, plabels)
        self.original_functions = FDataGrid([y1, y2], grid_points=x)
        self.warping_function = None
        self.adjusted_functions = None
        self.landmark_shift_deltas = None

    def __call__(self):
        """
        Calculate the warping function, registered curves and landmark shift deltas

        Returns
        -------
        self
        """
        self.warping_function = landmark_registration_warping(
            self.original_functions,
            self.landmarks,
            location=np.mean(self.landmarks, axis=0),
        )
        self.adjusted_functions = self.original_functions.compose(self.warping_function)
        self.landmark_shift_deltas = landmark_shift_deltas(self.original_functions, self.landmarks)
        return self

    def plot_warping(self, ax: list or None = None):
        """
        Generate a figure that plots the PDFs prior to landmark registration,
        the warping function, and the registered curves.

        Parameters
        ----------
        ax: Matplotlib.Axes, optional

        Returns
        -------
        Matplotlib.Axes
        """
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

    def shift_data(self, x: np.ndarray):
        """
        Provided the original vector of data to transform, use the warping
        function to normalise the data and align the reference.

        Parameters
        ----------
        x: numpy.ndarray

        Returns
        -------
        numpy.ndarray

        Raises
        ------
        AssertionError
            If the class has not been called and therefore a warping function has not
            been defined
        """
        assert self.warping_function is not None, "No warping function defined"
        return self.warping_function.evaluate(x)[1].reshape(-1)

    def plot_shift(self, x: np.ndarray, ax: plt.Axes or None = None):
        """
        Plot the reference PDF and overlay the target data before and after landmark
        registration.

        Parameters
        ----------
        x: numpy.ndarray
            Target data
        ax: Matplotlib.Axes, optional

        Returns
        -------
        Matplotlib.Axes
        """
        ax = ax or plt.subplots(figsize=(5, 5))[1]
        shifted = self.shift_data(x)
        x = np.linspace(np.min(x) - 0.1, np.max(x) + 0.1, 10000)
        y2 = FFTKDE(kernel="gaussian", bw="silverman").fit(shifted).evaluate(x)

        self.original_functions.plot(axes=ax)
        ax.plot(x, y2)
        ax.legend(labels=["Before", "Ref", "After"])
        return ax
