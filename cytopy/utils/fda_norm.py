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
from __future__ import annotations

from typing import List, Optional
from typing import Union

import matplotlib.pyplot as plt
import numpy as np
from detecta import detect_peaks
from KDEpy import FFTKDE
from skfda.preprocessing.registration import landmark_registration_warping
from skfda.representation.grid import FDataGrid


def merge_peaks(p: List, threshold: float = 0.1) -> List:
    """
    Merge peaks if values are within a certain distance to each other (controlled by threshold)

    Parameters
    ----------
    p: List
        Peaks for merging
    threshold: float (default=0.1)
        Maximum distance between peaks to prevent merger

    Returns
    -------
    List
    """
    to_merge = []
    for i in p:
        for j in p:
            if i != j and abs(i - j) <= threshold:
                to_merge.append(i)
    if not to_merge:
        return p
    return [i for i in p if i not in to_merge] + [np.mean(to_merge)]


def filter_peaks(x, grid, y, n):
    p = [(i, np.interp(i, grid, y)) for i in x]
    p = sorted(p, key=lambda l: l[1])[::-1]
    return np.array([x[0] for x in p])[0:n]


class LandmarkRegistration:
    """
    Align peaks of one or more distributions using landmark registration

    Parameters
    ----------
    kernel: str (default='gaussian')
        Kernel to use for kernel density estimation; see KDEpy.FFTKDE
    bw: Union[str, float], (default='ISJ')
        Kernel bandwidth; see KDEpy.FFTKDE
    min_peak_threshold: float (default=0.001)
        Peaks below this fraction of the maximum peak size will be ignored
    merge_peak_distance: float (default=0.1)
        Peaks within this distance will be merged
    min_peak_distance: float (default=0.1)
        Minimum peak distance passed to detecta.detect_peaks call
    grid_n: int (default=100)
        Grid size for kernel density estimate

    Attributes
    ----------
    landmarks: List[int]
        List of index for landmarks (peaks) in grid
    original_functions: FDataGrid
        Original data and grid space that landmark registration is performed on
    warping_functions: FDataGrid
        Results of landmark registration
    """
    def __init__(
        self,
        kernel: str = "gaussian",
        bw: Union[str, float] = "ISJ",
        min_peak_threshold: float = 0.001,
        merge_peak_distance: float = 0.1,
        min_peak_distance: float = 0.1,
        grid_n: int = 100,
    ):
        self.kernel = kernel
        self.bw = bw
        self.min_peak_threshold = min_peak_threshold
        self.merge_peak_distance = merge_peak_distance
        self.min_peak_distance = min_peak_distance
        self.original_functions = None
        self.landmarks = None
        self.warping_functions = None
        self.grid_n = grid_n

    def _compute_original_functions(self, data: List[np.ndarray], **peak_kwargs):
        """
        Fit kernel density estimate to data, calculate landmarks (peaks) and populate
        'original_functions' and 'landmarks'.

        Parameters
        ----------
        data: List[Numpy.Array]
            Two or more arrays to align
        peak_kwargs:
            Additional keyword arguments passed to detecta.detect_peaks call

        Returns
        -------
        None
        """
        x = np.linspace(np.min([np.min(x) for x in data]) - 0.1, np.max([np.max(x) for x in data]) + 0.1, self.grid_n)
        functions = [FFTKDE(kernel=self.kernel, bw=self.bw).fit(i).evaluate(x) for i in data]
        peak_kwargs = peak_kwargs or {}
        mpd = peak_kwargs.pop("mpd", self.min_peak_distance * np.max(x))
        landmarks = [peaks(y, x, mph=self.min_peak_threshold * y.max(), mpd=mpd, **peak_kwargs) for y in functions]
        landmarks = [merge_peaks(p, self.merge_peak_distance) for p in landmarks]
        n = np.min([len(p) for p in landmarks])
        self.landmarks = np.array([sorted(filter_peaks(p, x, y, n)) for p, y in zip(landmarks, functions)])
        self.original_functions = FDataGrid(functions, grid_points=x)

    def fit(self, data: List[np.ndarray]) -> LandmarkRegistration:
        """
        Perform landmark registration and compute the warping functions for the alignment of two
        or more arrays (arrays can be of unequal length)

        Parameters
        ----------
        data: List[Numpy.Array]

        Returns
        -------
        LandmarkRegistration
        """
        self._compute_original_functions(data=data)
        self.warping_functions = landmark_registration_warping(
            self.original_functions, np.array(self.landmarks), location=np.mean(self.landmarks, axis=0)
        )
        return self

    def transform(self, data: np.ndarray) -> np.ndarray:
        """
        Compute the warped data when applying landmark registration. Must call 'fit' first.

        Parameters
        ----------
        data: Numpy.Array

        Returns
        -------
        Numpy.Array
        """
        assert self.warping_functions is not None, "Call fit first!"
        return self.warping_functions.evaluate(data)[1].reshape(-1)

    def plot_warping(self, ax: Optional[plt.Axes] = None) -> plt.Axes:
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
        assert self.warping_functions is not None, "Call fit first!"
        ax = ax or plt.subplots(1, 3, figsize=(15, 4))[1]
        assert len(ax) == 3, "Must provide exactly 3 axis objects"
        self.original_functions.plot(axes=ax[0])
        ax[0].set_title("Before")
        self.warping_functions.plot(axes=ax[1])
        ax[1].set_title("Warping function")
        self.original_functions.compose(self.warping_functions).plot(axes=ax[2])
        ax[2].set_title("After")
        ax[0].legend(labels=["Target", "Reference"])
        return ax


def peaks(y: np.ndarray, x: np.ndarray, **kwargs) -> List:
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
