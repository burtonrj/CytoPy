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
from typing import List
from typing import Union

import matplotlib.pyplot as plt
import numpy as np
from detecta import detect_peaks
from KDEpy import FFTKDE
from skfda.preprocessing.registration import landmark_registration_warping
from skfda.representation.grid import FDataGrid


def merge_peaks(p, threshold: float = 0.1):
    to_merge = list()
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
    def __init__(
        self,
        kernel: str = "gaussian",
        bw: Union[str, float] = "silverman",
        min_peak_threshold: float = 0.001,
        merge_peak_distance: float = 0.1,
    ):
        self.kernel = kernel
        self.bw = bw
        self.min_peak_threshold = min_peak_threshold
        self.merge_peak_distance = merge_peak_distance
        self.original_functions = None
        self.landmarks = None
        self.warping_functions = None

    def _compute_original_functions(self, data: np.ndarray):
        x = np.linspace(np.min(data) - 0.1, np.max(data) + 0.1, 100000)
        functions = [FFTKDE(kernel=self.kernel, bw=self.bw).fit(data[i, :]).evaluate(x) for i in range(data.shape[0])]
        landmarks = [peaks(y, x, mph=0.001 * y.max()) for y in functions]
        landmarks = [merge_peaks(p, self.merge_peak_distance) for p in landmarks]
        n = np.min([len(p) for p in landmarks])
        self.landmarks = np.array([sorted(filter_peaks(p, x, y, n)) for p, y in zip(landmarks, functions)])
        self.original_functions = FDataGrid(functions, grid_points=x)

    def fit(self, data: np.ndarray):
        self._compute_original_functions(data=data)
        self.warping_functions = landmark_registration_warping(
            self.original_functions, np.array(self.landmarks), location=np.mean(self.landmarks, axis=0)
        )
        return self

    def transform(self, data: np.ndarray):
        assert self.warping_functions is not None, "Call fit first!"
        return self.warping_functions.evaluate(data)[1].reshape(-1)

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
