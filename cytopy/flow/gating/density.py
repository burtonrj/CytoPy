from .utilities import kde, check_peak, find_local_minima
from .defaults import ChildPopulationCollection
from .base import Gate, GateError
from scipy.signal import find_peaks
import pandas as pd
import numpy as np


class DensityThreshold(Gate):
    """
    Threshold gating estimated using properties of a Probability Density Function of events data as estimated
    using Gaussian Kernel Density Estimation

    Parameters
    ----------
    kde_bw: float, (default=0.01)
     Bandwidth to use for gaussian kernel density estimation
    ignore_double_pos: bool, (default=False)
        if True, in the case that multiple peaks are detected, peaks to the right of
        the highest peak will be ignored in the local minima calculation
    q: float, optional, (default=0.95)
        if only 1 peak is found, quartile gating is performed using this argument as the quartile
    std: float, optional
        alternative to quartile gating, the number of standard deviations from the mean can be used to
        determine the threshold
    peak_threshold: float, optional
        If not None, this decimal value represents what the minimum height of a peak should be relevant to the highest
        peak found (e.g. if peak_threshold=0.05, then all peaks with a height < 0.05 of the heighest peak will be
        ignored)
    kwargs:
        Gate constructor arguments (see cytopy.flow.gating.base)
    """
    def __init__(self,
                 kde_bw: float = 0.01,
                 ignore_double_pos: bool = False,
                 std: float or None = None,
                 q: float or None = 0.95,
                 peak_threshold: float or None = None,
                 **kwargs):
        super().__init__(**kwargs)
        self.kde_bw = kde_bw
        self.ignore_double_pos = ignore_double_pos
        self.std = std
        self.q = q
        self.peak_threshold = peak_threshold
        self.sample = self.sampling(self.data, 5000)

    def _find_peaks(self,
                    probs: np.array) -> np.array:
        """
        Internal function. Perform peak finding (see scipy.signal.find_peaks for details)

        Parameters
        -----------
        probs: Numpy.array
            array of probability estimates generated using flow.gating.utilities.kde

        Returns
        --------
        Numpy.array
            array of indices specifying location of peaks in `probs`
        """
        # Find peaks
        peaks = find_peaks(probs)[0]
        if self.peak_threshold:
            peaks = check_peak(peaks, probs, self.peak_threshold)
        return peaks

    def _evaluate_peaks(self,
                        data: pd.DataFrame,
                        peaks: np.array,
                        probs: np.array,
                        xx: np.array) -> float and str:
        """
        Internal function. Given the outputs of `__find_peaks` and `__smooth` calculate the threshold to generate
        for gating. If a single peak (one population) is found use quantile or standard deviation. If multiple peaks
        are found (multiple populations) then look for region of minimum density.

        Parameters
        ----------
        data: Pandas.DataFrame
            Events data
        peaks: Numpy.array
            array of indices specifying location of peaks in `probs`
        probs: Numpy.array
            array of probability estimates generated using flow.gating.utilities.kde
        xx: Numpy.array
            array of linear space kde calculated across

        Returns
        --------
        float and str
            (threshold, method used to generate threshold)
        """
        method = ''
        threshold = None
        # Evaluate peaks
        if len(peaks) == 1:
            # 1 peak found, use quantile or standard deviation to find threshold
            if self.q:
                threshold = data[self.x].quantile(self.q, interpolation='nearest')
                method = 'Quantile'
            elif self.std:
                u = data[self.x].mean()
                s = data[self.x].std()
                threshold = u + (s * self.std)
                method = 'Standard deviation'
            else:
                # No std or q provided so using default of 95th quantile
                threshold = data[self.x].quantile(0.95, interpolation='nearest')
                method = 'Quantile'
        if len(peaks) > 1:
            # Multiple peaks found, find the local minima between pair of highest peaks
            if self.ignore_double_pos:
                # Merge peaks of 'double positive' populations
                probs_peaks = probs[peaks]
                highest_peak = np.where(probs_peaks == max(probs_peaks))[0][0]
                if highest_peak < len(peaks):
                    peaks = peaks[:highest_peak + 1]
            # Merging of peaks if ignore_double_pos might result in one peak
            if len(peaks) > 1:
                threshold = find_local_minima(probs, xx, peaks)
                method = 'Local minima between pair of highest peaks'
            else:
                threshold = data[self.x].quantile(0.95, interpolation='nearest')
                method = 'Quantile'
        return threshold, method

    def _calc_threshold(self,
                        data: pd.DataFrame,
                        x: str) -> float and str:
        """
        Internal function Wrapper for calculating threshold for gating.

        data: Pandas.DataFrame
            Events data
        x: str
            feature of interest for threshold calculation

        Returns
        --------
        float and str
            (threshold, method used to generate threshold)
        """
        probs, xx = kde(data, x, self.kde_bw)
        peaks = self._find_peaks(probs)
        return self._evaluate_peaks(data, peaks, probs, xx)

    def gate_1d(self,
                merge_options: str = 'overwrite') -> ChildPopulationCollection:
        """
        Perform density based threshold gating in 1 dimensional space using the properties of a Probability
        Density Function of the events data as estimated using Gaussian Kernel Density Estimation.

        Parameters
        ----------
        merge_options: str
            must have value of 'overwrite' or 'merge'. Overwrite: existing index values in child
            populations will be overwritten by the results of the gating algorithm. Merge: index values generated from
            the gating algorithm will be merged with index values currently associated to child populations

        Returns
        --------
        ChildPopulationCollection
            Updated child population collection
        """
        # If parent is empty just return the child populations with empty index array
        if self.empty_parent:
            return self.child_populations
        if self.sample is not None:
            threshold, method = self._calc_threshold(self.sample, self.x)
        else:
            threshold, method = self._calc_threshold(self.data, self.x)
        if not threshold:
            raise GateError('Unexpected error whilst performing threshold gating. Calculated threshold is Null.')
        # Update child populations
        self.child_update_1d(threshold, method, merge_options)
        return self.child_populations

    def gate_2d(self) -> ChildPopulationCollection:
        """
        Perform density based threshold gating in 2 dimensional space using the properties of a Probability
        Density Function of the events data as estimated using Gaussian Kernel Density Estimation. KDE and threshold
        calculation performed on each dimension separately.

        Returns
        --------
        ChildPopulationCollection
            Updated child population collection
        """
        # If parent is empty just return the child populations with empty index array
        if self.empty_parent:
            return self.child_populations
        if not self.y:
            raise GateError('For a 2D threshold gate a value for `y` is required')
        if self.sample is not None:
            x_threshold, x_method = self._calc_threshold(self.sample, self.x)
            y_threshold, y_method = self._calc_threshold(self.sample, self.y)
        else:
            x_threshold, x_method = self._calc_threshold(self.data, self.x)
            y_threshold, y_method = self._calc_threshold(self.data, self.y)
        if not x_threshold or not y_threshold:
            raise GateError('Unexpected error whilst performing threshold gating. Calculated threshold is Null.')
        method = f'X: {x_method}, Y: {y_method}'
        self.child_update_2d(x_threshold, y_threshold, method)
        return self.child_populations
