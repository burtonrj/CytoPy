#!/usr/bin.env/python
# -*- coding: utf-8 -*-
"""
This module houses an adaption of the consensus clustering method
first described in https://link.springer.com/content/pdf/10.1023%2FA%3A1023949509487.pdf.
Python implementation is adapted from Žiga Sajovic with the original source code
found here: https://github.com/ZigaSajovic/Consensus_Clustering

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

from ...feedback import progress_bar
from itertools import combinations
import numpy as np
import bisect
np.random.seed(42)

__author__ = "Ross Burton"
__copyright__ = "Copyright 2020, cytopy"
__credits__ = ["Žiga Sajovic", "Ross Burton", "Simone Cuff", "Andreas Artemiou", "Matthias Eberl"]
__license__ = "MIT"
__version__ = "2.0.0"
__maintainer__ = "Ross Burton"
__email__ = "burtonrj@cardiff.ac.uk"
__status__ = "Production"


class ConsensusCluster:
    """
    Implementation of Consensus clustering, following the paper
    https://link.springer.com/content/pdf/10.1023%2FA%3A1023949509487.pdf
    Code is adapted from https://github.com/ZigaSajovic/Consensus_Clustering

    Parameters
    ----------
    cluster :
        clustering class (must be an instance of clustering algorithm with fit/fit_predict method (e.g. scikit-learn)
    L:
        smallest number of clusters to try
    K:
        biggest number of clusters to try
    H:
        number of resamplings for each cluster number
    resample_proportion :
        percentage to sample
    Mk:
        consensus matrices for each k (OTE: every consensus matrix is retained)
    Ak :
        area under CDF for each number of clusters (see paper)
    deltaK :
        changes in ares under CDF (see paper)
    """

    def __init__(self,
                 cluster: callable,
                 smallest_cluster_n: int,
                 largest_cluster_n: int,
                 n_resamples: int,
                 resample_proportion: float = 0.5,
                 verbose: bool = True):
        assert 0 <= resample_proportion <= 1, "proportion has to be between 0 and 1"
        self.verbose = verbose
        self.cluster_ = cluster
        self.resample_proportion_ = resample_proportion
        self.L_ = smallest_cluster_n
        self.K_ = largest_cluster_n
        self.H_ = n_resamples
        self.Mk = None
        self.Ak = None
        self.deltaK = None
        self.bestK = None

    @staticmethod
    def _internal_resample(data: np.array, proportion: float) -> (np.array, np.array):
        """Resampling array
        Parameters
        ----------
        data : numpy.ndarray
            data to be resampled
        proportion : float
            percentage to resample
        Returns
        -------
        numpy.ndarray, numpy.ndarray
            Resampled indices and numpy array of resampled data
        """
        resampled_indices = np.random.choice(
            range(data.shape[0]), size=int(data.shape[0]*proportion), replace=False)
        return resampled_indices, data[resampled_indices, :]

    def fit(self, data: np.array) -> None:
        """Fits a consensus matrix for each number of clusters
        Parameters
        ----------
        data : numpy.ndarray
            numpy array to fit clustering algorithm too
        Returns
        -------
        None
        """
        # Init a connectivity matrix and an indicator matrix with zeros
        Mk = np.zeros((self.K_-self.L_, data.shape[0], data.shape[0]))
        Is = np.zeros((data.shape[0],)*2)
        for k in progress_bar(range(self.L_, self.K_), verbose=self.verbose):  # for each number of clusters
            i_ = k-self.L_
            for h in range(self.H_):  # resample H times
                resampled_indices, resample_data = self._internal_resample(
                    data, self.resample_proportion_)
                self.cluster_.set_params(n_clusters=k)
                Mh = self.cluster_.fit_predict(resample_data)
                # find indexes of elements from same clusters with bisection
                # on sorted array => this is more efficient than brute force search
                id_clusts = np.argsort(Mh)
                sorted_ = Mh[id_clusts]
                for i in range(k):  # for each cluster
                    #
                    ia = bisect.bisect_left(sorted_, i)
                    ib = bisect.bisect_right(sorted_, i)
                    is_ = id_clusts[ia:ib]
                    ids_ = np.array(list(combinations(is_, 2))).T
                    # sometimes only one element is in a cluster (no combinations)
                    if ids_.size != 0:
                        Mk[i_, ids_[0], ids_[1]] += 1
                # increment counts
                ids_2 = np.array(list(combinations(resampled_indices, 2))).T
                Is[ids_2[0], ids_2[1]] += 1
            Mk[i_] /= Is+1e-8  # consensus matrix
            # Mk[i_] is upper triangular (with zeros on diagonal), we now make it symmetric
            Mk[i_] += Mk[i_].T
            Mk[i_, range(data.shape[0]), range(
                data.shape[0])] = 1  # always with self
            Is.fill(0)  # reset counter
        self.Mk = Mk
        # fits areas under the CDFs
        self.Ak = np.zeros(self.K_-self.L_)
        for i, m in enumerate(Mk):
            hist, bins = np.histogram(m.ravel(), density=True)
            self.Ak[i] = np.sum(h*(b-a) for b, a, h in zip(bins[1:], bins[:-1], np.cumsum(hist)))
        # fits differences between areas under CDFs
        self.deltaK = np.array([(Ab-Aa)/Aa if i > 2 else Aa
                                for Ab, Aa, i in zip(self.Ak[1:], self.Ak[:-1], range(self.L_, self.K_-1))])
        self.bestK = np.argmax(self.deltaK) + \
            self.L_ if self.deltaK.size > 0 else self.L_

    def predict(self):
        """Predicts on the consensus matrix, for best found cluster number
        Returns
        -------
            Clustering predictions
        """
        assert self.Mk is not None, "First run fit"
        self.cluster_.set_params(n_clusters=self.bestK)
        return self.cluster_.fit_predict(1-self.Mk[self.bestK-self.L_])

    def predict_data(self, data: np.array):
        """Predicts on the data, for best found cluster number
        Parameters
        ----------
        data: np.array :
            data to make predictions
        Returns
        -------
            Clustering predictions
        """
        assert self.Mk is not None, "First run fit"
        self.cluster_.set_params(n_clusters=self.bestK)
        return self.cluster_.fit_predict(data)
