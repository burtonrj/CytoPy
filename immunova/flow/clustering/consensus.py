from ..utilities import progress_bar
from itertools import combinations
import numpy as np
import bisect


class ConsensusCluster:
    """
      Implementation of Consensus clustering, following the paper
      https://link.springer.com/content/pdf/10.1023%2FA%3A1023949509487.pdf
      Code is adapted from https://github.com/ZigaSajovic/Consensus_Clustering
      Arguments:
        - cluster: clustering class (needs fit_predict method called with parameter n_clusters)
        - L: smallest number of clusters to try
        - K: biggest number of clusters to try
        - H: number of resamplings for each cluster number
        - resample_proportion -> percentage to sample
        - Mk: consensus matrices for each k (shape =(K,data.shape[0],data.shape[0]))
        (NOTE: every consensus matrix is retained, like specified in the paper)
        - Ak: area under CDF for each number of clusters
        (see paper: section 3.3.1. Consensus distribution.)
        - deltaK: changes in ares under CDF
         (see paper: section 3.3.1. Consensus distribution.)
        - self.bestK: number of clusters that was found to be best
      """

    def __init__(self, cluster: callable, smallest_cluster_n: int,
                 largest_cluster_n: int, n_resamples: int, resample_proportion: float = 0.5):
        assert 0 <= resample_proportion <= 1, "proportion has to be between 0 and 1"
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
        """
        Resampling array
        :param data: data to be resampled
        :param proportion: percentage to resample
        :return:
        """
        resampled_indices = np.random.choice(
            range(data.shape[0]), size=int(data.shape[0]*proportion), replace=False)
        return resampled_indices, data[resampled_indices, :]

    def fit(self, data: np.array) -> None:
        """
        Fits a consensus matrix for each number of clusters
        :param data: numpy array to fit clustering algorithm too
        """
        # Init a connectivity matrix and an indicator matrix with zeros
        Mk = np.zeros((self.K_-self.L_, data.shape[0], data.shape[0]))
        Is = np.zeros((data.shape[0],)*2)
        for k in progress_bar(range(self.L_, self.K_)):  # for each number of clusters
            i_ = k-self.L_
            for h in range(self.H_):  # resample H times
                resampled_indices, resample_data = self._internal_resample(
                    data, self.resample_proportion_)
                Mh = self.cluster_(n_clusters=k).fit_predict(resample_data)
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
            self.Ak[i] = np.sum(h*(b-a)
                             for b, a, h in zip(bins[1:], bins[:-1], np.cumsum(hist)))
        # fits differences between areas under CDFs
        self.deltaK = np.array([(Ab-Aa)/Aa if i > 2 else Aa
                                for Ab, Aa, i in zip(self.Ak[1:], self.Ak[:-1], range(self.L_, self.K_-1))])
        self.bestK = np.argmax(self.deltaK) + \
            self.L_ if self.deltaK.size > 0 else self.L_

    def predict(self):
        """
        Predicts on the consensus matrix, for best found cluster number
        :return clustering predictions
        """
        assert self.Mk is not None, "First run fit"
        return self.cluster_(n_clusters=self.bestK).fit_predict(1-self.Mk[self.bestK-self.L_])

    def predict_data(self, data: np.array):
        """
        Predicts on the data, for best found cluster number
        :param data: data to make predictions
        :return clustering predictions
        """
        assert self.Mk is not None, "First run fit"
        return self.cluster_(n_clusters=self.bestK).fit_predict(data)

