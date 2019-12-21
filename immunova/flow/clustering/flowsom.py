from immunova.flow.utilities import progress_bar
from sklearn.preprocessing import MinMaxScaler
from minisom import MiniSom
from itertools import combinations
import pandas as pd
import numpy as np
import bisect


class ConsensusCluster:
    """
      Implementation of Consensus clustering, following the paper
      https://link.springer.com/content/pdf/10.1023%2FA%3A1023949509487.pdf
      Args:
        * cluster -> clustering class
                    needs fit_predict method called with parameter n_clusters
        * L -> smallest number of clusters to try
        * K -> biggest number of clusters to try
        * H -> number of resamplings for each cluster number
        * resample_proportion -> percentage to sample
        * Mk -> consensus matrices for each k (shape =(K,data.shape[0],data.shape[0]))
                (NOTE: every consensus matrix is retained, like specified in the paper)
        * Ak -> area under CDF for each number of clusters
                (see paper: section 3.3.1. Consensus distribution.)
        * deltaK -> changes in ares under CDF
                (see paper: section 3.3.1. Consensus distribution.)
        * self.bestK -> number of clusters that was found to be best
      """

    def __init__(self, cluster, smallest_cluster_n, largest_cluster_n, n_resamples, resample_proportion=0.5):
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
    def _internal_resample(data, proportion):
        """
        Args:
          * data -> (examples,attributes) format
          * proportion -> percentage to sample
        """
        resampled_indices = np.random.choice(
            range(data.shape[0]), size=int(data.shape[0]*proportion), replace=False)
        return resampled_indices, data[resampled_indices, :]

    def fit(self, data, verbose=False):
        """
        Fits a consensus matrix for each number of clusters
        Args:
          * data -> (examples,attributes) format
          * verbose -> should print or not
        """
        Mk = np.zeros((self.K_-self.L_, data.shape[0], data.shape[0]))
        Is = np.zeros((data.shape[0],)*2)
        for k in range(self.L_, self.K_):  # for each number of clusters
            i_ = k-self.L_
            if verbose:
                print("At k = %d, aka. iteration = %d" % (k, i_))
            for h in range(self.H_):  # resample H times
                if verbose:
                    print("\tAt resampling h = %d, (k = %d)" % (h, k))
                resampled_indices, resample_data = self._internal_resample(
                    data, self.resample_proportion_)
                Mh = self.cluster_(n_clusters=k).fit_predict(resample_data)
                # find indexes of elements from same clusters with bisection
                # on sorted array => this is more efficient than brute force search
                id_clusts = np.argsort(Mh)
                sorted_ = Mh[id_clusts]
                for i in range(k):  # for each cluster
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
        """
        assert self.Mk is not None, "First run fit"
        return self.cluster_(n_clusters=self.bestK).fit_predict(
            1-self.Mk[self.bestK-self.L_])

    def predict_data(self, data):
        """
        Predicts on the data, for best found cluster number
        Args:
          * data -> (examples,attributes) format
        """
        assert self.Mk is not None, "First run fit"
        return self.cluster_(n_clusters=self.bestK).fit_predict(
            data)


class FlowSOM:
    """
    Python implementation of FlowSOM algorithm, adapted from https://github.com/Hatchin/FlowSOM
    """
    def __init__(self, data: pd.DataFrame,
                 features: list,
                 neighborhood_function: str = 'gaussian',
                 normalisation: bool = False):

        self.data = data[features].values
        self.normalisation = normalisation
        if normalisation:
            self.data = MinMaxScaler().fit_transform(self.data)
        self.dims = len(features)
        self.nf = neighborhood_function
        self.xn = None
        self.yn = None
        self.map = None
        self.weights = None
        self.flatten_weights = None
        self.meta_map = None
        self.meta_bestk = None
        self.meta_flatten = None
        self.meta_class = None

    def train(self, som_dim: tuple = (250, 250),
              sigma: float = 0.1,
              learning_rate: float = 0.5,
              batch_size: int = 500,
              seed: int = 42,
              weight_init: str = 'random'):

        som = MiniSom(som_dim[0], som_dim[1],
                      self.dims, sigma=sigma,
                      learning_rate=learning_rate,
                      neighborhood_function=self.nf,
                      random_seed=seed)
        if weight_init == 'random':
            som.random_weights_init(self.data)
        elif weight_init == 'pca':
            som.pca_weights_init(self.data)
        else:
            print('Warning: invalid value provided for "weight_init", valid input is either "random" or "pca". '
                  'Defaulting to random initialisation of weights')
            som.random_weights_init(self.data)

        print("------------- Training SOM -------------")
        som.train_batch(self.data, batch_size, verbose=True)  # random training
        self.xn = som_dim[0]
        self.yn = som_dim[1]
        self.map = som
        self.weights = som.get_weights()
        self.flatten_weights = self.weights.reshape(self.xn*self.yn, self.dims)
        print("\nTraining complete!")
        print("----------------------------------------")

    def meta_cluster(self, cluster_class: callable,
                     min_n: int,
                     max_n: int,
                     iter_n: int,
                     resample_proportion: float = 0.5):

        if self.map is None:
            raise ValueError('Error: SOM must be trained prior to meta-clustering.')
        # initialize cluster
        cluster_ = ConsensusCluster(cluster_class,
                                    min_n, max_n, iter_n,
                                    resample_proportion=resample_proportion)
        cluster_.fit(self.flatten_weights, verbose=True)  # fitting SOM weights into clustering algorithm

        self.meta_map = cluster_
        self.meta_bestk = cluster_.bestK  # the best number of clusters in range(min_n, max_n)

        # get the prediction of each weight vector on meta clusters (on bestK)
        self.meta_flatten = cluster_.predict_data(self.flatten_weights)
        self.meta_class = self.meta_flatten.reshape(self.xn, self.yn)

    def predict(self):
        if self.map is None:
            raise ValueError('Error: SOM must be trained prior to predicting cell clustering allegation.')
        labels = []
        print('---------- Predicting Labels ----------')
        for i in progress_bar(range(len(self.data))):
            xx = self.data[i, :]  # fetch the sample data
            winner = self.map.winner(xx)  # make prediction, prediction = the closest entry location in the SOM
            c = self.meta_class[winner]  # from the location info get cluster info
            labels.append(c)
        return labels



