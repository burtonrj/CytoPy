from immunova.flow.utilities import progress_bar
from immunova.flow.clustering.consensus import ConsensusCluster
from sklearn.preprocessing import MinMaxScaler
from minisom import MiniSom
import pandas as pd


class FlowSOM:
    """
    Python implementation of FlowSOM algorithm, adapted from https://github.com/Hatchin/FlowSOM

    Arguments:
        - data: training data (Pandas DataFrame)
        - features: list of columns to include
        - neighborhood_function: name of distribution for initialising weights (default = 'gaussian')
        - normalisation: if True, data is normalised prior to initialising weights (recommended if initialising
        weights using PCA).

    Methods:
        - train: train nodes of self-organising map
        - meta-cluster: using Consensus Clustering (see flow.clustering.consensus.ConsensusClustering) perform meta-clustering; finds the optimal number of
        meta clusters in a given range
        - predict: returns a list of clustering allocations where each row corresponds to the rows in the training data, predicted using the constructed SOM and
        results of meta-clustering (requires that 'train' and 'meta-cluster' have been called prior)
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
              sigma: float = 1.0,
              learning_rate: float = 0.5,
              batch_size: int = 500,
              seed: int = 42,
              weight_init: str = 'random'):
        """
        Train self-organising map.
        :param som_dim: dimensions of SOM embedding (number of nodes)
        :param sigma: the radius of the different neighbors in the SOM, default = 1.0
        :param learning_rate: alters the rate at which weights are updated
        :param batch_size: size of batches used in training (alters number of total iterations)
        :param seed: random seed
        :param weight_init: how to initialise weights: either 'random' or 'pca' (Initializes the weights to span the
        first two principal components)
        """

        som = MiniSom(som_dim[0], som_dim[1],
                      self.dims, sigma=sigma,
                      learning_rate=learning_rate,
                      neighborhood_function=self.nf,
                      random_seed=seed)
        if weight_init == 'random':
            som.random_weights_init(self.data)
        elif weight_init == 'pca':
            if not self.normalisation:
                print('Warning: It is strongly recommended to normalize the data before initializing '
                      'the weights if using PCA.')
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
        """
        Perform meta-clustering. Implementation of Consensus clustering, following the paper
        https://link.springer.com/content/pdf/10.1023%2FA%3A1023949509487.pdf

        :param cluster_class: clustering object (must follow Sklearn standard; needs fit_predict method called with
        parameter n_clusters)
        :param min_n: the min proposed number of clusters
        :param max_n: the max proposed number of clusters
        :param iter_n: the iteration times for each number of clusters
        :param resample_proportion: within (0, 1), the proportion of re-sampling when computing clustering
        """

        assert self.map is not None, 'SOM must be trained prior to meta-clustering; call train before meta_cluster'
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
        """
        Predict the cluster allocation for each cell in the associated dataset.
        (Requires that train and meta_cluster have been called previously)
        """
        err_msg = 'SOM must be trained prior to predicting cell clustering allegation; call train followed ' \
                  'by meta_cluster'
        assert self.map is not None, err_msg
        assert self.meta_class is not None, err_msg
        labels = []
        print('---------- Predicting Labels ----------')
        for i in progress_bar(range(len(self.data))):
            xx = self.data[i, :]  # fetch the sample data
            winner = self.map.winner(xx)  # make prediction, prediction = the closest entry location in the SOM
            c = self.meta_class[winner]  # from the location info get cluster info
            labels.append(c)
        print('---------------------------------------')
        return labels



