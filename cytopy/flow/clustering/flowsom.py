#!/usr/bin.env/python
# -*- coding: utf-8 -*-
"""
Here you will find cytopy's implementation of the FlowSOM algorithm, which
relies on the MiniSOM library for self-organising maps. The work was
adapted from https://github.com/Hatchin/FlowSOM for integration with cytopy and
the database architecture.

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
import logging

import pandas as pd
from minisom import MiniSom
from sklearn.preprocessing import MinMaxScaler

from ...feedback import progress_bar
from .consensus import ConsensusCluster

__author__ = "Ross Burton"
__copyright__ = "Copyright 2020, cytopy"
__credits__ = [
    "Žiga Sajovic",
    "Ross Burton",
    "Simone Cuff",
    "Andreas Artemiou",
    "Matthias Eberl",
]
__license__ = "MIT"
__version__ = "2.0.0"
__maintainer__ = "Ross Burton"
__email__ = "burtonrj@cardiff.ac.uk"
__status__ = "Production"
logger = logging.getLogger(__name__)


class FlowSOM:
    """
    Python implementation of FlowSOM algorithm, adapted from https://github.com/Hatchin/FlowSOM
    This class implements MiniSOM in an almost identical manner to the work by Hatchin, but removed all the
    of the data handling steps seen in Hatchin's original library, since these are handled by the infrastructure in
    cytopy. The FlowSOM algorithm is implemented here in such a way that it requires only a Pandas DataFrame,
    like that typically produced when retrieving data from the cytopy database, and gives access to methods
    of clustering and meta-clustering. In addition to Hatchin's work, the cytopy implementation has improved error
    handling and integrates better with the cytopy workflow.

    Parameters
    ----------
    data : Pandas.DataFrame
        training data
    features : List
        list of columns to include
    neighborhood_function : str
        name of distribution for initialising weights
    normalisation : bool
        if True, min max normalisation applied prior to computation
    """

    def __init__(
        self,
        data: pd.DataFrame,
        features: list,
        neighborhood_function: str = "gaussian",
        normalisation: bool = False,
        verbose: bool = True,
    ):

        self.data = data[features].values
        self.normalisation = normalisation
        if normalisation:
            self.data = MinMaxScaler().fit_transform(self.data)
        self.dims = len(features)
        assert neighborhood_function in [
            "gaussian",
            "mexican_hat",
            "bubble",
            "triangle",
        ], 'Invalid neighborhood function, must be one of "gaussian", "mexican_hat", "bubble", or "triangle"'
        self.verbose = verbose
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

    def train(
        self,
        som_dim: tuple = (50, 50),
        sigma: float = 1.0,
        learning_rate: float = 0.5,
        batch_size: int = 500,
        seed: int = 42,
        weight_init: str = "random",
    ):

        """Train self-organising map.
        Parameters
        ----------
        som_dim : tuple, (default=(250, 250))
            dimensions of SOM embedding (number of nodes)
        sigma : float, (default=1.0)
            the radius of the different neighbors in the SOM, default = 1.0
        learning_rate : float, (default=0.5)
            alters the rate at which weights are updated
        batch_size : int, (default=500)
            size of batches used in training (alters number of total iterations)
        seed : int, (default=42)
            random seed
        weight_init : str, (default='random')
            how to initialise weights: either 'random' or 'pca' (Initializes the weights to span the
            first two principal components)
        Returns
        -------
        None
        """

        som = MiniSom(
            som_dim[0],
            som_dim[1],
            self.dims,
            sigma=sigma,
            learning_rate=learning_rate,
            neighborhood_function=self.nf,
            random_seed=seed,
        )
        if weight_init == "random":
            som.random_weights_init(self.data)
        elif weight_init == "pca":
            if not self.normalisation:
                logger.warning(
                    "It is strongly recommended to normalize the data before initializing " "the weights if using PCA."
                )
            som.pca_weights_init(self.data)
        else:
            logger.warning(
                'Invalid value provided for "weight_init", valid input is either "random" or "pca". '
                "Defaulting to random initialisation of weights"
            )
            som.random_weights_init(self.data)

        logger.info("------------- Training SOM -------------")
        som.train_batch(self.data, batch_size, verbose=True)  # random training
        self.xn = som_dim[0]
        self.yn = som_dim[1]
        self.map = som
        self.weights = som.get_weights()
        self.flatten_weights = self.weights.reshape(self.xn * self.yn, self.dims)
        logger.info("Training complete!")
        logger.info("----------------------------------------")

    def meta_cluster(
        self,
        cluster_class: callable,
        min_n: int = 5,
        max_n: int = 50,
        iter_n: int = 10,
        resample_proportion: float = 0.5,
    ):
        """Perform meta-clustering. Implementation of Consensus clustering, following the paper
        https://link.springer.com/content/pdf/10.1023%2FA%3A1023949509487.pdf
        Parameters
        ----------
        cluster_class :
            clustering object (must follow Sklearn standard; needs fit_predict method called with
            parameter n_clusters)
        min_n : int
            the min proposed number of clusters
        max_n : int
            the max proposed number of clusters
        iter_n : int
            the iteration times for each number of clusters
        resample_proportion : float, (Default value = 0.5)
            within (0, 1), the proportion of re-sampling when computing clustering
        Returns
        -------
        None
        """

        assert self.map is not None, "SOM must be trained prior to meta-clustering; call train before meta_cluster"
        # initialize cluster
        cluster_ = ConsensusCluster(
            cluster_class,
            min_n,
            max_n,
            iter_n,
            resample_proportion=resample_proportion,
            verbose=self.verbose,
        )
        cluster_.fit(self.flatten_weights)  # fitting SOM weights into clustering algorithm

        self.meta_map = cluster_
        self.meta_bestk = cluster_.bestK  # the best number of clusters in range(min_n, max_n)

        # get the prediction of each weight vector on meta clusters (on bestK)
        self.meta_flatten = cluster_.predict_data(self.flatten_weights)
        self.meta_class = self.meta_flatten.reshape(self.xn, self.yn)

    def predict(self):
        """
        Predict the cluster allocation for each cell in the associated dataset.
        (Requires that train and meta_cluster have been called previously)
        Parameters
        ----------
        Returns
        -------
        numpy.ndarray
            Predicted labels
        """
        err_msg = (
            "SOM must be trained prior to predicting cell clustering allegation; call train followed "
            "by meta_cluster"
        )
        assert self.map is not None, err_msg
        assert self.meta_class is not None, err_msg
        labels = []
        logger.info("---------- Predicting Labels ----------")
        for i in progress_bar(range(len(self.data)), verbose=self.verbose):
            xx = self.data[i, :]  # fetch the sample data
            winner = self.map.winner(xx)  # make prediction, prediction = the closest entry location in the SOM
            c = self.meta_class[winner]  # from the location info get cluster info
            labels.append(c)
        logger.info("---------------------------------------")
        return labels
