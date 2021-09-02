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
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple
from typing import Type

import numpy as np
import pandas as pd
from minisom import MiniSom
from sklearn.cluster import AgglomerativeClustering
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
        neighborhood_function: str = "gaussian",
        normalisation: bool = False,
        verbose: bool = True,
        som_dim: Tuple[int, int] = (50, 50),
        sigma: float = 1.0,
        learning_rate: float = 0.5,
        batch_size: int = 500,
        random_seed: int = 42,
        weight_init: str = "random",
        meta_clusterer: Optional[Type] = None,
        meta_clusterer_kwargs: Optional[Dict] = None,
        min_n: int = 5,
        max_n: int = 50,
        iter_n: int = 10,
        resample_proportion: float = 0.5,
    ):
        self.normalisation = normalisation
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
        self.som_dim = som_dim
        self.sigma = sigma
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.random_seed = random_seed
        self.weight_init = weight_init
        meta_clusterer_kwargs = meta_clusterer_kwargs or {}
        self.meta_clusterer = meta_clusterer or AgglomerativeClustering
        self.meta_clusterer = self.meta_clusterer(**meta_clusterer_kwargs)
        self.min_n = min_n
        self.max_n = max_n
        self.iter_n = iter_n
        self.resample_proportion = resample_proportion

    def fit_predict(self, data: pd.DataFrame, features: List[str]):
        data = data[features].values
        if self.normalisation:
            data = MinMaxScaler().fit_transform(data)
        self.train(data=data)
        self.meta_cluster()
        return self.predict(data=data)

    def train(self, data: np.ndarray):

        """
        Train self-organising map.
        """
        som = MiniSom(
            self.som_dim[0],
            self.som_dim[1],
            data.shape[1],
            sigma=self.sigma,
            learning_rate=self.learning_rate,
            neighborhood_function=self.nf,
            random_seed=self.random_seed,
        )
        if self.weight_init == "random":
            som.random_weights_init(data)
        elif self.weight_init == "pca":
            if not self.normalisation:
                logger.warning(
                    "It is strongly recommended to normalize the data before initializing " "the weights if using PCA."
                )
            som.pca_weights_init(data)
        else:
            logger.warning(
                'Invalid value provided for "weight_init", valid input is either "random" or "pca". '
                "Defaulting to random initialisation of weights"
            )
            som.random_weights_init(data)

        logger.info("------------- Training SOM -------------")
        som.train_batch(data, self.batch_size, verbose=True)  # random training
        self.xn = self.som_dim[0]
        self.yn = self.som_dim[1]
        self.map = som
        self.weights = som.get_weights()
        self.flatten_weights = self.weights.reshape(self.xn * self.yn, data.shape[1])
        logger.info("Training complete!")
        logger.info("----------------------------------------")

    def meta_cluster(self):
        """Perform meta-clustering. Implementation of Consensus clustering, following the paper
        https://link.springer.com/content/pdf/10.1023%2FA%3A1023949509487.pdf

        Returns
        -------
        None
        """

        assert self.map is not None, "SOM must be trained prior to meta-clustering; call train before meta_cluster"
        # initialize cluster
        cluster_ = ConsensusCluster(
            self.meta_clusterer,
            self.min_n,
            self.max_n,
            self.iter_n,
            resample_proportion=self.resample_proportion,
            verbose=self.verbose,
        )
        cluster_.fit(self.flatten_weights)  # fitting SOM weights into clustering algorithm

        self.meta_map = cluster_
        self.meta_bestk = cluster_.bestK  # the best number of clusters in range(min_n, max_n)

        # get the prediction of each weight vector on meta clusters (on bestK)
        self.meta_flatten = cluster_.predict_data(self.flatten_weights)
        self.meta_class = self.meta_flatten.reshape(self.xn, self.yn)

    def predict(self, data: np.ndarray):
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
        for i in progress_bar(range(data.shape[0]), verbose=self.verbose):
            xx = data[i, :]  # fetch the sample data
            winner = self.map.winner(xx)  # make prediction, prediction = the closest entry location in the SOM
            c = self.meta_class[winner]  # from the location info get cluster info
            labels.append(c)
        logger.info("---------------------------------------")
        return labels
