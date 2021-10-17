import logging
from copy import deepcopy
from typing import Dict
from typing import Optional
from typing import Tuple
from typing import Union

import numpy as np
import pandas as pd
from numba import jit
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import MinMaxScaler

from cytopy.feedback import progress_bar

logger = logging.getLogger(__name__)


@jit(no_python=True)
def dividing_one_d(data: np.array, d2: int, bin_n: int, index: Optional[np.array] = None):
    if index is None:
        index = np.arange(data.shape[0])
    x = data[index, d2]
    bin_list = [None for i in bin_n]
    temp_index = np.arange(x.shape[0])
    st, end = np.min(x), np.max(x)
    for i in range(st, end + 1):
        cur_index = np.equal(x[temp_index], i)
        tem_index = temp_index[cur_index]
        if tem_index.shape[0] > 0:
            bin_list[i] = index[tem_index]
            temp_index = temp_index[np.logical_not(cur_index)]
    return bin_list


def combine(data: np.array, index_list: np.array, d2: int) -> Dict:
    new_index_dict = {}
    for index, value in enumerate(index_list):
        if value is not None:
            th_index_list = dividing_one_d(data, d2, value)
            for k, i in enumerate(th_index_list):
                if i is not None:
                    new_index_dict[(index, k)] = i
    return new_index_dict


def dividing_bins(data: np.array, bin_n: int, verbose: bool = True) -> Dict:
    d = data.shape[1]
    th_index_list = dividing_one_d(data, d2=0, bin_n=bin_n)
    new_index_dict = combine(data=data, index_list=th_index_list, d2=1)
    if d > 2:
        next_d = {}
        new_d = {}
        for key, value in progress_bar(new_index_dict.items(), verbose=verbose, total=len(new_index_dict)):
            temp_dict = {key: value}
            for cur_d in range(2, d):
                new_d = {}
                for cur_key, cur_index in temp_dict.items():
                    th_index_list = dividing_one_d(data=data, d2=cur_d, bin_n=bin_n, index=cur_index)
                    for k, i in enumerate(th_index_list):
                        if i is not None:
                            new_d[cur_key + (k,)] = i
                if cur_d != d - 1:
                    temp_dict = deepcopy(new_d)
            for k, i in new_d.items():
                next_d[k] = i
        return next_d
    else:
        return new_index_dict


def unique_bins(data: np.array, bin_n: int, verbose: bool):
    idd = dividing_bins(data=data, bin_n=bin_n, verbose=verbose)
    id_list = list(idd.keys())
    counts = np.zeros(len(id_list), dtype=int)
    unique_index = np.zeros(data.shape[0], dtype=int)
    for k, i in enumerate(id_list):
        counts[k] = idd[i].shape[0]
        unique_index[idd[i]] = k
    return np.array(id_list), unique_index, counts


class FlowGrid:
    """
    CytoPy implementation of the FlowGrid algorithm, originally described in:

    Ye, X., Ho, J. Ultrafast clustering of single-cell flow cytometry data using FlowGrid.
    BMC Syst Biol 13, 35 (2019).
    https://doi.org/10.1186/s12918-019-0690-2

    This code has been adapted from https://github.com/VCCRI/FlowGrid/blob/master/FlowGrid.py

    The FlowGrid algorithm offers a scalable and fast density-based clustering solution by
    implementing grid-based clustering. The following hyperparamters are required:

    * min_den_b - the minimum density for a high density (core) bin
    * min_den_c - the minimum collective density for core bins
    * bin_n - the number of equal sized bins for each dimension
    * eps - the maximum distance of connected bins
    """

    def __init__(
        self,
        min_den_b: int = 3,
        bin_n: int = 14,
        eps: float = 1.5,
        min_den_c: int = 40,
        verbose: bool = True,
        nn_params: Optional[Dict] = None,
    ):
        self.min_den_b = min_den_b
        self.min_den_c = min_den_c
        self.eps = eps
        self.bin_n = bin_n
        self.bins_number = None
        self.nn_params = nn_params or {}
        self.verbose = verbose
        self.nn_params["algorithm"] = self.nn_params.get("algorithm", "kd_tree")
        self.nn_params["n_neighbors"] = self.nn_params.get("n_neighbors", 5)
        self.nn_params["n_jobs"] = self.nn_params.get("n_jobs", -1)

    def dividing_bins(self, data: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Divide data into equal sized bins. First data is normalised to a range of
        0 to bin_n-10**(-7).

        Returns
        -------
        Tuple(Numpy.Array, Numpy.Array, Numpy.Array)
        """
        logging.info("Normalising data and assign to bins")
        n, d = data.shape
        scaler = MinMaxScaler(feature_range=(0, self.bin_n - 10 ** (-7)), copy=False)
        scaler.fit(data)
        x = scaler.transform(data).astype(u"int8")

        if np.log(n) / np.log(10) > 4 and d < 11 and self.bin_n < 11:
            unique_array, unique_index, counts = unique_bins(data=x, bin_n=self.bin_n, verbose=self.verbose)
        else:
            unique_array, unique_index, counts = np.unique(x, return_inverse=True, return_counts=True, axis=0)
        return unique_array, unique_index, counts

    def density_query(
        self, unique_array: np.ndarray, counts: np.ndarray, nn_model: NearestNeighbors
    ) -> Tuple[Dict[int, np.ndarray], np.ndarray]:
        """
        The density query function determines the core bins. It starts with filtering out bins whose
        density is lower than min_den_b, followed by nearest neighbor search using radius_neighbors from
        Scikit-Learn. Finally, the collective density is calculated if the number of samples
        located in the bin is larger than 85% of connected bins.

        Parameters
        ----------
        unique_array: Numpy.Array
        counts: Numpy.Array
        nn_model: sklearn.neighbors.NearestNeighbors

        Returns
        -------
        Tuple[Dict[Numpy.Array], Numpy.Array]
        """
        logger.info("Querying bin density to determine core bins")
        self.bins_number = unique_array.shape[0]
        tf_array = np.greater(counts, self.min_den_b)
        index_array = np.arange(self.bins_number)
        check_index = index_array[tf_array]
        check_nn = unique_array[tf_array]
        filterd_size = check_nn.shape[0]
        neighborhoods = nn_model.radius_neighbors(check_nn, radius=self.eps, return_distance=False)
        n_neighbors = np.zeros(filterd_size)
        for k, neighbors in progress_bar(enumerate(neighborhoods), verbose=self.verbose, total=len(neighborhoods)):
            nn_list = counts[neighbors]
            key_n = counts[check_index[k]]
            if key_n >= np.percentile(nn_list, 85):
                n_neighbors[k] = np.sum(nn_list)
        core_non = np.where(n_neighbors >= self.min_den_c)[0]
        core_or_non = np.zeros(self.bins_number, dtype=bool)
        core_bin_index = check_index[core_non]
        core_or_non[core_bin_index] = True
        query_d = {}
        for core in core_non:
            query_d[check_index[core]] = neighborhoods[core]
        return query_d, core_or_non

    def bfs(self, query_d: Dict[int, np.ndarray], core_non: np.ndarray) -> np.ndarray:
        """
        Breadth first search to group the core bins with their connected bins.
        The initial setting is assigning all bin label as -1 which stands for noise
        and building a set object for a queue.
        In the first while loop, if the core bin has not been labelled, it will be pushed
        into the queue, while if it is labelled, filter function is applied to remove all
        the labelled bins in bin_list.
        The second level while loop is to connect the core bins with their connected bins.
        cur_bin is poped from queue. If it is not to be labelled, it will be label by index
        and if it is a core bin, all non-labelled bin is put into queue for the next iteration.

        Parameters
        ----------
        query_d: Dict[Numpy.Array]
        core_non: Numpy.Array

        Returns
        -------
        Numpy.Array
        """
        logger.info("Grouping core bins with their connected bins")
        bin_labels = np.zeros(self.bins_number) - 1
        index = 0
        queue = set()
        bin_list = list(query_d.keys())
        while bin_list:
            core_bin = bin_list.pop()
            if bin_labels[core_bin] == -1:
                queue.add(core_bin)
                index += 1
                while len(queue) > 0:
                    cur_bin = queue.pop()
                    if bin_labels[cur_bin] == -1:
                        bin_labels[cur_bin] = index
                        if core_non[cur_bin]:
                            queue.update(list(filter(lambda x: bin_labels[x] == -1, query_d[cur_bin])))
            else:
                bin_list = list(filter(lambda x: bin_labels[x] == -1, bin_list))
        return bin_labels

    def density_scan(self, unique_array: np.ndarray, counts: np.ndarray) -> np.ndarray:
        """
        Group core bins with their connected bins. It starts with generating a nearest neighbors
        tree (uses KD-tree algorithm by default; can alter nearest neighbors algorithm with
        nn_params in constructor). Then calls 'density_query' to determine the core bins, then
        'bfs' function to group connected bins.

        Parameters
        ----------
        unique_array: Numpy.Array
        counts: Numpy.Array

        Returns
        -------
        Numpy.Array
        """
        logger.info("Computing nearest neighbors tree")
        nn_model = NearestNeighbors(**self.nn_params)
        nn_model.fit(unique_array)
        neighbors_d, core_non = self.density_query(unique_array=unique_array, counts=counts, nn_model=nn_model)
        return self.bfs(query_d=neighbors_d, core_non=core_non)

    def fit_predict(self, data: Union[pd.DataFrame, np.ndarray]):
        """
        Cluster given data using FlowGrid

        Parameters
        ----------
        data: Pandas.DataFrame or Numpy.Array

        Returns
        -------
        Numpy.Array
            Cluster labels
        """
        if isinstance(data, pd.DataFrame):
            data = data.values
        unique_array, unique_index, counts = self.dividing_bins(data=data)
        bin_labels = self.density_scan(unique_array, counts)
        logger.info("Clustering complete!")
        return bin_labels[unique_index]
