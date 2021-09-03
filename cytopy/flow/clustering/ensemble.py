import logging
from collections import defaultdict
from typing import Callable
from typing import Dict
from typing import Iterable
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import scipy.cluster.hierarchy as hierarchical_cluster
import seaborn as sns
from scipy.spatial import distance as ssd
from sklearn.metrics import adjusted_mutual_info_score
from sklearn.metrics import normalized_mutual_info_score

from ...feedback import progress_bar
from .main import ClusteringError

logger = logging.getLogger("clustering.ensemble")


def valid_labels(func: Callable):
    def wrapper(self, cluster_labels: List[int], *args, **kwargs):
        if len(cluster_labels) != self.data.shape[0]:
            raise ClusteringError(
                f"cluster_idx does not match the number of events. Did you use a valid "
                f"finishing technique? {len(cluster_labels)} != {self.data.shape[0]}"
            )
        return func(self, cluster_labels, *args, **kwargs)

    return wrapper


def get_adjacent_cliques(clique: frozenset, membership_dict: Dict):
    adjacent_cliques = set()
    for n in clique:
        for adj_clique in membership_dict[n]:
            if clique != adj_clique:
                adjacent_cliques.add(adj_clique)
    return adjacent_cliques


def get_percolated_cliques(graph: nx.Graph, k: int):
    perc_graph = nx.Graph()
    cliques = [frozenset(c) for c in nx.find_cliques(graph) if len(c) >= k]
    perc_graph.add_nodes_from(cliques)

    # First index which nodes are in which cliques
    membership_dict = defaultdict(list)
    for clique in cliques:
        for node in clique:
            membership_dict[node].append(clique)

    # For each clique, see which adjacent cliques percolate
    for clique in cliques:
        for adj_clique in get_adjacent_cliques(clique, membership_dict):
            if len(clique.intersection(adj_clique)) >= (k - 1):
                perc_graph.add_edge(clique, adj_clique)

    # Connected components of clique graph with perc edges
    # are the percolated cliques
    for component in nx.connected_components(perc_graph):
        yield frozenset.union(*component)


def gather_single_partition(labels: np.ndarray):
    n = len(labels)
    co_matrix = np.zeros(shape=(n, n))
    clusterid_list = np.unique(labels)
    for clusterid in clusterid_list:
        itemindex = np.where(labels == clusterid)[0]
        for i, x in enumerate(itemindex[0:]):
            co_matrix[x, x] += 1
            for j, y in enumerate(itemindex[i + 1 :]):
                co_matrix[x, y] += 1
                co_matrix[y, x] += 1
    return co_matrix


class CoMatrix:
    def __init__(
        self,
        data: pd.DataFrame,
        clustering_permuations: Dict,
        features: list,
        index: Optional[str] = None,
    ):
        self.data = data
        self.features = features
        self.clusterings = clustering_permuations
        self.n = data.shape[0]
        self.n_ensembles = len(clustering_permuations)
        self.co_matrix = self.gather_partitions()
        self.avg_dist = np.mean(ssd.squareform(1 - self.co_matrix))
        self.index = index

    def gather_partitions(self):
        co_matrix = np.zeros(shape=(self.n, self.n))
        for labels in [x["labels"] for x in self.clusterings.values()]:
            co_matrix += gather_single_partition(labels)
        co_matrix_f = co_matrix / self.n_ensembles
        header = np.arange(0, self.data.shape[0])
        co_matrix_df = pd.DataFrame(index=header, data=co_matrix_f, columns=header)
        return co_matrix_df

    def pairwise_list(self):
        """
        Reshapes the co-occurrence dataframe matrix into a list, so it can easily be ordered and explored

        Returns
        -------
        df: pandas dataframe
            with a row entry index of object1_object2 and a co-occurrence column labled 'pairwise'
        """
        # return a new dataframe in list with index equal to the pairs being
        # considered
        df = pd.DataFrame(columns=["site_1", "site_2", "pairwise"])
        header = list(self.co_matrix)

        for i in range(0, len(header) - 1):
            for x in range(i + 1, len(header)):
                s = pd.Series()
                h = header[i]
                j = header[x]
                s["pairwise"] = self.co_matrix.loc[h, j]
                s["site_1"] = h
                s["site_2"] = j
                s.name = f"{h}; {j}"
                df = df.append(s)
        return df

    def _linkage(self, linkage: str = "average", **kwargs):
        kwargs["metric"] = kwargs.get("metric", "euclidean")
        kwargs["optimal_ordering"] = kwargs.get("optimal_ordering", True)
        lnk = hierarchical_cluster.linkage(ssd.squareform(1 - self.co_matrix), method=linkage, **kwargs)
        return lnk

    def cluster_co_occurrence(self, threshold: Union[str, float] = "avg", linkage: str = "average", **kwargs):
        """
        Generates a final clustering solution for an ensemble. Cut the clustering at
        the given threshold and returns the labels from the resulting cut. Scipy Hierarchical clustering
        tools are used to generate finished labels.

        See 'plot_matrix' and 'plot_dendrogram' method to visually explore the cuts at different thresholds.

        Parameters
        ----------
        threshold: float or "avg" (default="avg")
            If given value "avg", will use the average pairwise distance as threshold
        linkage: str (default="average")

        Returns
        -------
        Numpy.Array
        """
        if threshold == "avg":
            threshold = self.avg_dist
        lnk = self._linkage(linkage=linkage, **kwargs)
        return hierarchical_cluster.fcluster(lnk, threshold, "distance")

    def majority_vote(self, threshold: float = 0.5):
        """
        Generates a final clustering solution for an ensemble.

        Returns
        -------

        """
        n = self.co_matrix.shape[0]
        labels = np.zeros(n).astype(int)
        curr_cluster = 1
        x = self.co_matrix.values

        for i in progress_bar(range(0, n)):
            for j in range(i + 1, n):
                if x[i, j] > threshold:
                    # the clusters should be set to the same value and if both belong to an existing cluster, all
                    # members of the clusters should be joined
                    if labels[i] and labels[j]:
                        cluster_num = min(labels[i], labels[j])
                        cluster_to_change = max(labels[i], labels[j])
                        idx = [ii for ii, c in enumerate(labels) if c == cluster_to_change]
                        labels[idx] = cluster_num
                    elif not labels[i] and not labels[j]:
                        # a new cluster
                        labels[i], labels[j] = curr_cluster, curr_cluster
                        curr_cluster += 1
                    else:
                        # one of them is in a cluster and the other is not, assign to the same cluster
                        cluster_num = max(labels[i], labels[j])
                        labels[i], labels[j] = cluster_num, cluster_num
                else:
                    # don't join them and give them an assignment the first time they are traversed
                    if not labels[i]:
                        labels[i] = curr_cluster
                        curr_cluster += 1
                    if not labels[j]:
                        labels[j] = curr_cluster
                        curr_cluster += 1
        # Cleanup: if there are any cluster numbers that jump, move them down
        clusters = np.sort(np.unique(labels))
        for i in range(0, len(clusters) - 1):
            if clusters[i + 1] != clusters[i] + 1:
                cluster_num = clusters[i] + 1
                idx = [j for j, v in enumerate(labels) if v == clusters[i + 1]]
                labels[idx] = cluster_num
                clusters[i + 1] = cluster_num
        return labels

    def graph_closure(self, threshold: float, clique_size: int = 3):
        binary_matrix = np.array(self.co_matrix.values >= threshold).astype(int)
        graph = nx.from_numpy_matrix(binary_matrix)
        y = get_percolated_cliques(graph=graph, k=clique_size)
        z = list(y)
        labels = np.empty(self.data.shape[0])
        cluster_num = 0
        while z:
            idx = list(z.pop())
            labels[idx] = int(cluster_num)
            cluster_num += 1
        return labels.astype(int)

    def plot_matrix(self, linkage: str = "average", **kwargs):
        kwargs["standard_scale"] = kwargs.get("standard_scale", 1)
        kwargs["figsize"] = kwargs.get("figsize", (10, 10))
        kwargs["cmap"] = kwargs.get("cmap", "Spectral_r")
        return sns.clustermap(
            data=self.data[self.features],
            row_linkage=self._linkage(linkage=linkage),
            **kwargs,
        )

    def plot_dendrogram(
        self,
        threshold: Union[str, float] = "avg",
        linkage: str = "average",
        figsize: Tuple[int, int] = (10, 5),
        linkage_kwargs: Optional[Dict] = None,
        **kwargs,
    ):
        fig, ax = plt.subplots(figsize=figsize)
        linkage_kwargs = linkage_kwargs or {}
        lnk = self._linkage(linkage=linkage, **linkage_kwargs)
        label_vec = None if self.index is None else self.data[self.index]
        hierarchical_cluster.dendrogram(
            lnk,
            orientation="top",
            color_threshold=threshold,
            ax=ax,
            labels=label_vec,
            **kwargs,
        )
        return fig


def gather_partitions(labels: List[List[int]], n: int):
    h = len(labels)
    list_parts = np.concatenate(labels).reshape(h, n)

    y = list()
    [y.append(list_parts[:, i]) for i in range(n)]

    y = pd.DataFrame(y, columns=np.arange(h))
    y.index.name = "objs"
    y.columns.name = "partition"
    return y


def generate_kj(y: pd.DataFrame):
    k = []
    for i in range(y.shape[1]):
        k.append(y[i].dropna().unique())
    return k


def init_v(y: pd.DataFrame, k: int, kj: List):
    v = list()
    [v.append([]) for _ in range(y.shape[1])]

    for i in range(y.shape[1]):
        for j in range(k):
            aux = abs(np.random.randn(len(kj[i])))
            v[i].append(aux / sum(aux))
    return v


def sigma(a: int, b: int):
    return 1 if a == b else 0


def vec_simga(x: Iterable, k: int):
    return np.asarray(list(map(lambda i: 1 if i == k else 0, x)))


def pi_consensus(expz: np.ndarray):
    max_expz_values = {}
    pi_finishing = {}
    labels = []
    for i in range(expz.shape[0]):
        max_expz_values[i] = max(expz[i, :])
        pi_finishing[i] = []

        for j in range(expz.shape[1]):
            if max_expz_values[i] == expz[i, j]:
                pi_finishing[i].append(j + 1)
                labels.append(j + 1)
    return pi_finishing, labels


class MixtureModel:
    def __init__(self, data: pd.DataFrame, clustering_permuations: Dict):
        self.labels = [x["labels"] for _, x in clustering_permuations.items()]
        self.data = data
        self.n = data.shape[0]
        self.y = gather_partitions(labels=self.labels, n=self.n)
        self.kj = generate_kj(self.y)
        self.expz = None
        self.alpha = None
        self.v = None

    def update_alpha(self):
        for m in range(self.alpha.shape[0]):
            self.alpha[m] = float(sum(self.expz[:, m]) / float(sum(sum(self.expz))))

    def update_v(self):
        for j in range(self.y.shape[1]):
            for m in range(self.alpha.shape[0]):
                ix = 0
                for k in self.kj[j]:
                    num = sum(vec_simga(self.y.iloc[:, j], k) * np.array(self.expz[:, m]))
                    den = 0
                    for kx in self.kj[j]:
                        den += sum(vec_simga(self.y.iloc[:, j], kx) * np.asarray(self.expz[:, m]))
                    self.v[j][m][ix] = float(num) / float(den)
                    ix += 1

    def expectation(self):
        M = self.expz.shape[0]
        n_elem = self.y.shape[0]

        for m in range(M):
            for i in range(n_elem):
                prod1 = 1
                num = 0

                for j in range(self.y.shape[1]):
                    ix1 = 0
                    for k in self.kj[j]:
                        prod1 *= self.v[j][m] ** sigma(self.y.iloc[i][j], k)
                        ix1 += 1
                num += self.alpha[m] * prod1

                den = 0
                for n in range(M):

                    prod2 = 1
                    for j2 in range(self.y.shape[1]):
                        ix2 = 0
                        for k in self.kj[j2]:
                            prod2 *= self.v[j2][n][ix2] ** sigma(self.y.iloc[i][j2], k)
                            ix2 += 1
                    den += self.alpha[n] * prod2

                self.expz[i][m] = float(num) / float(den)

    def __call__(self, k: int, iterations: int = 10):
        self.expz = np.zeros(self.y.shape[0] * k).reshape(self.y.shape[0], k)
        self.alpha = np.ones(k) / k
        self.v = init_v(y=self.y, k=k, kj=self.kj)
        for i in progress_bar(range(iterations)):
            # Expectation
            self.expectation()
            # Maximisation
            self.update_alpha()
            self.update_v()
        pi_finishing, labels = pi_consensus(expz=self.expz)
        return np.asarray(labels)


def mutual_info(a: List[int], b: List[int], method: str):
    methods = {
        "adjusted": adjusted_mutual_info_score,
        "normalized": normalized_mutual_info_score,
    }
    try:
        return methods[method](a, b)
    except KeyError:
        ClusteringError("Mutual information method must be either 'adjusted' or 'normalized'")


class MutualInfo:
    def __init__(self, clusterings: Dict, method: str):
        if method not in ["adjusted", "normalized"]:
            raise ClusteringError("Mutual information method must be either 'adjusted' or 'normalized'")
        self.labels = {cluster_name: data["labels"] for cluster_name, data in clusterings.items()}
        self.data = pd.DataFrame(columns=list(self.labels.keys()), index=list(self.labels.keys()))
        names = list(self.labels.keys())
        for n1 in progress_bar(names):
            for n2 in names:
                if np.isnan(self.data.loc[n1, n2]):
                    mi = mutual_info(self.labels[n1], self.labels[n2], method=method)
                    self.data.at[n1, n2] = mi
                    self.data.at[n2, n1] = mi
        self.avg_dist = np.mean(ssd.squareform(1 - self.data))

    def _linkage(self, method: str = "average", **kwargs):
        arr = 1 - self.data.values
        arr[np.where(arr < 0)] = 0.0
        dist_vec = arr[np.triu_indices(arr.shape[0], 1)]
        kwargs["metric"] = kwargs.get("metric", "euclidean")
        kwargs["optimal_ordering"] = kwargs.get("optimal_ordering", True)
        return hierarchical_cluster.linkage(dist_vec, method=method, **kwargs)

    def cluster_mutual_info(self, threshold: Union[str, float] = "avg", linkage: str = "average", **kwargs):
        if threshold == "avg":
            threshold = self.avg_dist
        lnk = self._linkage(linkage=linkage, **kwargs)
        return hierarchical_cluster.fcluster(lnk, threshold, "distance")

    def plot_matrix(self, linkage: str = "average", **kwargs):
        kwargs["standard_scale"] = kwargs.get("standard_scale", 1)
        kwargs["figsize"] = kwargs.get("figsize", (10, 10))
        kwargs["cmap"] = kwargs.get("cmap", "Spectral_r")
        return sns.clustermap(
            data=self.data,
            row_linkage=self._linkage(linkage=linkage),
            col_linkage=self._linkage(linkage=linkage),
            **kwargs,
        )

    def plot_dendrogram(
        self,
        threshold: Union[str, float] = "avg",
        linkage: str = "average",
        figsize: Tuple[int, int] = (10, 5),
        linkage_kwargs: Optional[Dict] = None,
        **kwargs,
    ):
        fig, ax = plt.subplots(figsize=figsize)
        linkage_kwargs = linkage_kwargs or {}
        lnk = self._linkage(linkage=linkage, **linkage_kwargs)
        hierarchical_cluster.dendrogram(lnk, orientation="top", color_threshold=threshold, ax=ax, **kwargs)
        return fig
