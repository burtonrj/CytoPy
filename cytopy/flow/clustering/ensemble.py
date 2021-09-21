import logging
from typing import Callable
from typing import Dict
from typing import Iterable
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union

import kahypar
import numpy as np
import pandas as pd
import pymetis
import seaborn as sns
from scipy import sparse
from sklearn.metrics import adjusted_mutual_info_score
from sklearn.metrics import adjusted_rand_score

from ...feedback import progress_bar
from .metrics import inbuilt_metrics
from .metrics import Metric

logger = logging.getLogger(__name__)


def adjusted_score(a: List[int], b: List[int], method: str):
    methods = {
        "adjusted_mutual_info": adjusted_mutual_info_score,
        "adjusted_rand_score": adjusted_rand_score,
    }
    try:
        return methods[method](a, b)
    except KeyError:
        ValueError(f"Method must be one of {methods.keys()}")


def comparison_matrix(clustering_permutations: Dict, method: str = "adjusted_mutual_info") -> pd.DataFrame:
    labels = {cluster_name: data["labels"] for cluster_name, data in clustering_permutations.items()}
    data = pd.DataFrame(columns=list(labels.keys()), index=list(labels.keys()), dtype=float)
    names = list(labels.keys())
    for n1 in progress_bar(names):
        for n2 in names:
            if np.isnan(data.loc[n1, n2]):
                mi = float(adjusted_score(labels[n1], labels[n2], method=method))
                data.at[n1, n2] = mi
                data.at[n2, n1] = mi
    return data


def create_hypergraph(label_matrix: np.ndarray) -> sparse.csc_matrix:
    """
    Create the incidence matrix of labels hypergraph.
    Code adapted from ClusterEnsembles authored by Takehiro Sano

    Parameters
    ----------
    label_matrix: Numpy.Array
        Matrix of clustering labels of shape k, n where k is the number of clustering solutions and
        n is the number of observations clustered

    Returns
    -------
    sparse.csc_matrix
        Incidence matrix
    """
    incidence_matrix = []
    for labels in label_matrix:
        labels = np.nan_to_num(labels, nan=float("inf"))
        unique_labels = np.unique(labels)
        label_id_mapping = dict(zip(unique_labels, np.arange(len(unique_labels))))
        converted = [label_id_mapping[x] for x in labels]
        h = np.identity(len(unique_labels), dtype=np.int8)[converted]
        if float("inf") in label_id_mapping.keys():
            h = np.delete(h, obj=label_id_mapping[float("inf")], axis=1)
        incidence_matrix.append(sparse.csc_matrix(h))
    return sparse.hstack(incidence_matrix)


def to_pymetis_format(adj_mat: sparse.csc_matrix) -> (List, List, List):
    xadj = [0]
    adjncy = []
    eweights = []
    n_rows = adj_mat.shape[0]
    adj_mat = adj_mat.tolil()

    for i in range(n_rows):
        row = adj_mat.getrow(i)
        idx_row, idx_col = row.nonzero()
        val = row[idx_row, idx_col]
        adjncy += list(idx_col)
        eweights += list(val.toarray()[0])
        xadj.append(len(adjncy))

    return xadj, adjncy, eweights


def cspa(label_matrix: np.ndarray, k: int) -> np.ndarray:
    """
    Cluster-based Similarity Partitioning Algorithm (CSPA). See http://strehl.com/diss/node80.html
    for detailed description. Implementation adapted from ClusterEnsembles authored by Takehiro Sano

    Parameters
    ----------
    label_matrix: Numpy.Array
        Matrix of clustering labels of shape k, n where k is the number of clustering solutions and
        n is the number of observations clustered
    k: int
        Number of clusters in consensus

    Returns
    -------
    Numpy.Array
        Consensus labels
    """
    if label_matrix.shape[1] > 5000:
        logger.warning("Other solutions are recommended for ")
    logger.info(f"Generating hypergraph with {label_matrix.shape[0]} partitions...")
    incidence_matrix = create_hypergraph(label_matrix=label_matrix)
    logger.info("Computing similarity matrix...")
    similarity_matrix = incidence_matrix * incidence_matrix.T
    logger.info("Formatting for PyMetis...")
    xadj, adjncy, eweights = to_pymetis_format(similarity_matrix)
    logger.info("Computing consensus")
    membership = pymetis.part_graph(nparts=k, xadj=xadj, adjncy=adjncy, eweights=eweights)[1]
    return np.array(membership)


def hgpa(label_matrix: np.ndarray, k: int, random_state: int = 42) -> np.ndarray:
    """
    HyperGraph Partitioning Algorithm

    Parameters
    ----------
    label_matrix: Numpy.Array
        Matrix of clustering labels of shape k, n where k is the number of clustering solutions and
        n is the number of observations clustered
    k: int
        Number of clusters in consensus
    random_state: int (default=42)

    Returns
    -------
    Numpy.Array
        Consensus labels
    """
    logger.info(f"Generating hypergraph with {label_matrix.shape[0]} partitions...")
    incidence_matrix = create_hypergraph(label_matrix)
    n_nodes, n_nets = incidence_matrix.shape

    node_weights = [1] * n_nodes
    edge_weights = [1] * n_nets

    hyperedge_indices = [0]
    hyperedges = []
    incidence_t = incidence_matrix.T
    for i in progress_bar(range(n_nets)):
        h = incidence_t.getrow(i)
        idx_row, idx_col = h.nonzero()
        hyperedges += list(idx_col)
        hyperedge_indices.append(len(hyperedges))

    hypergraph = kahypar.Hypergraph(n_nodes, n_nets, hyperedge_indices, hyperedges, k, edge_weights, node_weights)

    logger.info("")


def hbgf(label_matrix: np.ndarray, k: int) -> np.ndarray:
    """
    Hybrid Bipartite Graph Formulation (HBGF).

    Parameters
    ----------
    label_matrix: Numpy.Array
        Matrix of clustering labels of shape k, n where k is the number of clustering solutions and
        n is the number of observations clustered
    k: int
        Number of clusters in consensus

    Return
    -------
    Numpy.Array
        Consensus labels
    """
    logger.info(f"Generating hypergraph with {label_matrix.shape[0]} partitions...")
    incidence_matrix = create_hypergraph(label_matrix)
    n_rows, n_cols = incidence_matrix.shape
    logger.info("Creating bipartite graph...")
    bipart = sparse.bmat(
        [
            [sparse.dok_matrix((n_cols, n_cols)), incidence_matrix.T],
            [incidence_matrix, sparse.dok_matrix((n_rows, n_rows))],
        ]
    )
    logger.info("Formatting for PyMetis...")
    xadj, adjncy, _ = to_pymetis_format(bipart)
    logger.info("Computing consensus")
    membership = pymetis.part_graph(nparts=k, xadj=xadj, adjncy=adjncy, eweights=None)[1]
    label_ce = np.array(membership[n_cols:])
    return label_ce


def select_nclass(
    data: pd.DataFrame,
    k: List[int],
    consensus_method: str = "hbgf",
    metrics: Optional[List[Metric]] = None,
    resample: int = 20,
    sample_size: int = 1000,
    **kwargs,
) -> sns.FacetGrid:
    return sns.lineplot(**kwargs)
