from typing import *

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import scipy.cluster.hierarchy as hierarchical_cluster
import seaborn as sns
from scipy.spatial import distance as ssd
from sklearn.metrics import adjusted_mutual_info_score
from sklearn.metrics import normalized_mutual_info_score

from ...feedback import progress_bar
from .main import ClusteringError


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
            **kwargs
        )

    def plot_dendrogram(
        self,
        threshold: Union[str, float] = "avg",
        linkage: str = "average",
        figsize: Tuple[int, int] = (10, 5),
        linkage_kwargs: Optional[Dict] = None,
        **kwargs
    ):
        fig, ax = plt.subplots(figsize=figsize)
        linkage_kwargs = linkage_kwargs or {}
        lnk = self._linkage(linkage=linkage, **linkage_kwargs)
        hierarchical_cluster.dendrogram(lnk, orientation="top", color_threshold=threshold, ax=ax, **kwargs)
        return fig
