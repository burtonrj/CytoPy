import logging
from typing import Dict
from typing import Iterable
from typing import List
from typing import Optional
from typing import Union

import numpy as np
import pandas as pd
import polars as pl
from scipy.spatial import distance
from sklearn import metrics as sklearn_metrics
from sklearn.metrics import adjusted_mutual_info_score
from sklearn.metrics import adjusted_rand_score

from cytopy.feedback import progress_bar

logger = logging.getLogger(__name__)


class InternalMetric:
    def __init__(self, name: str, desc: str, **kwargs):
        self.name = name
        self.description = desc
        self.kwargs = kwargs

    def __call__(self, data: pd.DataFrame, features: List[str], labels: List[int]):
        return self


def center_dispersion(cluster: pl.DataFrame):
    cluster_center = cluster.mean()
    return pl.DataFrame(
        [
            cluster.apply(lambda x: np.linalg.norm(np.array(x) - cluster_center.to_numpy()) ** 2).sum()
            / cluster.shape[0]
        ]
    )


class BallHall(InternalMetric):
    def __init__(self, **kwargs):
        super().__init__(
            name="Ball Hall Index",
            desc="Ball-Hall Index is the mean of the mean dispersion across all clusters",
            **kwargs,
        )

    def __call__(self, data: pd.DataFrame, features: List[str], labels: List[int]):
        data = pl.DataFrame(data[features])
        data["labels"] = labels
        return data.groupby("labels").apply(center_dispersion).mean()[0, 0]


class BakerHubertGammaIndex(InternalMetric):
    def __init__(self, **kwargs):
        super().__init__(
            name="Baker-Hubert Gamma Index",
            desc="A measure of compactness, based on similarity between points in a cluster, "
            "compared to similarity with points in other clusters. Not memory efficient, use on small datasets.",
            **kwargs,
        )

    def __call__(self, data: pd.DataFrame, features: List[str], labels: List[int]):
        splus = 0
        sminus = 0
        data_matrix = data[features].copy()
        pair_dis = distance.pdist(data_matrix)
        num_pair = len(pair_dis)
        temp = np.zeros((len(labels), 2))
        temp[:, 0] = labels
        vec_b = distance.pdist(temp)
        # iterate through all the pairwise comparisons
        for i in range(num_pair - 1):
            for j in range(i + 1, num_pair):
                if vec_b[i] > 0 and vec_b[j] == 0:
                    if pair_dis[i] < pair_dis[j]:
                        splus += 1
                    if pair_dis[i] > vec_b[j]:
                        sminus += 1
                if vec_b[i] == 0 and vec_b[j] > 0:
                    if pair_dis[j] < pair_dis[i]:
                        splus += 1
                    if pair_dis[j] > vec_b[i]:
                        sminus += 1
        return (splus - sminus) / (splus + sminus)


class SilhouetteCoef(InternalMetric):
    def __init__(self, **kwargs):
        super().__init__(
            name="Silhouette Coefficient",
            desc="Compactness and connectedness combination that measures a ratio of within cluster "
            "distances to closest neighbors outside of cluster. This uses sklearn.metrics "
            "version of the Silhouette.",
            **kwargs,
        )

    def __call__(self, data: pd.DataFrame, features: List[str], labels: List[int]):
        return sklearn_metrics.silhouette_score(data[features].values, labels=labels, **self.kwargs)


class DaviesBouldinIndex(InternalMetric):
    def __init__(self, **kwargs):
        super().__init__(
            name="Davies-Bouldin index",
            desc="The average similarity between clusters. Similarity is defined as the "
            "ratio of within-cluster distances to between-cluster distances. Clusters further "
            "apart and less dispersed will result in a better score.",
            **kwargs,
        )

    def __call__(self, data: pd.DataFrame, features: List[str], labels: List[int]):
        return sklearn_metrics.davies_bouldin_score(data[features].values, labels=labels)


class GPlusIndex(InternalMetric):
    def __init__(self, **kwargs):
        super().__init__(
            name="G-plus index",
            desc="The proportion of discordant pairs among all the pairs of distinct points - "
            "a measure of connectedness. Not memory efficient, use on small datasets.",
            **kwargs,
        )

    def __call__(self, data: pd.DataFrame, features: List[str], labels: List[int]):
        sminus = 0
        data_matrix = data[features].copy()
        pair_dis = distance.pdist(data_matrix)
        num_pair = len(pair_dis)
        temp = np.zeros((len(labels), 2))
        temp[:, 0] = labels
        vec_b = distance.pdist(temp)
        # iterate through all the pairwise comparisons
        for i in range(num_pair - 1):
            for j in range(i + 1, num_pair):
                if vec_b[i] > 0 and vec_b[j] == 0:
                    if pair_dis[i] > vec_b[j]:
                        sminus += 1
                if vec_b[i] == 0 and vec_b[j] > 0:
                    if pair_dis[j] > vec_b[i]:
                        sminus += 1
        return (2 * sminus) / (num_pair * (num_pair - 1))


class CalinskiHarabaszScore(InternalMetric):
    def __init__(self, **kwargs):
        super().__init__(
            name="Calinski and Harabasz score",
            desc="The score is defined as ratio between the within-cluster dispersion and the "
            "between-cluster dispersion",
            **kwargs,
        )

    def __call__(self, data: pd.DataFrame, features: List[str], labels: List[int]):
        return sklearn_metrics.calinski_harabasz_score(data[features].values, labels=labels)


default_internal_metrics = {
    "ball_hall": BallHall,
    "silhouette_coef": SilhouetteCoef,
    "davies_bouldin_index": DaviesBouldinIndex,
    "calinski_harabasz_score": CalinskiHarabaszScore,
}


def init_internal_metrics(metrics: Optional[List[Union[str, InternalMetric]]] = None):
    if metrics is None:
        return [x() for x in default_internal_metrics.values()]
    metric_objs = list()
    try:
        for x in metrics:
            if isinstance(x, str):
                metric_objs.append(default_internal_metrics[x]())
            else:
                assert isinstance(x, InternalMetric)
                metric_objs.append(x)
    except KeyError:
        logger.error(f"Invalid metric, must be one of {default_internal_metrics.keys()}")
        raise
    except AssertionError:
        logger.error(
            f"metrics must be a list of strings corresponding to default metrics "
            f"({default_internal_metrics.keys()}) and/or Metric objects"
        )
        raise


class ComparisonMetric:
    def __init__(self, name: str, desc: str, **kwargs):
        self.name = name
        self.description = desc
        self.kwargs = kwargs

    def __call__(self, labels_x: Iterable[int], labels_y: Iterable[int]):
        return self


class AdjustedMutualInfo(ComparisonMetric):
    def __init__(self):
        super().__init__(
            name="Adjusted Mutual Information",
            desc="Measure of mutual dependence between two sets whilst accounting for chance.",
        )

    def __call__(self, labels_x: Iterable[int], labels_y: Iterable[int]):
        return adjusted_mutual_info_score(labels_x, labels_y)


class AdjustedRandIndex(ComparisonMetric):
    def __init__(self):
        super().__init__(
            name="Adjusted Rand Index",
            desc="Similarity between clustering by considering all pairs of data and counting the "
            "paris assigned to the same or different clusters in either set of labels. Adjusted "
            "to correct for random chance.",
        )

    def __call__(self, labels_x: Iterable[int], labels_y: Iterable[int]):
        return adjusted_rand_score(labels_x, labels_y)


def comparison_matrix(
    cluster_labels: Dict[str, np.ndarray], method: Union[str, ComparisonMetric] = "adjusted_mutual_info"
) -> pd.DataFrame:
    if isinstance(method, str):
        try:
            method = {"adjusted_rand_index": AdjustedRandIndex(), "adjusted_mutual_info": AdjustedMutualInfo()}[method]
        except KeyError:
            logger.error(
                f"Invalid method, must be constructed ComparisonMetric or one of the following: "
                f"'adjusted_rand_index' or 'adjusted_mutual_info'"
            )
            raise

    data = pd.DataFrame(columns=list(cluster_labels.keys()), index=list(cluster_labels.keys()), dtype=float)
    names = list(cluster_labels.keys())
    for n1 in progress_bar(names):
        for n2 in names:
            if np.isnan(data.loc[n1, n2]):
                mi = float(method(cluster_labels[n1], cluster_labels[n2]))
                data.at[n1, n2] = mi
                data.at[n2, n1] = mi
    return data
