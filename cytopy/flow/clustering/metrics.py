import logging
from typing import List
from typing import Optional
from typing import Union

import numpy as np
import pandas as pd
import polars as pl
from scipy.spatial import distance
from sklearn import metrics as sklearn_metrics

logger = logging.getLogger(__name__)


class Metric:
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


class BallHall(Metric):
    def __init__(self, **kwargs):
        super().__init__(
            name="Ball Hall Index (Compactness)",
            desc="Ball-Hall Index is the mean of the mean dispersion across all clusters",
            **kwargs,
        )

    def __call__(self, data: pd.DataFrame, features: List[str], labels: List[int]):
        data = pl.DataFrame(data[features])
        data["labels"] = labels
        return data.groupby("labels").apply(center_dispersion).mean()[0, 0]


class BakerHubertGammaIndex(Metric):
    def __init__(self, **kwargs):
        super().__init__(
            name="Baker-Hubert Gamma Index (Compactness)",
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


class SilhouetteCoef(Metric):
    def __init__(self, **kwargs):
        super().__init__(
            name="Silhouette Coefficient (Compactness/Separation)",
            desc="Compactness and connectedness combination that measures a ratio of within cluster "
            "distances to closest neighbors outside of cluster. This uses sklearn.metrics "
            "version of the Silhouette.",
            **kwargs,
        )

    def __call__(self, data: pd.DataFrame, features: List[str], labels: List[int]):
        return sklearn_metrics.silhouette_score(data[features].values, labels=labels, **self.kwargs)


class DaviesBouldinIndex(Metric):
    def __init__(self, **kwargs):
        super().__init__(
            name="Davies-Bouldin index (Compactness/Separation)",
            desc="The average similarity between clusters. Similarity is defined as the "
            "ratio of within-cluster distances to between-cluster distances. Clusters further "
            "apart and less dispersed will result in a better score.",
            **kwargs,
        )

    def __call__(self, data: pd.DataFrame, features: List[str], labels: List[int]):
        return sklearn_metrics.davies_bouldin_score(data[features].values, labels=labels)


class GPlusIndex(Metric):
    def __init__(self, **kwargs):
        super().__init__(
            name="G-plus index (Connectedness)",
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


class CalinskiHarabaszScore(Metric):
    def __init__(self, **kwargs):
        super().__init__(
            name="Calinski and Harabasz score (Compactness/Separation)",
            desc="The score is defined as ratio between the within-cluster dispersion and the "
            "between-cluster dispersion",
            **kwargs,
        )

    def __call__(self, data: pd.DataFrame, features: List[str], labels: List[int]):
        return sklearn_metrics.calinski_harabasz_score(data[features].values, labels=labels)


inbuilt_metrics = {
    "ball_hall": BallHall,
    "silhouette_coef": SilhouetteCoef,
    "davies_bouldin_index": DaviesBouldinIndex,
    "calinski_harabasz_score": CalinskiHarabaszScore,
}


def init_metrics(metrics: Optional[List[Union[str, Metric]]] = None):
    if metrics is None:
        return [x() for x in inbuilt_metrics.values()]
    metric_objs = list()
    try:
        for x in metrics:
            if isinstance(x, str):
                metric_objs.append(inbuilt_metrics[x]())
            else:
                assert isinstance(x, Metric)
                metric_objs.append(x)
    except KeyError:
        logger.error(f"Invalid metric, must be one of {inbuilt_metrics.keys()}")
        raise
    except AssertionError:
        logger.error(
            f"metrics must be a list of strings corresponding to default metrics "
            f"({inbuilt_metrics.keys()}) and/or Metric objects"
        )
        raise
