import logging
from collections import Counter
from collections import defaultdict
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union

import hdmedians as hd
import numpy as np
import pandas as pd
import seaborn as sns
import setuptools
from ClusterEnsembles import ClusterEnsembles
from joblib import delayed
from joblib import Parallel
from matplotlib import pyplot as plt
from numba import jit
from scipy.cluster.hierarchy import dendrogram
from scipy.cluster.hierarchy import fclusterdata
from scipy.cluster.hierarchy import linkage
from scipy.spatial import distance
from sklearn.base import ClusterMixin
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler

from ..plotting.general import box_swarm_plot
from ..plotting.general import ColumnWrapFigure
from .clustering import Clustering
from .clustering import ClusteringError
from .clustering import ClusterMethod
from .clustering import init_cluster_method
from .clustering import simpson_di
from .metrics import comparison_matrix
from .metrics import InternalMetric
from cytopy.data.experiment import Experiment
from cytopy.feedback import progress_bar

logger = logging.getLogger(__name__)


def distortion_score(data: pd.DataFrame, features: List[str], clusters: List[str], metric: str = "euclidean") -> Dict:
    score = {}
    clusters = [c for c in clusters if c in data.columns]
    for c in clusters:
        df = data[data[c] == 1]
        if df.shape[0] == 0:
            continue
        center = df[features].mean().values
        distances = pairwise_distances(df[features], center.reshape(1, -1), metric=metric)
        score[c] = (distances ** 2).sum() / df.shape[0]
    return score


def majority_vote_with_distortion_score_weighting(
    row: pd.Series,
    cluster_column_names: List[str],
    consensus_clusters: pd.DataFrame,
    distortion_score_dict: Dict[str, Dict[str, float]],
):
    clusters = row[cluster_column_names].replace({0: None}).dropna().index.tolist()
    clusters = [c for c in clusters if c in consensus_clusters.index.tolist()]

    consensus_cluster_score = []
    for cid, cc_data in consensus_clusters.loc[clusters].groupby("cluster_label"):
        consensus_cluster_score.append(
            [
                cid,
                cc_data.shape[0] / sum([distortion_score_dict.get(i) for i in cc_data.index.values]),
            ]
        )
    return sorted(consensus_cluster_score, key=lambda x: x[1])[::-1][0][0]


def majority_vote_with_distance_weighting(
    row: pd.Series,
    cluster_column_names: List[str],
    consensus_clusters: pd.DataFrame,
    distance_df: pd.DataFrame,
):
    clusters = row[cluster_column_names].replace({0: None}).dropna().index.tolist()
    clusters = [c for c in clusters if c in consensus_clusters.index.tolist()]

    distance_df = distance_df[distance_df.cell_id == row.name].set_index("cluster")

    consensus_cluster_score = []
    for cid, cc_data in consensus_clusters.loc[clusters].groupby("cluster_label"):
        consensus_cluster_score.append(
            [cid, cc_data.shape[0] / sum([distance_df.loc[i]["distance"] for i in cc_data.index.values])]
        )
    return sorted(consensus_cluster_score, key=lambda x: x[1])[::-1][0][0]


def majority_vote(row: pd.Series, cluster_column_names: List[str], consensus_clusters: pd.DataFrame):
    clusters = row[cluster_column_names].replace({0: None}).dropna().index.tolist()
    clusters = [c for c in clusters if c in consensus_clusters.index.tolist()]
    consensus_cluster_labels = Counter([consensus_clusters.loc[cid, "cluster_label"] for cid in clusters])
    score = 0
    winner = None

    for label, count in consensus_cluster_labels.items():
        if count == score:
            logger.warning(
                f"{label} and {winner} have equal scores, observation will be assigned to {winner}, "
                f"to avoid this provide weights for cluster voting."
            )
        if count > score:
            score = count
            winner = label
    return winner


def _gather_partitions(cluster_labels: List[List[int]]):
    h = len(cluster_labels)
    n = len(cluster_labels[0])
    parts = np.concatenate(cluster_labels).reshape(h, n)
    y = pd.DataFrame([parts[:, i] for i in range(n)], columns=np.arange(h))
    y.index.name = "objs"
    y.columns.name = "partition"
    return y


def _generate_kj(y: pd.DataFrame):
    k = y.apply(lambda x: np.unique(x.dropna())).values
    return [tuple(x) for x in k]


def _init_alpha(n_consensus_clusters: int):
    return np.ones(n_consensus_clusters) / n_consensus_clusters


def _calc_v(kj_i: Tuple):
    aux = abs(np.random.randn(len(kj_i)))
    return aux / sum(aux)


def _init_v(n_partitions: int, n_consensus_clusters: int, kj: List[Tuple]):
    return [[_calc_v(kj[i]) for _ in range(n_consensus_clusters)] for i in range(n_partitions)]


def _init_expz(n: int, n_consensus_clusters):
    return np.zeros(n * n_consensus_clusters).reshape(n, n_consensus_clusters)


def _init_parameters(n_consensus_clusters: int, n_obs: int, n_partitions: int, kj: List[Tuple]):
    alpha = _init_alpha(n_consensus_clusters=n_consensus_clusters)
    v = _init_v(n_partitions=n_partitions, n_consensus_clusters=n_consensus_clusters, kj=kj)
    exp_z = _init_expz(n=n_obs, n_consensus_clusters=n_consensus_clusters)
    return alpha, v, exp_z


def _compute_prod(y, kj, v, m):
    prod_values = []
    for i in range(y.shape[0]):
        prod = 1
        for j in range(y.shape[1]):
            ix = 0
            for k in kj[j]:
                if y[i, j] == k:
                    prod *= v[j][m][ix] ** 1
                else:
                    prod *= v[j][m][ix] ** 0
                ix += 1
        prod_values.append(prod)
    return prod_values


def _expectation(y: np.array, exp_z: np.array, kj: np.array, v: list, alpha: np.array):
    with Parallel(n_jobs=-1) as parallel:
        prod_values = parallel(delayed(_compute_prod)(y, kj, v, m) for m in range(exp_z.shape[1]))
    for m in range(exp_z.shape[1]):
        for i in range(y.shape[0]):
            prod = prod_values[m][i]
            num, den = 0, 0
            num += alpha[m] * prod
            for n in range(exp_z.shape[1]):
                prod = prod_values[n][i]
                den += alpha[n] * prod
            exp_z[i][m] = float(num) / float(den)
    return exp_z


class MixtureModel:
    def __init__(self, cluster_labels: List[List[int]], n_consensus_clusters: int, iterations: int = 10, verbose=True):
        if not len(set([len(x) for x in cluster_labels])) == 1:
            raise ValueError("Cluster solutions must be of equal length")
        self.cluster_labels = cluster_labels
        self.n = len(self.cluster_labels[0])
        self.iterations = iterations
        self.n_consensus_clusters = n_consensus_clusters
        self.y = _gather_partitions(cluster_labels=cluster_labels)
        self.kj = _generate_kj(y=self.y)
        self.alpha, self.v, self.exp_z = _init_parameters(
            n_consensus_clusters=n_consensus_clusters, n_obs=self.y.shape[0], n_partitions=self.y.shape[1], kj=self.kj
        )
        self.labels = []
        self.pi_finishing = {}
        self.verbose = verbose

    @staticmethod
    def _sigma(a, b):
        return 1 if a == b else 0

    def expectation(self):
        """
        Compute the Expectation (ExpZ) according to parameters.
        Obs: y(N,H) Kj(H) alpha(M) v(H,M,K(j)) ExpZ(N,M)
        """

        self.exp_z = _expectation(
            y=self.y.values.astype(np.int8), exp_z=self.exp_z, kj=self.kj, v=self.v, alpha=self.alpha
        )
        return self.exp_z

    @staticmethod
    def _vec_sigma(vec, k):
        aux = []
        for i in vec:
            if i == k:
                aux.append(1)
            else:
                aux.append(0)
        return np.asarray(aux)

    def update_alpha(self):
        for m in range(self.alpha.shape[0]):
            self.alpha[m] = float(sum(self.exp_z[:, m])) / float(sum(sum(self.exp_z)))
        return self.alpha

    def update_v(self):
        for j in range(self.y.shape[1]):
            for m in range(self.alpha.shape[0]):
                ix = 0
                for k in self.kj[j]:
                    num = sum(self._vec_sigma(self.y.iloc[:, j], k) * np.array(self.exp_z[:, m]))
                    den = 0
                    for kx in self.kj[j]:
                        den += sum(self._vec_sigma(self.y.iloc[:, j], kx) * np.asarray(self.exp_z[:, m]))
                    self.v[j][m][ix] = float(num) / float(den)
                    ix += 1
        return self.v

    def maximization(self):
        self.alpha = self.update_alpha()
        self.v = self.update_v()
        return self.alpha, self.v

    def _pi_consensus(self):
        max_expz_values = {}
        pi_finishing = {}
        labels = []
        for i in range(self.exp_z.shape[0]):
            max_expz_values[i] = max(self.exp_z[i, :])
            pi_finishing[i] = []

            for j in range(self.exp_z.shape[1]):
                if max_expz_values[i] == self.exp_z[i, j]:
                    pi_finishing[i].append(j + 1)
                    labels.append(j + 1)
        return pi_finishing, np.asarray(labels)

    def em_process(self):
        for _ in progress_bar(range(self.iterations), total=self.iterations, verbose=self.verbose):
            self.exp_z = self.expectation()
            self.alpha, self.v = self.maximization()

        self.pi_finishing, self.labels = self._pi_consensus()
        return self.labels


class EnsembleClustering(Clustering):
    def __init__(
        self, data: pd.DataFrame, experiment: Union[Experiment, List[Experiment]], features: List[str], *args, **kwargs
    ):
        ignore_clusters = kwargs.pop("ignore_clusters", None)
        super().__init__(data, experiment, features, *args, **kwargs)
        logger.info("Obtaining data about cluster membership")
        cluster_membership = self.experiment.population_membership_boolean_matrix(
            population_source="cluster", data_source="primary", ignore_clusters=ignore_clusters
        )
        self.data = self.data.merge(cluster_membership, on=["sample_id", "original_index"])
        self.clusters = [x for x in cluster_membership.columns if x not in ["sample_id", "original_index"]]
        self.cluster_groups = defaultdict(list)
        self.cluster_assignments = {}
        self._cluster_weights = {}
        self._event_distance_to_cluster_center = None
        self._event_distance_to_cluster_center_metric = None
        for sample_id in self.data.sample_id.unique():
            self.cluster_assignments[sample_id] = list(
                self.experiment.get_sample(sample_id=sample_id).list_populations(
                    population_source="cluster", data_source="primary"
                )
            )
        for cluster in self.clusters:
            prefix = cluster.split("_")[0]
            self.cluster_groups[prefix].append(cluster)

    @property
    def cell_cluster_assignments(self) -> pd.DataFrame:
        data = self.data[self.clusters].reset_index(drop=False).rename(columns={"index": "cell_id"})
        data = data.melt(id_vars="cell_id", value_name="assignment", var_name="cluster")
        data = data[data.assignment == 1]
        return data.drop("assignment", axis=1)

    def _reconstruct_labels(self, encoded: bool = False):
        labels = {}
        for prefix, clusters in self.cluster_groups.items():
            labels[prefix] = self.data[clusters].idxmax(axis=1)
        if encoded:
            return np.array([LabelEncoder().fit_transform(x) for x in labels.values()])
        return labels

    def _check_for_cluster_parents(self):
        for prefix, clusters in self.cluster_groups.items():
            if not (self.data[clusters].sum(axis=1) == 1).all():
                logger.warning(
                    f"Some observations are assigned to multiple clusters under the cluster prefix {prefix},"
                    f" either ensure cluster prefixes are unique to a cluster solution or remove parent "
                    f"populations with the 'ignore_clusters' method."
                )

    def ignore_clusters(self, clusters: List[str]):
        clusters = [x for x in clusters if x in self.clusters]
        self.clusters = [x for x in self.clusters if x not in clusters]
        self.data.drop(clusters, axis=1, inplace=True)
        self.cluster_assignments = {
            sample_id: [x for x in c if x not in clusters] for sample_id, c in self.cluster_assignments.items()
        }
        self.cluster_groups = {k: [x for x in c if x not in clusters] for k, c in self.cluster_groups.items()}
        self._cluster_weights = {}
        prefixes = set([c.split("_")[0] for c in clusters])
        for pf in prefixes:
            self.data = self.data[~(self.data[self.cluster_groups[pf]].sum(axis=1) == 0)]

    def simpsons_diversity_index(self, cell_identifier: str = "sample_id", groupby=None) -> pd.DataFrame:
        si_scores = {}
        for cluster in self.clusters:
            df = self.data[self.data[cluster] == 1]
            si_scores[cluster] = simpson_di(df[cell_identifier].value_counts().to_dict())
        return pd.DataFrame(si_scores, index=["SimpsonIndex"]).T

    def compute_cluster_centroids(self, diversity_threshold: Optional[float] = None):
        cluster_geometric_median = []
        for cluster in self.clusters:
            cluster_data = self.data[self.data[cluster] == 1][self.features].T.values
            x = np.array(hd.geomedian(cluster_data)).reshape(-1, 1)
            x = pd.DataFrame(x, columns=[cluster], index=self.features)
            cluster_geometric_median.append(x.T)
        centroids = pd.concat(cluster_geometric_median)
        if diversity_threshold:
            si_scores = self.simpsons_diversity_index()
            si_scores = si_scores[si_scores.SimpsonIndex <= diversity_threshold]
            centroids = centroids.loc[si_scores.index]
        return centroids

    def event_distance_to_cluster_center(self, metric: str = "cityblock", n_jobs: int = -1, verbose: bool = True):
        try:
            if (
                self._event_distance_to_cluster_center is not None
                and self._event_distance_to_cluster_center_metric == metric
            ):
                logger.info(f"Loading cell distance to cluster center data from cache")
                return self._event_distance_to_cluster_center
            logger.info(f"Computing distance to cluster centers for each event")
            data = self.cell_cluster_assignments.copy()
            centers = self.compute_cluster_centroids()
            with Parallel(n_jobs=n_jobs) as parallel:
                distance_to_centroid = parallel(
                    delayed(getattr(distance, metric))(
                        self.data[self.features].loc[x.cell_id].values, centers.loc[x.cluster].values
                    )
                    for _, x in progress_bar(data.iterrows(), total=data.shape[0], verbose=verbose)
                )
            data["distance"] = distance_to_centroid
            self._event_distance_to_cluster_center = data.copy()
            self._event_distance_to_cluster_center_metric = metric
            return data
        except AttributeError:
            raise AttributeError("Invalid metric, must be a valid metric available from Scipy spatial.distance module")

    def cluster_distortion_score(
        self, plot: bool = True, n_jobs=-1, distortion_metric: str = "euclidean", verbose: bool = True, **plot_kwargs
    ):
        if self._cluster_weights.get("metric", None) != distortion_metric:
            with Parallel(n_jobs=n_jobs) as parallel:
                sample_ids = self.data.sample_id.unique()
                weights = parallel(
                    delayed(distortion_score)(
                        data=self.data[self.data.sample_id == sid],
                        features=self.features,
                        metric=distortion_metric,
                        clusters=self.clusters,
                    )
                    for sid in progress_bar(sample_ids, verbose=verbose)
                )
                self._cluster_weights["metric"] = distortion_metric
                self._cluster_weights["weights"] = {sid: w for sid, w in zip(sample_ids, weights)}
        if plot:
            plot_kwargs = plot_kwargs or {}
            plot_df = (
                pd.DataFrame(self._cluster_weights["weights"])
                .T.melt(var_name="Cluster", value_name="Distortion score")
                .sort_values("Distortion score")
            )
            ax = box_swarm_plot(data=plot_df, x="Cluster", y="Distortion score", **plot_kwargs)
            return self._cluster_weights["weights"], ax
        return self._cluster_weights["weights"], None

    def _consensus_clusters_count_sources(self, consensus_results: pd.DataFrame):
        self._n_sources = {}
        for consensus_label, clusters in consensus_results.groupby("cluster_label"):
            self._n_sources[consensus_label] = len(set([c.split("_")[0] for c in clusters.index.unique()]))

    def similarity_matrix(self, diversity_threshold: Optional[float] = None, directional: bool = True):

        if diversity_threshold:
            si_scores = self.simpsons_diversity_index()
            si_scores = si_scores[si_scores.SimpsonIndex <= diversity_threshold]
            clusters = si_scores.index.values
        else:
            clusters = self.clusters

        matrix = np.zeros((len(clusters), len(clusters)))

        for i, cluster_i_id in enumerate(clusters):
            cluster_i_data = set(self.data[self.data[cluster_i_id] == 1].index.values)

            for j, cluster_j_id in enumerate(clusters):
                cluster_j_data = set(self.data[self.data[cluster_j_id] == 1].index.values)
                intersect = cluster_i_data.intersection(cluster_j_data)

                if directional:
                    if len(cluster_i_data) < len(cluster_j_data):
                        matrix[i, j] = len(intersect) / len(cluster_i_data)
                    else:
                        matrix[i, j] = len(intersect) / len(cluster_j_data)
                else:
                    union = cluster_i_data.union(cluster_j_data)
                    matrix[i, j] = len(intersect) / len(union)

        return pd.DataFrame(matrix, index=clusters, columns=clusters)

    def clustered_similarity_heatmap(
        self,
        directional: bool = True,
        diversity_threshold: Optional[float] = None,
        figsize: Tuple[int, int] = (10, 12),
        dendrogram_dimensions: Tuple[float, float, float, float] = (0.3, 0.9, 0.6, 0.2),
        similarity_heatmap_dimensions: Tuple[float, float, float, float] = (0.3, 0.5, 0.6, 0.4),
        centroid_heatmap_dimensions: Tuple[float, float, float, float] = (0.3, 0.08, 0.6, 0.4),
        method: str = "average",
        metric: str = "euclidean",
        cmap: str = "coolwarm",
        xticklabels: bool = False,
        scale: Optional[str] = "minmax",
    ):
        similarity_matrix = self.similarity_matrix(diversity_threshold=diversity_threshold, directional=directional)
        centroids = self.compute_cluster_centroids(diversity_threshold=diversity_threshold)

        if scale == "minmax":
            centroids[centroids.columns] = MinMaxScaler().fit_transform(centroids)
        elif scale == "standard":
            centroids[centroids.columns] = StandardScaler().fit_transform(centroids)

        fig = plt.figure(figsize=figsize)

        # Dendrogram
        dendrogram_ax = fig.add_axes(dendrogram_dimensions)
        linkage_matrix = linkage(similarity_matrix.values, method=method, metric=metric)
        dendro = dendrogram(linkage_matrix, color_threshold=0, above_threshold_color="black")
        dendrogram_ax.set_xticks([])
        dendrogram_ax.set_yticks([])

        # Similarity matrix
        axmatrix_sim = fig.add_axes(similarity_heatmap_dimensions)
        idx = dendro["leaves"]
        d = similarity_matrix.values[idx, :]
        axmatrix_sim.matshow(d[:, idx], aspect="auto", origin="lower", cmap=cmap)
        axmatrix_sim.set_xticks([])
        axmatrix_sim.set_yticks([])

        # Centroid heatmap
        axmatrix_centroids = fig.add_axes(centroid_heatmap_dimensions)
        idx = dendro["leaves"]
        d = centroids.iloc[idx].T
        sns.heatmap(data=d, ax=axmatrix_centroids, cmap=cmap, cbar=False, xticklabels=xticklabels)

        return fig

    def clustered_centroid_heatmap(
        self,
        method: str = "ward",
        metric: str = "euclidean",
        plot_orientation: str = "vertical",
        diversity_threshold: Optional[float] = None,
        **kwargs,
    ):
        centroids = self.compute_cluster_centroids(diversity_threshold=diversity_threshold)
        if plot_orientation == "horizontal":
            g = sns.clustermap(data=centroids.T, method=method, metric=metric, **kwargs)
        else:
            g = sns.clustermap(data=centroids, method=method, metric=metric, **kwargs)
        return g

    def _vote_filter(self, consensus_results: pd.DataFrame):
        cell_assignments = self.cell_cluster_assignments
        cell_assignments["cluster"] = cell_assignments["cluster"].apply(
            lambda x: consensus_results.loc[x]["cluster_label"]
        )
        cell_assignments.drop_duplicates(inplace=True)
        consensus_counts = cell_assignments.cell_id.value_counts()
        one_consensus_label = cell_assignments[
            cell_assignments.cell_id.isin(consensus_counts[consensus_counts == 1].index)
        ]
        labelled = self.data.loc[one_consensus_label.cell_id.values].copy()
        unlabelled = self.data[~self.data.index.isin(one_consensus_label.cell_id.values)].copy()
        labelled["cluster_label"] = one_consensus_label["cluster"].values
        return labelled, unlabelled

    def mixture_model_consensus_clustering(
        self, k: int, iterations: int = 10, labels: Optional[List[List]] = None, verbose: bool = True
    ):
        labels = labels if labels is not None else self._reconstruct_labels(encoded=True)
        mixture_model = MixtureModel(
            cluster_labels=labels, iterations=iterations, n_consensus_clusters=k, verbose=verbose
        )
        self.data["cluster_label"] = mixture_model.em_process()
        return self

    def _majority_vote(
        self,
        consensus_data: pd.DataFrame,
        vote_weighting_method: str,
        distortion_metric: str,
        distance_to_center_metric: str,
        verbose: bool,
        n_jobs: int,
    ):
        labelled, unlabelled = self._vote_filter(consensus_results=consensus_data)
        logger.info(
            f"{round(unlabelled.shape[0]/self.data.shape[0]*100, 3)}% of events assigned to more than one "
            f"consensus label, resolving by vote"
        )

        if vote_weighting_method == "distortion_score":
            weights = self.cluster_distortion_score(plot=False, distortion_metric=distortion_metric, verbose=verbose)[
                0
            ]
            with Parallel(n_jobs=n_jobs) as parallel:
                labels = parallel(
                    delayed(majority_vote_with_distortion_score_weighting)(
                        row=row,
                        cluster_column_names=self.clusters,
                        consensus_clusters=consensus_data[["cluster_label"]],
                        distortion_score_dict=weights[row["sample_id"]],
                    )
                    for _, row in progress_bar(unlabelled.iterrows(), verbose=verbose, total=unlabelled.shape[0])
                )
        elif vote_weighting_method == "distance_to_center":
            distance_df = self.event_distance_to_cluster_center(
                metric=distance_to_center_metric, n_jobs=n_jobs, verbose=verbose
            )
            with Parallel(n_jobs=n_jobs) as parallel:
                labels = parallel(
                    delayed(majority_vote_with_distance_weighting)(
                        row=row,
                        cluster_column_names=self.clusters,
                        consensus_clusters=consensus_data[["cluster_label"]],
                        distance_df=distance_df,
                    )
                    for _, row in progress_bar(unlabelled.iterrows(), verbose=verbose, total=unlabelled.shape[0])
                )
        else:
            with Parallel(n_jobs=n_jobs) as parallel:
                labels = parallel(
                    delayed(majority_vote)(
                        row=row,
                        cluster_column_names=self.clusters,
                        consensus_clusters=consensus_data[["cluster_label"]],
                    )
                    for _, row in progress_bar(unlabelled.iterrows(), verbose=verbose, total=unlabelled.shape[0])
                )
        unlabelled["cluster_label"] = labels
        return labels, labelled, unlabelled

    def consensus_vote_clustering(
        self,
        method: Union[str, ClusterMethod, ClusterMixin] = "k_consensus",
        method_kwargs: Optional[Dict] = None,
        n_jobs: int = -1,
        verbose: bool = True,
        cluster_data: str = "geometric_medians",
        diversity_threshold: Optional[float] = None,
        vote_weighting_method: Optional[str] = "distance_to_center",
        distortion_metric: str = "cityblock",
        distance_to_center_metric: str = "cityblock",
        return_labels: bool = False,
        directional_similarity: bool = True,
    ):
        method_kwargs = method_kwargs or {}
        method = init_cluster_method(method=method, verbose=self.verbose, **method_kwargs)
        self._check_for_cluster_parents()

        if cluster_data == "geometric_medians":

            logger.info("Calculating geometric median of each cluster")
            consensus_data = self.compute_cluster_centroids(diversity_threshold=diversity_threshold)

            logger.info("Performing consensus clustering on geometric medians")
            consensus_data["cluster_label"] = method.cluster(data=consensus_data, features=self.features)

        elif cluster_data == "similarity":

            logger.info("Calculating similarity matrix")
            consensus_data = self.similarity_matrix(
                diversity_threshold=diversity_threshold, directional=directional_similarity
            )

            logger.info("Performing consensus clustering on similarity matrix")
            consensus_data["cluster_label"] = method.cluster(data=consensus_data, features=self.clusters)
        else:
            raise ValueError("cluster_data must be either 'geometric_medians' or 'similarity'")
        logger.info(
            f"Generated {consensus_data.cluster_label.nunique()} consensus clusters: {consensus_data.cluster_label.unique()}"
        )
        self._consensus_clusters_count_sources(consensus_results=consensus_data)

        logger.info(f"Assigning clusters by majority vote (vote_weighting_method={vote_weighting_method})")
        labels, labelled, unlabelled = self._majority_vote(
            consensus_data=consensus_data,
            vote_weighting_method=vote_weighting_method,
            distortion_metric=distortion_metric,
            distance_to_center_metric=distance_to_center_metric,
            verbose=verbose,
            n_jobs=n_jobs,
        )
        self.data = pd.concat([labelled, unlabelled])
        logger.info("Consensus clustering complete!")
        if return_labels:
            return labels
        return self

    def _consensus_count_sources(self, original_labels: List):
        data = self.data.copy()
        data["original_cluster_label"] = original_labels
        for consensus_label, clusters in data.groupby("cluster_label"):
            self._n_sources[consensus_label] = clusters.original_cluster_label.nunique()

    def graph_consensus_clustering(
        self, consensus_method: str, k: int, random_state: int = 42, labels: Optional[List] = None
    ):
        labels = labels if labels is not None else self._reconstruct_labels(encoded=True)
        if consensus_method == "cspa" and self.data.shape[0] > 5000:
            logger.warning("CSPA is not recommended when n>5000, consider a different method")
            self.data["cluster_label"] = ClusterEnsembles.cspa(labels=labels, nclass=k)
        if consensus_method == "hgpa":
            self.data["cluster_label"] = ClusterEnsembles.hgpa(labels=labels, nclass=k, random_state=random_state)
        if consensus_method == "mcla":
            self.data["cluster_label"] = ClusterEnsembles.mcla(labels=labels, nclass=k, random_state=random_state)
        if consensus_method == "hbgf":
            self.data["cluster_label"] = ClusterEnsembles.hbgf(labels=labels, nclass=k)
        if consensus_method == "nmf":
            self.data["cluster_label"] = ClusterEnsembles.nmf(labels=labels, nclass=k, random_state=random_state)
        raise ClusteringError("Invalid consensus method, must be one of: cspa, hgpa, mcla, hbgf, or nmf")

    def comparison(self, method: str = "adjusted_mutual_info", **kwargs):
        kwargs["figsize"] = kwargs.get("figsize", (10, 10))
        kwargs["cmap"] = kwargs.get("cmap", "coolwarm")
        data = comparison_matrix(cluster_labels=self._reconstruct_labels(), method=method)
        return sns.clustermap(
            data=data,
            **kwargs,
        )

    def cluster_sizes(self):
        la = self._reconstruct_labels()
        return {k: x.value_counts() for k, x in la.items()}

    def smallest_cluster_n(self):
        return min([x.min() for _, x in self.cluster_sizes().items()])

    def largest_cluster_n(self):
        return max([x.max() for _, x in self.cluster_sizes().items()])

    def min_k(self):
        return min([len(x) for x in self.cluster_groups.values()])

    def max_k(self):
        return max([len(x) for x in self.cluster_groups.values()])

    def k_performance(
        self,
        k_range: Tuple[int, int],
        consensus_method: str,
        sample_n: int,
        resamples: int,
        balance_samples: bool = True,
        balance_clusters: bool = True,
        replace: bool = False,
        random_state: int = 42,
        features: Optional[List[str]] = None,
        metrics: Optional[List[Union[InternalMetric, str]]] = None,
        return_data: bool = True,
        **kwargs,
    ):
        results = []
        for k in range(*k_range):
            logger.info(f"Calculating consensus with k={k}...")
            if consensus_method == "mixture_model":
                self.mixture_model_consensus_clustering(k=k, **kwargs)
            elif consensus_method == "vote":
                self.consensus_vote_clustering(t=k, **kwargs)
            else:
                self.graph_consensus_clustering(consensus_method=consensus_method, k=k, random_state=random_state)
            perf = self.internal_performance(
                metrics=metrics,
                sample_n=sample_n,
                resamples=resamples,
                features=features,
                labels="cluster_label",
                balance_samples=balance_samples,
                balance_clusters=balance_clusters,
                replace=replace,
                random_state=random_state,
                verbose=True,
            )
            perf["K"] = k
            results.append(perf)
        results = pd.concat(results).reset_index(drop=True)
        results = pd.DataFrame(results).melt(id_vars="K", var_name="Metric", value_name="Value")
        return results

    def save(
        self,
        population_prefix: Optional[str] = "consensus",
        verbose: bool = True,
        parent_populations: Optional[Dict] = None,
    ):
        super()._save(
            population_prefix=population_prefix,
            verbose=verbose,
            population_var="cluster_label",
            parent_populations=parent_populations,
        )
