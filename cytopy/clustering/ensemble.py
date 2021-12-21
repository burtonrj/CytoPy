import logging
from collections import Counter
from collections import defaultdict
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union

import numpy as np
import pandas as pd
import seaborn as sns
from ClusterEnsembles import ClusterEnsembles
from joblib import delayed
from joblib import Parallel
from matplotlib.ticker import MaxNLocator
from scipy.cluster.hierarchy import fclusterdata
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.preprocessing import LabelEncoder

from ..plotting.general import box_swarm_plot
from ..plotting.general import ColumnWrapFigure
from .clustering import Clustering
from .clustering import ClusteringError
from .metrics import comparison_matrix
from .metrics import init_internal_metrics
from .metrics import InternalMetric
from cytopy.data.experiment import Experiment
from cytopy.feedback import progress_bar

logger = logging.getLogger(__name__)


def distortion_score(data: pd.DataFrame, features: List[str], clusters: List[str], metric: str = "euclidean"):
    score = {}
    for c in clusters:
        df = data[data[c] == 1]
        if df.shape[0] == 0:
            continue
        center = df[features].mean().values
        distances = pairwise_distances(df[features], center.reshape(1, -1), metric=metric)
        score[c] = (distances ** 2).sum() / df.shape[0]
    return score


def majority_vote(
    row: pd.Series,
    cluster_assignments: List[str],
    consensus_clusters: pd.DataFrame,
    weights: Optional[Dict[str, float]] = None,
):
    clusters = row[cluster_assignments].replace({0: None}).dropna().index.tolist()
    if weights is not None:
        consensus_cluster_score = []
        for cid, cc_data in consensus_clusters.loc[clusters].groupby("cluster_label"):
            consensus_cluster_score.append(
                [cid, cc_data.shape[0] / sum([weights.get(i) for i in cc_data.index.values])]
            )
        return sorted(consensus_cluster_score, key=lambda x: x[1])[::-1][0][0]

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


class EnsembleClustering(Clustering):
    def __init__(
        self, data: pd.DataFrame, experiment: Union[Experiment, List[Experiment]], features: List[str], *args, **kwargs
    ):
        super().__init__(data, experiment, features, *args, **kwargs)
        logger.info("Obtaining data about cluster membership")
        cluster_membership = self.experiment.population_membership_boolean_matrix(
            population_source="cluster", data_source="primary"
        )
        self.data = self.data.merge(cluster_membership, on=["sample_id", "original_index"])
        self.clusters = [x for x in cluster_membership.columns if x not in ["sample_id", "original_index"]]
        self.cluster_groups = defaultdict(list)
        self.cluster_assignments = {}
        self._cluster_weights = {}
        for sample_id in self.data.sample_id.unique():
            self.cluster_assignments[sample_id] = list(
                self.experiment.get_sample(sample_id=sample_id).list_populations(
                    source="cluster", data_source="primary"
                )
            )
        for cluster in self.clusters:
            prefix = cluster.split("_")[0]
            self.cluster_groups[prefix].append(cluster)

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

    def _compute_cluster_centroids(self, method: str = "median"):
        centroids = {}
        for cluster in self.clusters:
            cluster_data = self.data[self.data[cluster] == 1][self.features]
            if method == "median":
                centroids[cluster] = cluster_data.median().values
            else:
                centroids[cluster] = cluster_data.mean().values
        return pd.DataFrame(centroids, index=self.features).T

    def cluster_weights(
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
                        clusters=self.cluster_assignments[sid],
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
            ax = box_swarm_plot(plot_df=plot_df, x="Cluster", y="Distortion score", **plot_kwargs)
            return self._cluster_weights["weights"], ax
        return self._cluster_weights["weights"], None

    def _consensus_centroids_count_sources(self, centroids: pd.DataFrame):
        self._n_sources = {}
        for consensus_label, clusters in centroids.groupby("cluster_label"):
            self._n_sources[consensus_label] = len(set([c.split("_")[0] for c in clusters.index.unique()]))

    def centroid_clustered_heatmap(
        self,
        centroid_method: str = "median",
        method: str = "ward",
        metric: str = "euclidean",
        plot_orientation: str = "vertical",
        **kwargs,
    ):
        centroids = self._compute_cluster_centroids(method=centroid_method)
        if plot_orientation == "horizontal":
            g = sns.clustermap(data=centroids.T, method=method, metric=metric, **kwargs)
        else:
            g = sns.clustermap(data=centroids, method=method, metric=metric, **kwargs)
        return g

    def elbow_plot(
        self,
        min_k: int,
        max_k: int,
        sample_n: int = 10000,
        resamples: int = 20,
        method: str = "ward",
        cluster_metric: str = "euclidean",
        depth: int = 2,
        weight: bool = True,
        n_jobs: int = -1,
        distortion_metric: str = "euclidean",
        external_metrics: Optional[List[Union[str, InternalMetric]]] = None,
        centroid_method: str = "median",
        figsize: Optional[Tuple[int, int]] = None,
        verbose: bool = True,
        **kwargs,
    ):
        centroids = self._compute_cluster_centroids(method=centroid_method)
        results = []
        external_metrics = external_metrics or ["distortion_score"]
        metrics = init_internal_metrics(metrics=external_metrics)
        for k in range(min_k, max_k + 1):
            if verbose:
                print(f"N clusters = {k}")
            centroids["cluster_label"] = fclusterdata(
                X=centroids, t=k, criterion="maxclust", method=method, metric=cluster_metric, depth=depth
            )
            labels = self._assigning_events_to_clusters(
                centroids=centroids, distortion_metric=distortion_metric, n_jobs=n_jobs, weight=weight, verbose=verbose
            )
            tmp = pd.DataFrame(
                self.performance(
                    metrics=metrics, sample_n=sample_n, resamples=resamples, labels=labels, plot=False, verbose=verbose
                )
            )
            tmp["N clusters"] = k
            results.append(tmp)
        results = pd.concat(results).sort_values("N clusters")
        metrics = [x for x in results.columns if x != "N clusters"]
        figsize = figsize or (10, 5 * len(metrics))
        fig = ColumnWrapFigure(n=len(metrics), figsize=figsize, col_wrap=1)
        for i, metric in enumerate(metrics):
            ax = fig.add_wrapped_subplot()
            box_swarm_plot(plot_df=results, x="N clusters", y=metric, ax=ax, **kwargs)
            if i < len(metrics) - 1:
                ax.set_xlabel("")
                ax.set_xticklabels([])
        return fig

    def _assigning_events_to_clusters(
        self,
        centroids: pd.DataFrame,
        distortion_metric: str = "euclidean",
        n_jobs: int = -1,
        weight: bool = True,
        verbose: bool = True,
    ):
        with Parallel(n_jobs=n_jobs) as parallel:
            weights = {}
            if weight:
                weights, _ = self.cluster_weights(plot=False, distortion_metric=distortion_metric, verbose=verbose)
            labels = parallel(
                delayed(majority_vote)(
                    row=row,
                    cluster_assignments=self.cluster_assignments[row.sample_id],
                    consensus_clusters=centroids[["cluster_label"]],
                    weights=weights.get(row.sample_id, None),
                )
                for _, row in progress_bar(self.data.iterrows(), verbose=verbose, total=self.data.shape[0])
            )
        return labels

    def consensus_centroid_clustering(
        self,
        t: int,
        centroid_method: str = "median",
        method: str = "ward",
        metric: str = "euclidean",
        criterion: str = "maxclust",
        depth: int = 2,
        weight: bool = True,
        n_jobs: int = -1,
        verbose: bool = True,
        distortion_metric: str = "euclidean",
        return_labels: bool = False,
    ):
        self._check_for_cluster_parents()
        logger.info("Calculating cluster centroids")
        centroids = self._compute_cluster_centroids(method=centroid_method)
        logger.info("Performing hierarchical clustering of centroids")
        centroids["cluster_label"] = fclusterdata(
            X=centroids, t=t, criterion=criterion, method=method, metric=metric, depth=depth
        )
        logger.info(
            f"Clustered centroids into {centroids.cluster_label.nunique()} clusters: "
            f"{centroids.cluster_label.unique()}"
        )
        self._consensus_centroids_count_sources(centroids=centroids)

        logger.info("Assigning clusters by majority vote")
        labels = self._assigning_events_to_clusters(
            centroids=centroids, distortion_metric=distortion_metric, n_jobs=n_jobs, weight=weight, verbose=verbose
        )
        if return_labels:
            return labels
        self.data["cluster_label"] = labels
        return self

    def _consensus_count_sources(self, original_labels: List):
        data = self.data.copy()
        data["original_cluster_label"] = original_labels
        for consensus_label, clusters in data.groupby("cluster_label"):
            self._n_sources[consensus_label] = clusters.original_cluster_label.nunique()

    def consensus_clustering(
        self, consensus_method: str, k: int, random_state: int = 42, labels: Optional[List] = None
    ):
        if consensus_method not in ["cdpa", "hgpa", "mcla", "hbgf", "nmf"]:
            raise ClusteringError("Invalid consensus method, must be one of: cdpa, hgpa, mcla, hbgf, or nmf")
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
        # self._consensus_count_sources(labels)

    def comparison(self, method: str = "adjusted_mutual_info", **kwargs):
        kwargs["figsize"] = kwargs.get("figsize", (10, 10))
        kwargs["cmap"] = kwargs.get("cmap", "coolwarm")
        data = comparison_matrix(cluster_labels=self._reconstruct_labels(), method=method)
        return sns.clustermap(
            data=data,
            **kwargs,
        )

    def min_k(self):
        return min([len(x) for x in self._reconstruct_labels().values()])

    def max_k(self):
        return max([len(x) for x in self._reconstruct_labels().values()])

    def k_performance(
        self,
        k_range: Tuple[int, int],
        consensus_method: str,
        sample_size: int,
        resamples: int,
        random_state: int = 42,
        metrics: Optional[List[Union[InternalMetric, str]]] = None,
        return_data: bool = True,
        **kwargs,
    ):
        if sample_size > self.data.shape[0]:
            raise ClusteringError(f"Sample size cannot exceed size of data ({self.data.shape[0]})")
        logger.info("Sampling...")
        metrics = init_internal_metrics(metrics=metrics)
        labels = []
        data = []
        for _ in progress_bar(range(resamples), total=resamples):
            idx = np.random.randint(0, self.data.shape[0], sample_size)
            labels.append(np.array([np.array(x)[idx] for x in self._reconstruct_labels().values()]))
            data.append(self.data.iloc[idx])
        k_range = np.arange(k_range[0], k_range[1] + 1)
        results = defaultdict(list)
        for k in k_range:
            logger.info(f"Calculating consensus with k={k}...")
            for la, df in progress_bar(zip(labels, data), total=len(data)):
                la = self.consensus_clustering(
                    consensus_method=consensus_method, k=k, labels=la, random_state=random_state
                )
                results["K"].append(k)
                for m in metrics:
                    results[m.name].append(m(data=df, features=self.features, labels=la))
        results = pd.DataFrame(results).melt(id_vars="K", var_name="Metric", value_name="Value")
        facet_kws = kwargs.pop("facet_kws", {})
        facet_kws["sharey"] = facet_kws.get("sharey", False)
        g = sns.relplot(data=results, x="K", y="Value", kind="line", col="Metric", facet_kws=facet_kws, **kwargs)
        g.set_titles("{col_name}")
        for ax in g.axes:
            ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        if return_data:
            return g, results
        return g

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
