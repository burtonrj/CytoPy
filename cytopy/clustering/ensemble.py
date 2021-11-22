import itertools
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

from .clustering import Clustering
from .clustering import ClusteringError
from .metrics import comparison_matrix
from .metrics import init_internal_metrics
from .metrics import InternalMetric
from cytopy.data.experiment import Experiment
from cytopy.feedback import progress_bar

logger = logging.getLogger(__name__)


def distortion_score(data: pd.DataFrame, metric: str = "euclidean"):
    if data.shape[0] == 0:
        return None
    center = data.mean()
    distances = pairwise_distances(data, center, metric=metric)
    return (distances ** 2).sum()


def majority_vote(
    cluster_assignments: List[str], consensus_clusters: pd.DataFrame, weights: Optional[Dict[str, float]] = None
):
    if weights is not None:
        scores = []
        consensus_clusters = consensus_clusters.loc[cluster_assignments]
        for cc, cc_data in consensus_clusters.groupby("cluster_label"):
            scores.append([cc, cc_data.shape[0] / sum([weights.get(i) for i in cc_data.index.values])])
        return sorted(scores, key=lambda x: x[1])[::-1][0][0]

    cluster_labels = Counter([consensus_clusters.loc[cluster, "cluster_label"] for cluster in cluster_assignments])
    score = 0
    winner = None
    for label, count in cluster_labels.items():
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
        cluster_membership = self.experiment.population_membership(
            population_source="cluster", data_source="primary", as_boolean=True
        )
        self.data = self.data.merge(cluster_membership, on=["sample_id", "original_index"])
        self.clusters = list(cluster_membership.columns)
        self.cluster_groups = defaultdict(list)
        self.cluster_assignments = {}
        self._labels = self._reconstruct_labels()
        for sample_id in self.data.sample_id.unique():
            self.cluster_assignments[sample_id] = list(
                self.experiment.get_sample(sample_id=sample_id).list_populations(
                    source="cluster", data_source="primary"
                )
            )
        for cluster in cluster_membership.columns:
            prefix = cluster.split("_")[0]
            self.cluster_groups[prefix].append(cluster)

    def _check_for_cluster_parents(self):
        for prefix, clusters in self.cluster_groups.items():
            if not (self.data[clusters].sum(axis=1) == 0).all():
                logger.warning(
                    f"Some observations are assigned to multiple clusters under the cluster prefix {prefix},"
                    f" either ensure cluster prefixes are unique to a cluster solution or remove parent "
                    f"populations with the 'ignore_clusters' method."
                )

    def ignore_clusters(self, clusters: List[str]):
        if not all([x in self.data.columns for x in clusters]):
            raise ValueError("One or more provided clusters does not exist.")
        self.data.drop(clusters, axis=1, inplace=True)
        self.cluster_assignments = {
            sample_id: [x for x in c if x not in clusters] for sample_id, c in self.cluster_assignments.items()
        }

    def _compute_cluster_centroids(self, method: str = "median"):
        centroids = {}
        for cluster in self.clusters:
            cluster_data = self.data[self.data[cluster] == 1][self.features]
            if method == "median":
                centroids[cluster] = pd.concat(cluster_data).median().values
            else:
                centroids[cluster] = pd.concat(cluster_data).mean().values
        return pd.DataFrame(centroids, index=self.features).T

    def cluster_centroids(
        self,
        t: int,
        centroid_method: str = "median",
        method: str = "average",
        metric: str = "euclidean",
        criterion: str = "maxclust",
        plot_only: bool = True,
        depth: Optional[int] = None,
        weight: bool = True,
        n_jobs: int = -1,
        verbose: bool = True,
        distortion_metric: str = "euclidean",
        **kwargs,
    ):
        # Check for parent clusters and warn
        self._check_for_cluster_parents()
        # Calculate centroids
        centroids = self._compute_cluster_centroids(method=centroid_method)
        g = sns.clustermap(data=centroids, method=method, metric=metric, **kwargs)
        if plot_only:
            return g
        # Perform clustering
        centroids["cluster_label"] = fclusterdata(
            X=centroids, t=t, criterion=criterion, method=method, metric=metric, depth=depth
        )
        with Parallel(n_jobs=n_jobs) as parallel:
            weights = None
            if weight:
                sample_cluster = list(itertools.product(self.data.sample_id.unique(), self.clusters))
                weights = parallel(
                    delayed(distortion_score)(
                        data=self.data[(self.data.sample_id == sample_id) & (self.data[cluster] == 1)],
                        metric=distortion_metric,
                    )
                    for sample_id, cluster in sample_cluster
                )
            self.data["cluster_label"] = parallel(
                delayed(majority_vote)(
                    cluster_assignments=self.cluster_assignments[sample_id],
                    consensus_clusters=centroids[["cluster_label"]],
                    weights=weights,
                )
                for sample_id in progress_bar(self.data.sample_id.unique(), verbose=verbose)
            )
        return g

    def _reconstruct_labels(self):
        labels = {}
        for prefix, clusters in self.cluster_groups:
            labels[prefix] = self.data[clusters].idxmax(axis=1)
        return labels

    def consensus_clustering(
        self, consensus_method: str, k: int, random_state: int = 42, labels: Optional[List] = None
    ):
        labels = labels if labels is not None else list(self._labels.values())
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
        raise ClusteringError("Invalid consensus method, must be one of: cdpa, hgpa, mcla, hbgf, or nmf")

    def comparison(self, method: str = "adjusted_mutual_info", **kwargs):
        kwargs["figsize"] = kwargs.get("figsize", (10, 10))
        kwargs["cmap"] = kwargs.get("cmap", "coolwarm")
        data = comparison_matrix(cluster_labels=self._labels, method=method)
        return sns.clustermap(
            data=data,
            **kwargs,
        )

    def min_k(self):
        return min([len(x) for x in self._labels.values()])

    def max_k(self):
        return max([len(x) for x in self._labels.values()])

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
            labels.append(np.array([np.array(x)[idx] for x in self._labels.values()]))
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
