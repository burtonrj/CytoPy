from typing import Dict
from typing import List
from typing import Optional
from typing import Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.base import ClusterMixin

from .. import Experiment
from ..plotting.general import build_plot_grid
from .clustering import ClusterMethod
from .clustering import init_cluster_method


def example_clusters(
    data: pd.DataFrame,
    n: int = 5,
    metric: str = "euclidean",
    linkage: str = "average",
    col_wrap: int = 2,
    figure_kwargs: Optional[Dict] = None,
) -> plt.Figure:
    sample_ids = np.random.choice(data.sample_id.values, size=n, replace=False)
    figure_kwargs = figure_kwargs or {}
    fig, axes = build_plot_grid(n=n, col_wrap=col_wrap, **figure_kwargs)
    for _id in sample_ids:
        df = data[data.sample_id == _id]
        sns.clustermap(
            data=df,
        )


class BCSU:
    def __init__(
        self, clustering_methods: List[Union[str, ClusterMethod, ClusterMixin]], clustering_params: List[Dict]
    ):
        pass

    def add_definitions(
        self,
        start_cut: int,
        start_definitions: Dict,
        subcluster_features: List[str],
        subcluster_divisions: Dict[str, int],
    ):
        err = "One or more features missing from subcluster_divisions"
        assert all([x in subcluster_divisions.keys() for x in subcluster_features]), err

    def cluster(self):
        pass

    def meta_cluster(self):
        pass

    def plot_umap(self):
        pass
