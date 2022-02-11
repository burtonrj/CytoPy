import logging
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union

import hdmedians as hd
import numpy as np
import pandas as pd
import polars as pl
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib.cm import nipy_spectral
from sklearn.metrics import silhouette_samples
from sklearn.metrics import silhouette_score

from cytopy.data.read_write import polars_to_pandas
from cytopy.plotting.general import box_swarm_plot
from cytopy.plotting.general import discrete_label
from cytopy.plotting.general import discrete_palette
from cytopy.utils.dim_reduction import DimensionReduction

logger = logging.getLogger(__name__)


def geomedian(data: pd.DataFrame, label: str, features: List[str]):
    cluster_geometric_median = []
    for cluster, cluster_data in data.groupby(label):
        x = np.array(hd.geomedian(cluster_data[features].T.values)).reshape(-1, 1)
        x = pd.DataFrame(x, columns=[cluster], index=features)
        cluster_geometric_median.append(x.T)
    return pd.concat(cluster_geometric_median)


def clustered_heatmap(
    data: pd.DataFrame,
    features: List[str],
    sample_id: Optional[str] = None,
    meta_label: bool = True,
    plot_orientation="vertical",
    **kwargs,
):
    """
    Generate a clustered heatmap (using Seaborn Clustermap function). This function is capable of producing
    different types of heatmaps depending on the input and the clustered dataframe (data attribute):

    * sample_id is None and meta_label is True, will plot the median intensity of each feature for
    each meta cluster
    * sample_id is None and meta_label is False, will plot the median intensity of each feature
    for each cluster label
    * if sample_id is not None, then will plot the median intensity for each feature for each cluster
    for the specified sample

    Default parameters passed to clustermap (overwrite using kwargs):
    * col_cluster = True
    * figsize = (10, 15)
    * standard_scale = 1
    * cmap = "viridis"

    Parameters
    ----------
    features: list
    sample_id: str, optional
    meta_label: bool (default=True)
    kwargs:
        Additional keyword arguments passed to Seaborn.clustermap

    Returns
    -------
    Seaborn.ClusterGrid
    """
    if sample_id is None and meta_label:
        data = geomedian(data=data, label="meta_label", features=features)
    elif sample_id is None and not meta_label:
        data = geomedian(data=data, label="cluster_label", features=features)
    else:
        data = geomedian(data=data[data.sample_id == sample_id], label="cluster_label", features=features)
    data[features] = data[features].apply(pd.to_numeric)
    kwargs = kwargs or {}
    kwargs["col_cluster"] = kwargs.get("col_cluster", True)
    kwargs["figsize"] = kwargs.get("figsize", (10, 15))
    kwargs["standard_scale"] = kwargs.get("standard_scale", 1)
    kwargs["cmap"] = kwargs.get("cmap", "viridis")
    if plot_orientation == "vertical":
        return sns.clustermap(data[features], **kwargs)
    return sns.clustermap(data[features].T, **kwargs)


def plot_meta_clusters(
    data: pd.DataFrame,
    features: List[str],
    colour_label: str = "meta_label",
    discrete: bool = True,
    method: str = "UMAP",
    dim_reduction_kwargs: dict or None = None,
    **kwargs,
):
    """
    Generate a cluster bubble plot (see cytopy.utils.plotting.cluster_bubble_plot) where each
    data point (bubble) is a single cluster centroid from a unique patient. Size of the data points represents
    the fraction of cells with membership to the sample relative to the total number of events
    in that sample. By default data points are coloured by meta label membership.

    Parameters
    ----------
    method: str
        Dimensionality reduction technique; available methods are: UMAP, PCA, PHATE, KernelPCA or tSNE
    dim_reduction_kwargs: dict, optional
        Additional keyword arguments passed to dimension reduction (see cytopy.utils.dim_reduction)
    colour_label: str, (default='meta_label')
        How to colour cluster centroids
    discrete: bool (default=True)
        If True, label is treated as a discrete variable. If False, continuous colourmap will be applied.
    kwargs:
        Additional keyword arguments passed to cytopy.utils.plotting.cluster_bubble_plot

    Returns
    -------
    Matplotlib.Axes
    """
    return cluster_bubble_plot(
        data=data,
        features=features,
        cluster_label="cluster_label",
        sample_label="sample_id",
        colour_label=colour_label,
        discrete=discrete,
        dim_reduction_method=method,
        dim_reduction_kwargs=dim_reduction_kwargs,
        **kwargs,
    )


def _assert_unique_label(x):
    assert len(x) == 1, "Chosen label is not unique within clusters"
    return x[0]


def _cluster_centroids(data: pd.DataFrame, features: list, sample_label: str, cluster_label: str):
    return data.groupby([sample_label, cluster_label])[features].median().reset_index()


def _sample_n(data: pd.DataFrame, sample_label: str):
    sample_size = data[sample_label].value_counts()
    sample_size.name = "sample_n"
    return pd.DataFrame(sample_size).reset_index().rename({"index": sample_label}, axis=1)


def _cluster_n(data: pd.DataFrame, cluster_label: str, sample_label: str):
    sample_cluster_counts = data.groupby(sample_label)[cluster_label].value_counts()
    sample_cluster_counts.name = "cluster_n"
    return pd.DataFrame(sample_cluster_counts).reset_index()


def _cluster_size(sample_n: pd.DataFrame, cluster_n: pd.DataFrame):
    cluster_size = cluster_n.merge(sample_n, on="sample_id")
    cluster_size["cluster_size"] = cluster_size["cluster_n"] / cluster_size["sample_n"]
    return cluster_size


def _label_centroids(
    data: pd.DataFrame,
    centroids: pd.DataFrame,
    sample_label: str,
    cluster_label: str,
    target_label: str,
):
    data = data[[sample_label, cluster_label, target_label]].drop_duplicates()
    return centroids.merge(data, on=[sample_label, cluster_label])


def _generate_cluster_centroids(
    data: pd.DataFrame,
    features: list,
    cluster_label: str,
    sample_label: str,
    colour_label: str or None,
    dim_reduction_method: str or None,
    n_components: int = 2,
    dim_reduction_kwargs: dict or None = None,
):
    """
    Generate centroids for clusters in given dataframe

    Parameters
    ----------
    data: Pandas.DataFrame
        DataFrame of single cell data
    features: list
        List of features
    cluster_label: str
        Column that corresponds to the name of cluster
    sample_label: str
        Column that contains sample unique identifiers
    colour_label: str, optional
        Column that contains variable used to colour data points
    dim_reduction_method: str, optional
        Dimension reduction method to be applied
    n_components: int (default=2)
        Number of components to generate in dimension reduction
    dim_reduction_kwargs: dict, optional
        Additional keyword arguments to pass to dimension reduction method

    Returns
    -------
    Pandas.DataFrame
        Centroids

    Raises
    ------
    AssertionError
        Invalid number of components, should be 2 or 3
    """
    dim_reduction_kwargs = dim_reduction_kwargs or {}
    assert n_components in [2, 3], "n_components must be 2 or 3"
    data = data.dropna(axis=1, how="any")
    centroids = _cluster_centroids(
        data=data,
        features=features,
        sample_label=sample_label,
        cluster_label=cluster_label,
    )
    if dim_reduction_method is not None:
        reducer = DimensionReduction(
            method=dim_reduction_method,
            n_components=n_components,
            **dim_reduction_kwargs,
        )
        centroids = reducer.fit_transform(data=centroids, features=features)
    centroids = centroids.merge(
        _cluster_size(
            _sample_n(data=data, sample_label=sample_label),
            _cluster_n(data=data, sample_label=sample_label, cluster_label=cluster_label),
        )
    )
    if colour_label is not None:
        if colour_label != cluster_label:
            centroids = _label_centroids(
                data=data,
                centroids=centroids,
                sample_label=sample_label,
                cluster_label=cluster_label,
                target_label=colour_label,
            )
    return centroids


def cluster_bubble_plot(
    data: Union[pd.DataFrame, pl.DataFrame],
    features: list,
    cluster_label: str,
    sample_label: str,
    colour_label: Optional[str] = "meta_label",
    palette: Optional[str] = None,
    discrete: bool = True,
    style: Optional[str] = None,
    hue_norm: Optional[Union[plt.Normalize, Tuple[int, int]]] = (0, 1),
    dim_reduction_method: str = "UMAP",
    n_components: int = 2,
    dim_reduction_kwargs: Optional[Dict] = None,
    figsize: tuple = (8, 8),
    ax: Optional[plt.Axes] = None,
    legend_kwargs: Optional[Dict] = None,
    cbar_kwargs: Optional[Dict] = None,
    **kwargs,
):
    """
    Generate a cluster 'bubble' plot where each data point (bubble) is a single cluster centroid
    from a unique patient. Size of the data points represents the fraction of events with membership
    to the sample relative to the total number of events in that sample.
    By default data points are coloured by meta label membership.

    Parameters
    ----------
    data: Pandas.DataFrame
        Single cell data
    features: list
        Features to include in dimension reduction and cluster centroid summarisation
    cluster_label: str
        Name of the column containing cluster identifier
    sample_label: str
        Name of the column containing sample unique identifiers
    colour_label: str, optional (default='meta_label')
        Column used to assign colours to data points
    zscore: bool (default=False)
        z-score normalisation performed
    discrete: bool (default=True)
        Treat label as categorical
    cmap: str (default="tab20")
        Colourmap (must be a valid Matplotlib colourmap)
    dim_reduction_method: str, optional (default="UMAP")
        Dimensionality reduction technique; available methods are: UMAP, PCA, PHATE, KernelPCA or tSNE
    n_components: int (default=2)
        Number of components to generate from dimension reduction
    dim_reduction_kwargs: dict, optional
        Additional keyword arguments passed to dimension reduction (see cytopy.utils.dim_reduction)
    figsize: tuple (default=(8,8))
    legend_kwargs: dict, optional
        Keyword arguments passed to legend
    cbar_kwargs: dict, optional
        Keyword arguments passed to colorbar
    kwargs:
        Additional keyword arguments passed to Matplotlib.Axes.scatter call

    Returns
    -------
    Matplotlib.Axes
    """
    if isinstance(data, pl.DataFrame):
        data = polars_to_pandas(data=data)
    legend_kwargs = legend_kwargs or {}
    cbar_kwargs = cbar_kwargs or {}
    kwargs = _bubbleplot_defaults(**kwargs)
    centroids = _generate_cluster_centroids(
        data=data,
        features=features,
        cluster_label=cluster_label,
        sample_label=sample_label,
        colour_label=colour_label,
        dim_reduction_method=dim_reduction_method,
        n_components=n_components,
        dim_reduction_kwargs=dim_reduction_kwargs,
    )

    if colour_label is not None:
        discrete = discrete_label(data=data, label=colour_label, discrete=discrete)

    if palette is None:
        if discrete:
            palette = discrete_palette(n=data.shape[0])
        else:
            palette = "coolwarm"

    ax = ax or plt.figure(figsize=figsize)[1]
    ax = sns.scatterplot(
        data=centroids,
        x=f"{dim_reduction_method}1",
        y=f"{dim_reduction_method}2",
        hue=colour_label,
        size="cluster_size",
        style=style,
        palette=palette,
        hue_norm=hue_norm,
        legend=False,
        ax=ax,
        **kwargs,
    )

    if not discrete:
        plt.gcf().colorbar(ax, ax=ax, **cbar_kwargs)

    legend_kwargs = legend_kwargs or {}
    legend_kwargs["bbox_to_anchor"] = legend_kwargs.get("bbox_to_anchor", (1.1, 1.0))
    ax.legend(*ax.get_legend_handles_labels(), **legend_kwargs)
    return ax


def _bubbleplot_defaults(**kwargs):
    updated_kwargs = {k: v for k, v in kwargs.items()}
    defaults = {
        "edgecolor": "black",
        "alpha": 0.75,
        "linewidth": 2,
        "sizes": (100, 1000),
    }
    for k, v in defaults.items():
        if k not in updated_kwargs.keys():
            updated_kwargs[k] = v
    return updated_kwargs


def silhouette_analysis(
    data: pd.DataFrame,
    features: List[str],
    ax: Optional[plt.Axes] = None,
    figsize: Optional[Tuple[int, int]] = (7.5, 7.5),
    xlim: Tuple[int, int] = (-1, 1),
):
    n_clusters = data.cluster_label.nunique()
    cluster_labels = data.cluster_label.values
    if data.shape[0] > 10000:
        logger.warning("For data with more than 10000 observations downsampling is suggested.")
    ax = ax if ax is not None else plt.subplots(figsize=figsize)[1]
    ax.set_xlim(xlim)
    ax.set_ylim([0, data.shape[0] + (n_clusters + 1) * 10])
    silhouette_avg = silhouette_score(data[features], cluster_labels)
    sample_silhouette_values = silhouette_samples(data[features], cluster_labels)

    y_lower = 10
    for i, cl in enumerate(np.unique(cluster_labels)):
        ith_cluster_silhouette_values = sample_silhouette_values[cluster_labels == cl]
        ith_cluster_silhouette_values.sort()

        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i

        color = nipy_spectral(float(i) / n_clusters)
        ax.fill_betweenx(
            np.arange(y_lower, y_upper),
            0,
            ith_cluster_silhouette_values,
            facecolor=color,
            edgecolor=color,
            alpha=0.7,
        )

        # Label the silhouette plots with their cluster numbers at the middle
        ax.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

        # Compute the new y_lower for next plot
        y_lower = y_upper + 10  # 10 for the 0 samples
    ax.set_xlabel("The silhouette coefficient values")
    ax.set_ylabel("Cluster label")
    ax.axvline(x=silhouette_avg, color="red", linestyle="--")
    ax.set_yticks([])  # Clear the yaxis labels / ticks
    ax.set_xticks([round(i, 2) for i in np.arange(xlim[0], xlim[1] + 0.1, 0.1)])
    return ax


def boxswarm_and_source_count(
    plot_data: pd.DataFrame,
    x: str,
    y: str,
    hue: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 8),
    **plot_kwargs,
):
    n_sources = plot_data[[x, "n_sources"]].drop_duplicates().sort_values("n_sources", ascending=False)
    fig, ax = plt.subplots(figsize=figsize)
    axes = [ax, ax.twinx()]
    axes[0].bar(
        n_sources[x].values, n_sources.n_sources.values, color="grey", edgecolor=None, align="center", alpha=0.3
    )
    plot_kwargs = plot_kwargs or {}
    plot_kwargs["legend_anchor"] = plot_kwargs.get("legend_anchor", (1.2, 1))
    boxplot_kwargs = plot_kwargs.pop("boxplot_kwargs", {})
    boxplot_kwargs["order"] = n_sources[x].values
    overlay_kwargs = plot_kwargs.pop("overlay_kwargs", {})
    overlay_kwargs["order"] = n_sources[x].values
    box_swarm_plot(
        data=plot_data,
        x=x,
        y=y,
        hue=hue,
        ax=axes[1],
        boxplot_kwargs=boxplot_kwargs,
        overlay_kwargs=overlay_kwargs,
        **plot_kwargs,
    )
    axes[0].set_xticklabels(n_sources[x].values, rotation=90)
    axes[0].set_xlabel("")
    axes[1].set_ylabel(y)
    axes[0].set_ylabel("# of clustering algorithms")
    fig.tight_layout()
    return fig
