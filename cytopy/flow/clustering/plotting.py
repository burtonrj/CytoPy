from typing import List
from typing import Optional
from typing import Type
from typing import Union

import pandas as pd
import seaborn as sns

from ..plotting import cluster_bubble_plot
from ..plotting import single_cell_plot
from cytopy.flow.dim_reduction import DimensionReduction
from cytopy.flow.sampling import sample_dataframe_uniform_groups


def plot_cluster_membership(
    data: pd.DataFrame,
    features: List[str],
    sample_size: Union[int, None] = 100000,
    sampling_method: str = "uniform",
    method: Union[str, Type] = "UMAP",
    dim_reduction_kwargs: dict or None = None,
    label: str = "cluster_label",
    discrete: bool = True,
    **kwargs,
):
    """
    Plot the entire single cell dataframe (the data attribute). WARNING: this can be computationally
    expensive so if you have limited resource, try specifying a sample size first.

    Parameters
    ----------
    sample_size: int, optional (default=100000)
    sampling_method: str (default="uniform")
        Can either be uniform or absolute:
        * uniform: take a uniform sample of the same size from each sample
        * absolute: sample uniformly from the whole dataframe without accounting for individual
        biological sample size
    method: Union[str, Type]
        Dimensionality reduction technique; available in-built methods are: UMAP, PCA, PHATE, KernelPCA or tSNE.
        (see cytopy.flow.dimension_reduction)
    dim_reduction_kwargs: dict, optional
        Additional keyword arguments passed to dimension reduction (see cytopy.flow.dim_reduction)
    label: str, (default='cluster_label')
        How to colour single cells
    discrete: bool (default=True)
        If True, label is treated as a discrete variable. If False, continuous colourmap will be applied.
    kwargs:
        Additional keyword arguments passed to cytopy.flow.plotting.single_cell_plot

    Returns
    -------
    Matplotlib.Axes
    """
    plot_data = data
    if sample_size is not None:
        if sampling_method == "uniform":
            plot_data = sample_dataframe_uniform_groups(data=plot_data, group_id="sample_id", sample_size=sample_size)
        else:
            if sample_size < data.shape[0]:
                plot_data = data.sample(sample_size)

    dim_reduction_kwargs = dim_reduction_kwargs or {}
    reducer = DimensionReduction(method=method, n_components=2, **dim_reduction_kwargs)
    df = reducer.fit_transform(data=plot_data, features=features)
    return single_cell_plot(data=df, x=f"{method}1", y=f"{method}2", label=label, discrete=discrete, **kwargs)


def plot_cluster_membership_sample(
    data: pd.DataFrame,
    features: List[str],
    sample_id: str,
    method: Union[str, Type] = "UMAP",
    dim_reduction_kwargs: dict or None = None,
    label: str = "cluster_label",
    discrete: bool = True,
    **kwargs,
):
    """
    Generate a single cell plot (see cytopy.flow.plotting.single_cell_plot) for a single sample,
    with cells coloured by cluster membership (default)

    Parameters
    ----------
    sample_id: str
    method: Union[str, Type]
        Dimensionality reduction technique; available in-built methods are: UMAP, PCA, PHATE, KernelPCA or tSNE.
        (see cytopy.flow.dimension_reduction)
    dim_reduction_kwargs: dict, optional
        Additional keyword arguments passed to dimension reduction (see cytopy.flow.dim_reduction)
    label: str, (default='cluster_label')
        How to colour single cells
    discrete: bool (default=True)
        If True, label is treated as a discrete variable. If False, continuous colourmap will be applied.
    kwargs:
        Additional keyword arguments passed to cytopy.flow.plotting.single_cell_plot

    Returns
    -------
    Matplotlib.Axes
    """
    dim_reduction_kwargs = dim_reduction_kwargs or {}
    df = data[data.sample_id == sample_id].copy()
    reducer = DimensionReduction(method=method, n_components=2, **dim_reduction_kwargs)
    df = reducer.fit_transform(data=df, features=features)
    return single_cell_plot(
        data=df,
        x=f"{method}1",
        y=f"{method}2",
        label=label,
        discrete=discrete,
        **kwargs,
    )


def clustered_heatmap(
    data: pd.DataFrame,
    features: List[str],
    sample_id: Optional[str] = None,
    meta_label: bool = True,
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
        data = data.groupby(["meta_label"])[features].median()
    elif sample_id is None and not meta_label:
        data = data.groupby(["cluster_label"])[features].median()
    else:
        data = data[data.sample_id == sample_id].groupby(["cluster_label"]).median()
    data[features] = data[features].apply(pd.to_numeric)
    kwargs = kwargs or {}
    kwargs["col_cluster"] = kwargs.get("col_cluster", True)
    kwargs["figsize"] = kwargs.get("figsize", (10, 15))
    kwargs["standard_scale"] = kwargs.get("standard_scale", 1)
    kwargs["cmap"] = kwargs.get("cmap", "viridis")
    return sns.clustermap(data[features], **kwargs)


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
    Generate a cluster bubble plot (see cytopy.flow.plotting.cluster_bubble_plot) where each
    data point (bubble) is a single cluster centroid from a unique patient. Size of the data points represents
    the fraction of cells with membership to the sample relative to the total number of events
    in that sample. By default data points are coloured by meta label membership.

    Parameters
    ----------
    method: str
        Dimensionality reduction technique; available methods are: UMAP, PCA, PHATE, KernelPCA or tSNE
    dim_reduction_kwargs: dict, optional
        Additional keyword arguments passed to dimension reduction (see cytopy.flow.dim_reduction)
    colour_label: str, (default='meta_label')
        How to colour cluster centroids
    discrete: bool (default=True)
        If True, label is treated as a discrete variable. If False, continuous colourmap will be applied.
    kwargs:
        Additional keyword arguments passed to cytopy.flow.plotting.cluster_bubble_plot

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
