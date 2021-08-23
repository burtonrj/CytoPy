#!/usr/bin.env/python
# -*- coding: utf-8 -*-
"""
This module houses plotting functions for global views of an Experiment data, for example
single cell or cluster centroid plots after dimension reduction has been performed.

Copyright 2020 Ross Burton

Permission is hereby granted, free of charge, to any person
obtaining a copy of this software and associated documentation
files (the "Software"), to deal in the Software without restriction,
including without limitation the rights to use, copy, modify,
merge, publish, distribute, sublicense, and/or sell copies of the
Software, and to permit persons to whom the Software is furnished
to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included
in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""
from itertools import cycle
from typing import Union

import matplotlib.colors as cm
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
import polars as pl
import seaborn as sns
from scipy.spatial.distance import pdist
from scipy.spatial.distance import squareform
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler

from ..dim_reduction import DimensionReduction


def _scatterplot_defaults(**kwargs):
    updated_kwargs = {k: v for k, v in kwargs.items()}
    defaults = {"alpha": 0.75, "linewidth": 0}
    for k, v in defaults.items():
        if k not in updated_kwargs.keys():
            updated_kwargs[k] = v
    return updated_kwargs


def discrete_scatterplot(
    data: pl.DataFrame,
    x: str,
    y: str,
    z: str or None,
    label: str,
    cmap: str,
    size: int or str or None,
    fig: plt.Figure,
    **kwargs,
):
    """
    Scatterplot with discrete label

    Parameters
    ----------
    data: Pandas.DataFrame
    x: str
    y: str
    z: str, optional
    label: str
    cmap: str
    size: int or str, optional
    fig: Matplotlib.Figure
    kwargs:
        Additional keyword arguments passed to Matplotlib.Axes.scatter call

    Returns
    -------
    Matplotlib.Axes
    """
    colours = cycle(plt.get_cmap(cmap).colors)
    data[label] = data[label].astype(str)
    if z is not None:
        ax = fig.add_subplot(111, projection="3d")
        for (l, df), c in zip(data.to_pandas().groupby(label), colours):
            s = size
            if isinstance(size, str):
                s = df[size]
            ax.scatter(
                df[x],
                df[y],
                df[z],
                s=s,
                color=c,
                label=l,
                **kwargs,
            )
        return ax
    ax = fig.add_subplot(111)
    for (l, df), c in zip(data.groupby(label), colours):
        s = size
        if isinstance(size, str):
            s = df[size]
        ax.scatter(df[x], df[y], color=c, label=l, s=s, **kwargs)
    return ax


def cont_scatterplot(
    data: pl.DataFrame,
    x: str,
    y: str,
    z: str or None,
    label: str,
    cmap: str,
    size: int or str or None,
    fig: plt.Figure,
    cbar_kwargs: dict,
    **kwargs,
):
    """
    Scatterplot with continuous label

    Parameters
    ----------
    data: Pandas.DataFrame
    x: str
    y: str
    z: str, optional
    label: str
    cmap: str
    size: int or str, optional
    fig: Matplotlib.Figure
    cbar_kwargs: dict
        Keyword arguments passed to colorbar
    kwargs:
        Additional keyword arguments passed to Matplotlib.Axes.scatter call

    Returns
    -------
    Matplotlib.Axes
    """
    if isinstance(size, str):
        size = data[size].to_numpy()
    if z is not None:
        ax = fig.add_subplot(111, projection="3d")
        im = ax.scatter(
            data[x].to_numpy(),
            data[y].to_numpy(),
            data[z].to_numpy(),
            c=data[label].to_numpy(),
            s=size,
            cmap=cmap,
            **kwargs,
        )
    else:
        ax = fig.add_subplot(111)
        im = ax.scatter(
            data[x].to_numpy(),
            data[y].to_numpy(),
            c=data[label].to_numpy(),
            s=size,
            cmap=cmap,
            **kwargs,
        )
    fig.colorbar(im, ax=ax, **cbar_kwargs)
    return ax


def single_cell_plot(
    data: pl.DataFrame,
    x: str,
    y: str,
    z: str or None = None,
    label: str or None = None,
    discrete: bool or None = None,
    scale: str or None = None,
    figsize: tuple = (8, 8),
    include_legend: bool = False,
    cmap: str = "tab20",
    size: int or str or None = 10,
    legend_kwargs: dict or None = None,
    cbar_kwargs: dict or None = None,
    **kwargs,
):
    """
    Single cell plot, to be used with a dimensionality reduction method for example. Takes a
    DataFrame of single cell data and the name of two or three columns (generates a 3D plot if third
    is given). Specify discrete as True to treat label (column used to colour data points) as
    categorical, otherwise treated as continuous; be sure to supply a suitable colourmap, we
    recommend 'tab20' for discrete plots and 'coolwarm' for continuous.

    Parameters
    ----------
    data: Pandas.DataFrame
    x: str
        X-axis variable
    y: str
        Y-axis variable
    z: str, optional
        Z-axis variable (induces 3D plot)
    label: str
        Variable used to colour data points
    discrete: bool (default=True)
        Treat label as categorical
    cmap: str (default="tab20")
        Colourmap (must be a valid Matplotlib colourmap)
    scale: str, optional
        Scale data prior to plotting. Valid methods are 'zscore' or 'minmax'.
    figsize: tuple (default=(8,8))
        Figure size
    size: int or str, optional
        Size of the data points. Either an integer for uniform size or name of the column
        to infer datapoint size from
    include_legend: bool (default=False)
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
    data = data.copy()
    kwargs = _scatterplot_defaults(**kwargs)
    cbar_kwargs = cbar_kwargs or {}
    data = data.dropna(axis=1, how="any")
    legend_kwargs = legend_kwargs or {}
    fig = plt.figure(figsize=figsize)
    if label is not None:
        if discrete:
            ax = discrete_scatterplot(
                data=data.to_pandas(),
                x=x,
                y=y,
                z=z,
                label=label,
                size=size,
                cmap=cmap,
                fig=fig,
                **kwargs,
            )
        else:
            if scale == "zscore":
                data[label] = StandardScaler().fit_transform(data[label].to_numpy().reshape(-1, 1))
            elif scale == "minmax":
                data[label] = MinMaxScaler().fit_transform(data[label].to_numpy().reshape(-1, 1))
            ax = cont_scatterplot(
                data=data.to_pandas(),
                x=x,
                y=y,
                z=z,
                label=label,
                size=size,
                cmap=cmap,
                fig=fig,
                cbar_kwargs=cbar_kwargs,
                **kwargs,
            )
    else:
        if isinstance(size, str):
            size = data[size].to_numpy()
        if z is not None:
            ax = fig.add_subplot(111, projection="3d")
            ax.scatter(data[x], data[y], data[z], s=size, **kwargs)
        else:
            ax = fig.add_subplot(111)
            ax.scatter(data[x], data[y], s=size, **kwargs)
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    if discrete:
        if include_legend:
            ax.legend(*ax.get_legend_handles_labels(), **legend_kwargs)
        else:
            ax.legend().remove()
    return fig, ax


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
    colour_label: str or None = "meta_label",
    zscore: bool = False,
    discrete: bool = True,
    cmap: str = "tab20",
    dim_reduction_method: str or None = "UMAP",
    n_components: int = 2,
    dim_reduction_kwargs: dict or None = None,
    figsize: tuple = (8, 8),
    legend_kwargs: dict or None = None,
    cbar_kwargs: dict or None = None,
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
        Additional keyword arguments passed to dimension reduction (see cytopy.flow.dim_reduction)
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
        data = data.to_pandas()
    fig = plt.figure(figsize=figsize)
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
    if colour_label:
        if discrete:
            centroids[colour_label] = centroids[colour_label].astype(str)
        elif zscore:
            centroids[colour_label] = StandardScaler().fit_transform(centroids[colour_label].to_numpy().reshape(-1, 1))
        if n_components == 2:
            ax = fig.add_subplot(111)
            ax = sns.scatterplot(
                data=centroids,
                x=f"{dim_reduction_method}1",
                y=f"{dim_reduction_method}2",
                hue=colour_label,
                palette=cmap,
                ax=ax,
                size="cluster_size",
                **kwargs,
            )
        else:
            ax = discrete_scatterplot(
                data=centroids,
                x=f"{dim_reduction_method}1",
                y=f"{dim_reduction_method}2",
                z=f"{dim_reduction_method}3",
                size="cluster_size",
                label=colour_label,
                cmap=cmap,
                fig=fig,
                **kwargs,
            )
    else:
        if n_components == 2:
            ax = fig.add_subplot(111)
            ax = sns.scatterplot(
                data=centroids,
                x=f"{dim_reduction_method}1",
                y=f"{dim_reduction_method}2",
                ax=ax,
                size="cluster_size",
                **kwargs,
            )
        else:
            ax = cont_scatterplot(
                data=data,
                x=f"{dim_reduction_method}1",
                y=f"{dim_reduction_method}2",
                z=f"{dim_reduction_method}3",
                label=colour_label,
                size="cluster_size",
                cmap=cmap,
                fig=fig,
                cbar_kwargs=cbar_kwargs,
                **kwargs,
            )
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


def plot_min_spanning_tree(
    data: Union[pd.DataFrame, pl.DataFrame],
    features: list,
    cluster_label: str,
    sample_label: str,
    colour_label: str or None = "meta_label",
    **kwargs,
):
    """
    Experimental method in version 2.0. Generates a minimum spanning tree of cluster centroids.

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
    kwargs:
        Keyword arguments passed to NetworkX.draw

    Returns
    -------
    Matplotlib.Axes
    """
    if isinstance(data, pl.DataFrame):
        data = data.to_pandas()
    centroids = _generate_cluster_centroids(
        data=data,
        features=features,
        cluster_label=cluster_label,
        sample_label=sample_label,
        colour_label=colour_label,
        dim_reduction_method=None,
        n_components=2,
        dim_reduction_kwargs={},
    )
    graph = nx.Graph()
    distance_matrix = squareform(pdist(centroids[features]), "minkowski")
    for i in range(len(distance_matrix)):
        for j in range(i + 1, len(distance_matrix)):
            w = distance_matrix[i][j]
            graph.add_edge(i, j, weight=w)
    mst = nx.minimum_spanning_tree(graph)
    norm = cm.Normalize(vmin=0, vmax=21, clip=True)
    mapper = plt.cm.ScalarMappable(norm=norm, cmap=plt.get_cmap("tab10"))
    colours = centroids[colour_label].apply(lambda x: mcolors.to_hex(mapper.to_rgba(x)))

    fig, ax = plt.subplots(figsize=(10, 10))
    pos = nx.spring_layout(mst, iterations=300, scale=3, dim=2)
    sizes = centroids["cluster_size"].to_numpy() * 2000
    nx.draw(
        mst,
        pos=pos,
        with_labels=False,
        node_size=sizes,
        node_color=colours,
        width=2,
        alpha=0.5,
        ax=ax,
        **kwargs,
    )
    ax.legend()
    return ax
