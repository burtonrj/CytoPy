from ..dim_reduction import dimensionality_reduction
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

np.random.seed(42)
TAB = list(sns.color_palette("tab20") +
           sns.color_palette("tab20b") +
           sns.color_palette("tab20c"))


def colour_palette(cmap: str,
                   discrete: bool):
    if cmap == "tab60":
        return TAB
    return sns.color_palette(palette=cmap, as_cmap=discrete is True)


def single_cell_plot(data: pd.DataFrame,
                     features: list,
                     label: str or None = None,
                     discrete: bool or None = None,
                     zscore: bool = False,
                     downsample: int = 1,
                     downsample_n: int = 1e5,
                     method: str = "UMAP",
                     dim_reduction_kwargs: dict or None = None,
                     ax: plt.Axes or None = None,
                     figsize: tuple or None = (8, 8),
                     include_legend: bool = False,
                     cmap: str = "tab60",
                     legend_kwargs: dict or None = None,
                     **kwargs):
    data = data.dropna(axis=1, how="any")
    if (method != "PCA" and data.shape[0] > downsample_n and downsample == 1) or downsample == 2:
        if data.shape[0] > downsample_n:
            data = data.sample(int(downsample_n))
    dim_reduction_kwargs = dim_reduction_kwargs or {}
    legend_kwargs = legend_kwargs or {}
    ax = ax or plt.subplots(figsize=figsize)[1]
    palette = colour_palette(cmap=cmap, discrete=discrete)
    data = dimensionality_reduction(data=data,
                                    features=features,
                                    method=method,
                                    n_components=2,
                                    return_reducer=False,
                                    return_embeddings_only=False,
                                    **dim_reduction_kwargs)
    if label is not None:
        if discrete:
            data[label] = data[label].astype(str)
        elif zscore:
            data[label] = StandardScaler().fit_transform(data[label].values.reshape(-1, 1))
        ax = sns.scatterplot(data=data,
                             x=f"{method}1",
                             y=f"{method}2",
                             hue=label,
                             palette=palette,
                             ax=ax,
                             **kwargs)
    else:
        ax = sns.scatterplot(data=data,
                             x=f"{method}1",
                             y=f"{method}2",
                             ax=ax,
                             **kwargs)
    ax.set_xlabel(f"{method}1")
    ax.set_ylabel(f"{method}2")
    if discrete:
        if include_legend:
            ax.legend(*ax.get_legend_handles_labels(), **legend_kwargs)
        else:
            ax.legend().remove()
    else:
        ax.legend(*ax.get_legend_handles_labels(), **legend_kwargs)
    return ax


def _cluster_centroids(data: pd.DataFrame,
                       features: list,
                       sample_label: str,
                       meta_label: str):
    return data.groupby([sample_label, meta_label])[features].median().reset_index()


def _assert_unique_label(x):
    assert len(x) == 1, "Chosen label is not unique within clusters"
    return x[0]


def _reassign_labels(data: pd.DataFrame,
                     sample_id: str,
                     cluster_label: str,
                     centroids: pd.DataFrame,
                     colour_label: str):
    lookup = data.groupby([sample_id, cluster_label])[colour_label].unique().apply(_assert_unique_label)
    centroids[colour_label] = centroids[[sample_id, cluster_label]].apply(lambda x: lookup.loc[x[0], x[1]], axis=1)
    return centroids


def _cluster_size(data: pd.DataFrame,
                  centroids: pd.DataFrame,
                  sample_id: str,
                  cluster_label: str):
    sample_size = data[sample_id].value_counts()
    sample_cluster_counts = data.groupby(sample_id)[cluster_label].value_counts()
    cluster_n = centroids[[sample_id, cluster_label]].apply(lambda x: sample_cluster_counts.loc[x[sample_id],
                                                                                                x[cluster_label]],
                                                            axis=1)
    sample_n = centroids["sample_id"].apply(lambda x: sample_size.loc[x])
    centroids["sample_n"], centroids["cluster_n"] = sample_n, cluster_n
    centroids["cluster_size"] = cluster_n / sample_n * 100
    return centroids


def meta_cluster_plot(data: pd.DataFrame,
                      features: list,
                      meta_label: str,
                      cluster_label: str,
                      sample_label: str,
                      colour_label: str or None = None,
                      discrete: bool = True,
                      cmap: str = "tab20",
                      dim_reduction_method="UMAP",
                      dim_reduction_kwargs: dict or None = None,
                      ax: plt.Axes or None = None,
                      figsize: tuple = (8, 8),
                      **kwargs):
    ax = ax or plt.subplots(figsize=figsize)[1]
    dim_reduction_kwargs = dim_reduction_kwargs or {}
    data = data.dropna(axis=1, how="any")
    centroids = _cluster_centroids(data=data, features=features, sample_label=sample_label, meta_label=meta_label)
    centroids = dimensionality_reduction(data=centroids,
                                         features=features,
                                         method=dim_reduction_method,
                                         n_components=2,
                                         return_reducer=False,
                                         return_embeddings_only=False,
                                         **dim_reduction_kwargs)
    if colour_label:
        centroids = _reassign_labels(data=data, centroids=centroids, colour_label=colour_label,
                                     sample_id=sample_label, cluster_label=cluster_label)
    centroids = _cluster_size(data=data, centroids=centroids, cluster_label=cluster_label,
                              sample_id=sample_label)
    kwargs = _bubbleplot_defaults(**kwargs)
    if colour_label:
        palette = colour_palette(cmap=cmap, discrete=discrete)
        return sns.scatterplot(data=centroids,
                               x=f"{dim_reduction_method}1",
                               y=f"{dim_reduction_method}2",
                               hue=colour_label,
                               palette=palette,
                               ax=ax,
                               size="cluster_size",
                               **kwargs)
    return sns.scatterplot(data=centroids,
                           x=f"{dim_reduction_method}1",
                           y=f"{dim_reduction_method}2",
                           ax=ax,
                           size="cluster_size",
                           **kwargs)


def _bubbleplot_defaults(**kwargs):
    updated_kwargs = {k: v for k, v in kwargs.items()}
    defaults = {"edgecolor": "black",
                "alpha": 0.75,
                "linewidth": 2,
                "sizes": (20, 400)}
    for k, v in defaults.items():
        if k not in updated_kwargs.keys():
            updated_kwargs[k] = v
    return updated_kwargs
