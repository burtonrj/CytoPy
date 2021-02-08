from ..dim_reduction import dimensionality_reduction
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from itertools import cycle
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


def _scatterplot_defaults(**kwargs):
    updated_kwargs = {k: v for k, v in kwargs.items()}
    defaults = {"alpha": 0.75,
                "linewidth": 0,
                "s": 10}
    for k, v in defaults.items():
        if k not in updated_kwargs.keys():
            updated_kwargs[k] = v
    return updated_kwargs


def _discrete_scatterplot(data: pd.DataFrame,
                          x: str,
                          y: str,
                          label: str,
                          cmap: str,
                          ax: plt.Axes,
                          **kwargs):
    colours = cycle(plt.get_cmap(cmap).colors)
    data[label] = data[label].astype(str)
    for (l, df), c in zip(data.groupby(label), colours):
        ax.scatter(df[x].values,
                   df[y].values,
                   color=c,
                   label=l,
                   **kwargs)
    return ax


def _cont_scatterplot(data: pd.DataFrame,
                      x: str,
                      y: str,
                      label: str,
                      cmap: str,
                      ax: plt.Axes,
                      fig: plt.Figure,
                      cbar_kwargs: dict,
                      **kwargs):
    im = ax.scatter(data[x].values,
                    data[y].values,
                    c=data[label].values,
                    cmap=cmap,
                    **kwargs)
    fig.colorbar(im, ax=ax, **cbar_kwargs)
    return ax


def single_cell_plot(data: pd.DataFrame,
                     x: str,
                     y: str,
                     label: str or None = None,
                     discrete: bool or None = None,
                     scale: str or None = None,
                     figsize: tuple = (8, 8),
                     include_legend: bool = False,
                     cmap: str = "tab20",
                     legend_kwargs: dict or None = None,
                     cbar_kwargs: dict or None = None,
                     **kwargs):
    data = data.copy()
    kwargs = _scatterplot_defaults(**kwargs)
    cbar_kwargs = cbar_kwargs or {}
    data = data.dropna(axis=1, how="any")
    legend_kwargs = legend_kwargs or {}
    fig, ax = plt.subplots(figsize=figsize)
    if label is not None:
        if discrete:
            ax = _discrete_scatterplot(data=data,
                                       x=x,
                                       y=y,
                                       label=label,
                                       cmap=cmap,
                                       ax=ax,
                                       **kwargs)
        else:
            if scale == "zscore":
                data[label] = StandardScaler().fit_transform(data[label].values.reshape(-1, 1))
            elif scale == "minmax":
                data[label] = MinMaxScaler().fit_transform(data[label].values.reshape(-1, 1))
            ax = _cont_scatterplot(data=data,
                                   x=x,
                                   y=y,
                                   label=label,
                                   cmap=cmap,
                                   ax=ax,
                                   fig=fig,
                                   cbar_kwargs=cbar_kwargs,
                                   **kwargs)
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    if discrete:
        if include_legend:
            ax.legend(*ax.get_legend_handles_labels(), **legend_kwargs)
        else:
            ax.legend().remove()
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
                      zscore: bool = False,
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
        if discrete:
            centroids[colour_label] = data[colour_label].astype(str)
        elif zscore:
            centroids[colour_label] = StandardScaler().fit_transform(centroids[colour_label].values.reshape(-1, 1))
        return sns.scatterplot(data=centroids,
                               x=f"{dim_reduction_method}1",
                               y=f"{dim_reduction_method}2",
                               hue=colour_label,
                               palette=cmap,
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
