from typing import Dict
from typing import Optional

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


class ColumnWrapFigure(plt.Figure):
    def __init__(self, n: int, col_wrap: int, *args, **kwargs):
        rows = n // col_wrap
        rows += n % col_wrap
        kwargs = kwargs or {}
        figsize = kwargs.pop("figsize", (rows * 2.5, col_wrap * 2.5))
        super().__init__(*args, **kwargs, figsize=figsize)
        self.rows = rows
        self._i = 1
        self.n = n
        self.col_wrap = col_wrap

    def add_subplot(self, *args, **kwargs):
        if self._i > self.n:
            raise ValueError("Figure is full! Define figure with more rows to add more subplots.")
        ax = super().add_subplot(self.rows, self.col_wrap, self._i, *args, **kwargs)
        self._i += 1
        return ax


def build_plot_grid(n: int, col_wrap: int, **kwargs):
    kwargs = kwargs or {}
    rows = n // col_wrap
    rows += n % col_wrap
    figsize = kwargs.pop("figsize", (rows * 5, col_wrap * 5))
    fig = plt.figure(figsize=figsize, **kwargs)
    axes = []
    for i in range(n):
        axes.append(fig.add_subplot(rows, col_wrap, i + 1))
    fig.tight_layout()
    return fig, axes


def box_swarm_plot(
    plot_df: pd.DataFrame,
    x: str,
    y: str,
    hue: Optional[str] = None,
    ax: Optional[plt.Axes] = None,
    palette: Optional[str] = None,
    overlay: bool = True,
    boxplot_kwargs: Optional[Dict] = None,
    overlay_kwargs: Optional[Dict] = None,
):
    """
    Convenience function for generating a boxplot with a swarmplot/stripplot overlaid showing
    individual datapoints (using tools from Seaborn library)

    Parameters
    ----------
    plot_df: polars.DataFrame
        Data to plot
    x: str
        Name of the column to use as x-axis variable
    y: str
        Name of the column to use as y-axis variable
    hue: str, optional
        Name of the column to use as factor to colour plot
    overlay: bool (default=True)
        Overlay swarm plot on boxplot
    ax: Matplotlib.Axes, optional
        Axis object to plot on. If None, will generate new axis of figure size (10,5)
    palette: str, optional
        Palette to use
    boxplot_kwargs: dict, optional
        Additional keyword arguments passed to Seaborn.boxplot
    overlay_kwargs: dict, optional
        Additional keyword arguments passed to Seaborn.swarmplot/stripplot

    Returns
    -------
    Matplotlib.Axes
    """
    boxplot_kwargs = boxplot_kwargs or {}
    overlay_kwargs = overlay_kwargs or {}
    ax = ax if ax is not None else plt.subplots(figsize=(10, 5))[1]
    sns.boxplot(
        data=plot_df,
        x=x,
        y=y,
        hue=hue,
        ax=ax,
        showfliers=False,
        boxprops=dict(alpha=0.3),
        palette=palette,
        **boxplot_kwargs,
    )
    if overlay:
        sns.stripplot(
            data=plot_df,
            x=x,
            y=y,
            hue=hue,
            ax=ax,
            dodge=True,
            palette=palette,
            **overlay_kwargs,
        )
    return ax
