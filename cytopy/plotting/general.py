#!/usr/bin.env/python
# -*- coding: utf-8 -*-
"""
The 'general' module contains functions for general plotting, such as scatter and density plots for tSNE,
UMAP, PHATE plots, and functions for box and swarm plots.
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
import logging
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union

import numpy as np
import pandas as pd
import polars as pl
import seaborn as sns
from matplotlib import colors as cm
from matplotlib import pyplot as plt
from matplotlib.cm import ScalarMappable

from cytopy.data.read_write import polars_to_pandas


logger = logging.getLogger(__name__)


class ColumnWrapFigure(plt.Figure):
    """
    Extension of the Matplotlib Figure class for column wrapped figures. The user should specify the size of
    the grid (i.e. the number of subplots) as 'n', the maximum number of columns as 'col_wrap' and optionally
    the figure size. Subplots are then dynamically added using the 'add_wrapped_subplot' method, returning a
    new Axes object within the constraints of the column wrapping.
    Parameters
    ----------
    n: int
        The maximum number of subplots
    col_wrap: int
        The maximum number of columns
    figsize: Tuple[float, float], optional
        If not given, figsize defaults to (rows * 2.5, col_wrap * 2.5) where rows is n/col_wrap plus the remainder.
    args: optional
        Additional positional arguments passed to Matplotlib.Figure
    kwargs: optional
        Additional keyword arguments passed to Matplotlib.Figure
    """

    def __init__(self, n: int, col_wrap: int, figsize: Optional[Tuple[float, float]] = None, *args, **kwargs):
        rows = n // col_wrap
        rows += n % col_wrap
        kwargs = kwargs or {}
        figsize = figsize or (rows * 2.5, col_wrap * 2.5)
        super().__init__(*args, **kwargs, figsize=figsize)
        self.rows = rows
        self._i = 1
        self.n = n
        self.col_wrap = col_wrap

    def add_wrapped_subplot(self, *args, **kwargs) -> plt.Axes:
        """
        Add a new column wrapped subplot to Figure
        Parameters
        ----------
        args: optional
            Additional positional arguments passed to Matplotlib.Figure.add_subplot
        kwargs: optional
            Additional keyword arguments passed to Matplotlib.Figure.add_subplot
        Returns
        -------
        Matplotlib.Axes
        """
        if self._i > self.n:
            raise ValueError("Figure is full! Define figure with more rows to add more subplots.")
        ax = super().add_subplot(self.rows, self.col_wrap, self._i, *args, **kwargs)
        self._i += 1
        return ax


def box_swarm_plot(
    data: pd.DataFrame,
    x: str,
    y: str,
    hue: Optional[str] = None,
    ax: Optional[plt.Axes] = None,
    palette: Optional[str] = None,
    figsize: Tuple[int, int] = (5, 5),
    overlay: bool = True,
    order: Optional[List[str]] = None,
    hue_order: Optional[List[str]] = None,
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
    xticklabels: Optional[List[Any]] = None,
    yticklabels: Optional[List[Any]] = None,
    boxplot_kwargs: Optional[Dict] = None,
    overlay_kwargs: Optional[Dict] = None,
    legend_anchor: Tuple[int, int] = (1.05, 1),
):
    """
    This is a convenience function that wraps Seaborn's boxplot and swarmplot functions. It will generate/plot a single
    axis with the boxplot in the background and swarmplot in the foreground.
    Parameters
    ----------
    data: Pandas.DataFrame
    x: str
        Name of column to plot on x-axis
    y: str
        Name of column to plot on y-axis
    hue: str, optional
        Name of column used to specify the colour of boxplot and swarm points
    ax: Matplotlib.Axes, optional
        If provided, plotting will be performed on existing Axes
    palette: palette name, list, or dict, optional
        Passed to palette argument of seaborn functions
    figsize: Tuple[int, int], (default=(5, 5))
        Ignored if 'ax' provided, otherwise new figure generated with this figure size.
    overlay: bool (default=True)
        If True, swarmplot generated overlaying the boxplot
    order: List[str], optional
        If provided, will specify the order of the x-axis values
    hue_order: List[str], optional
        If provided, will specify the hue order
    xlabel: str, optional
    ylabel: str, optional
    xticklabels: List[Any], optional
    yticklabels: List[Any], optional
    boxplot_kwargs: Dict, optional
        Additional keyword arguments passed to seaborn.boxplot
    overlay_kwargs: Dict, optional
        Additional keyword arguments passed to seaborn.swarmplot
    legend_anchor: Tuple[int, int], (default=(1.05, 1))
        Position of the legend relative to plot axes
    Returns
    -------
    Matplotlib.Axes
    """
    boxplot_kwargs = boxplot_kwargs or {}
    overlay_kwargs = overlay_kwargs or {}
    if order:
        boxplot_kwargs["order"] = boxplot_kwargs.get("order", order)
        overlay_kwargs["order"] = overlay_kwargs.get("order", order)
    if hue_order:
        boxplot_kwargs["hue_order"] = boxplot_kwargs.get("hue_order", hue_order)
        overlay_kwargs["hue_order"] = overlay_kwargs.get("hue_order", hue_order)
    ax = ax if ax is not None else plt.subplots(figsize=figsize)[1]
    sns.boxplot(
        data=data,
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
            data=data,
            x=x,
            y=y,
            hue=hue,
            ax=ax,
            dodge=True,
            palette=palette,
            **overlay_kwargs,
        )
        if hue:
            handles, labels = ax.get_legend_handles_labels()
            n = data[hue].nunique()
            ax.legend(handles[0:n], labels[0:n], bbox_to_anchor=legend_anchor)
    if xlabel is not None:
        ax.set_xlabel(xlabel=xlabel)
    if ylabel is not None:
        ax.set_ylabel(ylabel=ylabel)
    if xticklabels is not None:
        ax.set_xticklabels(xticklabels)
    if yticklabels is not None:
        ax.set_yticklabels(yticklabels)
    return ax


def discrete_palette(n: int) -> str:
    """
    Default discrete palette based on 'n' categories.
    * If n <= 10 then palette is 'tab10'
    * if 10 > n >= 20 the palette is 'tab20'
    * if n > 20, then a warning will be raised and palette is 'tab20'
    Parameters
    ----------
    n: int
    Returns
    -------
    str
    """
    if n <= 10:
        return "tab10"
    if n <= 20:
        return "tab20"
    logger.warning("Palette requires more than 20 unique colours, be careful interpreting results!")
    hsv = plt.get_cmap("Spectral")
    return hsv(np.linspace(0, 1.0, n))


def discrete_label(data: pd.DataFrame, label: str, discrete: Optional[bool] = None) -> bool:
    """
    Performs checks regarding the colour variable in the plotting data. If the user has not specified
    whether the variable should be treated as discrete, the column will be checked for numeric data types
    and if found, the label will be treated as discrete.
    If the label has been specified to be continuous by the user or found to contain only numeric data, then
    returns False, otherwise the 'label' column is coerced to a string data type and returns True.
    Parameters
    ----------
    data: Pandas.DataFrame
    label: str
    discrete: bool, optional
    Returns
    -------
    bool
    """
    if discrete is None:
        if pd.api.types.is_numeric_dtype(data[label]):
            return False
        discrete = True
    if discrete:
        data[label] = data[label].astype(str)
        return True
    return False


def scatterplot_defaults(**kwargs) -> Dict:
    """
    Yields default settings for scatterplot
    Parameters
    ----------
    kwargs: optional
        User defined settings
    Returns
    -------
    Dict
    """
    updated_kwargs = {k: v for k, v in kwargs.items()}
    defaults = {"alpha": 0.75, "linewidth": 0, "edgecolors": None, "s": 5}
    for k, v in defaults.items():
        if k not in updated_kwargs.keys():
            updated_kwargs[k] = v
    return updated_kwargs


def scatterplot(
    data: Union[pl.DataFrame, pd.DataFrame],
    x: str,
    y: str,
    label: Optional[str] = None,
    discrete: Optional[bool] = None,
    hue_norm: Optional[Union[plt.Normalize, Tuple[int, int]]] = (0, 1),
    include_legend: bool = False,
    palette: Optional[str] = None,
    size: Optional[Union[int, str]] = 5,
    style: Optional[str] = None,
    legend_kwargs: Optional[Dict] = None,
    cbar_kwargs: Optional[Dict] = None,
    ax: Optional[plt.Axes] = None,
    figsize: Tuple[int, int] = (8, 8),
    xscale: Optional[str] = None,
    yscale: Optional[str] = None,
    **kwargs,
):
    """
    General function for scatterplot. Wraps seaborn.scatterplot for convenient plotting and some default settings
    inforced.
    Parameters
    ----------
    data: Pandas.DataFrame or Polars.DataFrame
    x: str
        Name of the column to plot on the x-axis
    y: str
        Name of the column to plot on the y-axis
    label: str, optional
        Column to use for colouring the data points
    discrete: bool, optional
        Specify whether the label should be treated as discrete for the purposes of the colour palette. If not discrete
        then axis will be accompanied by a colour bar. If not specified (default) then the label will be treated as
        continuous if all values in the label column are numeric.
    hue_norm: Union[plt.Normalize, Tuple[int, int]], optional
        Either a Normalize object or a Tuple of integers, passed to seaborn.scatterplot hue_norm argument
    include_legend: bool (default=False)
    palette: str, optional
        Passed to seaborn.scatterplot palette argument. If not provided, will be selected from the data:
            * If label is not discrete, defaults to 'coolwarm'
            * If label is discrete and n categories less than or equal to 10, will be 'tab10'
            * If label is discrete and n categories is more than 10 but less than or equal to 20, will be 'tab20'
            * If label is discrete and n categories is more than 20, will raise a warning and palette will be
            'tab20'
    size: str, optional
        Passed to seaborn.scatterplot 'size' argument
    style: str, optional
        Passed to seaborn.scatterplot 'style' argument
    legend_kwargs: optional
        Additional keyword arguments passed to legend, ignored if 'include_legend' is False
    cbar_kwargs: optional
        Additional keyword arguments passed to 'colorbar', ignored if label is discrete
    ax: Matplotlib.Axes, optional
        If provided, plotting will be performed on existing Axes
    figsize: Tuple[int, int], (default=(5, 5))
        Ignored if 'ax' provided, otherwise new figure generated with this figure size.
    xscale: str, optional
        X-axis scale
    yscale: str, optional
        Y-axis scale
    kwargs: optional
        Additional keyword arguments passed to seaborn.scatterplot
    Returns
    -------
    Matplotlib.Axes
    """
    hue_order = None
    if isinstance(data, pl.DataFrame):
        data = polars_to_pandas(data=data)
    if label is not None:
        discrete = discrete_label(data=data, label=label, discrete=discrete)
        hue_order = data[label].value_counts().sort_values(ascending=False).index.values
        data = data.set_index(label).loc[hue_order].reset_index()
    if palette is None:
        if discrete and label is not None:
            palette = discrete_palette(n=data[label].nunique())
            hue_norm = None
        else:
            palette = "coolwarm"
    if size is not None:
        if isinstance(size, int):
            kwargs["s"] = size
        else:
            kwargs.pop("s", None)

    kwargs = scatterplot_defaults(**kwargs)
    cbar_kwargs = cbar_kwargs or {}
    legend_kwargs = legend_kwargs or {}

    key = [x, y] if label is None else [x, y, label]
    data = data[~data[key].isnull().any(axis=1)]
    ax = ax or plt.subplots(figsize=figsize)[1]

    ax = sns.scatterplot(
        data=data,
        x=x,
        y=y,
        hue=label,
        size=size if isinstance(size, str) else None,
        style=style,
        palette=palette,
        hue_norm=hue_norm,
        hue_order=hue_order,
        legend=include_legend,
        ax=ax,
        **kwargs,
    )

    if not discrete:
        if isinstance(hue_norm, tuple):
            hue_norm = cm.Normalize(*hue_norm)
        plt.gcf().colorbar(ScalarMappable(norm=hue_norm, cmap=palette), ax=ax, **cbar_kwargs)

    ax.set_xlabel(x)
    ax.set_ylabel(y)

    if xscale is not None:
        ax.set_xscale(xscale)
    if yscale is not None:
        ax.set_xscale(yscale)

    if discrete:
        if include_legend:
            ax.legend(*ax.get_legend_handles_labels(), **legend_kwargs)
        else:
            ax.legend().remove()

    return ax


def density_plot(
    data: Union[pl.DataFrame, pd.DataFrame],
    x: str,
    y: str,
    bins: Optional[int] = None,
    palette: str = "jet",
    ax: Optional[plt.Axes] = None,
    figsize: Tuple[int, int] = (8, 8),
    norm: Optional[plt.Normalize] = None,
    **kwargs,
):
    """
    Plot a two-dimensional histogram
    Parameters
    ----------
    data: Pandas.DataFrame or Polars.DataFrame
    x: str
        Column to plot on x-axis
    y: str
        Column to plot on y-axis
    bins: int, optional
        Number of bins. If not provided will default to sqrt(n) where n is the number of observations
    palette: str, (default='jet')
    ax: Matplotlib.Axes, optional
        If provided, plotting will be performed on existing Axes
    figsize: Tuple[int, int], (default=(5, 5))
        Ignored if 'ax' provided, otherwise new figure generated with this figure size.
    norm: Matplotlib.Normalize, optional
        If not given, defaults to LogNorm
    kwargs: optional
        Additional keyword arguments passed to Matplotlib.hist2d
    Returns
    -------
    Matplotlib.Axes
    """
    bins = bins if isinstance(bins, int) else int(np.sqrt(data.shape[0]))
    ax = ax or plt.subplots(figsize=figsize)[1]
    norm = norm or cm.LogNorm()
    ax.autoscale(enable=True)
    ax.hist2d(data[x], data[y], bins=bins, cmap=palette, norm=norm, **kwargs)
    return ax
