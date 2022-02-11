#!/usr/bin.env/python
# -*- coding: utf-8 -*-
"""
This module concerns plotting functions for one or two-dimensional 'flow plots' common to software such as FlowJo.
These plots support common transform methods in cytometry such as logicle (biexponential), log, hyper-log and
inverse hyperbolic arc-sine. These plotting functions have application in traditional gating and visualising
populations in low dimensional space.
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
from itertools import cycle
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union

import numpy as np
import seaborn as sns
from KDEpy import FFTKDE
from matplotlib import pyplot as plt
from matplotlib.collections import QuadMesh
from matplotlib.colors import LogNorm
from mongoengine import DoesNotExist

from cytopy.data import FileGroup
from cytopy.data import Population
from cytopy.gating.base import Child
from cytopy.gating.base import Gate
from cytopy.gating.threshold import ThresholdBase
from cytopy.gating.threshold import ThresholdGate
from cytopy.plotting.asinh_transform import *
from cytopy.plotting.hlog_transform import *
from cytopy.plotting.logicle_transform import *
from cytopy.utils.transform import apply_transform
from cytopy.utils.transform import TRANSFORMERS

logger = logging.getLogger(__name__)


class PlotError(Exception):
    def __init__(self, message: Exception):
        logger.error(message)
        super().__init__(message)


def _auto_plot_kind(
    data: pd.DataFrame,
    y: Optional[str] = None,
) -> str:
    """
    Determine the best plotting method. If the number of observations is less than 1000, returns 'hist2d' otherwise
    returns 'scatter_kde'. If 'y' is None returns 'kde'.
    Parameters
    ----------
    data: Pandas.DataFrame
    y: str
    Returns
    -------
    str
    """
    if y:
        if data.shape[0] > 1000:
            return "hist2d"
        return "scatter_kde"
    return "kde"


def _hist2d_axis_limits(x: np.ndarray, y: np.ndarray) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Generate DataFrames for axis limits. DataFrames have the columns 'Min' and 'Max', and values
    are the min and max of the provided arrays
    Parameters
    ----------
    x: Numpy.Array
    y: Numpy.Array
    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame]
        X-axis limits, Y-axis limits
    """
    xlim = [np.min(x), np.max(x)]
    ylim = [np.min(y), np.max(y)]
    xlim = pd.DataFrame({"Min": [xlim[0]], "Max": [xlim[1]]})
    ylim = pd.DataFrame({"Min": [ylim[0]], "Max": [ylim[1]]})
    return xlim, ylim


def _hist2d_bins(
    x: np.ndarray,
    y: np.ndarray,
    bins: Optional[int],
    transform_x: Optional[str],
    transform_x_kwargs: Optional[Dict],
    transform_y: Optional[str],
    transform_y_kwargs: Optional[Dict],
    xlim: Optional[Tuple[float, float]] = None,
    ylim: Optional[Tuple[float, float]] = None,
) -> List[np.ndarray]:
    """
    Calculate bins and edges for 2D histogram
    Parameters
    ----------
    x: Numpy.Array
    y: Numpy.Array
    bins: int, optional
    transform_x: str, optional
    transform_x_kwargs: Dict, optional
    transform_y: str, optional
    transform_y_kwargs: Dict, optional
    Returns
    -------
    List[Numpy.Array]
    """
    nbins = bins or int(np.sqrt(x.shape[0]))
    if xlim is not None:
        if ylim is not None:
            xlim, ylim = (
                pd.DataFrame({"Min": [xlim[0]], "Max": [xlim[1]]}),
                pd.DataFrame({"Min": [ylim[0]], "Max": [ylim[1]]}),
            )
        else:
            xlim, ylim = (pd.DataFrame({"Min": [xlim[0]], "Max": [xlim[1]]}), _hist2d_axis_limits(x, y)[1])
    elif ylim is not None:
        xlim, ylim = (_hist2d_axis_limits(x, y)[0], pd.DataFrame({"Min": [ylim[0]], "Max": [ylim[1]]}))
    else:
        xlim, ylim = _hist2d_axis_limits(x, y)
    bins = []
    for lim, transform_method, transform_kwargs in zip(
        [xlim, ylim], [transform_x, transform_y], [transform_x_kwargs, transform_y_kwargs]
    ):
        if transform_method:
            transform_kwargs = transform_kwargs or {}
            lim, transformer = apply_transform(
                data=lim, features=["Min", "Max"], method=transform_method, return_transformer=True, **transform_kwargs
            )
            grid = pd.DataFrame({"x": np.linspace(lim["Min"].values[0], lim["Max"].values[0], nbins)})
            bins.append(transformer.inverse_scale(grid, features=["x"]).x.to_numpy())
        else:
            bins.append(np.linspace(lim["Min"].values[0], lim["Max"].values[0], nbins))
    return bins


def kde1d(
    data: pd.DataFrame,
    x: str,
    transform_method: Optional[str] = None,
    bw: Union[str, float] = "silverman",
    **transform_kwargs,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute one-dimensional kernel density estimation
    Parameters
    ----------
    data: Pandas.DataFrame
    x: str
    bw: Union[str, float], (default='silverman')
    transform_method: str, optional
    transform_kwargs: optional
        Additional keyword arguments passed to transform method
    Returns
    -------
    Tuple[Numpy.Array, Numpy.Array]
        The grid space and the density array
    """
    transformer = None
    if transform_method:
        data, transformer = apply_transform(
            data=data, features=[x], method=transform_method, return_transformer=True, **transform_kwargs
        )
    x_grid, y = FFTKDE(kernel="gaussian", bw=bw).fit(data[x].values).evaluate()
    data = pd.DataFrame({"x": x_grid, "y": y})
    if transform_method:
        data = transformer.inverse_scale(data=pd.DataFrame({"x": x_grid, "y": y}), features=["x"])
    return data["x"].values, data["y"].values


def hist2d(
    data: pd.DataFrame,
    x: str,
    y: str,
    transform_x: Optional[str],
    transform_y: Optional[str],
    transform_x_kwargs: Optional[Dict],
    transform_y_kwargs: Optional[Dict],
    ax: plt.Axes,
    bins: Optional[int],
    cmap: str,
    xlim: Optional[Tuple[float, float]] = None,
    ylim: Optional[Tuple[float, float]] = None,
    **kwargs,
) -> None:
    """
    Plot two-dimensional histogram on axes
    Parameters
    ----------
    data: Pandas.DataFrame
    x: str
        Column to plot on x-axis
    y: str
        Column to plot on y-axis
    transform_x: str, optional
        Transform of the x-axis data, can be one of: 'log', 'hyperlog', 'asinh' or 'logicle'
    transform_y: str, optional
        Transform of the y-axis data, can be one of: 'log', 'hyperlog', 'asinh' or 'logicle'
    transform_x_kwargs: Dict, optional
        Additional keyword arguments passed to transform method
    transform_y_kwargs: Dict, optional
        Additional keyword arguments passed to transform method
    ax: Matplotlib.Axes
        Axes to plot on
    bins: int, optional
        Number of bins to use, if not given defaults to square root of the number of observations
    cmap: str (default='jet')
    xlim: Tuple[float, float], optional
        Limit the x-axis between this range
    ylim: Tuple[float, float], optional
        Limit the y-axis between this range
    kwargs: optional
        Additional keyword arguments passed to Matplotlib.hist2d
    Returns
    -------
    None
    """
    xbins, ybins = _hist2d_bins(
        x=data[x].values,
        y=data[y].values,
        bins=bins,
        transform_x=transform_x,
        transform_y=transform_y,
        transform_x_kwargs=transform_x_kwargs,
        transform_y_kwargs=transform_y_kwargs,
        xlim=xlim,
        ylim=ylim,
    )
    ax.hist2d(data[x].values, data[y].values, bins=[xbins, ybins], norm=LogNorm(), cmap=cmap, **kwargs)


def cyto_plot(
    data: pd.DataFrame,
    x: str,
    y: Optional[str] = None,
    kind: str = "auto",
    transform_x: Optional[str] = "asinh",
    transform_y: Optional[str] = "asinh",
    transform_x_kwargs: Optional[Dict] = None,
    transform_y_kwargs: Optional[Dict] = None,
    xlim: Optional[Tuple[float, float]] = None,
    ylim: Optional[Tuple[float, float]] = None,
    ax: Optional[plt.Axes] = None,
    figsize: Tuple[float, float] = (5.0, 5.0),
    bins: Optional[int] = None,
    cmap: str = "jet",
    autoscale: bool = True,
    **kwargs,
) -> plt.Axes:
    """
    Generate a generic 'flow plot', that is a one or two-dimensional plot identical to that generated by common
    cytometry software like FlowJo. These plots support common cytometry data transformations like logicle
    (biexponential), log, hyperlog, or hyperbolic arc-sine transformations, whilst translating values back
    to a linear scale on axis for improved interpretability.
    Parameters
    ----------
    data: Pandas.DataFrame
    x: str
        Column to plot on x-axis
    y: str, optional
        Column to plot on y-axis, will generate a one-dimensional KDE plot if not provided
    kind: str, (default='auto)
        Should be one of: 'hist2d', 'scatter', 'scatter_kde', 'kde' or 'auto'. If 'auto' then plot type is
        determined from data; If the number of observations is less than 1000, will use 'hist2d' otherwise
        kind is 'scatter_kde'. If data is one-dimensional (i.e. y is not provided), then will use 'kde'.
    transform_x: str, optional, (default='asinh')
        Transform of the x-axis data, can be one of: 'log', 'hyperlog', 'asinh' or 'logicle'
    transform_y: str, optional, (default='asinh')
        Transform of the y-axis data, can be one of: 'log', 'hyperlog', 'asinh' or 'logicle'
    transform_x_kwargs: Dict, optional
        Additional keyword arguments passed to transform method
    transform_y_kwargs: Dict, optional
        Additional keyword arguments passed to transform method
    xlim: Tuple[float, float], optional
        Limit the x-axis between this range
    ylim: Tuple[float, float], optional
        Limit the y-axis between this range
    ax: Matplotlib.Axes
        Axes to plot on
    bins: int, optional
        Number of bins to use, if not given defaults to square root of the number of observations
    cmap: str (default='jet')
        Colour palette to use for two-dimensional histogram
    figsize: Tuple[int, int], (default=(5, 5))
        Ignored if 'ax' provided, otherwise new figure generated with this figure size.
    autoscale: bool (default=True)
        Allow matplotlib to autoscale the axis view to the data
    kwargs:
        Additional keyword arguments passed to plotting method:
            * seaborn.scatterplot for 'scatter' or 'scatter_kde'
            * matplotlib.Axes.hist2d for 'hist2d'
            * matplotlib.Axes.plot for 'kde'
    Returns
    -------
    Matplotlib.Axes
    """
    try:
        ax = ax or plt.subplots(figsize=figsize)[1]
        ax.xaxis.labelpad = 20
        ax.yaxis.labelpad = 20
        kind = kind if kind != "auto" else _auto_plot_kind(data=data, y=y)
        kwargs = kwargs or {}
        if kind == "hist2d":
            assert y, "No y-axis variable provided"
            hist2d(
                data=data,
                x=x,
                y=y,
                transform_x=transform_x,
                transform_y=transform_y,
                transform_x_kwargs=transform_x_kwargs,
                transform_y_kwargs=transform_y_kwargs,
                ax=ax,
                bins=bins,
                cmap=cmap,
                xlim=xlim,
                ylim=ylim,
            )
        elif kind == "scatter":
            assert y, "No y-axis variable provided"
            kwargs["s"] = kwargs.get("s", 10)
            kwargs["edgecolor"] = kwargs.get("edgecolor", None)
            kwargs["linewidth"] = kwargs.get("linewidth", 0)
            sns.scatterplot(data=data, x=x, y=y, **kwargs)
        elif kind == "scatter_kde":
            assert y, "No y-axis variable provided"
            scatter_kwargs = kwargs.get("scatter_kwargs", {})
            kde_kwargs = kwargs.get("kde_kwargs", {})
            sns.kdeplot(data=data, x=x, y=y, **kde_kwargs)
            scatter_kwargs["s"] = scatter_kwargs.get("s", 10)
            scatter_kwargs["edgecolor"] = scatter_kwargs.get("edgecolor", None)
            scatter_kwargs["linewidth"] = scatter_kwargs.get("linewidth", 0)
            scatter_kwargs["color"] = scatter_kwargs.get("color", "black")
            sns.scatterplot(data=data, x=x, y=y, **scatter_kwargs)
        elif kind == "kde":
            if y:
                sns.kdeplot(data=data, x=x, y=y, **kwargs)
            else:
                bw = kwargs.pop("bw", "silverman")
                transform_x_kwargs = transform_x_kwargs or {}
                xx, pdf = kde1d(data=data, x=x, transform_method=transform_x, bw=bw, **transform_x_kwargs)
                ax.plot(xx, pdf, linewidth=kwargs.get("linewidth", 2), color=kwargs.get("color", "black"))
                ax.fill_between(xx, pdf, color=kwargs.get("fill", "#8A8A8A"), alpha=kwargs.get("alpha", 0.5))
        else:
            raise KeyError("Invalid value for 'kind', must be one of: 'auto', 'hist2d', 'scatter_kde', or 'kde'.")
        ax.set_xlabel(x)
        ax.set_ylabel(y)
        if autoscale:
            ax.autoscale(enable=True)
        else:
            if xlim is None:
                xlim = (data[x].quantile(q=0.001), data[x].quantile(q=0.999))
            if ylim is None:
                ylim = (data[y].quantile(q=0.001), data[y].quantile(q=0.999))
            ax.set_xlim(*xlim)
            if y is not None:
                ax.set_ylim(*ylim)
        if transform_x:
            transform_x_kwargs = transform_x_kwargs or {}
            ax.set_xscale(transform_x, **transform_x_kwargs)
        if y and transform_y:
            transform_y_kwargs = transform_y_kwargs or {}
            ax.set_yscale(transform_y, **transform_y_kwargs)
        plt.xticks(rotation=90)
        plt.tight_layout()
        return ax
    except AssertionError as e:
        raise PlotError(e)
    except KeyError as e:
        raise PlotError(e)


def overlay(
    x: str,
    y: str,
    background_data: pd.DataFrame,
    overlay_data: Dict[str, pd.DataFrame],
    background_colour: str = "#323232",
    transform_x: Optional[str] = "asinh",
    transform_y: Optional[str] = "asinh",
    legend_kwargs: Optional[Dict] = None,
    **plot_kwargs,
) -> plt.Axes:
    """
    Generates a two-dimensional scatterplot as background data and overlays a histogram, KDE, or scatterplot
    in the foreground. Can be useful for comparing populations and is commonly referred to as 'back-gating' in
    traditional cytometry analysis.
    Parameters
    ----------
    x: str
        Column to plot on x-axis, must be common to both 'background_data' and 'overlay_data'
    y: str
        Column to plot on y-axis, must be common to both 'background_data' and 'overlay_data'
    background_data: Pandas.DataFrame
        Data to plot in the background
    overlay_data: Pandas.DataFrame
        Data to plot in the foreground
    background_colour: str (default='#323232')
        How to colour the background data points (defaults to a grey-ish black)
    transform_x: str, optional, (default='asinh')
        Transform of the x-axis data, can be one of: 'log', 'hyperlog', 'asinh' or 'logicle'
    transform_y: str, optional, (default='asinh')
        Transform of the y-axis data, can be one of: 'log', 'hyperlog', 'asinh' or 'logicle'
    legend_kwargs: optional
        Additional keyword arguments passed to legend
    plot_kwargs: optional
        Additional keyword arguments passed to cytopy.plotting.cyto.cyto_plot for foreground plot
    Returns
    -------
    Matplotlib.Axes
    """
    colours = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#bcbd22", "#17becf"]
    if len(overlay_data) > len(colours):
        raise ValueError(f"Maximum of {len(colours)} overlaid populations.")
    ax = cyto_plot(
        data=background_data,
        x=x,
        y=y,
        transform_x=transform_x,
        transform_y=transform_y,
        color=background_colour,
        kind="scatter",
        **plot_kwargs,
    )
    legend_kwargs = legend_kwargs or {}
    for label, df in overlay_data.items():
        cyto_plot(data=df, x=x, y=y, transform_x=transform_x, transform_y=transform_y, ax=ax, **plot_kwargs)
    _default_legend(ax=ax, **legend_kwargs)
    return ax


def _threshold_annotation(x: float, y: float, text: str, ax: plt.Axes):
    """
    Annotate an Axes with the text label of a threshold gate
    Parameters
    ----------
    x: float
    y: float
    text: str
    ax: Matplotlib.Axes
    Returns
    -------
    None
    """
    ax.text(
        x,
        y,
        text,
        ha="center",
        va="center",
        transform=ax.transAxes,
        backgroundcolor="white",
        bbox=dict(facecolor="white", edgecolor="black", pad=5.0),
    )


def _1dthreshold_annot(labels: Dict, ax: plt.Axes):
    """
    Annotate an axis with the labels for a one-dimensional threshold gate
    Parameters
    ----------
    labels: Dict
        The population names of a one-dimensional threshold; expects the keys '-' and '+'
    ax: Matplotlib.Axes
    Returns
    -------
    None
    """
    try:
        legend_labels = {"A": labels["-"], "B": labels["+"]}
    except KeyError:
        raise KeyError(f"Definitions for 1D threshold gate must be either '-' or '+', not: {labels.keys()}")
    _threshold_annotation(0.05, 0.92, "A", ax=ax)
    _threshold_annotation(0.95, 0.92, "B", ax=ax)
    ax.text(1.15, 0.95, f"A: {legend_labels.get('A')}", transform=ax.transAxes)
    ax.text(1.15, 0.85, f"B: {legend_labels.get('B')}", transform=ax.transAxes)


def _2dthreshold_annot(labels: Dict, ax: plt.Axes):
    """
    Annotate an axis with the labels for a two-dimensional threshold gate
    Parameters
    ----------
    labels: Dict
        The population names of a one-dimensional threshold; expects the keys '--', '++', '+-' and '-+'
    ax: Matplotlib.Axes
    Returns
    -------
    None
    """
    labels = labels or {"-+": "-+", "++": "++", "--": "--", "+-": "+-"}
    legend_labels = {}
    for definition, label in labels.items():
        for d in definition.split(","):
            if "-+" == d:
                legend_labels["A"] = label
            elif "++" == d:
                legend_labels["B"] = label
            elif "--" == d:
                legend_labels["C"] = label
            elif "+-" == d:
                legend_labels["D"] = label
            else:
                raise KeyError(f"Definition {d} is invalid for a 2D threshold gate.")
    _threshold_annotation(0.05, 0.92, "A", ax=ax)
    _threshold_annotation(0.95, 0.92, "B", ax=ax)
    _threshold_annotation(0.05, 0.08, "C", ax=ax)
    _threshold_annotation(0.95, 0.08, "D", ax=ax)
    ax.text(1.15, 0.95, f"A: {legend_labels.get('A')}", transform=ax.transAxes)
    ax.text(1.15, 0.85, f"B: {legend_labels.get('B')}", transform=ax.transAxes)
    ax.text(1.15, 0.75, f"C: {legend_labels.get('C')}", transform=ax.transAxes)
    ax.text(1.15, 0.65, f"D: {legend_labels.get('D')}", transform=ax.transAxes)


def _plot_thresholds(geom_objs: Union[List[Population], List[Child]], ax: plt.Axes):
    """
    Plot threshold geoms (lines that define the boundary of a threshold gate)
    Parameters
    ----------
    geom_objs: List[Population] or List[Child]
        One or more Population objects or Child objects; where Child is from a Gate
    ax: Matplotlib.Axes
    Returns
    -------
    None
    """
    x, y = geom_objs[0].geom.transform_to_linear()
    labels = {}
    for g in geom_objs:
        if isinstance(g, Population):
            labels[g.definition] = g.population_name
        else:
            labels[g.definition] = g.name
    ax.axvline(x, lw=2.5, c="#c92c2c")
    if y:
        ax.axhline(y, lw=2.5, c="#c92c2c")
        _2dthreshold_annot(labels=labels, ax=ax)
    else:
        _1dthreshold_annot(labels=labels, ax=ax)


def _default_legend(ax: plt.Axes, **legend_kwargs):
    """
    Default setting for plot legend
    Parameters
    ----------
    ax: Matplotlib.Axes
    legend_kwargs: optional
        User defined legend keyword arguments
    Returns
    -------
    None
    """
    legend_kwargs = legend_kwargs or {}
    anchor = legend_kwargs.get("bbox_to_anchor", (1.1, 0.95))
    loc = legend_kwargs.get("loc", 2)
    ncol = legend_kwargs.get("ncol", 3)
    fancy = legend_kwargs.get("fancybox", True)
    shadow = legend_kwargs.get("shadow", False)
    ax.legend(loc=loc, bbox_to_anchor=anchor, ncol=ncol, fancybox=fancy, shadow=shadow)


def _plot_polygons(geom_objs: Union[List[Population], List[Child]], ax: plt.Axes, **legend_kwargs):
    """
    Plot polygon geoms
    Parameters
    ----------
    geom_objs: List[Population] or List[Child]
        One or more Population objects or Child objects; where Child is from a Gate
    ax: Matplotlib.Axes
    legend_kwargs: optional
        User defined legend keyword arguments
    Returns
    -------
    None
    """
    colours = cycle(["#c92c2c", "#2df74e", "#e0d572", "#000000", "#64b9c4", "#9e3657"])
    for g in geom_objs:
        x_values, y_values = g.geom.transform_to_linear()
        if isinstance(g, Population):
            name = g.population_name
        else:
            name = g.name
        ax.plot(x_values, y_values, c=next(colours), lw=2.5, label=name)
    _default_legend(ax=ax, **legend_kwargs)


def _inverse_gate_transform(gate: Gate, data: pd.DataFrame) -> pd.DataFrame:
    """
    Perform an inverse transformation of the given data using the transform definition of a gate
    Parameters
    ----------
    gate: Gate
    data: Pandas.DataFrame
    Returns
    -------
    Pandas.DataFrame
    """
    if gate.transform_x:
        x_transformer = TRANSFORMERS.get(gate.transform_x)(**gate.transform_x_kwargs or {})
        data = x_transformer.inverse_scale(data=data, features=[gate.x])
    if gate.transform_y:
        y_transformer = TRANSFORMERS.get(gate.transform_y)(**gate.transform_y_kwargs or {})
        data = y_transformer.inverse_scale(data=data, features=[gate.y])
    return data


def _plot_ctrl_gate_1d(
    primary_data: np.ndarray,
    ctrl_data: np.ndarray,
    ax: Optional[plt.Axes] = None,
    figsize: Tuple[float, float] = (5.0, 5.0),
    **kwargs,
) -> plt.Axes:
    """
    Generate a KDE plot over a grid space that spans the primary and control data
    Parameters
    ----------
    primary_data: Numpy.Array
    ctrl_data: Numpy.Array
    ax: Matplotlib.Axes
        Axes to plot on
    figsize: Tuple[int, int], (default=(5, 5))
        Ignored if 'ax' provided, otherwise new figure generated with this figure size.
    kwargs: optional
        Additional keyword arguments passed to Matplotlib.Axes.plot
    Returns
    -------
    Matplotlib.Axes
    """
    kwargs = kwargs or {}
    ax = ax if ax is not None else plt.subplots(figsize=figsize)[1]
    x = np.linspace(
        np.min([np.min(primary_data), np.min(ctrl_data)]) - 0.01,
        np.max([np.max(primary_data), np.max(ctrl_data)]) + 0.01,
        1000,
    )
    y = FFTKDE(kernel=kwargs.get("kernel", "gaussian"), bw=kwargs.get("bw", "ISJ")).fit(ctrl_data).evaluate(x)
    lw = kwargs.get("linewidth", 2)
    color = kwargs.get("ctrl_linecolor", "black")
    fill = kwargs.get("ctrl_fill", "#1F77B4")
    alpha = kwargs.get("alpha", 0.5)
    ax.plot(x, y, linewidth=lw, color=color)
    ax.fill_between(x, y, color=fill, alpha=alpha)

    y = FFTKDE(kernel=kwargs.get("kernel", "gaussian"), bw=kwargs.get("bw", "ISJ")).fit(primary_data).evaluate(x)
    lw = kwargs.get("linewidth", 2)
    color = kwargs.get("linecolor", "black")
    fill = kwargs.get("fill", "#8A8A8A")
    alpha = kwargs.get("alpha", 0.5)
    ax.plot(x, y, linewidth=lw, color=color)
    ax.fill_between(x, y, color=fill, alpha=alpha)
    return ax


def plot_ctrl_gate_1d(
    gate: ThresholdBase,
    filegroup: FileGroup,
    ax: Optional[plt.Axes] = None,
    figsize: Tuple[float, float] = (5.0, 5.0),
    autoscale: bool = True,
    xlim: Optional[List] = None,
    **kwargs,
) -> plt.Axes:
    """
    Plot a one-dimensional control gate
    Parameters
    ----------
    gate: ThresholdGThresholdBaseate
        The Gate to plot, should be a threshold gate with a control defined
    filegroup: FileGroup
        The FileGroup to plot the gate on
    ax: Matplotlib.Axes
        Axes to plot on
    figsize: Tuple[int, int], (default=(5, 5))
        Ignored if 'ax' provided, otherwise new figure generated with this figure size.
    xlim: Tuple[float, float], optional
        Limit the x-axis between this range
    autoscale: bool (default=True)
        Allow matplotlib to autoscale the axis view to the data
    kwargs: optional
        Additional keyword arguments passed to Matplotlib.Axes.plot
    Returns
    -------
    Matplotlib.Axes
    """
    try:
        assert isinstance(
            gate, ThresholdBase
        ), "plot_ctrl_gate expects a threshold-like gate with 'ctrl' attribute defined."
        assert gate.ctrl, "plot_ctrl_gate expects a threshold-like gate with 'ctrl' attribute defined."
        primary_data = filegroup.load_population_df(population=gate.parent, transform=None, data_source="primary")
        ctrl_data = filegroup.load_population_df(population=gate.parent, transform=None, data_source=gate.ctrl)
        ax = _plot_ctrl_gate_1d(
            primary_data=primary_data[gate.x].values,
            ctrl_data=ctrl_data[gate.x].values,
            ax=ax,
            figsize=figsize,
            **kwargs,
        )
        ax.set_xlabel(gate.x)
        try:
            geom_objs = [filegroup.populations.get(population_name=c.name) for c in gate.children]
        except DoesNotExist:
            logger.info(
                "Filegroup does not contain populations generated by the given gate. Will use Gate children geoms."
            )
            geom_objs = gate.children
        _plot_thresholds(geom_objs=geom_objs, ax=ax)
        if autoscale:
            ax.autoscale(enable=True)
        else:
            if xlim is None:
                xlim = (primary_data[gate.x].quantile(q=0.001), primary_data[gate.x].quantile(q=0.999))
                ax.set_xlim(*xlim)
        if gate.transform_x:
            transform_x_kwargs = gate.transform_x_kwargs or {}
            ax.set_xscale(gate.transform_x, **transform_x_kwargs)
        plt.xticks(rotation=90)
        plt.tight_layout()
        return ax
    except AssertionError as e:
        raise PlotError(e)


def plot_gate(
    gate: Gate,
    filegroup: Optional[FileGroup] = None,
    data_source: str = "primary",
    legend_kwargs: Optional[Dict] = None,
    ax: Optional[plt.Axes] = None,
    n_limit: Optional[int] = None,
    figsize: Tuple[float, float] = (5.0, 5.0),
    **kwargs,
) -> plt.Axes:
    """
    Plot a Gate geometry on the data of a FileGroup. If the gate geometry has been defined for the FileGroup, then
    population gate geometry will be plotted, otherwise the Gate definition geometry will be plotted on the
    FileGroup data.
    Parameters
    ----------
    gate: Gate
    filegroup: FileGroup
    data_source: str (default='primary')
    legend_kwargs: optional
        User defined legend keyword arguments
    ax: Matplotlib.Axes
        Axes to plot on
    figsize: Tuple[int, int], (default=(5, 5))
        Ignored if 'ax' provided, otherwise new figure generated with this figure size.
    n_limit: int, optional
        If provided, data will be down-sampled prior to plotting to not exceed this size
    kwargs: optinal
        Additional keyword arguments passed to cytopy.plotting.cyto.cyto_plot
    Returns
    -------
    Matplotlib.Axes
    """
    n_limit = n_limit or np.inf
    try:
        ax = ax if ax is not None else plt.subplots(figsize=figsize)[1]
        kwargs = kwargs or {}
        y = kwargs.pop("y", gate.y)
        transform_y = kwargs.pop("transform_y", gate.transform_y)
        transform_y_kwargs = kwargs.pop("transform_y_kwargs", gate.transform_y_kwargs)
        if filegroup is not None:
            data = filegroup.load_population_df(population=gate.parent, transform=None, data_source=data_source)
            if data.shape[0] > n_limit:
                data = data.sample(n=n_limit)
            geom_objs = [
                filegroup.populations.get(population_name=c.name, data_source=data_source) for c in gate.children
            ]
            if gate.reference_alignment:
                data = _inverse_gate_transform(gate, gate.preprocess(data=data, transform=True))
        else:
            data = gate.reference
            if data.shape[0] > n_limit:
                data = data.sample(n=n_limit)
            data = _inverse_gate_transform(gate, data)
            geom_objs = gate.children
        ax = cyto_plot(
            data=data,
            x=gate.x,
            y=y,
            transform_x=gate.transform_x,
            transform_y=transform_y,
            transform_x_kwargs=gate.transform_x_kwargs,
            transform_y_kwargs=transform_y_kwargs,
            ax=ax,
            **kwargs,
        )
        if isinstance(gate, ThresholdBase):
            _plot_thresholds(geom_objs=geom_objs, ax=ax)
        else:
            legend_kwargs = legend_kwargs or {}
            _plot_polygons(geom_objs=geom_objs, ax=ax, **legend_kwargs)
        return ax
    except DoesNotExist as e:
        logger.error("One or more populations generated from this Gate are not present in the given FileGroup.")
        raise PlotError(e)
