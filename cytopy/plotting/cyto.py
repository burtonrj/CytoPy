import logging
from itertools import cycle
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union

import numpy as np
import seaborn as sns
from fast_histogram import histogram2d
from KDEpy import FFTKDE
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm
from mongoengine import DoesNotExist

from cytopy.data import FileGroup
from cytopy.data import Population
from cytopy.gating.base import Child
from cytopy.gating.base import Gate
from cytopy.gating.threshold import ThresholdBase
from cytopy.plotting.asinh_transform import *
from cytopy.plotting.hlog_transform import *
from cytopy.plotting.logicle_transform import *
from cytopy.utils.transform import apply_transform
from cytopy.utils.transform import TRANSFORMERS

logger = logging.getLogger(__name__)


class PlotError(Exception):
    def __init__(self, message: str):
        logger.error(message)
        super().__init__(message)


def _auto_plot_kind(
    data: pd.DataFrame,
    y: Optional[str] = None,
):
    if y:
        if data.shape[0] > 1000:
            return "hist2d"
        return "scatter_kde"
    return "kde"


def _hist2d_axis_limits(x: np.ndarray, y: np.ndarray) -> Tuple[pd.DataFrame, pd.DataFrame]:
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
):
    nbins = bins or int(np.sqrt(x.shape[0]))
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
):
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
    **kwargs,
):
    xbins, ybins = _hist2d_bins(
        x=data[x].values,
        y=data[y].values,
        bins=bins,
        transform_x=transform_x,
        transform_y=transform_y,
        transform_x_kwargs=transform_x_kwargs,
        transform_y_kwargs=transform_y_kwargs,
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
):
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
            ax.set_xlim((xlim[0] or data[x].quantile(q=0.001), xlim[1] or data[x].quantile(q=0.999)))
            if y is not None:
                ax.set_ylim((ylim[0] or data[y].quantile(q=0.001), ylim[1] or data[y].quantile(q=0.999)))
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
    transform_x: str = "asinh",
    transform_y: str = "asinh",
    legend_kwargs: Optional[Dict] = None,
    **plot_kwargs,
):
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
    try:
        legend_labels = {"A": labels["-"], "B": labels["+"]}
    except KeyError:
        raise KeyError(f"Definitions for 1D threshold gate must be either '-' or '+', not: {labels.keys()}")
    _threshold_annotation(0.05, 0.92, "A", ax=ax)
    _threshold_annotation(0.95, 0.92, "B", ax=ax)
    ax.text(1.15, 0.95, f"A: {legend_labels.get('A')}", transform=ax.transAxes)
    ax.text(1.15, 0.85, f"B: {legend_labels.get('B')}", transform=ax.transAxes)


def _2dthreshold_annot(labels: Dict, ax: plt.Axes):
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
    legend_kwargs = legend_kwargs or {}
    anchor = legend_kwargs.get("bbox_to_anchor", (1.1, 0.95))
    loc = legend_kwargs.get("loc", 2)
    ncol = legend_kwargs.get("ncol", 3)
    fancy = legend_kwargs.get("fancybox", True)
    shadow = legend_kwargs.get("shadow", False)
    ax.legend(loc=loc, bbox_to_anchor=anchor, ncol=ncol, fancybox=fancy, shadow=shadow)


def _plot_polygons(geom_objs: Union[List[Population], List[Child]], ax: plt.Axes, **legend_kwargs):
    colours = cycle(["#c92c2c", "#2df74e", "#e0d572", "#000000", "#64b9c4", "#9e3657"])
    for g in geom_objs:
        x_values, y_values = g.geom.transform_to_linear()
        if isinstance(g, Population):
            name = g.population_name
        else:
            name = g.name
        ax.plot(x_values, y_values, "-k", c=next(colours), lw=2.5, label=name)
    _default_legend(ax=ax, **legend_kwargs)


def _inverse_gate_transform(gate: Gate, data: pd.DataFrame):
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
):
    kwargs = kwargs or {}
    ax = ax if ax is not None else plt.subplots(figsize=figsize)[1]
    x = np.linspace(
        np.min([np.min(primary_data), np.min(ctrl_data)]) - 0.01,
        np.max([np.max(primary_data), np.max(ctrl_data)]) + 0.01,
        1000,
    )
    y = FFTKDE(kernel=kwargs.get("kernel", "gaussian"), bw=kwargs.get("bw", "ISJ")).fit(primary_data).evaluate(x)
    lw = kwargs.get("linewidth", 2)
    color = kwargs.get("linecolor", "black")
    fill = kwargs.get("fill", "#8A8A8A")
    alpha = kwargs.get("alpha", 0.5)
    ax.plot(x, y, linewidth=lw, color=color)
    ax.fill_between(x, y, color=fill, alpha=alpha)
    return ax


def plot_ctrl_gate_1d(
    gate: Gate,
    filegroup: FileGroup,
    ax: Optional[plt.Axes] = None,
    figsize: Tuple[float, float] = (5.0, 5.0),
    autoscale: bool = True,
    xlim: Optional[List] = None,
    **kwargs,
):
    try:
        assert isinstance(gate, ThresholdBase)
        assert gate.ctrl
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
                "Filegroup does not contain populations generated by the given gate. Will use " "Gate children geoms."
            )
            geom_objs = gate.children
        _plot_thresholds(geom_objs=geom_objs, ax=ax)
        if autoscale:
            ax.autoscale(enable=True)
        else:
            ax.set_xlim(
                (xlim[0] or primary_data[gate.x].quantile(q=0.001), xlim[1] or primary_data[gate.x].quantile(q=0.999))
            )
        if gate.transform_x:
            transform_x_kwargs = gate.transform_x_kwargs or {}
            ax.set_xscale(gate.transform_x, **transform_x_kwargs)
        plt.xticks(rotation=90)
        plt.tight_layout()
        return ax
    except AssertionError:
        raise PlotError("plot_ctrl_gate expects a threshold-like gate with 'ctrl' attribute defined.")


def plot_gate(
    gate: Gate,
    filegroup: Optional[FileGroup] = None,
    data_source: str = "primary",
    legend_kwargs: Optional[Dict] = None,
    ax: Optional[plt.Axes] = None,
    n_limit: Optional[int] = None,
    figsize: Tuple[float, float] = (5.0, 5.0),
    **kwargs,
):
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
    except DoesNotExist:
        raise PlotError("One or more populations generated from this Gate are not present in the given FileGroup.")
