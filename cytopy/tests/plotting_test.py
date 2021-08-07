import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.colors import LogNorm

from ..flow import transform
from ..flow.plotting import FlowPlot
from .conftest import create_linear_data
from .conftest import create_lognormal_data

sns.set(style="white", font_scale=1.2)
sns.set_style("ticks", {"xtick.major.size": 8, "ytick.major.size": 8})


def test_create_linear_data():
    data = create_linear_data()
    fig, ax = plt.subplots(figsize=(5, 5))
    bins = [
        np.histogram_bin_edges(data.x.values, bins="sqrt"),
        np.histogram_bin_edges(data.y.values, bins="sqrt"),
    ]
    ax.hist2d(data.x.values, data.y.values, bins=bins, norm=LogNorm(), cmap="jet")
    ax.set_title("Linear data example")
    plt.show()


def test_create_lognormal_data():
    data = create_lognormal_data()
    n = int(np.sqrt(data.shape[0]))
    transformer = transform.LogTransformer()
    fig, ax = plt.subplots(figsize=(5, 5))
    xlim = transform.safe_range(data, "x")
    ylim = transform.safe_range(data, "y")
    xlim = pd.DataFrame({"Min": [xlim[0]], "Max": [xlim[1]]})
    ylim = pd.DataFrame({"Min": [ylim[0]], "Max": [ylim[1]]})
    xlim, ylim = transformer.scale(xlim, ["Min", "Max"]), transformer.scale(ylim, ["Min", "Max"])
    xgrid = pd.DataFrame({"x": np.linspace(xlim["Min"].iloc[0], xlim["Max"].iloc[0], n)})
    ygrid = pd.DataFrame({"y": np.linspace(ylim["Min"].iloc[0], ylim["Max"].iloc[0], n)})
    xbins = transformer.inverse_scale(xgrid, features=["x"]).x.values
    ybins = transformer.inverse_scale(ygrid, features=["y"]).y.values
    ax.hist2d(data.x.values, data.y.values, bins=[xbins, ybins], norm=LogNorm(), cmap="jet")
    ax.set_xscale("log", base=10)
    ax.set_yscale("log", base=10)
    ax.set_title("Log-normal data example")
    plt.show()


def test_valid_transform():
    valid = ["log", "logicle", "hyperlog", "asinh", None]
    invalid = ["LOG", "hyper-log", "Sine"]
    for x in valid:
        plotter = FlowPlot(transform_x=x, transform_y=x)
        assert plotter.transform_x == x
        assert plotter.transform_y == x
    for x in invalid:
        with pytest.raises(AssertionError) as e:
            FlowPlot(transform_x=x, transform_y=x)
        assert str(e.value) == f"Unsupported transform, must be one of: {valid}"


def test_create_flowplot_object():
    plotter = FlowPlot()
    assert plotter.transform_x == "logicle"
    assert plotter.transform_y == "logicle"
    assert plotter.transform_x_kwargs == {}
    assert plotter.transform_y_kwargs == {}
    assert plotter.labels.get("x", None) is None
    assert plotter.labels.get("y", None) is None
    assert plotter.autoscale
    assert plotter.lims.get("x") == [None, None]
    assert plotter.lims.get("y") == [None, None]
    assert plotter.title is None
    assert plotter.bw == "silverman"
    assert plotter.bins is None
    assert isinstance(plotter._ax, plt.Axes)
    assert isinstance(plotter.cmap, LinearSegmentedColormap)
    assert plotter._ax.xaxis.labelpad == 20
    assert plotter._ax.yaxis.labelpad == 20


TEST_CONDITIONS_1D = [
    (None, {}),
    ("logicle", {}),
    ("logicle", {"m": 5.0, "w": 0.25, "t": 270000}),
    ("asinh", {}),
    ("hyperlog", {}),
    ("log", {}),
    ("log", {"base": 10}),
    ("log", {"base": 2}),
]


@pytest.mark.parametrize("t,kwargs", TEST_CONDITIONS_1D)
def test_plot_1dhistogram(t, kwargs):
    data = create_lognormal_data()
    plotter = FlowPlot(transform_x=t, transform_x_kwargs=kwargs, title=f"{t}; {kwargs}")
    plotter.plot(data=data, x="x")
    plt.tight_layout()
    plt.show()


TEST_CONDITIONS_2D = [
    (None, None, {}, {}),
    ("logicle", None, {}, {}),
    ("logicle", "logicle", {}, {}),
    ("logicle", "logicle", {"m": 5.0, "w": 0.25, "t": 270000}, {}),
    (
        "logicle",
        "logicle",
        {"m": 5.0, "w": 0.25, "t": 270000},
        {"m": 3.0, "w": 0.1, "t": 270000},
    ),
    ("asinh", "logicle", {}, {}),
    ("asinh", "asinh", {}, {}),
    ("hyperlog", "hyperlog", {}, {}),
    ("log", None, {}, {}),
    ("log", "logicle", {"base": 10}, {}),
    ("log", "log", {"base": 2}, {}),
]


@pytest.mark.parametrize("tx,ty,xkwargs,ykwargs", TEST_CONDITIONS_2D)
def test_plot_2dhistogram(tx, ty, xkwargs, ykwargs):
    data = create_lognormal_data()
    plotter = FlowPlot(
        transform_x=tx,
        transform_y=ty,
        transform_x_kwargs=xkwargs,
        title=f"{tx}, {ty}; " f"{xkwargs}, {ykwargs}",
    )
    plotter.plot(data=data, x="x", y="y")
    plt.tight_layout()
    plt.show()


def test_plot_axis_limits():
    data = create_lognormal_data()
    plotter = FlowPlot(
        transform_x="logicle",
        transform_y="logicle",
        title="Axis limits",
        xlim=(1000, 10000),
    )
    plotter.plot(data=data, x="x", y="y")
    plt.tight_layout()
    plt.show()
