# CytoPy Imports
import numpy as np
import pandas as pd

from cytopy.data.setup import global_init, setup_logs
from cytopy.data.gating_strategy import GatingStrategy, Gate
from cytopy.data.experiment import Experiment
from cytopy.data.geometry import ThresholdGeom, PolygonGeom
# Bokeh imports
from bokeh.models import ColumnDataSource, Div, Select, Slider, TextInput
from bokeh.layouts import column, row
from bokeh.plotting import figure, curdoc
from bokeh.transform import log_cmap
from bokeh.util.hex import hexbin
# Other imports
from mongoengine.errors import DoesNotExist
from loguru import logger
from KDEpy import FFTKDE
import argparse
import inspect
import typing
import os

# A bokeh server that uses the PolyEditTool to edit Patches on a Hextile plot
setup_logs()

# Data structures
GATING_STRATEGY = None
EXPERIMENT = None
SOURCE = ColumnDataSource(data=dict(x=[], y=[]))
GEOMS = dict()

# Page layout
MAIN = column(row(), row(), name='mainLayout')

# Figure
PLOT = figure(title="Gate editor",
              tools="poly_edit,wheel_zoom,reset,save",
              plot_width=500,
              plot_height=500,
              name="FACSplot")
PLOT.toolbar.active_drag = None

# Controls
GATE_SELECT = Select(title="Gate")


def load_geometry_from_gate(gate: Gate):
    return {
        child.name: GATING_STRATEGY.filegroup.get_population(population_name=child.name).geom
        for child in gate.children
    }


def load_data_from_gate(gate: Gate):
    parent = GATING_STRATEGY.filegroup.load_population_df(population=gate.parent,
                                                          transform=gate.transform_info())
    PLOT.xaxis.axis_label = gate.x
    if gate.y:
        PLOT.yaxis.axis_label = gate.y
        return dict(x=parent[gate.x], y=parent[gate.y])
    x, y = kde_1d(parent[gate.x])
    return dict(x=x, y=y)


def kde_1d(data: np.ndarray, bw: typing.Union[float, str] = "silverman") -> (np.ndarray, np.ndarray):
    return (FFTKDE(kernel="gaussian", bw=bw)
            .fit(data)
            .evaluate())


def plot_2dhex():
    bins = hexbin(SOURCE.data["x"], SOURCE.data["y"], size=0.01)
    PLOT.hex_tile(q="q", r="r", size=0.1, line_color=None, source=bins,
                  fill_color=log_cmap("counts", "RdYlBu", 0, max(bins.counts)))


def plot_1dkde():
    PLOT.line(x="x", y="y", source=SOURCE, line_width=4, alpha=0.7)


def plot_polygon_geom(population_name: str,
                      geom: PolygonGeom):
    pass


def plot_threshold_geom(population_name: str,
                        geom: ThresholdGeom):
    pass


def update_plot():
    sub_layouts = curdoc().get_model_by_name('mainLayout').children
    sub_layouts.remove(curdoc().get_model_by_name("FACSplot"))
    new_plot = figure(title="Gate editor",
                      tools="poly_edit,wheel_zoom,reset,save",
                      plot_width=500,
                      plot_height=500,
                      background_fill_color="#fafafa",
                      name="FACSplot")
    sub_layouts.append(new_plot)
    PLOT.toolbar.active_drag = None
    gate = GATING_STRATEGY.get_gate(gate=GATE_SELECT.value)
    SOURCE.data = load_data_from_gate(gate=gate)
    GEOMS.clear()
    GEOMS.update(load_geometry_from_gate(gate=gate))

    for population_name, geom in GEOMS.items():
        if isinstance(geom, ThresholdGeom):
            plot_threshold_geom(population_name=population_name, geom=geom)
        else:
            plot_polygon_geom(population_name=population_name, geom=geom)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CytoPy interactive gating tool")
    parser.add_argument("database",
                        type=str,
                        help="Name of the database to connect.")
    parser.add_argument("gating_strategy",
                        type=str,
                        help="Name of the GatingStrategy to use.")
    parser.add_argument("experiment",
                        type=str,
                        help="Name of the experiment to edit.")
    parser.add_argument("filegroup",
                        type=str,
                        help="Name of sample (FileGroup) to edit.")
    args = parser.parse_args()
    logger.info(f"Interactive gating; database = {args.database}")

    # Initialise data
    global_init(database_name=args.database)
    try:
        GATING_STRATEGY = GatingStrategy.objects(name=args.gating_strategy).get()
        EXPERIMENT = Experiment.objects(experiment_id=args.experiment).get()
    except DoesNotExist:
        e = f"Either {args.gating_strategy} is not a valid GatingStrategy or {args.experiment} is not a valid Experiment"
        logger.error(e)
        raise DoesNotExist(e)
    GATING_STRATEGY.load_data(experiment=EXPERIMENT,
                              sample_id=args.filegroup)
    assert len(GATING_STRATEGY.gates) > 0, "No gates associated to chosen GatingStrategy"

    # Initialise controls
    GATE_SELECT.options = GATING_STRATEGY.list_gates()
    GATE_SELECT.value = GATING_STRATEGY.list_gates()[0]
