import logging
from itertools import cycle
from typing import Dict
from typing import Optional
from typing import Tuple
from typing import Union

import numpy as np
import pandas as pd
from ipywidgets import widgets
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.widgets import PolygonSelector

from .gate import PolygonGate
from .gate import ThresholdGate
from .gating_strategy import FileGroup
from .gating_strategy import GatingStrategy

logger = logging.getLogger(__name__)


def make_box_layout():
    return widgets.Layout(border="solid 1px black", margin="0px 10px 10px 0px", padding="5px 5px 5px 5px")


class InteractiveGateEditor(widgets.HBox):
    def __init__(
        self,
        gating_strategy: GatingStrategy,
        default_y: Optional[str] = "FSC-A",
        default_y_transform: Optional[str] = None,
        default_y_transform_kwargs: Optional[str] = None,
        figsize: Tuple[int, int] = (5, 5),
        xlim: Optional[Tuple[int]] = None,
        ylim: Optional[Tuple[int]] = None,
        cmap: str = "jet",
        min_bins: int = 250,
        downsample: Optional[float] = None,
    ):
        super().__init__()
        if gating_strategy.filegroup is None:
            raise ValueError(
                "Gating strategy must be populated, call load_data before using " "interactive gate editor."
            )
        # Organise data
        self.selector = None
        self.default_y = default_y
        self.default_y_transform = default_y_transform
        self.default_y_transform_kwargs = default_y_transform_kwargs
        self.gs = gating_strategy
        self.min_bins = min_bins
        self.downsample = downsample
        self.cmap = cmap
        self.xlim = xlim
        self.ylim = ylim
        self.gate = None
        self.artists = {}

        # Define canvas
        output = widgets.Output()
        with output:
            self.fig, self.ax = plt.subplots(constrained_layout=True, figsize=figsize)
        self.fig.canvas.toolbar_position = "bottom"

        # Define widgets
        self.selector = None
        self.progress_bar = widgets.IntProgress(description="Loading:", value=5, min=0, max=5)
        self.gate_select = widgets.Dropdown(
            description="Gate",
            disabled=False,
            options=[g.gate_name for g in self.gs.gates],
            value=self.gs.gates[0].gate_name,
        )
        self.gate_select.observe(self._load_gate, "value")
        self.child_select = widgets.Dropdown(
            description="Child population",
            disabled=True,
        )
        self.update_button = widgets.Button(
            description="Update", disable=True, tooltop="Update population geometry", button_style="info"
        )
        self.update_button.on_click(self._poly_update)
        self.x_text = widgets.Text(disabled=True)
        self.x_text.observe(self._update_x_threshold, "value")
        self.y_text = widgets.Text(disabled=True)
        self.y_text.observe(self._update_y_threshold, "value")
        self.apply_button = widgets.Button(
            description="Apply", disabled=True, tooltip="Apply changed to GatingStrategy", button_style="warning"
        )
        self.apply_button.on_click(self._apply_click)
        self.save_button = widgets.Button(
            description="Save", disabled=False, tooltip="Save changes", button_style="danger"
        )
        self.save_button.on_click(self._save_click)
        controls = widgets.VBox(
            [
                self.gate_select,
                self.child_select,
                self.x_text,
                self.y_text,
                self.update_button,
                self.apply_button,
                self.save_button,
                self.progress_bar,
            ]
        )
        controls.layout = make_box_layout()
        _ = widgets.Box([output])
        output.layout = make_box_layout()

        self.children = [controls, output]

        self._load_gate(change={"new": self.gs.gates[0].gate_name})

    def _update_hexbin_plot(self, plot_data: np.ndarray):
        self.ax.cla()
        bins = int(np.sqrt(self.parent_data.shape[0]))
        if bins < self.min_bins:
            bins = self.min_bins
        self.ax.hist2d(plot_data[:, 0], plot_data[:, 1], bins=[bins, bins], cmap=self.cmap, norm=LogNorm())

        if self.xlim is not None:
            self.ax.set_xlim(self.xlim)

        if self.ylim is not None:
            self.ax.set_ylim(self.ylim)
        self.ax.set_xlabel(self.gate.x)

        if self.gate.y is not None:
            self.ax.set_ylabel(self.gate.y)
        else:
            self.ax.set_ylabel(self.default_y)
        self.ax.set_title(f"{self.gate.gate_name} (Parent={self.gate.parent})")

    def _update_widgets(self):
        if isinstance(self.gate, PolygonGate):
            self.selector = PolygonSelector(self.ax, lambda x: None)
            self.child_select.disabled = False
            self.child_select.options = [child.name for child in self.gate.children]
            self.child_select.value = self.gate.children[0].name
            self.update_button.disabled = False
            self.x_text.disabled = True
            self.y_text.disabled = True
        else:
            self.x_text.disabled = False
            self.x_text.description = f"{self.gate.x} threshold"
            self.x_text.value = str(self.gate_geometry["x_threshold"])
            if self.gate_geometry["y_threshold"] is not None:
                self.y_text.disabled = False
                self.y_text.value = str(self.gate_geometry["y_threshold"])
                self.y_text.description = f"{self.gate.y} threshold"
            self.update_button.disabled = True
            self.child_select.disabled = True
            if isinstance(self.selector, PolygonSelector):
                self.selector.disconnect_events()
            self.selector = None
        self.apply_button.disabled = False

    def _load_gate(self, change: Dict):
        self.gate = self.gs.gate_children_present_in_filegroup(self.gs.get_gate(gate=change["new"]))
        self.progress_bar.value = 1
        transforms, transform_kwargs = self.gate.transform_info()
        n = self.gs.filegroup.get_population(population_name=self.gate.parent).n
        sample_size = self.downsample if n > 10000 else None
        self.parent_data = self.gs.filegroup.load_population_df(
            population=self.gate.parent,
            transform=transforms,
            transform_kwargs=transform_kwargs,
            sample_size=sample_size,
            sampling_method="uniform",
        )
        self.progress_bar.value = 2

        if self.default_y not in self.parent_data.columns:
            raise ValueError(
                f"Chosen default Y-axis variable {self.default_y} does not exist for this data. "
                f"Make sure to chose a suitable default y-axis variable to be used with 1 dimensional "
                f"gates."
            )
        y = self.gate.y or self.default_y
        self.gate_geometry = self._obtain_gate_geometry()
        self.progress_bar.value = 3
        self._update_hexbin_plot(self.parent_data[[self.gate.x, y]].values)
        self._draw_artists()
        self.progress_bar.value = 4
        self._update_widgets()
        self.progress_bar.value = 5

    def _draw_artists(self):
        if isinstance(self.gate, ThresholdGate):
            self.artists["x"] = self.ax.axvline(self.gate_geometry["x_threshold"], c="red")
            if self.gate_geometry["y_threshold"] is not None:
                self.artists["y"] = self.ax.axhline(self.gate_geometry["y_threshold"], c="red")
        else:
            self.artists = {
                child.name: self.ax.plot(
                    self.gate_geometry[child.name][self.gate.x].values,
                    self.gate_geometry[child.name][self.gate.y].values,
                    c=self.gate_geometry[child.name]["colour"].values[0],
                    lw=1.5,
                )[0]
                for child in self.gate.children
            }

    def _obtain_gate_geometry(self):
        gate_colours = cycle(
            [
                "#c92c2c",
                "#2df74e",
                "#e0d572",
                "#000000",
                "#64b9c4",
                "#9e3657",
                "#d531f2",
                "#cf0077",
                "#5c37bd",
                "#52b58c",
            ]
        )
        if isinstance(self.gate, ThresholdGate):
            pop = self.gs.filegroup.get_population(population_name=self.gate.children[0].name)
            return {"x_threshold": pop.geom.x_threshold, "y_threshold": pop.geom.y_threshold}
        geom = {}
        for child in self.gate.children:
            pop = self.gs.filegroup.get_population(population_name=child.name)
            c = [next(gate_colours) for _ in range(len(pop.geom.x_values))]
            geom[pop.population_name] = pd.DataFrame(
                {self.gate.x: pop.geom.x_values, self.gate.y: pop.geom.y_values, "colour": c}
            )
        return geom

    def _poly_update(self, _):
        self.progress_bar.value = 1
        verts = self.selector.verts
        verts.append(verts[0])
        verts = np.array(verts)
        c = self.gate_geometry[self.child_select.value]["colour"].values[0]
        self.progress_bar.value = 2
        self.gate_geometry[self.child_select.value] = pd.DataFrame(
            {self.gate.x: verts[:, 0], self.gate.y: verts[:, 1], "colour": [c for _ in range(verts.shape[0])]}
        )
        self.progress_bar.value = 3
        geom = self.gate_geometry[self.child_select.value]
        self.artists[self.child_select.value].set_data(
            geom[[self.gate.x, self.gate.y]].values[:, 0], geom[[self.gate.x, self.gate.y]].values[:, 1]
        )
        self.progress_bar.value = 4
        self.fig.canvas.draw()
        self.progress_bar.value = 5

    def _update_threshold(self, value: Union[str, int, float], axis: str):
        self.progress_bar.value = 1
        try:
            self.gate_geometry[f"{axis}_threshold"] = float(value)
            set_data = getattr(self.artists[axis], f"set_{axis}data")
            self.progress_bar.value = 3
            set_data(np.array([float(value), float(value)]))
            self.progress_bar.value = 4
            self.fig.canvas.draw()
            self.progress_bar.value = 5
        except ValueError:
            logger.debug("Invalid value passed to text field")
            self.progress_bar.value = 5

    def _update_x_threshold(self, change: Dict):
        self._update_threshold(value=change["new"], axis="x")

    def _update_y_threshold(self, change: Dict):
        self._update_threshold(value=change["new"], axis="y")

    def _apply_click(self, _):
        self.progress_bar.value = 2
        if isinstance(self.gate, ThresholdGate):
            self.gs.edit_threshold_gate(
                gate_name=self.gate.gate_name,
                x_threshold=self.gate_geometry["x_threshold"],
                y_threshold=self.gate_geometry["y_threshold"],
                transform=False,
            )
            self.progress_bar.value = 5
        else:
            self.gs.edit_polygon_gate(
                gate_name=self.gate.gate_name,
                coords={
                    pop_name: df[[self.gate.x, self.gate.y]].values for pop_name, df in self.gate_geometry.items()
                },
                transform=False,
            )
            self.progress_bar.value = 5

    def _save_click(self, _):
        self.progress_bar.value = 1
        self.gs.save()
        self.progress_bar.value = 5
        logger.info("Changes saved!")


class ManualLatentGating:
    def __init__(
        self,
        filegroup: FileGroup,
        parent_population: str = "root",
        source: str = "primary",
        transform: str = "logicle",
        transform_kwargs: Optional[Dict] = None,
        figsize: Tuple[int, int] = (5, 5),
        min_bins: int = 250,
    ):
        super().__init__()
        # -- Organise data
        transform_kwargs = transform_kwargs or {}
        self.fg = filegroup
        self.data = self.fg.load_population_df(
            population=parent_population,
        )
        self.min_bins = min_bins
        self.artists = {}

        # -- Define canvas
        output = widgets.Output()
        with output:
            self.fig, self.ax = plt.subplots(constrained_layout=True, figsize=figsize)
        self.fig.canvas.toolbar_position = "bottom"

        # -- Define widgets

        # UMAP settings
        self.sample_size = widgets.IntSlider(
            value=10000, min=10000, max=1000000, step=100, description="Training sample size"
        )
        self.n_neighbors = widgets.IntSlider(value=15, min=3, max=1000, step=1, description="N neighbours")
        self.min_dist = widgets.IntSlider(value=0.1, min=0.0, max=0.99, step=0.05, description="N neighbours")
        self.metric = widgets.Dropdown(
            description="Metric",
            disabled=False,
            options=["euclidean", "manhattan", "chebyshev", "minkowski", "cosine"],
            value="euclidean",
        )
        self.umap_settings = {"sample_size": 10000, "n_neighbors": 15, "min_dist": 0.1, "metric": "euclidean"}

        # Other plotting settings
        self.plot_colour = widgets.Dropdown(
            description="Plot colour/type", disabled=False, options=["Density", "Scatter"] + self.gs.filegroup.d
        )

        # Population settings
        self.pop_select = widgets.Dropdown(
            description="Population",
            disabled=False,
            options=self.gs.filegroup.list_populations(),
            value="root",
        )
        self.population_name = widgets.Text(disabled=True, description="Population name")

        # Buttons
        self.plot_button = widgets.Button(
            description="Plot", disabled=True, tooltip="Compute and plot embeddings", button_style="warning"
        )
        self.apply_button = widgets.Button(
            description="Apply", disabled=True, tooltip="Create/edit population", button_style="warning"
        )
        self.save_button = widgets.Button(
            description="Save", disabled=False, tooltip="Save changes", button_style="danger"
        )

        # -- Actions
        self.selector = None
        self.progress_bar = widgets.IntProgress(description="Loading:", value=5, min=0, max=5)

        self.plot_button.on_click(self._plot)
        self.apply_button.on_click(self._apply_click)
        self.save_button.on_click(self._save_click)
