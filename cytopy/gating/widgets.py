import logging
from itertools import cycle
from typing import Any
from typing import Callable
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union

import numpy as np
import pandas as pd
from ipywidgets import widgets
from KDEpy import FFTKDE
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.widgets import PolygonSelector

from ..data.errors import MissingPopulationError
from .base import Gate
from .polygon import PolygonGate
from .template import GatingStrategy
from .threshold import ThresholdBase

logger = logging.getLogger(__name__)


def make_box_layout():
    return widgets.Layout(border="solid 1px black", margin="0px 10px 10px 0px", padding="5px 5px 5px 5px")


class InteractiveGateTool(widgets.HBox):
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
        self.parent_data = None
        self.ctrl_data = None
        self.artists = {}

        # Define canvas
        self.output = widgets.Output()
        self.output.layout = make_box_layout()
        with self.output:
            self.fig, self.ax = plt.subplots(constrained_layout=True, figsize=figsize)
        self.fig.canvas.toolbar_position = "bottom"
        self._define_children(self.output)

    @staticmethod
    def _add_dropdown(
        description: str,
        disabled: bool = False,
        options: Optional[List[Any]] = None,
        value: Optional[Any] = None,
        func: Optional[Callable] = None,
    ) -> widgets.Dropdown:
        options = options or []
        dropdown = widgets.Dropdown(
            description=description,
            disabled=disabled,
            options=options,
            value=value,
        )
        if func is not None:
            dropdown.observe(func, "value")
        return dropdown

    @staticmethod
    def _add_button(
        description: str,
        disabled: bool = False,
        tooltip: Optional[str] = None,
        button_style: str = "info",
        func: Optional[Callable] = None,
    ) -> widgets.Button:
        button = widgets.Button(description=description, disabled=disabled, tooltip=tooltip, button_style=button_style)
        if func is not None:
            button.on_click(func)
        return button

    @staticmethod
    def _add_text(disabled: bool = False, func: Optional[Callable] = None) -> widgets.Text:
        txt = widgets.Text(disabled=disabled)
        if func is not None:
            txt.observe(func, "value")
        return txt

    @staticmethod
    def _build_vbox(*args) -> widgets.VBox:
        items = widgets.VBox([*args])
        items.layout = make_box_layout()
        return items

    def _apply_and_save_button(self):
        # Button for applying changes to GatingStrategy
        apply_button = self._add_button(
            disabled=True,
            description="Apply",
            tooltip="Apply changed to GatingStrategy",
            button_style="warning",
            func=self._apply_click,
        )
        # Button for saving changes to database
        save_button = self._add_button(
            disabled=False, description="Save", tooltip="Save changes", button_style="danger", func=self._save_click
        )
        return apply_button, save_button

    def _define_children(self, *args):
        self.children = [*args]

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

    def _update_kde_plot(self, primary_data: np.ndarray, ctrl_data: np.ndarray):
        self.ax.cla()
        primary_x, primary_y = FFTKDE(kernel="gaussian", bw="ISJ").fit(primary_data).evaluate()
        ctrl_x, ctrl_y = FFTKDE(kernel="gaussian", bw="ISJ").fit(ctrl_data).evaluate()
        self.ax.plot(ctrl_x, ctrl_y, linewidth=2, color="black")
        self.ax.fill_between(ctrl_x, ctrl_y, color="#25958A", alpha=0.5)
        self.ax.plot(primary_x, primary_y, linewidth=2, color="black")
        self.ax.fill_between(primary_x, primary_y, color="#8A8A8A", alpha=0.5)
        self.ax.set_xlabel(self.gate.x)
        if self.xlim is not None:
            self.ax.set_xlim(self.xlim)
        self.ax.set_title(f"{self.gate.gate_name} (Parent={self.gate.parent})")


class GateEditor(InteractiveGateTool):
    def _define_children(self, *args):
        self.progress_bar = widgets.IntProgress(description="Loading:", value=5, min=0, max=5)
        # Dropdown for choosing a gate
        self.gate_select = self._add_dropdown(
            description="Gate",
            disabled=False,
            options=[g.gate_name for g in self.gs.gates],
            value=self.gs.gates[0].gate_name,
            func=self._load_gate,
        )
        # Dropdown for choosing population to edit
        self.child_select = self._add_dropdown(description="Child population", disabled=True)
        # Button to update polygon geometry when editing a polygon gate
        self.update_button = self._add_button(
            disabled=True,
            description="Update",
            tooltip="Update population geometry",
            button_style="info",
            func=self._poly_update,
        )
        # X and Y axis values when editing a threshold gate
        self.x_text = self._add_text(disabled=True, func=self._update_x_threshold)
        self.y_text = self._add_text(disabled=True, func=self._update_y_threshold)
        # Button for applying and saving changes to GatingStrategy
        self.apply_button, self.save_button = self._apply_and_save_button()
        # Package controls into VBox
        controls = self._build_vbox(
            self.gate_select,
            self.child_select,
            self.x_text,
            self.y_text,
            self.update_button,
            self.apply_button,
            self.save_button,
            self.progress_bar,
        )
        self.children = [controls, self.output]
        # Load the first gate
        self._load_gate(change={"new": self.gs.gates[0].gate_name})

    def _load_gate(self, change: Dict):
        self.gate = self._load_and_check_children(gate=change["new"])
        self.progress_bar.value = 1
        data = self.gs.population_data(population_name=self.gate.parent)
        if data.shape[0] > 10000 and self.downsample is not None:
            if self.downsample < data.shape[0]:
                data = data.sample(n=self.downsample)
        self.parent_data = self.gate.preprocess(data=data, transform=True)
        if isinstance(self.gate, ThresholdBase):
            if self.gate.ctrl:
                self.ctrl_data = self.gate.preprocess(
                    data=self.gs.population_data(population_name=self.gate.parent, data_source=self.gate.ctrl)
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

    def _load_and_check_children(self, gate: str) -> Gate:
        gate = self.gs.get_gate(gate=gate)
        for child in gate.children:
            if child.name not in self.gs.filegroup.list_populations():
                raise MissingPopulationError(f"{child.name} not found in associated filegroup!")
        return gate

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

    def _draw_artists(self):
        if isinstance(self.gate, ThresholdBase):
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
        if isinstance(self.gate, ThresholdBase):
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
        if isinstance(self.gate, ThresholdBase):
            self.gs.edit_threshold_populations(
                gate=self.gate.gate_name,
                x_threshold=self.gate_geometry["x_threshold"],
                y_threshold=self.gate_geometry["y_threshold"],
                transform=False,
            )
            self.progress_bar.value = 5
        else:
            self.gs.edit_polygon_populations(
                gate=self.gate.gate_name,
                coords={
                    pop_name: df[[self.gate.x, self.gate.y]].values for pop_name, df in self.gate_geometry.items()
                },
                transform=False,
            )
            self.progress_bar.value = 5
        logger.info("New gate applied.")

    def _save_click(self, _):
        self.progress_bar.value = 1
        self.gs.save()
        self.progress_bar.value = 5
        logger.info("Changes saved!")
