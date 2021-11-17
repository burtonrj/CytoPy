import logging
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union

import matplotlib.pyplot as plt
import numpy as np
from ipywidgets import widgets
from KDEpy import FFTKDE

from cytopy.data import Population
from cytopy.data.errors import MissingPopulationError
from cytopy.data.experiment import Experiment
from cytopy.data.population import ThresholdGeom
from cytopy.gating.threshold import apply_threshold
from cytopy.gating.threshold import find_threshold

logger = logging.getLogger(__name__)


def error(txt):
    return f"<p style='color:#FF0000;'><b>{txt}</b></p>"


def info(txt):
    return f"<p style='color:#1B5DA4;'><b>{txt}</b></p>"


class InteractiveControlGate1D(widgets.HBox):
    def __init__(
        self,
        experiment: Experiment,
        x: str,
        x_ctrl: str,
        parent_population: str,
        positive_population: str,
        negative_population: str,
        figsize: Tuple[int, int] = (5, 5),
        transform: str = "asinh",
        transform_kwargs: Optional[Dict] = None,
        kernel: str = "gaussian",
        bw: Union[str, float] = "ISJ",
        q: Optional[float] = None,
    ):
        super(InteractiveControlGate1D, self).__init__()
        self.experiment = experiment
        self._samples = experiment.list_samples()
        self._sample_i = -1
        self.x = x
        self.x_ctrl = x_ctrl
        self.parent_population = parent_population
        self.positive_population = positive_population
        self.negative_population = negative_population
        self.transform = transform
        self.transform_kwargs = transform_kwargs
        self.kernel = kernel
        self.bw = bw
        self.q = q
        self.threshold = None
        self._new_population_data = None
        self._primary_data = None
        self._ctrl_data = None

        # Define canvas
        output = widgets.Output()
        with output:
            self.fig, self.ax = plt.subplots(constrained_layout=True, figsize=figsize)
        self.fig.canvas.toolbar_position = "bottom"
        self.warnings = widgets.HTML(value="")
        self.x_text = widgets.Text(description="Threshold value")
        self.x_text.observe(self._update_x_threshold, "value")
        self.current_file = widgets.HTML(value="")
        self.parent_population_text = widgets.Label()
        self.pos_population_text = widgets.Label()
        self.neg_population_text = widgets.Label()
        self.progress_bar = widgets.IntProgress(description="Loading:", value=10, min=0, max=10)
        self.save_button = widgets.Button(
            description="Save", disabled=False, tooltip="Save changes", button_style="danger"
        )
        self.save_button.on_click(self._save_click)
        self.next_button = widgets.Button(
            description="Next", disabled=False, tooltip="Next sample", button_style="info"
        )
        self.next_button.on_click(self._next_sample)
        self.prev_button = widgets.Button(
            description="Previous", disabled=False, tooltip="Previous sample", button_style="info"
        )
        self.prev_button.on_click(self._previous_sample)

        controls = widgets.VBox(
            [
                self.current_file,
                self.warnings,
                self.parent_population_text,
                self.pos_population_text,
                self.neg_population_text,
                self.x_text,
                self.save_button,
                self.progress_bar,
            ]
        )
        navigation = widgets.HBox([self.prev_button, self.next_button])
        navigation.layout = widgets.Layout(
            border="solid 1px black", margin="0px 5px 5px 0px", padding="2px 2px 2px 2px"
        )
        controls.layout = widgets.Layout(border="solid 1px black", margin="0px 5px 5px 0px", padding="2px 2px 2px 2px")
        container = widgets.VBox([navigation, controls])
        container.layout = widgets.Layout(
            border="solid 1px black", margin="0px 5px 5px 0px", padding="5px 5px 5px 5px"
        )
        _ = widgets.Box([output])
        output.layout = widgets.Layout(border="solid 1px black", margin="0px 10px 10px 0px", padding="5px 5px 5px 5px")

        self.children = [container, output]
        self._next_sample(None)

    def _load_data(self, sample_id: str):
        try:
            self.progress_bar.value = 2
            fg = self.experiment.get_sample(sample_id)
            self.current_file.value = f"<b>{sample_id} ({self._sample_i})</b>"
            if self.positive_population in fg.list_populations():
                self.warnings.value = error("Control populations already exist for this sample!")
                self.save_button.disabled = True
            else:
                self.save_button.disabled = False
                self.warnings.value = ""
            self._primary_data = fg.load_population_df(
                population=self.parent_population, transform=self.transform, transform_kwargs=self.transform_kwargs
            )
            self._ctrl_data = fg.load_population_df(
                population=self.parent_population,
                transform=self.transform,
                transform_kwargs=self.transform_kwargs,
                data_source=self.x_ctrl,
            )
            self.progress_bar.value = 3
        except KeyError:
            self._primary_data = None
            self._ctrl_data = None
            self.warnings.value = error(f"{self._samples[self._sample_i]} missing requested control {self.x_ctrl}")
            self.ax.cla()
        except MissingPopulationError:
            self._primary_data = None
            self._ctrl_data = None
            self.warnings.value = error(
                f"{self._samples[self._sample_i]} missing requested parent {self.parent_population}"
            )
            self.ax.cla()

    def _next_sample(self, _):
        self.progress_bar.value = 1
        self._sample_i = self._sample_i + 1
        if self._sample_i > (len(self._samples) - 1):
            self._sample_i = 0
        self._load_sample()
        self.progress_bar.value = 10

    def _previous_sample(self, _):
        self.progress_bar.value = 1
        self._sample_i = self._sample_i - 1
        if self._sample_i < 0:
            self._sample_i = len(self._samples) - 1
        self._load_sample()
        self.progress_bar.value = 10

    def _load_sample(self):
        self._load_data(self._samples[self._sample_i])
        if self._primary_data is not None:
            self._plot_kde()
            self.parent_population_text.value = f"{self.parent_population} (n={self._primary_data.shape[0]})"
            self._apply_gate()

    def _plot_kde(self):
        self.ax.cla()
        self.progress_bar.value = 4
        x = np.linspace(
            np.min([self._primary_data[self.x].min(), self._ctrl_data[self.x].min()]) - 0.01,
            np.max([self._primary_data[self.x].max(), self._ctrl_data[self.x].max()]) + 0.01,
            1000,
        )
        y = FFTKDE(kernel=self.kernel, bw=self.bw).fit(self._primary_data[self.x].values).evaluate(x)
        self.ax.plot(x, y, linewidth=2, color="black")
        self.progress_bar.value = 5
        self.ax.fill_between(x, y, color="#8A8A8A", alpha=0.5)
        y = FFTKDE(kernel=self.kernel, bw=self.bw).fit(self._ctrl_data[self.x].values).evaluate(x)
        self.ax.plot(x, y, linewidth=2, color="blue")
        self.progress_bar.value = 6

    def _apply_gate(self):
        self.progress_bar.value = 7
        if self.q:
            self.threshold = self._ctrl_data[self.x].quantile(self.q)
        else:
            self.threshold = find_threshold(
                x=self._ctrl_data[self.x].values,
                bw="silverman",
                min_peak_threshold=0.05,
                peak_boundary=0.1,
                incline=False,
                kernel="gaussian",
                q=None,
            )
        self.progress_bar.value = 8
        self._new_population_data = apply_threshold(
            data=self._primary_data,
            x=self.x,
            x_threshold=self.threshold,
        )
        self.progress_bar.value = 9
        self.ax.axvline(self.threshold, linewidth=2, linestyle="-", color="black")
        self._update_labels()
        self.x_text.value = str(round(self.threshold, 4))

    def _update_labels(self):
        pos, neg = self._new_population_data.get("+"), self._new_population_data.get("-")
        pos_perc = round(pos.shape[0] / self._primary_data.shape[0] * 100, 3)
        neg_perc = round(neg.shape[0] / self._primary_data.shape[0] * 100, 3)
        self.pos_population_text.value = f"{self.positive_population} (n={pos.shape[0]}; {pos_perc}% of parent)"
        self.neg_population_text.value = f"{self.negative_population} (n={neg.shape[0]}; {neg_perc}% of parent)"

    def _update_x_threshold(self, change: Dict):
        try:
            self.progress_bar.value = 1
            self._plot_kde()
            self.threshold = change["new"]
            self.ax.axvline(float(self.threshold), linewidth=2, linestyle="-", color="black")
            self.progress_bar.value = 7
            self._new_population_data = apply_threshold(
                data=self._primary_data,
                x=self.x,
                x_threshold=float(self.threshold),
            )
            self.progress_bar.value = 9
            self._update_labels()
            self.progress_bar.value = 10
        except ValueError:
            self.progress_bar.value = 10

    def _save_click(self, _):
        self.progress_bar.value = 1
        fg = self.experiment.get_sample(self._samples[self._sample_i])
        if self.positive_population in fg.list_populations():
            logger.error(f"Populations {self.positive_population} and {self.negative_population} already exist!")
            return
        self.progress_bar.value = 2
        for definition, population_name in zip(["-", "+"], [self.negative_population, self.positive_population]):
            self.progress_bar.value = self.progress_bar.value + 1
            try:
                pop = Population(
                    population_name=population_name,
                    definition=definition,
                    parent=self.parent_population,
                    n=self._new_population_data[definition].shape[0],
                    source="gate",
                    geom=ThresholdGeom(
                        x=self.x,
                        y=None,
                        transform_x=self.transform,
                        transform_y=None,
                        transform_x_kwargs=self.transform_kwargs,
                        transform_y_kwargs=None,
                        x_threshold=float(self.threshold),
                        y_threshold=None,
                    ),
                )
                pop.index = self._new_population_data[definition].index.to_list()
                fg.add_population(population=pop)
            except ValueError:
                self.warnings.value = error(f"Invalid threshold: {self.threshold} is not a valid float")
            self.progress_bar.value = self.progress_bar.value + 1
        fg.save()
        self.warnings.value = info(f"Changes saved to {fg.primary_id}!")
        self.progress_bar.value = 10
