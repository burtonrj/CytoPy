import logging
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from ipywidgets import widgets
from KDEpy import FFTKDE
from sklearn.base import BaseEstimator
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.utils import compute_sample_weight

from cytopy.data import FileGroup
from cytopy.data import Population
from cytopy.data.errors import MissingPopulationError
from cytopy.data.experiment import Experiment
from cytopy.data.population import ThresholdGeom
from cytopy.feedback import progress_bar
from cytopy.gating.threshold import apply_threshold
from cytopy.gating.threshold import find_threshold
from cytopy.plotting import scatterplot
from cytopy.utils import DimensionReduction

logger = logging.getLogger(__name__)


class ControlPopulationPrediction:
    def __init__(
        self,
        filegroup: FileGroup,
        ctrl: str,
        populations: List[str],
        model: Optional[BaseEstimator] = None,
        features: Optional[List[str]] = None,
        transform: str = "asinh",
        transform_kwargs: Optional[Dict] = None,
        downsample_background: Optional[int] = 10000,
    ):
        self.filegroup = filegroup
        self.data = filegroup.load_population_df(
            population="root", transform=transform, transform_kwargs=transform_kwargs
        )
        self.populations = populations
        self.data["label"] = 0
        for i, pop in enumerate(populations):
            self.data.loc[filegroup.get_population(pop).index, "label"] = i + 1

        if downsample_background:
            tmp = self.data[self.data["label"] != 0]
            background = self.data[self.data["label"] == 0].sample(downsample_background)
            self.data = pd.concat([tmp, background]).reset_index(drop=True)

        self.features = features or [i for i in self.data.columns if i.lower() not in ["time", "label"]]
        self.model = (
            model
            if model is not None
            else HistGradientBoostingClassifier(max_iter=100, loss="categorical_crossentropy", max_depth=6)
        )
        self.ctrl = ctrl
        self.control = filegroup.load_population_df(
            population="root", transform=transform, transform_kwargs=transform_kwargs, data_source=ctrl
        )

    def cross_val_performance(self, n_splits: int = 10, verbose: bool = True, balance: bool = True) -> pd.DataFrame:
        performance = []
        skf = StratifiedKFold(n_splits=n_splits)
        for train_idx, test_idx in progress_bar(
            skf.split(self.data[self.features], self.data["label"]), total=n_splits, verbose=verbose
        ):
            train_x, test_x = self.data[self.features].values[train_idx], self.data[self.features].values[test_idx]
            train_y, test_y = self.data["label"].values[train_idx], self.data["label"].values[test_idx]

            # Fit model
            weights = None if not balance else compute_sample_weight(class_weight="balanced", y=train_y)
            self.model.fit(train_x, train_y, sample_weight=weights)

            # Log performance
            for x, y, key in zip([train_x, test_x], [train_y, test_y], ["Training", "Testing"]):
                yhat = self.model.predict(x)
                yscore = self.model.predict_proba(x)
                performance.append(
                    {
                        "Dataset": key,
                        "Balance accuracy": balanced_accuracy_score(y_pred=yhat, y_true=y),
                        "Weighted F1 score": f1_score(y_pred=yhat, y_true=y, average="weighted"),
                        "ROC AUC Score": roc_auc_score(y_true=y, y_score=yscore, average="macro", multi_class="ovo"),
                    }
                )
        performance = pd.DataFrame(performance)
        performance = performance.melt(id_vars="Dataset", var_name="Metric", value_name="Value")
        return performance

    def fit(self, balance: bool = True):
        weights = None if not balance else compute_sample_weight(class_weight="balanced", y=self.data["label"].values)
        self.model.fit(self.data[self.features].values, self.data["label"].values, sample_weight=weights)
        return self.model

    def fit_predict(self, balance: bool = True):
        self.control["label"] = self.fit(balance=balance).predict(self.control[self.features].values)
        return self.control

    def umap_comparison(self, n: int = 100000, save_path: Optional[str] = None, **plot_kwargs) -> plt.Figure:
        if "label" not in self.control.columns:
            logger.info("Predicting labels for control data")
            self.fit_predict(balance=True)

        logger.info("Computing UMAP embeddings for all data")
        umap = DimensionReduction(method="UMAP", n_components=2)
        primary_plot_all = umap.fit_transform(self.data.sample(n), features=self.features)
        ctrl_plot_all = umap.transform(self.control.sample(n), features=self.features)

        logger.info("Computing UMAP embeddings for populations only")
        umap = DimensionReduction(method="UMAP", n_components=2)

        primary_plot_pops = self.data[self.data["label"] != 0].copy()
        if primary_plot_pops.shape[0] > n:
            primary_plot_pops = primary_plot_pops.sample(n)

        ctrl_plot_pops = self.control[self.control["label"] != 0].copy()
        if ctrl_plot_pops.shape[0] > n:
            ctrl_plot_pops = ctrl_plot_pops.sample(n)

        primary_plot_pops = umap.fit_transform(primary_plot_pops, features=self.features)
        ctrl_plot_pops = umap.transform(ctrl_plot_pops, features=self.features)

        logger.info("Plotting data")
        fig, axes = plt.subplots(2, 2, figsize=(12.5, 15))
        for df, ax, title in zip(
            [primary_plot_all, ctrl_plot_all, primary_plot_pops, ctrl_plot_pops],
            axes.flatten(),
            [
                "Original data (all)",
                "Control data (all)",
                "Original data (populations only)",
                "Control data (populations only)",
            ],
        ):
            scatterplot(
                data=df.sort_values("label"), x="UMAP1", y="UMAP2", label="label", discrete=True, ax=ax, **plot_kwargs
            )
            ax.set_title(title)
        if save_path is not None:
            fig.savefig(save_path, dpi=100, facecolor="white", bbox_inches="tight")
        return fig

    def save(self):
        if "label" not in self.control.columns:
            raise ValueError("Call 'fit_predict' first.")
        self.control["label"] = self.control["label"].replace({i + 1: p for i, p in enumerate(self.populations)})
        for label, df in self.control.groupby("label"):
            if label == 0:
                continue
            ctrl_pop = Population(
                population_name=label,
                parent="root",
                n=df.shape[0],
                source="classifier",
                data_source=self.ctrl,
            )
            ctrl_pop.index = df.index.to_list()
            self.filegroup.add_population(population=ctrl_pop)
        self.filegroup.save()


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
