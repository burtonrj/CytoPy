import logging
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from dtw import dtw
from ipywidgets import widgets
from KDEpy import FFTKDE
from skfda import FDataGrid
from skfda.preprocessing.registration import landmark_registration_warping

from cytopy.feedback import progress_bar
from cytopy.plotting.general import ColumnWrapFigure
from cytopy.utils.transform import Scaler
from cytopy.utils.transform import Transformer
from cytopy.variance.base import BatchCorrector

logger = logging.getLogger(__name__)


class LandmarkEditor(widgets.HBox):
    def __init__(self, x: np.ndarray, y: np.ndarray, landmarks: Optional[List] = None):
        super().__init__()
        self.landmarks = landmarks or []
        output = widgets.Output()
        with output:
            self.fig, self.ax = plt.subplots(constrained_layout=True, figsize=(4, 2.5))
        self.fig.canvas.toolbar_position = "bottom"
        self.ax.plot(x, y, color="black")
        self._plot_landmarks()

        self.landmark_text = widgets.Text(disabled=True)
        self.add_landmark_button = widgets.Button(description="Add landmark", button_style="info")
        self.add_landmark_button.on_click(self._add_landmark)
        self.landmark_dropbox = widgets.Dropdown(description="Select landmark", options=self.landmarks)
        self.remove_landmark_button = widgets.Button(description="Remove landmark", button_style="warning")
        self.remove_landmark_button.on_click(self._removed_landmark)

        self.close_button = widgets.Button(description="Close", button_style="warning")
        self.close_button.on_click(lambda _: self.close())

        controls = widgets.VBox(
            [
                self.landmark_text,
                self.add_landmark_button,
                self.landmark_dropbox,
                self.remove_landmark_button,
                self.close_button,
            ]
        )
        controls.layout = widgets.Layout(
            border="solid 1px black", margin="0px 10px 10px 0px", padding="5px 5px 5px 5px"
        )
        _ = widgets.Box([output])
        output.layout = widgets.Layout(border="solid 1px black", margin="0px 10px 10px 0px", padding="5px 5px 5px 5px")

        self.children = [controls, output]

    def _plot_landmarks(self):
        for lm in self.landmarks:
            self.ax.axvline(lm, ls="--", color="black")

    def _add_landmark(self, _):
        self.landmarks.append(self.landmark_text.value)
        self.landmarks = list(set(self.landmarks))
        self._plot_landmarks()
        self.landmark_dropbox.options = self.landmarks
        self.landmark_dropbox.value = self.landmarks[0]

    def _remove_landmark(self, _):
        self.landmarks = [x for x in self.landmarks if x != self.landmark_dropbox.value]
        self._plot_landmarks()


def kde(data: List[np.ndarray], grid_size: int, kernel: str, bw: Union[str, float]):
    x = np.linspace(np.min([np.min(x) for x in data]) - 0.1, np.max([np.max(x) for x in data]) + 0.1, grid_size)
    return FDataGrid([FFTKDE(kernel=kernel, bw=bw).fit(i).evaluate(x) for i in data], grid_points=x)


def differential(x):
    idx = np.arange(1, len(x) - 1)
    return (x[idx] - x[idx - 1]) + ((x[idx + 1] - x[idx - 1]) / 2) / 2


class FunctionalBatchCorrection(BatchCorrector):
    def __init__(
        self,
        data: pd.DataFrame,
        features: List[str],
        transformer: Optional[Transformer] = None,
        scaler: Optional[Scaler] = None,
        verbose: bool = True,
        grid_size: int = 100,
        kernel: str = "gaussian",
        bw: Union[str, float] = "ISJ",
    ):
        super().__init__(data, features, transformer, scaler, verbose)
        self.grid_size = grid_size
        self.kernel = kernel
        self.bw = bw
        self._feature_data = {}
        self._landmark_editor = None
        for f in progress_bar(features, verbose=self.verbose):
            self._feature_data[f] = {}
            labelled_data = [(i, x.values) for i, x in self.original_data[["sample_id", f]].groupby("sample_id")[f]]
            self._feature_data[f]["index"] = [i[0] for i in labelled_data]
            self._feature_data[f]["kde"] = kde(
                data=[i[1] for i in labelled_data], kernel=kernel, bw=bw, grid_size=grid_size
            )
            self._feature_data[f]["df"] = pd.DataFrame(
                self._feature_data[f]["kde"].data_matrix[:, :, 0], index=self._feature_data[f]["index"]
            ).T
            self._feature_data[f]["warping_functions"] = None
            self._feature_data[f]["corrected_functions"] = None
            self._feature_data[f]["landmarks"] = None
            self._feature_data[f]["reference"] = None

    def prepare_reference(self, feature: str):
        original = self._feature_data[feature]["df"]
        dtw_distances = np.zeros((original.shape[1], original.shape[1]))
        for i, x in progress_bar(enumerate(original.columns), total=original.shape[1], verbose=self.verbose):
            for j, y in enumerate(original.columns):
                dtw_distances[i, j] = dtw(differential(original[x].values), differential(original[y].values)).distance
        dtw_distances = pd.DataFrame(dtw_distances, index=original.columns, columns=original.columns)
        ref = dtw_distances.mean().sort_values().index[0]
        self._feature_data[feature]["reference"] = ref
        self._landmark_editor = LandmarkEditor(
            x=self._feature_data[feature]["kde"].grid_points[0], y=original[ref].values
        )
        return self._landmark_editor

    def collect_landmarks(self, feature: str):
        ref = self._feature_data[feature]["reference"]
        if ref is None:
            raise ValueError("Call 'prepare reference' first")
        ref_landmarks = self._landmark_editor.landmarks
        ref_data = self._feature_data[feature]["df"][ref].values
        landmarks = []
        for _id in self._feature_data[feature]["df"].columns:
            if _id == ref:
                landmarks.append(ref_landmarks)
                continue
            alignment = dtw(differential(ref_data), differential(self._feature_data[feature]["df"][_id].values))
            xgrid = self._feature_data[feature]["kde"].grid_points[0]
            for lm in ref_landmarks:
                tmp = []
                idx = np.where(abs(xgrid - lm) == np.min(abs(xgrid - lm)))[0][0]
                tmp.append(np.mean(xgrid[alignment.index2[alignment.index1 == idx]]))
                landmarks.append(tmp)
        self._feature_data[feature]["landmarks"] = landmarks
        return self

    def plot_landmarks(self, feature: str):
        landmarks = self._feature_data[feature]["landmarks"]
        if landmarks is None:
            raise ValueError("Call 'collect_landmarks' first")
        original = self._feature_data[feature]["df"]
        fig = ColumnWrapFigure(n=original.shape[1], col_wrap=1, figsize=(5, original.shape[1] * 1))
        x = self._feature_data[feature]["kde"].grid_points[0]
        for i, y_col in progress_bar(enumerate(original.columns), total=original.shape[1], verbose=self.verbose):
            ax = fig.add_subplot()
            ax.plot(x, original[y_col], color="blue")
            for la in landmarks[i]:
                ax.axvline(la, ls="--", color="black")
            ax.set_title(f"{y_col} ({[round(la, 4) for la in landmarks[i]]})")
        fig.tight_layout()
        return fig

    def landmark_editor(self, feature: str, sample_id: str):
        landmarks = self._feature_data[feature]["landmarks"]
        if landmarks is None:
            raise ValueError("Call 'collect_landmarks' first")
        original = self._feature_data[feature]["df"]
        landmarks = pd.DataFrame(landmarks, index=self._feature_data[feature]["index"])
        landmarks = landmarks.loc[sample_id].values
        self._landmark_editor = LandmarkEditor(
            x=self._feature_data[feature]["kde"].grid_points[0], y=original[sample_id].values, landmarks=landmarks
        )
        return self._landmark_editor

    def confirm_edits(self, feature: str, sample_id: str):
        landmarks = self._feature_data[feature]["landmarks"]
        if landmarks is None:
            raise ValueError("Call 'collect_landmarks' first")
        if self._landmark_editor is None:
            raise ValueError("Call 'landmark_editor' first")
        i = self._feature_data[feature]["index"].index(sample_id)
        self._feature_data[feature]["landmarks"][i] = self._landmark_editor.landmarks

    def landmark_registration(self, feature: str):
        self._feature_data[feature]["warping_functions"] = landmark_registration_warping(
            self._feature_data[feature]["kde"],
            self._feature_data[feature]["landmarks"],
            location=np.mean(self._feature_data[feature]["landmarks"], axis=0),
        )
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        self._feature_data[feature]["kde"].plot(axes=axes[0])
        axes[0].set_title("Before")
        self._feature_data[feature]["warping_functions"].plot(axes=axes[1])
        axes[1].set_title("Warping function")
        self._feature_data[feature]["kde"].compose(self._feature_data[feature]["warping_functions"]).plot(axes=axes[2])
        axes[2].set_title("After")
        return fig

    def plot_corrections(self, feature: str):
        original = self._feature_data[feature]["df"]
        corrected = pd.DataFrame(
            self._feature_data[feature]["kde"]
            .compose(self._feature_data[feature]["warping_functions"])
            .data_matrix[:, :, 0],
            index=self._feature_data[feature]["index"],
        )
        fig = ColumnWrapFigure(n=original.shape[1], col_wrap=1, figsize=(5, original.shape[1] * 1))
        x = self._feature_data[feature]["kde"].grid_points[0]
        landmarks = np.mean(self._feature_data[feature]["landmarks"], axis=0)
        for y_col in progress_bar(original.columns, total=original.shape[1], verbose=self.verbose):
            ax = fig.add_subplot()
            ax.plot(x, original[y_col], color="red")
            ax.plot(x, corrected[y_col], color="blue")
            for lm in landmarks:
                ax.axvline(lm, ls="--", color="blue")
            ax.set_title(y_col)
        fig.tight_layout()
        return fig

    def correct_channel(self, feature: str):
        corrected = self._feature_data[feature]["warping_functions"]
        if corrected is None:
            raise ValueError("No warping functions defined for given feature")
        if self.corrected_data is None:
            self.corrected_data = self.original_data[["sample_id", "subject_id", feature]].copy()
            self.corrected_data[feature] = self.corrected_data.groupby("sample_id")[feature].apply(
                lambda x: corrected.evaluate(x.values)[1].reshape(-1)
            )
        else:
            self.corrected_data[feature] = self.original_data.groupby("sample_id")[feature].apply(
                lambda x: corrected.evaluate(x.values)[1].reshape(-1)
            )

    def plot_feature(
        self,
        feature: str,
        kde_kwargs: Optional[Dict] = None,
        plot_overlaid: bool = True,
        figsize: Optional[Tuple[int]] = None,
        plot_corrected: bool = True,
        col_wrap: int = 1,
    ):
        if plot_corrected:
            if feature not in self.corrected_data.columns:
                raise ValueError(f"{feature} has not been corrected yet!")
        super(FunctionalBatchCorrection, self).plot_feature(
            feature=feature,
            kde_kwargs=kde_kwargs,
            plot_overlaid=plot_overlaid,
            plot_corrected=plot_corrected,
            figsize=figsize,
            col_wrap=col_wrap,
        )

    def umap_plot(
        self,
        n: int = 10000,
        hue: str = "sample_id",
        dim_reduction_kwargs: Optional[Dict] = None,
        figsize: Tuple[int] = (12, 7),
        legend: bool = False,
        features: Optional[List[str]] = None,
        **plot_kwargs,
    ):
        features = features or self.features
        features = [x for x in features if x in self.corrected_data.columns]
        if len(features) < 3:
            raise ValueError(f"Require more than 3 features; corrected features = {features}")
        super(FunctionalBatchCorrection, self).umap_plot(
            n=n,
            hue=hue,
            dim_reduction_kwargs=dim_reduction_kwargs,
            figsize=figsize,
            legend=legend,
            features=features,
            **plot_kwargs,
        )
