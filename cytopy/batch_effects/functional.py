from typing import Dict, Tuple
from typing import List
from typing import Optional
from typing import Union

import logging
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from KDEpy import FFTKDE
from skfda import FDataGrid

from cytopy import Project
from cytopy.data.experiment import Experiment
from cytopy.data.experiment import single_cell_dataframe
from cytopy.feedback import progress_bar
from cytopy.plotting.general import ColumnWrapFigure
from cytopy.utils import DimensionReduction
from cytopy.utils.transform import TRANSFORMERS

logger = logging.getLogger(__name__)


class LandmarkEditor(widgets.HBox):
    def __init__(
            self,
            x: np.ndarray,
            y: np.ndarray,
            landmarks: Optional[List] = None
    ):
        super().__init__()
        self.landmarks = landmarks or []
        output = widgets.Output()
        with output:
            self.fig, self.ax = plt.subplots(
                constrained_layout=True,
                figsize=(4, 2.5)
            )
        self.fig.canvas.toolbar_position = "bottom"
        self.ax.plot(x, y, color="black")
        self._plot_landmarks()

        self.landmark_text = widgets.Text(disabled=True)
        self.add_landmark_button = widgets.Button(
            description="Add landmark",
            button_style="info"
        )
        self.add_landmark_button.on_click(self._add_landmark)
        self.landmark_dropbox = widgets.Dropdown(
            description="Select landmark",
            options=self.landmarks
        )
        self.remove_landmark_button = widgets.Button(
            description="Remove landmark",
            button_style="warning"
        )
        self.remove_landmark_button.on_click(self._removed_landmark)

        self.close_button = widgets.Button(
            description="Close",
            button_style="warning"
        )
        self.close_button.on_click(lambda _: self.close())

        controls = widgets.VBox(
            [
                self.landmark_text,
                self.add_landmark_button,
                self.landmark_dropbox,
                self.remove_landmark_button
                self.close_button
            ]
        )
        controls.layout = widgets.Layout(
            border="solid 1px black",
            margin="0px 10px 10px 0px",
            padding="5px 5px 5px 5px"
        )
        _ = widgets.Box([output])
        output.layout = make_box_layout()

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


class FunctionalBatchCorrection:
    def __init__(
        self,
        experiment: Experiment,
        population: str,
        features: List[str],
        transform: str = "asinh",
        transform_kwargs: Optional[Dict] = None,
        grid_size: int = 100,
        kernel: str = "gaussian",
        bw: Union[str, float] = "ISJ",
        verbose: bool = True,
        **data_kwargs,
    ):
        self.original_data = single_cell_dataframe(
            experiment=experiment,
            populations=population,
            transform=transform,
            transform_kwargs=transform_kwargs,
            **data_kwargs,
        )[["sample_id", "subject_id"] + features]
        self.transformer = TRANSFORMERS.get(transform)(**transform_kwargs)
        self.corrected_data = None
        self.verbose = verbose
        self.features = features
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
            x=self._feature_data[feature]["kde"].grid_points[0],
            y=original[ref].values
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
            alignment = dtw(
                differential(ref_data),
                differential(self._feature_data[feature]["df"][_id].values)
            )
            xgrid = self._feature_data[feature]["kde"].grid_points[0]
            for lm in ref_landmarks:
                tmp = []
                idx = np.where(abs(xgrid - lm) == np.min(abs(xgrid - lm)))[0][0]
                tmp.append(np.mean(
                    xgrid[alignment.index2[alignment.index1 == idx]]
                ))
                landmarks.append(tmp)
        self._feature_data[feature]["landmarks"] = landmarks
        return self

    def plot_landmarks(self, feature: str):
        landmarks = self._feature_data[feature]["landmarks"]
        if landmarks is None:
            raise ValueError("Call 'collect_landmarks' first")
        original = self._feature_data[feature]["df"]
        fig = ColumnWrapFigure(
            n=original.shape[1],
            col_wrap=1,
            figsize=(5, original.shape[1] * 1)
        )
        x = self._feature_data[feature]["kde"].grid_points[0]
        for i, y_col in progress_bar(
                enumerate(original.columns),
                total=original.shape[1],
                verbose=self.verbose
        ):
            ax = fig.add_subplot()
            ax.plot(x, original[y_col], color="blue")
            for l in landmarks[i]:
                ax.axvline(l, ls="--", color="black")
            ax.set_title(f"{y_col} ({[round(l, 4) for l in landmarks[i]]})")
        fig.tight_layout()
        return fig

    def landmark_editor(self, feature: str, sample_id: str):
        landmarks = self._feature_data[feature]["landmarks"]
        if landmarks is None:
            raise ValueError("Call 'collect_landmarks' first")
        original = self._feature_data[feature]["df"]
        landmarks = pd.DataFrame(landmarks,
                                 index=self._feature_data[feature]["index"])
        landmarks = landmarks.loc[sample_id].values
        self._landmark_editor = LandmarkEditor(
            x=self._feature_data[feature]["kde"].grid_points[0],
            y=original[sample_id].values,
            landmarks=landmarks
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
            location=np.mean(self._feature_data[feature]["landmarks"], axis=0)
        )
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        self._feature_data[feature]["kde"].plot(axes=axes[0])
        axes[0].set_title("Before")
        self._feature_data[feature]["warping_functions"].plot(axes=axes[1])
        axes[1].set_title("Warping function")
        self._feature_data[feature]["kde"].compose(
            self._feature_data[feature]["warping_functions"]
        ).plot(axes=axes[2])
        axes[2].set_title("After")
        return fig

    def plot_corrections(self, feature: str):
        original = self._feature_data[feature]["df"]
        corrected = pd.DataFrame(
            self._feature_data[feature]["kde"]
                .compose(self._feature_data[feature]["warping_functions"])
                .data_matrix[:, :, 0],
            index=self._feature_data[feature]["index"])
        fig = ColumnWrapFigure(
            n=original.shape[1],
            col_wrap=1,
            figsize=(5, original.shape[1] * 1)
        )
        x = self._feature_data[feature]["kde"].grid_points[0]
        landmarks = np.mean(self._feature_data[feature]["landmarks"], axis=0)
        for i, y_col in progress_bar(
            enumerate(original.columns),
            total=original.shape[1],
            verbose=self.verbose
        ):
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

    def umap_plot(
            self,
            sample_n: int = 10000,
            figsize: Tuple[int] = (5, 5),
            legend: bool = False,
            dim_reduction_kwargs: Optional[Dict] = None,
            plot_kwargs: Optional[Dict] = None
    ):
        plot_kwargs = plot_kwargs or {}
        dim_reduction_kwargs = dim_reduction_kwargs or {}

        features = []
        for f in self.features:
            if f not in self.corrected_data.columns:
                logger.warning(f"{f} has not been corrected and will be omitted")
            else:
                features.append(f)
        if len(features) < 3:
            raise ValueError("umap_plot expect at least 3 features")

        plot_kwargs["hue"] = plot_kwargs.get("hue", "sample_id")
        plot_kwargs["linewidth"] = plot_kwargs.get("linewidth", 0)
        plot_kwargs["s"] = plot_kwargs.get("s", 1)
        reducer = DimensionReduction(method="UMAP", **dim_reduction_kwargs)
        data = self.original_data[features].sample(sample_n)
        before = reducer.fit_transform(data=data, features=features)
        after = reducer.transform(data=self.corrected_data.iloc[before.index], features=features)
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        sns.scatterplot(data=before, x="UMAP1", y="UMAP2", ax=axes[0], **plot_kwargs)
        sns.scatterplot(data=after, x="UMAP1", y="UMAP2", ax=axes[1], **plot_kwargs)
        axes[0].set_title("Before")
        axes[1].set_title("After")
        if not legend:
            for ax in axes:
                ax.legend().remove()
        return fig

    def save(
            self,
            project: Project,
            fcs_dir: str,
            experiment_id: str,
            suffix: str = "corrected",
            subject_mappings: Union[Dict[str, str], None] = None
    ):
        assert experiment_id not in project.list_experiments(), f"Experiment with ID {experiment_id} already exists!"
        if not os.path.isdir(fcs_dir):
            raise FileNotFoundError(f"Directory {fcs_dir} not found.")
        if any([x not in self.corrected_data.columns for x in self.features]):
            raise KeyError("One or more features have not been corrected")

        logger.info(f"Creating {experiment_id}...")
        exp = Experiment(experiment_id=experiment_id)
        exp.panel = Panel()
        for channel in self.features:
            exp.panel.channels.append(Channel(channel=channel, name=channel))
        exp.save()
        project.experiments.append(exp)
        project.save()

        subject_mappings = subject_mappings or {}
        if len(subject_mappings) == 0 and "subject_id" in self.corrected_data.columns:
            logger.info("Inferring subject mappings from data attribute")
            subject_mappings = self.corrected_data[["sample_id", "subject_id"]]
            subject_mappings = dict(zip(subject_mappings.sample_id, subject_mappings.subject_id))

        logger.info(f"Saving corrected data to disk and associating to {experiment_id}")
        try:
            for sample_id, df in progress_bar(
                self.corrected_data.groupby("sample_id"),
                verbose=True,
                total=self.corrected_data.sample_id.nunique(),
            ):
                df = df[self.features]
                if self.transformer is not None:
                    df = self.transformer.inverse_scale(data=df, features=self.features)
                filepath = os.path.join(fcs_dir, f"{prefix}_{sample_id}.fcs")
                with open(filepath, "wb") as f:
                    flowio.create_fcs(
                        event_data=df.to_numpy().flatten(),
                        file_handle=f,
                        channel_names=self.features,
                        opt_channel_names=self.features,
                    )
                exp.add_filegroup(
                    sample_id=sample_id,
                    paths={"primary": filepath},
                    compensate=False,
                    subject_id=subject_mappings.get(sample_id, None),
                )
            project.save()
        except (TypeError, ValueError, AttributeError, AssertionError) as e:
            logger.error("Failed to save data. Rolling back changes.")
            logger.exception(e)
            project.delete_experiment(experiment_id=experiment_id)
            project.save()


