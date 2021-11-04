from typing import Dict
from typing import List
from typing import Optional
from typing import Union

import numpy as np
import pandas as pd
from KDEpy import FFTKDE
from skfda import FDataGrid

from cytopy.data.experiment import Experiment
from cytopy.data.experiment import single_cell_dataframe
from cytopy.feedback import progress_bar


class Landmarks:
    def __init__(self):
        pass

    def review(self):
        sample_id = "sep70_t1"
        x = lr.original_functions.grid_points[0]
        fig, ax = plt.subplots(figsize=(4, 2.5))
        ax.plot(x, original[sample_id])
        ax.axvline(landmarks.loc[sample_id][0])
        ax.axvline(landmarks.loc[sample_id][1])
        ax.set_title(f"{sample_id} ({[round(l, 4) for l in landmarks.loc[sample_id].values]})")
        plt.show()

        landmarks.loc[sample_id][0] = 5
        landmarks.loc[sample_id][1] = 6.6

    def plot_all(self):
        fig = ColumnWrapFigure(n=original.shape[1], col_wrap=1, figsize=(5, original.shape[1] * 1))
        for i, y_col in tqdm(enumerate(original.columns), total=original.shape[1]):
            ax = fig.add_subplot()
            ax.plot(x, original[y_col], color="blue")
            for l in landmarks[i]:
                ax.axvline(l, ls="--", color="black")
            ax.set_title(f"{y_col} ({[round(l, 4) for l in landmarks[i]]})")
        fig.tight_layout()
        fig

    def mean(self):
        np.mean(lr.landmarks, axis=0)


def kde(data: List[np.ndarray], grid_size: int, kernel: str, bw: Union[str, float]):
    x = np.linspace(np.min([np.min(x) for x in data]) - 0.1, np.max([np.max(x) for x in data]) + 0.1, grid_size)
    return FDataGrid([FFTKDE(kernel=kernel, bw=bw).fit(i).evaluate(x) for i in data], grid_points=x)


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
        )
        self.verbose = verbose
        self._feature_data = {}
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
            self._feature_data[f]["landmarks"] = self.collect_landmarks()

    def prepare_reference(self):
        dtw_distances = np.zeros((original.shape[1], original.shape[1]))
        for i, x in tqdm(enumerate(original.columns), total=original.shape[1]):
            for j, y in enumerate(original.columns):
                dtw_distances[i, j] = dtw(differential(original[x].values), differential(original[y].values)).distance
        dtw_distances = pd.DataFrame(dtw_distances, index=original.columns, columns=original.columns)
        ref = dtw_distances.mean().sort_values().index[0]
        x = lr.original_functions.grid_points[0]
        fig, ax = plt.subplots(figsize=(4, 2.5))
        ax.plot(x, original["sep243_t1"])
        ax.axvline(0.7)
        ax.axvline(5.5)
        plt.show()

    def collect_landmarks(self) -> Landmarks:
        xgrid = lr.original_functions.grid_points[0]
        ref = "sep243_t1"
        ref_landmarks = [0.7, 5.5]
        landmarks = []
        for _id in tqdm(original.columns, total=original.shape[1]):
            temp_landmarks = []
            if _id == ref:
                landmarks.append(ref_landmarks)
                continue
            alignment = dtw(
                differential(original[ref].values), differential(original[_id].values), window_type="itakura"
            )
            for l in ref_landmarks:
                nearest_idx = np.where(abs(xgrid - l) == np.min(abs(xgrid - l)))[0][0]
                temp_landmarks.append(np.mean(xgrid[alignment.index2[alignment.index1 == nearest_idx]]))
            landmarks.append(temp_landmarks)
        landmarks = pd.DataFrame(landmarks, index=original.columns)

    def landmark_registration(self):
        lr.warping_functions = landmark_registration_warping(
            lr.original_functions, np.array(landmarks.values), location=landmarks.mean()
        )
        lr.plot_warping()
        plt.show()

    def plot_corrections(self):
        fig = ColumnWrapFigure(n=original.shape[1], col_wrap=1, figsize=(5, original.shape[1] * 1))
        for i, y_col in tqdm(enumerate(original.columns), total=original.shape[1]):
            ax = fig.add_subplot()
            ax.plot(x, original[y_col], color="red")
            ax.plot(x, corrected[y_col], color="blue")
            ax.axvline(0.8258527, ls="--", color="blue")
            ax.axvline(5.42631849, ls="--", color="blue")
            ax.set_title(y_col)
        fig.tight_layout()
        fig

    def correct_channel(self):
        pass

    def plot_embeddings(self):
        pass

    def save(self):
        pass
