from ...flow.dim_reduction import DimensionReduction
import matplotlib.pyplot as plt
import seaborn as sns
from typing import *
import pandas as pd
import numpy as np
import harmonypy
import logging

logger = logging.getLogger("calibrator")


class CalibrationError(Exception):
    def __init__(self, message: str):
        logger.error(message)
        super(self).__init__(message)


def resample(x: pd.DataFrame, y: pd.DataFrame):
    if y.shape[0] > x.shape[0]:
        n = y.shape[0] - x.shape[0]
        return pd.concat([x, x.sample(n)]), y
    elif y.shape[0] < x.shape[0]:
        return x.sample(y.shape[0]), y


def shared_features(x: pd.DataFrame, y: pd.DataFrame):
    return set(x.columns) == set(y.columns)


def check_called(func):
    def wrapper(*args, **kwargs):
        assert args[0].harmony is not None, "Class HarmonyCalibrator must be called first"
        return func(*args, **kwargs)
    return wrapper


class HarmonyCalibrator:
    """
    Use the Harmony algorithm, first described by Korsunsky et al [1] and implemented in Python by Kamil
    Slowikowski [2], to calibrate data for supervised classification - by calibration we mean alignment of
    some target dataframe (y) to training data (x) to negate batch effects.

    [1] Korsunsky, I., Millard, N., Fan, J. et al. Fast, sensitive and accurate integration of single-cell data
    with Harmony. Nat Methods 16, 1289–1296 (2019). https://doi.org/10.1038/s41592-019-0619-0
    [2] https://github.com/slowkow/harmonypy

    Parameters
    ----------
    x: Pandas.DataFrame
        Training DataFrame
    y: Pandas.DataFrame
        Target DataFrame

    Attributes
    ----------
    meta: Pandas.DataFrame
        Identifies origin of data points - used for alignment
    features: List[str]
        List of features
    data: Pandas.DataFrame
        Matrix of merged data before alignment
    harmony: harmonypy.Harmony
        Instance of Harmony fitted to data
    corrected: Pandas.DataFrame
        Merged DataFrame of aligned training and target data
    corrected_target: Pandas.DataFrame
        Corrected target data
    """
    def __init__(self, x: pd.DataFrame, y: pd.DataFrame):
        if not shared_features(x, y):
            logger.error("x and y must have the same columns")
            raise ValueError("x and y must have the same columns")
        x["sample_id"] = "ref"
        y["sample_id"] = "target"
        data = pd.DataFrame(resample(x, y)).reset_index()
        self.meta = data[["sample_id"]].copy()
        self.features = [x for x in data.columns if x != "sample_id"]
        self.data = data[self.features].astype(float).values
        self.harmony = None
        self.corrected = None

    @check_called
    @property
    def corrected_target(self):
        return (self.corrected[self.corrected.sample_id == "target"]
                .drop("sample_id", axis=1)
                .reset_index(drop=True))

    def _before_and_after(self):
        before = pd.DataFrame(self.data, columns=self.features)
        before["sample_id"] = self.meta["sample_id"]
        return before, self.corrected

    def overlay_plot(self,
                     dim_reduction_method: str = "UMAP",
                     dim_reduction_kwargs: Optional[Dict] = None,
                     figsize: Tuple[int, int] = (10, 5),
                     **kwargs):
        """
        Use dimension reduction to plot an overlay of the data before and
        after calibration.

        Parameters
        ----------
        dim_reduction_method: str (default="UMAP")
        dim_reduction_kwargs: Dict, optional
        figsize: Tuple (default=(10, 5))
        kwargs:
            Additional keyword arguments passed to seaborn.scatterplot

        Returns
        -------
        Matplotlib.Figure
        """
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        dim_reduction_kwargs = dim_reduction_kwargs or {}
        reducer = DimensionReduction(method=dim_reduction_method,
                                     **dim_reduction_kwargs)

        before, after = self._before_and_after()

        before = reducer.fit_transform(data=before, features=self.features)
        axes[0].set_title("Before")
        sns.scatterplot(data=before,
                        x=f"{dim_reduction_method}1",
                        y=f"{dim_reduction_method}2",
                        hue="sample_id",
                        ax=axes[0],
                        **kwargs)

        after = reducer.transform(after, features=self.features)
        axes[1].set_title("After")
        sns.scatterplot(data=after,
                        x=f"{dim_reduction_method}1",
                        y=f"{dim_reduction_method}2",
                        hue="sample_id",
                        ax=axes[0],
                        **kwargs)

        fig.tight_layout()
        return fig

    def lisi_distribution(self,
                          sample_size: int = 10000,
                          ax: Optional[plt.Axes] = None,
                          **kwargs):
        """
        Plot the distribution of LISI using the given meta_var as label

        Parameters
        ----------
        sample_size: float (default=10000)
            Number of events to sample prior to calculating LISI. Downsampling is recommended
            for large datasets, since LISI is computationally expensive
        ax: Matplotlib.Axes, optional
        kwargs:
            Additional keyword arguments passed to Seaborn.histplot

        Returns
        -------
        Matplotlib.Axes
        """
        before, after = self._before_and_after()
        if before.shape[0] > sample_size:
            before = before.sample(sample_size)
            after = after.sample(sample_size)

        before = harmonypy.lisi.compute_lisi(before[self.features],
                                             metadata=before[["sample_id"]],
                                             label_colnames=["sample_id"])
        after = harmonypy.lisi.compute_lisi(after[self.features],
                                            metadata=after[["sample_id"]],
                                            label_colnames=["sample_id"])
        plot_data = pd.DataFrame({"Before": before.reshape(-1),
                                  "After": after.reshape(-1)}).melt(var_name="Data", value_name="LISI")
        sns.histplot(data=plot_data, x="LISI", hue="Data", ax=ax, **kwargs)

    def __call__(self,
                 **kwargs) -> pd.DataFrame:
        """
        Align the given target to the reference using Harmony.

        Parameters
        ----------
        kwargs:
            Additional keyword arguments passed to harmonypy.Harmony

        Returns
        -------
        Pandas.DataFrame
        """
        logger.info(f"Calibrating data")
        self.harmony = harmonypy.run_harmony(data_mat=self.data,
                                             meta_data=self.meta,
                                             vars_use="sample_id",
                                             **kwargs)
        self.corrected = pd.DataFrame(self.harmony.Z_corr.T, columns=self.features)
        self.corrected["sample_id"] = self.meta["sample_id"]
        return (self.corrected[self.corrected.sample_id == "target"]
                .drop("sample_id", axis=1)
                .reset_index(drop=True))
