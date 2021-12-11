import logging
import os
from itertools import cycle
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union

import flowio
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from KDEpy import FFTKDE
from matplotlib.lines import Line2D

from cytopy import Experiment
from cytopy import Project
from cytopy.data import Panel
from cytopy.data import single_cell_dataframe
from cytopy.data import Subject
from cytopy.data.panel import Channel
from cytopy.feedback import progress_bar
from cytopy.plotting import single_cell_density
from cytopy.plotting.general import ColumnWrapFigure
from cytopy.utils import DimensionReduction
from cytopy.utils.transform import Scaler
from cytopy.utils.transform import Transformer
from cytopy.utils.transform import TRANSFORMERS

np.random.seed(42)
logger = logging.getLogger(__name__)


def _init_transformers(
    transform: Optional[Union[Transformer, str]] = None,
    transform_kwargs: Optional[Dict] = None,
    scale: Optional[Union[str, Scaler]] = None,
    scale_kwargs: Optional[Dict] = None,
):
    transform_kwargs = transform_kwargs or {}
    scale_kwargs = scale_kwargs or {}
    transformer, scaler = None, None
    if transform is not None:
        transformer = (
            transform if isinstance(transform, Transformer) else TRANSFORMERS.get(transform)(**transform_kwargs)
        )
    if scale is not None:
        scaler = scale if isinstance(scale, Scaler) else Scaler(method=scale, **scale_kwargs)
    return transformer, scaler


class BatchCorrector:
    def __init__(
        self,
        data: pd.DataFrame,
        features: List[str],
        transformer: Optional[Transformer] = None,
        scaler: Optional[Scaler] = None,
        verbose: bool = True,
    ):
        for x in features + ["sample_id", "subject_id"]:
            if x not in data.columns:
                raise KeyError(f"Data does not contain {x} column")

        self.original_data = data[~data[features].isnull().any(axis=1)].reset_index(drop=True)
        self._corrected_data = None
        self.features = features
        self.transformer = transformer
        self.scaler = scaler
        self.verbose = verbose
        self._umap_cache = None
        if self.transformer is not None:
            self.original_data = self.transformer.scale(data=self.original_data, features=self.features)
        if self.scaler is not None:
            self.original_data = self.scaler.fit_transform(data=data, features=features)

    @property
    def corrected_data(self):
        if self._corrected_data is None:
            raise ValueError("Data has not been batch corrected")
        return self._corrected_data

    @corrected_data.setter
    def corrected_data(self, data: pd.DataFrame):
        self._corrected_data = data

    @classmethod
    def from_experiment(
        cls,
        experiment: Experiment,
        population: str,
        features: List[str],
        transform: Union[Transformer, str] = "asinh",
        transform_kwargs: Optional[Dict] = None,
        scale: Optional[str] = None,
        scale_kwargs: Optional[Dict] = None,
        sample_size: Optional[int] = None,
        sampling_method: str = "uniform",
        sampling_level: str = "file",
        sample_ids: Optional[List[str]] = None,
        verbose: bool = True,
        **kwargs,
    ):
        logger.info("Loading data...")
        data = single_cell_dataframe(
            sample_ids=sample_ids,
            experiment=experiment,
            populations=population,
            sample_size=sample_size,
            sampling_method=sampling_method,
            sampling_level=sampling_level,
            transform=None,
        )
        transformer, scaler = _init_transformers(
            transform=transform, transform_kwargs=transform_kwargs, scale=scale, scale_kwargs=scale_kwargs
        )
        return cls(data=data, features=features, transformer=transformer, scaler=scaler, verbose=verbose, **kwargs)

    @classmethod
    def from_dataframe(
        cls,
        data: pd.DataFrame,
        features: List[str],
        transform: Union[Transformer, str] = "asinh",
        transform_kwargs: Optional[Dict] = None,
        scale: Optional[str] = None,
        scale_kwargs: Optional[Dict] = None,
        verbose: bool = True,
        **kwargs,
    ):
        transformer, scaler = _init_transformers(
            transform=transform, transform_kwargs=transform_kwargs, scale=scale, scale_kwargs=scale_kwargs
        )
        return cls(data=data, features=features, transformer=transformer, scaler=scaler, verbose=verbose, **kwargs)

    def add_meta_variable(self, key: Union[str, List[str]], var_name: str):
        meta = {}
        for subject in self.original_data.subject_id.unique():
            subject = Subject.objects(subject_id=subject).get()
            meta[subject.subject_id] = subject.lookup_var(key=key)
        self.original_data[var_name] = self.original_data.subject_id.apply(lambda x: meta.get(x, None))
        if self._umap_cache is not None:
            self._umap_cache[var_name] = self._umap_cache.subject_id.apply(lambda x: meta.get(x, None))
        try:
            self.corrected_data[var_name] = self.corrected_data.subject_id.apply(lambda x: meta.get(x, None))
        except ValueError:
            logger.warning(
                f"No corrected data obtained yet. Call 'add_meta_variable' again after batch correction "
                f"to match meta-variable to corrected data."
            )

    @staticmethod
    def _kde(data: pd.DataFrame, feature: str, x: Optional[np.ndarray] = None, **kde_kwargs):
        kde = {}
        x = x if x is not None else np.linspace(data[feature].min() - 0.01, data[feature].min() + 0.01, 1000)
        for sample_id, df in data.groupby("sample_id"):
            kde[sample_id] = FFTKDE(**kde_kwargs).fit(df[feature]).evaluate(x)
        return kde, x

    def plot_feature(
        self,
        feature: str,
        kde_kwargs: Optional[Dict] = None,
        plot_overlaid: bool = True,
        figsize: Optional[Tuple[int]] = None,
        plot_corrected: bool = True,
        col_wrap: int = 1,
    ):
        if feature not in self.features:
            raise KeyError(f"No such column {feature}")
        kde_kwargs = kde_kwargs or {}
        kde_kwargs["kernel"] = kde_kwargs.get("kernel", "gaussian")
        kde_kwargs["bw"] = kde_kwargs.get("bw", "ISJ")
        corrected_kde = {}
        logger.info(f"Computing KDE for original data")
        original_kde, x = self._kde(data=self.original_data, feature=feature, **kde_kwargs)

        if plot_corrected:
            logger.info(f"Computing KDE for corrected data")
            corrected_kde = self._kde(data=self.corrected_data, feature=feature, x=x, **kde_kwargs)

        if plot_overlaid:
            colours = cycle(sns.color_palette("tab20").as_hex())
            fig, axes = plt.subplots(figsize=figsize or (8, 5))
            for _id, y in original_kde.items():
                axes.plot(x, y, color=next(colours), linestyle="-")
                yc = corrected_kde.get(_id, None)
                if yc is not None:
                    axes.plot(x, yc, color=next(colours), linestyle="--")
        else:
            fig = ColumnWrapFigure(
                n=len(original_kde), col_wrap=col_wrap, figsize=figsize or (5, len(original_kde) * 1)
            )
            for _id, y in original_kde.items():
                ax = fig.add_subplot()
                ax.plot(x, y, color="black", linestyle="-")
                yc = corrected_kde.get(_id, None)
                if yc is not None:
                    ax.plot(x, yc, color="black", linestyle="--")
                ax.set_title(_id)
        fig.legend(
            [
                Line2D([0], [0], color="black", linestyle="-", linewidth=2),
                Line2D([0], [0], color="black", linestyle="--", linewidth=2),
            ],
            ["Original", "Corrected"],
            bbox_to_anchor=(1.05, 1.0),
        )
        return fig

    def _umap(self, n: int, features: List[str], **dim_reduction_kwargs):
        features = features or self.features
        reducer = DimensionReduction(method="UMAP", **dim_reduction_kwargs)
        logger.info("Performing dimension reduction on original data")
        before = reducer.fit_transform(data=self.original_data.sample(n), features=features)
        logger.info("Performing dimension reduction on batch corrected data")
        after = reducer.transform(data=self.corrected_data.iloc[before.index], features=features)
        logger.info("Plotting comparison")
        before["Source"] = "Before"
        after["Source"] = "After"
        self._umap_cache = pd.concat([before, after])
        return self._umap_cache

    def umap_plot(
        self,
        n: int = 10000,
        hue: str = "sample_id",
        density: bool = False,
        dim_reduction_kwargs: Optional[Dict] = None,
        figsize: Tuple[int] = (15, 7),
        legend: bool = False,
        features: Optional[List[str]] = None,
        overwrite_cache: bool = False,
        **plot_kwargs,
    ):
        dim_reduction_kwargs = dim_reduction_kwargs or {}
        if overwrite_cache or self._umap_cache is None:
            data = self._umap(n=n, features=features, **dim_reduction_kwargs)
        else:
            data = self._umap_cache
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        if density:
            single_cell_density(data=data[data.Source == "Before"], x="UMAP1", y="UMAP2", ax=axes[0], **plot_kwargs)
            single_cell_density(data=data[data.Source == "After"], x="UMAP1", y="UMAP2", ax=axes[1], **plot_kwargs)
        else:
            plot_kwargs["linewidth"] = plot_kwargs.get("linewidth", 0)
            plot_kwargs["s"] = plot_kwargs.get("s", 1)
            sns.scatterplot(
                data=data[data.Source == "Before"], x="UMAP1", y="UMAP2", ax=axes[0], hue=hue, **plot_kwargs
            )
            sns.scatterplot(
                data=data[data.Source == "After"], x="UMAP1", y="UMAP2", ax=axes[1], hue=hue, **plot_kwargs
            )
        axes[0].set_title("Before")
        axes[1].set_title("After")
        if not legend:
            for ax in axes:
                ax.legend().remove()
        return fig

    def save(self, project: Project, fcs_dir: str, experiment_id: str, suffix: str = "corrected"):
        corrected_data = self.corrected_data
        assert experiment_id not in project.list_experiments(), f"Experiment with ID {experiment_id} already exists!"
        if not os.path.isdir(fcs_dir):
            raise FileNotFoundError(f"Directory {fcs_dir} not found.")

        logger.info(f"Creating {experiment_id}...")
        exp = Experiment(experiment_id=experiment_id)
        exp.panel = Panel()
        for channel in self.features:
            exp.panel.channels.append(Channel(channel=channel, name=channel))
        exp.save()
        project.experiments.append(exp)
        project.save()

        subject_mappings = self.original_data[["sample_id", "subject_id"]]
        subject_mappings = dict(zip(subject_mappings.sample_id, subject_mappings.subject_id))

        logger.info(f"Saving corrected data to disk and associating to {experiment_id}")
        try:
            for sample_id, df in progress_bar(
                corrected_data.groupby("sample_id"),
                verbose=True,
                total=self.corrected_data.sample_id.nunique(),
            ):
                df = df[self.features]
                if self.scaler is not None:
                    df = self.scaler.inverse(data=df, features=self.features)
                if self.transformer is not None:
                    df = self.transformer.inverse_scale(data=df, features=self.features)
                filepath = os.path.join(fcs_dir, f"{sample_id}_{suffix}.fcs")
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
