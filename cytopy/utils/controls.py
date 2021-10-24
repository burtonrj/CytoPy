import logging
import os
from collections import defaultdict
from typing import Dict
from typing import Iterable
from typing import List
from typing import Optional
from typing import Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pingouin as pg
import seaborn as sns

from cytopy.data.experiment import Experiment
from cytopy.feedback import progress_bar
from cytopy.utils import DimensionReduction

logger = logging.getLogger(__name__)


def plot_embeddings(
    primary: pd.DataFrame,
    primary_idx: Iterable[int],
    ctrl: pd.DataFrame,
    ctrl_idx: Iterable[int],
    population_label: str,
    features: Optional[List[str]] = None,
    **kwargs,
) -> plt.Figure:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    primary["Population"] = "Other"
    primary.loc[primary_idx, ["Population"]] = population_label
    ctrl["Population"] = "Other"
    ctrl.loc[ctrl_idx, ["Population"]] = population_label

    features = features or primary.columns.tolist()
    reducer = DimensionReduction(method="UMAP", **kwargs)
    primary = reducer.fit_transform(data=primary, features=features)
    ctrl = reducer.transform(data=ctrl, features=features)

    for ax, df in zip(axes, [primary, ctrl]):
        for population, colour in zip(["Other", population_label], ["#4240CE", "#D53A3A"]):
            plt_df = df[df.Population == population]
            ax.scatter(
                plt_df["UMAP1"].values,
                plt_df["UMAP2"].values,
                s=4,
                c=colour,
                alpha=0.85,
                linewidth=0,
                edgecolors=None,
                label=population,
            )
    axes[0].legend().remove()
    axes[1].legend(bbox_to_anchor=(1.5, 1.0))
    lgd = axes[1].get_legend()
    for handle in lgd.legendHandles:
        handle.set_sizes([20.0])
    axes[0].set_title("Primary")
    axes[1].set_title("Control")
    return fig


def plot_distributions(primary: np.ndarray, ctrl: np.ndarray) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(5, 5))
    sns.kdeplot(primary, ax=ax, shade=True, color="#4240CE", label="Primary")
    sns.kdeplot(ctrl, ax=ax, shade=True, color="#D53A3A", label="Control")
    ax.axvline(np.quantile(ctrl, q=0.995), color="black", linestyle="--")
    ax.legend(bbox_to_anchor=(1.15, 1.0))
    return fig


def statistics(x: np.ndarray, y: np.ndarray, ctrl: str, var: str, sample_id: str):
    stats = {"sample_id": [sample_id], "ctrl": [ctrl], "var": [var]}
    datax = pd.DataFrame({"X": x})
    datax["Type"] = "Primary"
    datay = pd.DataFrame({"X": y})
    datay["Type"] = "Control"
    data = pd.concat([datax, datay]).reset_index(drop=True)

    stats["% > Control 95th quantile"] = x[np.where(x > np.quantile(y, q=0.95))[0]].shape[0] / x.shape[0] * 100
    stats["Normal"] = [pg.normality(data=data, dv="X", group="Type").iloc[0]["normal"]]
    stats["Equal var"] = [pg.homoscedasticity(data=data, dv="X", group="Type").iloc[0]["equal_var"]]
    ttest = pg.ttest(x, y, paired=False)
    ttest.rename(columns={c: f"{c} (T-test)" for c in ttest.columns}, inplace=True)
    ttest["sample_id"], ttest["ctrl"], ttest["var"] = sample_id, ctrl, var
    mwu = pg.mwu(x, y)
    mwu.rename(columns={c: f"{c} (MWU)" for c in ttest.columns}, inplace=True)
    mwu["sample_id"], mwu["ctrl"], mwu["var"] = sample_id, ctrl, var
    return (
        pd.DataFrame(stats).merge(ttest, on=["sample_id", "ctrl", "var"]).merge(mwu, on=["sample_id", "ctrl", "var"])
    )


class ControlComparison:
    def __init__(
        self,
        experiment: Experiment,
        ctrl: str,
        population: str,
        transform: str = "asinh",
        transform_kwargs: Optional[Dict] = None,
        meta_var: Optional[Union[str, List[str]]] = None,
    ):
        self.ctrl = ctrl
        self.population = population
        self.transform = transform
        self.transform_kwargs = transform_kwargs or {}
        self.filegroups = []
        self.meta_var = meta_var
        self.groups = defaultdict(list)
        for fg in experiment.fcs_files:
            if ctrl not in fg.file_paths.keys():
                logger.warning(f"{fg.primary_id} missing {ctrl} file.")
            elif population not in fg.list_populations(data_source="primary"):
                logger.warning(f"Population '{population}' missing in primary staining for {fg.primary_id}.")
            elif population not in fg.list_populations(data_source=ctrl):
                logger.warning(f"Population '{population}' missing in {ctrl} control staining for {fg.primary_id}.")
            else:
                self.filegroups.append(fg)
        self.create_groups()

    def filter_filegroups(self, filter_: Optional[List[str]] = None):
        return self.filegroups if filter_ is None else [f for f in self.filegroups if f.primary_id in filter_]

    def create_groups(self):
        for fg in self.filegroups:
            if not fg.subject:
                continue
            var = fg.subject.lookup_var(key=self.meta_var)
            if var is None:
                self.groups["undefined"].append(fg.primary_id)
            else:
                self.groups[var].append(fg.primary_id)

    def compare_embeddings(
        self,
        parent: str,
        sample_size: int,
        umap_kwargs: Optional[Dict] = None,
        features: Optional[List[str]] = None,
        filegroups: Optional[List[str]] = None,
        verbose: bool = True,
        plot_dir: Optional[str] = None,
    ):
        plot_dir = plot_dir or os.getcwd()
        filegroups = self.filter_filegroups(filter_=filegroups)
        umap_kwargs = umap_kwargs or {}
        if not os.path.isdir(plot_dir):
            raise FileNotFoundError(f"No such directory {plot_dir}")
        for fg in progress_bar(filegroups, verbose=verbose):
            if parent not in fg.list_populations(data_source=self.ctrl):
                logger.error(
                    f"Chosen parent {parent} does not exist for {self.ctrl} control "
                    f"in {fg.primary_id}. Skipping embedding plot"
                )
                continue
            primary_parent = fg.load_population_df(
                population=parent,
                transform=self.transform,
                data_source="primary",
                sample_size=sample_size,
            )
            primary_idx = fg.get_population(self.population, data_source="primary").index
            primary_idx = [i for i in primary_idx if i in primary_parent.index]
            ctrl_parent = fg.load_population_df(
                population=parent,
                transform=self.transform,
                data_source=self.ctrl,
                sample_size=sample_size,
            )
            ctrl_idx = fg.get_population(self.population, data_source=self.ctrl).index
            ctrl_idx = [i for i in ctrl_idx if i in ctrl_parent.index]
            fig = plot_embeddings(
                primary=primary_parent,
                primary_idx=primary_idx,
                ctrl=ctrl_parent,
                ctrl_idx=ctrl_idx,
                features=features,
                population_label=self.population,
                **umap_kwargs,
            )
            fig.savefig(os.path.join(plot_dir, f"{fg.primary_id}.png"), bbox_inches="tight")
            plt.close("all")

    def compare_distributions(
        self,
        var: Union[str, List[str]],
        plot_dir: Optional[str] = None,
        sample_size: int = 10000,
        filegroups: Optional[List[str]] = None,
        verbose: bool = True,
    ):
        plot_dir = plot_dir or os.getcwd()
        filegroups = self.filter_filegroups(filter_=filegroups)
        for fg in progress_bar(filegroups, verbose=verbose):
            x = fg.load_population_df(
                self.population,
                transform=self.transform,
                transform_kwargs=self.transform_kwargs,
                sample_size=sample_size,
                data_source="primary",
            )[var].values
            y = fg.load_population_df(
                self.population,
                transform=self.transform,
                transform_kwargs=self.transform_kwargs,
                sample_size=sample_size,
                data_source=self.ctrl,
            )[var].values
            fig = plot_distributions(primary=x, ctrl=y)
            fig.savefig(os.path.join(plot_dir, f"{fg.primary_id}_{var}.png"), bbox_inches="tight")
            plt.close("all")
