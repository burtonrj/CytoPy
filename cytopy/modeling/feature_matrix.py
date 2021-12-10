"""
1. The generation of a feature matrix needs improving, I should be able to:
    1. Choose the populations of interest
    2. Choose how I wish to represent those populations - N events, frac of parent, ratio to some other pop
    3. I should be able to easily append meta data of varying complexity
2. Feature selection should be linear in the form of: filter → wrapper (feature importance) → embedded selection (Lasso) → partial dependence and feature interaction.
3. Dimension reduction should be focused on feature contribution in embedding
"""
import re
from collections import defaultdict
from typing import *

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pingouin as pg
import seaborn as sns
from mlxtend.evaluate import permutation_test

from ..feedback import progress_bar
from .hypothesis_testing import hypothesis_test
from cytopy.data.experiment import FileGroup
from cytopy.data.project import Project


def get_population_as_perc_of(files: List[FileGroup], population: str, stat: str, column_name: str):
    pattern = re.compile("% of (.*)")
    parent_name = pattern.search(stat).group(1)
    data = defaultdict(list)
    for fg in files:
        if parent_name == "parent":
            parent_name = fg.get_population(population).parent
        elif parent_name == "grandparent":
            parent_name = fg.get_population(fg.get_population(population).parent).parent
        parent_n = fg.population_stats(population=parent_name)["n"]
        if parent_n == 0:
            data["subject_id"].append(fg.subject.subject_id)
            data[column_name].append(0)
        else:
            pop_n = fg.population_stats(population=population)["n"]
            data["subject_id"].append(fg.subject.subject_id)
            data[column_name].append(pop_n / parent_n * 100)
    return pd.DataFrame(data)


def get_population_n(files: List[FileGroup], population: str, column_name: str):
    data = defaultdict(list)
    for fg in files:
        data["sample_id"].append(fg.primary_id)
        data[column_name].append(fg.population_stats(population=population)["n"])
    return pd.DataFrame(data)


class FeatureSpace:
    def __init__(self, project: Project, verbose: bool = True, data: Optional[pd.DataFrame] = None):
        self.project = project
        self.verbose = verbose
        self.subjects = self.project.list_subjects()
        if data is not None:
            if "subject_id" not in data.columns:
                raise KeyError("DataFrame must contain column 'subject_id'")
            self._data = data
        else:
            self._data = pd.DataFrame({"subject_id": self.subjects})

    @classmethod
    def from_csv(cls, project: Project, csv_path: str, verbose: bool = True):
        return cls(project=project, data=pd.read_csv(csv_path), verbose=verbose)

    @classmethod
    def from_dataframe(cls, project: Project, data: pd.DataFrame, verbose: bool = True):
        return cls(project=project, data=data, verbose=verbose)

    @property
    def data(self):
        return self._data

    @data.setter
    def data(self, new_data: pd.DataFrame):
        self._data = self._data.merge(new_data, on=["subject_id"], how="outer")

    def __setitem__(self, key, value):
        self._data[key] = value

    def __getitem__(self, item):
        return self._data[item]

    def drop(self, key: Union[str, List[str]], axis: int = 1):
        self._data.drop(key, axis=axis, inplace=True)
        return self

    def subset(self, query: str):
        self._data = self._data.query(query).copy()

    def to_csv(self, csv_path: str, **kwargs):
        kwargs = kwargs or {}
        kwargs["index"] = kwargs.get("index", False)
        self.data.to_csv(csv_path, **kwargs)

    def add_subject_variable(self, column_name: str, key: Union[str, Iterable[str]], summary: str = "mean"):
        values = [
            self.project.get_subject(subject_id=subject).lookup_var(key=key, summary=summary)
            for subject in self.subjects
        ]
        self._data[column_name] = values
        return self

    def add_many_subject_variables(self, column_key: Dict[str, Union[str, Iterable[str]]], summary: str = "mean"):
        for column, key in column_key.items():
            self.add_subject_variable(column_name=column, key=key, summary=summary)
        return self

    def list_experiment_populations(self, experiment: str, **kwargs):
        return self.project.get_experiment(experiment_id=experiment).list_populations(**kwargs)

    def add_population_statistics(
        self, experiment: Union[str, List[str]], population: str, stat: str, prefix: Optional[str] = None
    ):
        column_name = f"{prefix}_{population}_{stat}" if prefix else f"{population}_{stat}"
        if isinstance(experiment, str):
            exp = self.project.get_experiment(experiment_id=experiment)
            if "% of " in stat:
                data = get_population_as_perc_of(
                    files=exp.fcs_files, population=population, stat=stat, column_name=column_name
                )
                self.data = data
            elif stat == "n":
                data = get_population_n(files=exp.fcs_files, population=population, column_name=column_name)
                self.data = data
            else:
                raise ValueError(
                    "stat must be either '% of parent', '% of grandparent', 'n' or '% of NAME' where "
                    "'NAME' is the name of a valid population."
                )
            return self
        else:
            data = []
            for exp in experiment:
                exp = self.project.get_experiment(experiment_id=exp)
                if "% of " in stat:
                    data.append(
                        get_population_as_perc_of(
                            files=exp.fcs_files, population=population, stat=stat, column_name=column_name
                        )
                    )
                elif stat == "n":
                    data.append(get_population_n(files=exp.fcs_files, population=population, column_name=column_name))
                else:
                    raise ValueError(
                        "stat must be either '% of parent', '% of grandparent', 'n' or '% of NAME' where "
                        "'NAME' is the name of a valid population."
                    )
            data = pd.concat(data).reset_index(drop=True)
            data = data.groupby("subject_id")[column_name].mean().reset_index()
            self.data = data
            return self

    def add_ratio(self, column_name: str, x1: str, x2: str):
        for x in [x1, x2]:
            if x not in self.data.columns:
                raise KeyError(f"{x} is not a valid column")
        self._data[column_name] = self.data[x1] / self.data[x2]
        return self

    def replace_columns_with_average(self, column_name: str, columns: List[str], drop: bool = False):
        self._data[column_name] = self.data[columns].mean(axis=1)
        if drop:
            self._data.drop(columns, axis=1, inplace=True)
        return self

    def add_embedded_subject_variable(self, column_name: str, key: str, id_var: str, value_var: str):
        data = []
        for subject in self.project.subjects:
            subject_data = []
            if key in subject.fields:
                df = subject.field_to_df(key)
                values = {}
                for _id, value in df[[id_var, value_var]].values:
                    values[f"{_id}_{column_name}"] = value
                subject_data.append(values)
            subject_data = pd.DataFrame(subject_data)
            subject_data["subject_id"] = subject.subject_id
            data.append(subject_data)
        self.data = pd.concat(data).reset_index(drop=True)
        return self

    def encode_categorical(self, column: str):
        self._data[column] = self._data[column].astype("category")
        self._data[column] = self._data[column].cat.codes
        self._data[column] = self._data[column].apply(lambda x: None if x == -1 else x)
        return self

    def rank_variance(self):
        data = self.data.set_index("subject_id")
        data = data / data.mean()
        return (
            pd.DataFrame(data, columns=data.columns)
            .var()
            .sort_values()
            .reset_index()
            .rename(columns={"index": "Feature", 0: "var"})
        )

    def binary_pval(
        self,
        endpoint: str,
        multicomp_alpha: float = 0.05,
        multicomp_method: str = "fdr_bh",
        features: Optional[List[str]] = None,
        **kwargs,
    ):
        if self.data[endpoint].nunique() != 2:
            raise ValueError("Endpoint must be binary")
        x1, x2 = self.data[endpoint].unique()
        features = features or [x for x in self.data.columns if x not in ["subject_id", endpoint]]
        results = []
        for f in progress_bar(features, verbose=self.verbose):
            x = self.data[self.data[endpoint] == x1][f].dropna().values
            y = self.data[self.data[endpoint] == x2][f].dropna().values
            if len(x) < 3:
                continue
            if len(y) < 3:
                continue
            results.append(
                {
                    "feature": f,
                    "pval": permutation_test(x=x, y=y, **kwargs),
                    "CLES": pg.compute_effsize(x=x, y=y, paired=False, eftype="CLES"),
                }
            )
        data = pd.DataFrame(results)
        data["corrected_pval"] = pg.multicomp(
            pvals=data["pval"].values, method=multicomp_method, alpha=multicomp_alpha
        )[1]
        return data

    def multi_pval(
        self,
        endpoint: str,
        multicomp_alpha: float = 0.05,
        multicomp_method: str = "fdr_bh",
        features: Optional[List[str]] = None,
        **kwargs,
    ):
        if self.data[endpoint].nunique() <= 2:
            raise ValueError("Endpoint must have more than 2 unique values")
        features = features or [x for x in self.data.columns if x not in ["subject_id", endpoint]]
        results = []
        for f in progress_bar(features, verbose=self.verbose):
            results.append(
                {
                    "feature": f,
                    "pval": hypothesis_test(
                        data=self.data[[endpoint, f]], dv=f, between_group=endpoint, **kwargs
                    ).iloc[0]["p-unc"],
                }
            )
        data = pd.DataFrame(results)
        data["corrected_pval"] = pg.multicomp(pvals=data["pval"], method=multicomp_method, alpha=multicomp_alpha)[1]
        return data

    def correlation_matrix(self, features: List[str], method: str = "spearman", **kwargs) -> sns.FacetGrid:
        kwargs = kwargs or {}
        kwargs["vmin"] = kwargs.get("vmin", -1)
        kwargs["vmax"] = kwargs.get("vmax", 1)
        kwargs["figsize"] = kwargs.get("figsize", (8.5, 8.5))
        kwargs["cmap"] = kwargs.get("cmap", "RdBu_r")
        kwargs["cbar_pos"] = kwargs.get("cbar_pos", (0.85, 1, 0.1, 0.025))
        kwargs["cbar_kws"] = kwargs.get("cbar_kws", {"orientation": "horizontal"})
        kwargs["dendrogram_ratio"] = kwargs.get("dendrogram_ratio", 0.05)

        return sns.clustermap(self.data[features].corr(method=method), **kwargs)

    def vif(self, features: List[str], **kwargs) -> pd.DataFrame:
        kwargs = kwargs or {}
        kwargs["remove_na"] = kwargs.get("remove_na", True)
        vif = []
        for f in features:
            x = self.data[[i for i in features if i != f]].values
            y = self.data[f].values
            lr = pg.linear_regression(x, y, **kwargs)
            vif.append({"feature": f, "VIF": 1 / (1 - lr.loc[1]["r2"])})
        return pd.DataFrame(vif)

    def high_correlates(
        self, features: List[str], threshold: float = 0.5, method: str = "spearman"
    ) -> Dict[str, List[str]]:
        high_correlates = defaultdict(list)
        for feature, row in self.data[features].corr(method=method).iterrows():
            for x in features:
                if x == feature:
                    continue
                if abs(row[x]) > threshold:
                    high_correlates[feature].append(x)
        return high_correlates

    @staticmethod
    def _lr_r2_pval(data: pd.DataFrame, x: str, y: str, **kwargs):
        linreg = pg.linear_regression(data[x], data[y], **kwargs).set_index("names")
        p = round(linreg.loc[x].pval, 3)
        p = f"={p}" if p > 0 else "<0.0001"
        return "$r^{2}$=" + f"{round(linreg.loc[x].r2, 3)}, p{p}"

    def plot_correlation(
        self,
        x: str,
        y: str,
        group_var: Optional[str] = None,
        legend_kwargs: Optional[Dict] = None,
        xlabel: Optional[str] = None,
        ylabel: Optional[str] = None,
        ax: Optional[plt.Axes] = None,
        lr_kwargs: Optional[Dict] = None,
        **kwargs,
    ):
        data = self.data[~(self.data[x].isnull() | self.data[y].isnull())]
        legend_kwargs = legend_kwargs or {}
        lr_kwargs = lr_kwargs or {}
        ax = ax if ax is not None else plt.subplots(figsize=(5, 5))[1]
        if group_var:
            for grp_id, group_df in data.groupby(group_var):
                r2_pval = self._lr_r2_pval(data=group_df, x=x, y=y, **lr_kwargs)
                sns.regplot(data=group_df, x=x, y=y, ax=ax, label=f"{grp_id} ({r2_pval})", **kwargs)
        else:
            r2_pval = self._lr_r2_pval(data=data, x=x, y=y, **lr_kwargs)
            sns.regplot(data=data, x=x, y=y, ax=ax, label=r2_pval, **kwargs)
        ax.legend(**legend_kwargs)
        if xlabel:
            ax.set_xlabel(xlabel)
        if ylabel:
            ax.set_ylabel(ylabel)
        return ax

    def percentage_missing_data(self, subset: Optional[str] = None):
        if subset:
            data = self.data.query(subset)
        else:
            data = self.data
        return (
            (data.isnull().sum() / data.shape[0] * 100)
            .reset_index()
            .rename(columns={"index": "feature", 0: "% missing"})
            .sort_values("% missing", ascending=False)
        )

    def missing_drop_above_threshold(self, threshold: float):
        missing = self.percentage_missing_data()
        drop = missing[missing["% missing"] > threshold]["features"].values
        return self.drop(key=drop, axis=1)
