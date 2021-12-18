"""
1. The generation of a feature matrix needs improving, I should be able to:
    1. Choose the populations of interest
    2. Choose how I wish to represent those populations - N events, frac of parent, ratio to some other pop
    3. I should be able to easily append meta data of varying complexity
2. Feature selection should be linear in the form of: filter → wrapper (feature importance) → embedded selection (Lasso) → partial dependence and feature interaction.
3. Dimension reduction should be focused on feature contribution in embedding
"""
import logging
import re
from collections import defaultdict
from copy import deepcopy
from functools import reduce
from typing import Dict
from typing import Iterable
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union

import matplotlib.pyplot as plt
import miceforest as mf
import numpy as np
import pandas as pd
import pingouin as pg
import seaborn as sns
from mlxtend.evaluate import permutation_test
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

from ..feedback import progress_bar
from ..plotting.general import box_swarm_plot
from ..plotting.general import ColumnWrapFigure
from .hypothesis_testing import hypothesis_test
from cytopy.data.fcs import FileGroup
from cytopy.data.project import Project

logger = logging.getLogger(__name__)


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
        data["subject_id"].append(fg.subject.subject_id)
        data[column_name].append(fg.population_stats(population=population)["n"])
    return pd.DataFrame(data)


class DuplicateColumnError(Exception):
    pass


class FeatureSpace:
    def __init__(self, project: Project, verbose: bool = True, data: Optional[Dict[str, pd.DataFrame]] = None):
        self.data = {}
        self.project = project
        self.verbose = verbose
        self.subjects = self.project.list_subjects()
        self.train_idx, self.test_idx, self.target = None, None, None
        if data is not None:
            if not all([isinstance(df, pd.DataFrame) for df in data.values()]):
                raise ValueError("All elements of data must be a Pandas DataFrame")
            for df in data.values():
                self._has_subject_id(df)
            self.data = {k: df.set_index("subject_id") for k, df in data.items()}

    @classmethod
    def from_excel(cls, project: Project, filepath: str, verbose: bool = True, **kwargs):
        data = {}
        with pd.ExcelFile(filepath) as xls:
            for sheet_name in xls.sheet_names:
                data[sheet_name] = pd.read_excel(xls, sheet_name=sheet_name, **kwargs)
        return cls(project=project, data=data, verbose=verbose)

    @property
    def sheet_names(self) -> List[str]:
        return list(self.data.keys())

    @property
    def sheet_features(self) -> Dict[str, List[str]]:
        features = {}
        for sheet_name, df in self.data.items():
            features[sheet_name] = df.columns.tolist()
        return features

    @property
    def all_features(self) -> List[str]:
        features = []
        for df in self.data.values():
            features = features + df.columns.tolist()
        return features

    def lookup_sheet_name(self, column_name: str) -> str:
        for sheet_name, columns in self.sheet_features.items():
            if column_name in columns:
                return sheet_name
        raise KeyError(f"No such column {column_name}")

    @staticmethod
    def _has_subject_id(data: pd.DataFrame):
        if "subject_id" not in data.columns:
            raise KeyError("DataFrame must contain column 'subject_id'")

    def _duplicate_column_check(self, data: Optional[pd.DataFrame] = None, column: Optional[str] = None) -> None:
        if data is not None:
            for col in data.columns.tolist():
                for sheet, columns in self.sheet_features.items():
                    if col in columns:
                        raise DuplicateColumnError(f"{col} already exists in sheet {sheet}")
            return
        if column is not None:
            for sheet, columns in self.sheet_features.items():
                if column in columns:
                    raise DuplicateColumnError(f"{column} already exists in sheet {sheet}")
            return
        raise ValueError("Must provide either dataframe or column")

    def _data_exists(self):
        if len(self.data) == 0:
            raise ValueError(f"Data has not been populated")

    def _sheet_exists(self, sheet_name: str):
        if sheet_name not in self.data.keys():
            raise ValueError(f"No such sheet {sheet_name}")

    def _column_exists(self, sheet_name: str, column: str):
        self._sheet_exists(sheet_name=sheet_name)
        if column not in self.sheet_features[sheet_name]:
            raise ValueError(f"No such column {column} in sheet {sheet_name}")

    def add_dataframe_as_sheet(self, sheet_name: str, data: pd.DataFrame, overwrite: bool = True):
        self._has_subject_id(data=data)
        if sheet_name in self.data.keys():
            if overwrite:
                logger.warning(f"{sheet_name} already exists, data will be overwritten")
            else:
                raise ValueError(f"{sheet_name} already exists, set 'overwrite' to True to overwrite data")
        self._duplicate_column_check(data=data)
        self.data[sheet_name] = data
        return self

    def merge_dataframe_into_sheet(self, sheet_name: str, data: pd.DataFrame):
        self._has_subject_id(data=data)
        self._sheet_exists(sheet_name=sheet_name)
        self._duplicate_column_check(data=data)
        self.data[sheet_name] = self.data[sheet_name].merge(
            data.set_index("subject_id"), right_index=True, left_index=True, how="outer"
        )
        return self

    def new_sheet(self, sheet_name: str):
        self.data[sheet_name] = pd.DataFrame(index=pd.Series(data=self.subjects, name="subject_id"))
        return self

    def as_dataframe(
        self,
        features: List[str],
        dataset: str = "original",
        subset: Optional[str] = None,
        subset_features: Optional[List[str]] = None,
        **kwargs,
    ) -> Union[pd.DataFrame, List[pd.DataFrame]]:
        if isinstance(features, str):
            features = [features]
        if self.train_idx is None and dataset != "original":
            raise ValueError("Must call stratified_kfold first before requesting train/test data")
        self._data_exists()
        data = []
        sheets = [x for x in features if x in self.sheet_names]
        features = [x for x in features if x not in sheets]
        if subset:
            if subset_features is None:
                raise ValueError("Must provide a list of features to subset on")
            features = features + subset_features
        if sheets:
            data = [df.copy() for sheet_name, df in self.data.items() if sheet_name in sheets]
            features = [x for x in features if not any([x in self.sheet_features[s] for s in sheets])]
        for df in self.data.values():
            columns = [x for x in features if x in df.columns.tolist()]
            if columns:
                data.append(df[columns].copy())
            features = [x for x in features if x not in columns]
        if len(features) > 0:
            logger.warning(f"The following features/sheet names were not recognised: {features}")
        data = [df for df in data if df.shape[1] > 0]
        if len(data) > 1:
            data = reduce(
                lambda left, right: pd.merge(left, right, how="outer", left_index=True, right_index=True), data
            )
        else:
            data = data[0]
        if subset:
            data = data.query(subset, **kwargs).copy()
            data.drop(subset_features, axis=1, inplace=True)
        if dataset == "original":
            return data
        if dataset == "train":
            return [data.iloc[i] for i in self.train_idx]
        if dataset == "test":
            return [data.iloc[i] for i in self.test_idx]
        raise ValueError("dataset should be one of 'original', 'train' or 'test'")

    def save(self, filepath: str, **kwargs):
        with pd.ExcelWriter(filepath) as writer:
            for sheet_name, df in self.data.items():
                df.to_excel(writer, sheet_name=sheet_name, **kwargs)
        return self

    def __getitem__(self, sheet_name: str) -> pd.DataFrame:
        self._sheet_exists(sheet_name=sheet_name)
        return self.data[sheet_name]

    def drop(self, sheet_name: str, key: Union[str, List[str]], axis: int = 1):
        self._sheet_exists(sheet_name=sheet_name)
        self.data[sheet_name].drop(key, axis=axis, inplace=True)
        return self

    def subset(self, sheet_name: str, query: str, **kwargs):
        self._sheet_exists(sheet_name=sheet_name)
        self.data[sheet_name] = self.data[sheet_name].query(query, **kwargs).copy()
        return self

    def overwrite_column(self, sheet_name: str, column: str, values: Iterable):
        self._column_exists(sheet_name=sheet_name, column=column)
        self.data[sheet_name][column] = values
        return self

    def add_column(self, sheet_name: str, column_name: str, values: Iterable):
        self._sheet_exists(sheet_name=sheet_name)
        self._duplicate_column_check(column=column_name)
        self.data[sheet_name][column_name] = values

    def add_subject_variable(
        self,
        sheet_name: str,
        column_name: str,
        key: Union[str, Iterable[str]],
        summary: str = "mean",
        overwrite: bool = False,
    ):
        self._sheet_exists(sheet_name)
        if not overwrite:
            self._duplicate_column_check(column=column_name)
        values = [
            self.project.get_subject(subject_id=subject).lookup_var(key=key, summary=summary)
            for subject in self.subjects
        ]
        self.data[sheet_name][column_name] = values
        return self

    def add_many_subject_variables(
        self,
        sheet_name: str,
        column_key: Dict[str, Union[str, Iterable[str]]],
        summary: str = "mean",
        overwrite: bool = False,
    ):
        for column, key in column_key.items():
            self.add_subject_variable(
                sheet_name=sheet_name, column_name=column, key=key, summary=summary, overwrite=overwrite
            )
        return self

    def list_experiment_populations(self, experiment: str, **kwargs):
        return self.project.get_experiment(experiment_id=experiment).list_populations(**kwargs)

    def add_population_statistics(
        self,
        sheet_name: str,
        experiment: Union[str, List[str]],
        population: str,
        stat: str,
        prefix: Optional[str] = None,
        overwrite: bool = False,
    ):
        self._sheet_exists(sheet_name)
        column_name = f"{prefix}_{population}_{stat}" if prefix else f"{population}_{stat}"
        if not overwrite:
            self._duplicate_column_check(column=column_name)
        data = []
        if isinstance(experiment, str):
            experiment = [experiment]
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
        data = data.groupby("subject_id")[column_name].mean()
        self.data[sheet_name] = self.data[sheet_name].merge(data, left_index=True, right_index=True, how="outer")
        return self

    def add_ratio(self, sheet_name: str, column_name: str, x1: str, x2: str, overwrite: bool = False):
        self._sheet_exists(sheet_name)
        if not overwrite:
            self._duplicate_column_check(column=column_name)
        for x in [x1, x2]:
            if x not in self.data[sheet_name].columns:
                raise KeyError(f"{x} is not a valid column")
        self.data[sheet_name][column_name] = self.data[x1] / self.data[x2]
        return self

    def replace_columns_with_average(
        self, sheet_name: str, column_name: str, columns: List[str], drop: bool = False, overwrite: bool = False
    ):
        self._sheet_exists(sheet_name)
        if not overwrite:
            self._duplicate_column_check(column=column_name)
        self.data[sheet_name][column_name] = self.data[columns].mean(axis=1)
        if drop:
            self.data[sheet_name].drop(columns, axis=1, inplace=True)
        return self

    def add_embedded_subject_variable(
        self, sheet_name: str, column_name: str, key: str, id_var: str, value_var: str, overwrite: bool = False
    ):
        self._sheet_exists(sheet_name)
        if not overwrite:
            self._duplicate_column_check(column=column_name)
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
        data = pd.concat(data).reset_index(drop=True).set_index("subject_id")
        self.data[sheet_name] = self.data[sheet_name].merge(data, left_index=True, right_index=True, how="outer")
        return self

    def encode_categorical(self, sheet_name: str, column_name: str):
        self._column_exists(sheet_name=sheet_name, column=column_name)
        self.data[sheet_name][column_name] = self.data[sheet_name][column_name].astype("category")
        self.data[sheet_name][column_name] = self.data[sheet_name][column_name].cat.codes
        self.data[sheet_name][column_name] = self.data[sheet_name][column_name].apply(lambda x: None if x == -1 else x)
        return self

    def rank_variance(
        self,
        features: List[str],
        subset: Optional[str] = None,
        subset_features: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        data = self.as_dataframe(features=features, subset=subset, subset_features=subset_features)
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
        features: List[str],
        target: str,
        multicomp_alpha: float = 0.05,
        multicomp_method: str = "fdr_bh",
        subset: Optional[str] = None,
        subset_features: Optional[List[str]] = None,
        **kwargs,
    ):
        data = self.as_dataframe(features=features, subset=subset, subset_features=subset_features)
        if data[target].nunique() != 2:
            raise ValueError("Endpoint must be binary")
        x1, x2 = data[target].dropna().unique()
        features = [x for x in data.columns.tolist() if x != target]
        results = []
        for f in progress_bar(features, verbose=self.verbose):
            x = data[data[target] == x1][f].dropna().values
            y = data[data[target] == x2][f].dropna().values
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
        features: List[str],
        target: str,
        multicomp_alpha: float = 0.05,
        multicomp_method: str = "fdr_bh",
        subset: Optional[str] = None,
        subset_features: Optional[List[str]] = None,
        **kwargs,
    ):
        data = self.as_dataframe(features=features, subset=subset, subset_features=subset_features)
        if self.data[target].nunique() <= 2:
            raise ValueError("Endpoint must have more than 2 unique values")
        features = [x for x in data.columns.tolist() if x != target]
        results = []
        for f in progress_bar(features, verbose=self.verbose):
            results.append(
                {
                    "feature": f,
                    "pval": hypothesis_test(data=self.data[[target, f]], dv=f, between_group=target, **kwargs).iloc[0][
                        "p-unc"
                    ],
                }
            )
        data = pd.DataFrame(results)
        data["corrected_pval"] = pg.multicomp(pvals=data["pval"], method=multicomp_method, alpha=multicomp_alpha)[1]
        return data

    def correlation_matrix(
        self,
        features: List[str],
        method: str = "spearman",
        subset: Optional[str] = None,
        subset_features: Optional[List[str]] = None,
        **kwargs,
    ) -> sns.FacetGrid:
        data = self.as_dataframe(features=features, subset=subset, subset_features=subset_features)
        kwargs = kwargs or {}
        kwargs["vmin"] = kwargs.get("vmin", -1)
        kwargs["vmax"] = kwargs.get("vmax", 1)
        kwargs["figsize"] = kwargs.get("figsize", (8.5, 8.5))
        kwargs["cmap"] = kwargs.get("cmap", "RdBu_r")
        kwargs["cbar_pos"] = kwargs.get("cbar_pos", (0.85, 1, 0.1, 0.025))
        kwargs["cbar_kws"] = kwargs.get("cbar_kws", {"orientation": "horizontal"})
        kwargs["dendrogram_ratio"] = kwargs.get("dendrogram_ratio", 0.05)
        return sns.clustermap(data.corr(method=method), **kwargs)

    def vif(
        self, features: List[str], subset: Optional[str] = None, subset_features: Optional[List[str]] = None, **kwargs
    ) -> pd.DataFrame:
        data = self.as_dataframe(features=features, subset=subset, subset_features=subset_features)
        kwargs = kwargs or {}
        kwargs["remove_na"] = kwargs.get("remove_na", True)
        vif = []
        data = pd.DataFrame(StandardScaler().fit_transform(data), columns=features)
        for f in features:
            x = data[[i for i in features if i != f]].values
            y = data[f].values
            lr = pg.linear_regression(x, y, **kwargs)
            vif.append({"feature": f, "VIF": 1 / (1 - lr.loc[1]["adj_r2"])})
        return pd.DataFrame(vif)

    def high_correlates(
        self,
        features: List[str],
        threshold: float = 0.5,
        method: str = "spearman",
        subset: Optional[str] = None,
        subset_features: Optional[List[str]] = None,
    ) -> Dict[str, List[str]]:
        data = self.as_dataframe(features=features, subset=subset, subset_features=subset_features)
        high_correlates = defaultdict(list)
        for feature, row in data.corr(method=method).iterrows():
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
        standard_scale: bool = False,
        subset: Optional[str] = None,
        subset_features: Optional[List[str]] = None,
        **kwargs,
    ):
        data = self.as_dataframe(features=[x, y, group_var], subset=subset, subset_features=subset_features)
        data = data[~(data[x].isnull() | data[y].isnull())].copy()
        if standard_scale:
            data[[x, y]] = StandardScaler().fit_transform(data[[x, y]])
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

    def percentage_missing_data(
        self,
        features: List[str],
        subset: Optional[str] = None,
        subset_features: Optional[List[str]] = None,
    ):
        data = self.as_dataframe(features=features, subset=subset, subset_features=subset_features)
        return (
            (data.isnull().sum() / data.shape[0] * 100)
            .reset_index()
            .rename(columns={"index": "feature", 0: "% missing"})
            .sort_values("% missing", ascending=False)
        )

    def missing_drop_above_threshold(
        self,
        features: List[str],
        threshold: float,
        subset: Optional[str] = None,
        subset_features: Optional[List[str]] = None,
    ):
        missing = self.percentage_missing_data(features=features, subset=subset, subset_features=subset_features)
        drop = missing[missing["% missing"] > threshold]["feature"].values
        return self.drop(key=drop, axis=1)

    def stratified_kfold(self, target: str, n_splits: int = 5, random_state: int = 42):
        self.train_idx, self.test_idx, self.target = [], [], target
        if target not in self.all_features:
            raise KeyError(f"{target} is not a valid column")
        target_sheet = self.lookup_sheet_name(column_name=target)
        y = self.data[target_sheet][target].values
        x = self.data[target_sheet]
        skf = StratifiedKFold(n_splits=n_splits, random_state=random_state)
        for train_idx, test_idx in skf.split(x, y):
            self.train_idx.append(train_idx)
            self.test_idx.append(test_idx)


class ImputedFeatureSpace:
    def __init__(self, feature_space: FeatureSpace, features: List[str], datasets: int, random_state: int = 42):
        self.original_feature_space = feature_space
        self.imputated_feature_space = []
        self.optimal_parameters = None
        self.losses = None
        self.original_data = feature_space.as_dataframe(features=features)
        self.features = self.original_data.columns.tolist()
        self.kernel = mf.ImputationKernel(
            data=self.original_data.values, datasets=datasets, random_state=random_state, save_all_iterations=True
        )

    def _construct_imputed_feature_space(self):
        for i in list(range(self.kernel.dataset_count())):
            data = pd.DataFrame(
                self.kernel.complete_data(dataset=i),
                columns=self.original_data.columns,
                index=self.original_data.index,
            )
            imputed_space = deepcopy(self.original_feature_space)
            for col in data.columns:
                sheet_name = self.original_feature_space.lookup_sheet_name(column_name=col)
                imputed_space.overwrite_column(sheet_name=sheet_name, column=col, values=data[col])
            self.imputated_feature_space.append(imputed_space)

    def optimise_parameters(self, dataset: int = 0, optimization_steps: int = 10, **kwargs):
        self.optimal_parameters, self.losses = self.kernel.tune_parameters(
            dataset=dataset, optimization_steps=optimization_steps, **kwargs
        )
        return self

    def impute(self, iterations: int = 5, **kwargs):
        kwargs = kwargs or {}
        if self.optimal_parameters is not None and "variable_parameters" not in kwargs.keys():
            kwargs["variable_parameters"] = self.optimal_parameters
        self.kernel.mice(iterations=iterations, **kwargs)
        self._construct_imputed_feature_space()
        return self

    def __iter__(self) -> FeatureSpace:
        for fs in self.imputated_feature_space:
            yield fs

    def plot_distributions(self, col_wrap: int = 3, figsize: Tuple[int, int] = (15, 15)):
        fig = ColumnWrapFigure(n=self.original_data.shape[1], figsize=figsize, col_wrap=col_wrap)
        for i, col in enumerate(self.features):
            ax = fig.add_wrapped_subplot()
            sns.kdeplot(self.original_data[col], ax=ax, color="red")
            for j in range(self.kernel.dataset_count()):
                x = self.kernel.complete_data(dataset=j)[:, i]
                sns.kdeplot(x, ax=ax, color="black")
        fig.tight_layout()
        return fig

    def plot_correlation_convergence(self, col_wrap: int = 3, figsize: Tuple[int, int] = (15, 15)):
        fig = ColumnWrapFigure(n=self.original_data.shape[1], figsize=figsize, col_wrap=col_wrap)
        var_indx = self.kernel._get_variable_index(None)
        num_vars = self.kernel._get_num_vars(var_indx)
        corr_dict = self.kernel.get_correlations(datasets=list(range(self.kernel.dataset_count())), variables=num_vars)
        for i, col in enumerate(self.features):
            ax = fig.add_wrapped_subplot()
            corr = pd.DataFrame(corr_dict[i]).melt(var_name="Iteration", value_name="Mean")
            box_swarm_plot(
                corr,
                x="Iteration",
                y="Correlation",
                overlay_kwargs={"color": "white", "linewidth": 1, "edgecolor": "black"},
                boxplot_kwargs={"color": "grey"},
                ax=ax,
            )
            ax.set_title(col)
        fig.tight_layout()
        return fig
