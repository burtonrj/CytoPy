#!/usr/bin.env/python
# -*- coding: utf-8 -*-
"""
High-dimensional clustering offers the advantage of an unbiased approach
to classification of single cells whilst also exploiting all available variables
in your data (all your fluorochromes/isotypes). In cytopy, the clustering is
performed on a Population of a FileGroup. The resulting clusters are saved
as new Populations.

In CytoPy, we refer to three different types of clustering:
* Per-sample clustering, where each FileGroup (sample) is clustered individually
* Global clustering, where FileGroup's (sample's) are combined into the same space and clustering is
performed for all events - this is computationally expensive and requires that batch effects have been
minimised or corrected prior to clustering
* Meta-clustering, where the clustering results of individual FileGroup's are clustered to
match clusters between FileGroup's; essentially 'clustering the clusters'

In this module you will find the Clustering class, which is the apparatus to apply a
clustering method in cytopy and save the results to the database. We also
provide implementations of PhenoGraph, FlowSOM and provide access to any
of the clustering methods available through the Scikit-Learn API.

The Clustering class is algorithm agnostic and only requires that a function be
provided that accepts a Pandas DataFrame with a column name 'sample_id' as the
sample identifier, 'cluster_label' as the clustering results, and 'meta_label'
as the meta clustering results. The function should also accept 'features' as
a list of columns to use to construct the input space to the clustering algorithm.
This function must return a Pandas DataFrame with the cluster_label/meta_label
columns populated accordingly. It should also return two null value OR can optionally
return a graph object, and modularity or equivalent score. These will be saved
to the Clustering attributes.

Copyright 2020 Ross Burton

Permission is hereby granted, free of charge, to any person
obtaining a copy of this software and associated documentation
files (the "Software"), to deal in the Software without restriction,
including without limitation the rights to use, copy, modify,
merge, publish, distribute, sublicense, and/or sell copies of the
Software, and to permit persons to whom the Software is furnished
to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included
in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""
import logging
import math
from collections import defaultdict
from typing import Callable
from typing import Dict
from typing import Iterable
from typing import List
from typing import Optional
from typing import Tuple
from typing import Type
from typing import Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import phenograph
import seaborn as sns
from sklearn.cluster import AgglomerativeClustering

from ..modeling.hypothesis_testing import hypothesis_test
from ..plotting import single_cell_density
from ..plotting import single_cell_plot
from ..plotting.general import box_swarm_plot
from ..plotting.general import ColumnWrapFigure
from .consensus_k import KConsensusClustering
from .flowgrid import FlowGrid
from .flowsom import FlowSOM
from .latent import LatentClustering
from .metrics import init_internal_metrics
from .metrics import InternalMetric
from .parc import PARC
from .plotting import boxswarm_and_source_count
from .plotting import clustered_heatmap
from .plotting import plot_meta_clusters
from .plotting import silhouette_analysis
from .spade import CytoSPADE
from cytopy.data.experiment import Experiment
from cytopy.data.experiment import single_cell_dataframe
from cytopy.data.population import Population
from cytopy.data.subject import Subject
from cytopy.feedback import progress_bar
from cytopy.utils.dim_reduction import dimension_reduction_with_sampling
from cytopy.utils.dim_reduction import DimensionReduction
from cytopy.utils.transform import Scaler

logger = logging.getLogger(__name__)


class ClusteringError(Exception):
    def __init__(self, message: str):
        logger.error(message)
        super().__init__(message)


def remove_null_features(data: pd.DataFrame, features: Optional[List[str]] = None) -> List[str]:
    """
    Check for null values in the dataframe.
    Returns a list of column names for columns with no missing values.

    Parameters
    ----------
    data: Pandas.DataFrame
    features: List[str], optional

    Returns
    -------
    List
        List of valid columns
    """
    features = features or data.columns.tolist()
    null_cols = data[features].isnull().sum()[data[features].isnull().sum() > 0].index.values
    if null_cols.size != 0:
        logger.warning(
            f"The following columns contain null values and will be excluded from clustering analysis: {null_cols}"
        )
    return [x for x in features if x not in null_cols]


def assign_metalabels(data: pd.DataFrame, metadata: pd.DataFrame):
    """
    Given the original clustered data (data) and the meta-clustering results of
    clustering the clusters of this original data (metadata), assign the meta-cluster
    labels to the original data and return the modified dataframe with the meta cluster
    labels in a new column called 'meta_label'

    Parameters
    ----------
    data: Pandas.DataFrame
    metadata: Pandas.DataFrame

    Returns
    -------
    Pandas.DataFrame
    """
    data = data.drop("meta_label", axis=1)
    return data.merge(
        metadata[["sample_id", "cluster_label", "meta_label"]],
        on=["sample_id", "cluster_label"],
    )


def summarise_clusters(
    data: pd.DataFrame,
    features: list,
    scale: Optional[str] = None,
    scale_kwargs: Optional[Dict] = None,
    summary_method: str = "median",
):
    """
    Average cluster parameters along columns average to generated a centroid for
    meta-clustering

    Parameters
    ----------
    data: Pandas.DataFrame
        Clustering results to average
    features: list
        List of features to use when generating centroid
    summary_method: str (default='median')
        Average method, should be mean or median
    scale: str, optional
        Perform scaling of centroids; see cytopy.transform.Scaler
    scale_kwargs: dict, optional
        Additional keyword arguments passed to Scaler

    Returns
    -------
    Pandas.DataFrame

    Raises
    ------
    ValueError
        If invalid method provided
    """
    if summary_method == "median":
        data = data.groupby(["sample_id", "cluster_label"])[features].median().reset_index()
    elif summary_method == "mean":
        data = data.groupby(["sample_id", "cluster_label"])[features].mean().reset_index()
    else:
        raise ValueError("summary_method should be 'mean' or 'median'")
    scale_kwargs = scale_kwargs or {}
    if scale is not None:
        scaler = Scaler(method=scale, **scale_kwargs)
        data = scaler.fit_transform(data=data, features=features)
    return data


class Phenograph:
    def __init__(self, **params):
        params = params or {}
        self.params = params

    def fit_predict(self, data: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        communities, graph, q = phenograph.cluster(data, **self.params)
        return communities


class ClusterMethod:
    def __init__(self, klass: Type, params: Optional[Dict] = None, verbose: bool = True):
        params = params or {}
        self.verbose = verbose
        self.method = klass(**params)
        self.params = params
        self.valid_method()

    def valid_method(self):
        try:
            fit_predict = getattr(self.method, "fit_predict", None)
            assert fit_predict is not None
            assert callable(fit_predict)
        except AssertionError:
            raise ClusteringError("Invalid Class as clustering method, must have function 'fit_predict'")

    def _cluster(self, data: pd.DataFrame, features: List[str]):
        return self.method.fit_predict(data[features])

    def cluster(self, data: pd.DataFrame, features: List[str]):
        data["cluster_label"] = None
        for _id, df in progress_bar(data.groupby("sample_id"), verbose=self.verbose):
            labels = self._cluster(df, features)
            data.loc[df.index, ["cluster_label"]] = labels
        return data

    def global_clustering(self, data: pd.DataFrame, features: List[str]):
        data["cluster_label"] = self._cluster(data, features)
        return data

    def meta_clustering(
        self,
        data: pd.DataFrame,
        features: List[str],
        summary_method: str = "median",
        scale_method: Optional[str] = None,
        scale_kwargs: Optional[Dict] = None,
    ):
        data = data.copy()
        metadata = summarise_clusters(
            data=data, features=features, summary_method=summary_method, scale=scale_method, scale_kwargs=scale_kwargs
        )
        metadata["meta_label"] = self._cluster(metadata, features)
        data = assign_metalabels(data, metadata)
        return data


def init_cluster_method(
    method: Union[str, ClusterMethod],
    verbose: bool,
    **kwargs,
) -> ClusterMethod:
    if method == "phenograph":
        method = ClusterMethod(klass=Phenograph, params=kwargs, verbose=verbose)
    elif method == "flowsom":
        method = ClusterMethod(klass=FlowSOM, params=kwargs, verbose=verbose)
    elif method == "consensus":
        method = ClusterMethod(klass=KConsensusClustering, params=kwargs, verbose=verbose)
    elif method == "flowgrid":
        method = ClusterMethod(klass=FlowGrid, params=kwargs, verbose=verbose)
    elif method == "spade":
        method = ClusterMethod(klass=CytoSPADE, params=kwargs, verbose=verbose)
    elif method == "latent":
        method = ClusterMethod(klass=LatentClustering, params=kwargs, verbose=verbose)
    elif method == "parc":
        method = ClusterMethod(klass=PARC, params=kwargs, verbose=verbose)
    elif isinstance(method, str):
        valid_str_methods = ["phenograph", "flowsom", "spade", "latent", "consensus", "parc"]
        raise ValueError(f"If a string is given must be one of {valid_str_methods}")
    elif not isinstance(method, ClusterMethod):
        method = ClusterMethod(klass=method, params=kwargs, verbose=verbose)
    if not isinstance(method, ClusterMethod):
        raise ValueError(
            "Must provide a valid string, a ClusterMethod object, or a valid Scikit-Learn like "
            "clustering class (must have 'fit_predict' method)."
        )
    return method


class Clustering:
    def __init__(
        self,
        data: pd.DataFrame,
        experiment: Optional[Experiment],
        features: List[str],
        sample_ids: Optional[List[str]] = None,
        root_population: str = "root",
        transform: str = "asinh",
        transform_kwargs: Optional[Dict] = None,
        verbose: bool = True,
        random_state: int = 42,
        n_sources: Optional[Dict] = None,
    ):
        np.random.seed(random_state)
        self.experiment = experiment
        self.verbose = verbose
        self.features = features
        self.transform = transform
        self.transform_kwargs = transform_kwargs
        self.root_population = root_population
        self.sample_ids = sample_ids
        self.data = data
        self._embedding_cache = None
        self._n_sources = n_sources or {}

    @classmethod
    def from_experiment(
        cls,
        experiment: Experiment,
        features: list,
        sample_ids: list or None = None,
        root_population: str = "root",
        transform: str = "asinh",
        transform_kwargs: dict or None = None,
        verbose: bool = True,
        random_state: int = 42,
    ):
        logger.info(f"Obtaining data for clustering for population {root_population}")
        data = single_cell_dataframe(
            experiment=experiment,
            sample_ids=sample_ids,
            transform=transform,
            transform_kwargs=transform_kwargs,
            populations=root_population,
        )
        data["meta_label"] = None
        data["cluster_label"] = None
        return cls(
            data=data,
            experiment=experiment,
            features=features,
            sample_ids=sample_ids,
            root_population=root_population,
            transform=transform,
            transform_kwargs=transform_kwargs,
            verbose=verbose,
            random_state=random_state,
        )

    @classmethod
    def from_dataframe(
        cls,
        data: pd.DataFrame,
        experiment: Optional[Experiment],
        features: list,
        sample_ids: list or None = None,
        root_population: str = "root",
        transform: str = "asinh",
        transform_kwargs: dict or None = None,
        verbose: bool = True,
        random_state: int = 42,
    ):
        n_sources = None
        if "meta_label" not in data.columns:
            data["meta_label"] = None
        if "cluster_label" not in data.columns:
            data["cluster_label"] = None
        else:
            if "n_sources" in data.columns:
                n_sources = {
                    cluster: n
                    for cluster, n in data[["cluster_label", "n_sources"]].drop_duplicates().itertuples(index=False)
                }
        return cls(
            data=data,
            experiment=experiment,
            features=features,
            sample_ids=sample_ids,
            root_population=root_population,
            transform=transform,
            transform_kwargs=transform_kwargs,
            verbose=verbose,
            random_state=random_state,
            n_sources=n_sources,
        )

    def scale_data(self, features: List[str], scale_method: Optional[str] = None, scale_kwargs: Optional[Dict] = None):
        scale_kwargs = scale_kwargs or {}
        scalar = None
        data = self.data.copy()
        if scale_method is not None:
            scalar = Scaler(scale_method, **scale_kwargs)
            data = scalar.fit_transform(data=self.data, features=features)
        return data, scalar

    def scale_and_reduce(
        self,
        features: List[str],
        scale_method: Optional[str] = None,
        scale_kwargs: Optional[Dict] = None,
        dim_reduction: Optional[str] = None,
        dim_reduction_kwargs: Optional[Dict] = None,
    ):
        dim_reduction_kwargs = dim_reduction_kwargs or {}
        downsample = dim_reduction_kwargs.pop("downsample", True)
        sample_n = dim_reduction_kwargs.pop("sample_n", 10000)
        if sample_n >= self.data.shape[0]:
            downsample = None
        scale_kwargs = scale_kwargs or {}
        data, _ = self.scale_data(features=features, scale_method=scale_method, scale_kwargs=scale_kwargs)
        if dim_reduction is not None:
            if downsample:
                logger.info(f"Embeddings will be calculated on a subsample of {sample_n} events.")
                data, _ = dimension_reduction_with_sampling(
                    data=self.data, features=features, method=dim_reduction, **dim_reduction_kwargs
                )
            else:
                if self.data.shape[0] > 500000:
                    logger.warning(
                        f"No downsampling specified, yet your data is rather big! This might "
                        f"take a while or might use a lot of memory! Specify 'downsample' as "
                        f"True and a sample_n to perform embedding on a subsample."
                    )
                    reducer = DimensionReduction(method=dim_reduction, **dim_reduction_kwargs)
                    data = reducer.fit_transform(data=data, features=features)
            features = [x for x in data.columns if dim_reduction in x]
        return data, features

    def reset_clusters(self):
        """
        Resets cluster and meta cluster labels to None

        Returns
        -------
        self
        """
        self.data["cluster_label"] = None
        self.data["meta_label"] = None
        return self

    def rename_clusters(self, sample_id: str, mappings: dict):
        """
        Given a dictionary of mappings, replace the current IDs stored
        in cluster_label column for a particular sample

        Parameters
        ----------
        sample_id: str
        mappings: dict
            Mappings; {current ID: new ID}

        Returns
        -------
        None
        """
        if sample_id != "all":
            idx = self.data[self.data.sample_id == sample_id].index
            self.data.loc[idx, "cluster_label"] = self.data.loc[idx]["cluster_label"].replace(mappings)
        else:
            self.data["cluster_label"] = self.data["cluster_label"].replace(mappings)

    def load_meta_variable(self, variable_name: str, key: Union[str, List[str]], verbose: bool = True):
        """
        Load a meta-variable for each Subject, adding this variable as a new column. If a sample
        is not associated to a Subject or the meta variable is missing from a Subject, value will be
        None.
        Parameters
        ----------
        variable: str
            Name of the meta-variable
        verbose: bool (default=True)
        embedded: list
            If the meta-variable is embedded, this should be a list of keys that
            preceed the variable

        Returns
        -------
        None
        """
        self.data[variable_name] = None
        for _id in progress_bar(self.data.subject_id.unique(), verbose=verbose):
            if _id is None:
                continue
            try:
                p = Subject.objects(subject_id=_id).get()
                self.data.loc[self.data.subject_id == _id, variable_name] = p.lookup_var(key=key)
            except ValueError as e:
                logger.error(f"Failed to load meta variable for {_id}")
                logger.exception(e)

    def dimension_reduction(
        self,
        n: Optional[int] = 1000,
        sample_id: Optional[str] = None,
        overwrite_cache: bool = False,
        method: str = "UMAP",
        replace: bool = False,
        weights: Optional[Iterable] = None,
        random_state: int = 42,
        **dim_reduction_kwargs,
    ):
        reducer = DimensionReduction(method=method, n_components=2, **dim_reduction_kwargs)
        if sample_id and self._embedding_cache is not None:
            if self._embedding_cache.sample_id.nunique() > 1:
                # Embedding previously captures multiple samples
                overwrite_cache = True
            elif self.data.sample_id.unique()[0] != sample_id:
                # Embedding previously captures another sample
                overwrite_cache = True
        if self._embedding_cache is not None and not overwrite_cache:
            if f"{method}1" not in self._embedding_cache.columns:
                self._embedding_cache = reducer.fit_transform(data=self._embedding_cache, features=self.features)
            else:
                return self._embedding_cache
        if overwrite_cache or self._embedding_cache is None:
            data = self.data.copy()
            if sample_id:
                data = data[data.sample_id == sample_id]
                if self.data.shape[0] > n:
                    data = self.data.sample(n)
            elif n is not None:
                data = data.groupby("sample_id").sample(
                    n=n, replace=replace, weights=weights, random_state=random_state
                )
            self._embedding_cache = reducer.fit_transform(data=data, features=self.features)
        if sample_id:
            self._embedding_cache["cluster_label"] = self.data[self.data.sample_id == sample_id]["cluster_label"]
            self._embedding_cache["meta_label"] = self.data[self.data.sample_id == sample_id]["meta_label"]
        else:
            self._embedding_cache["cluster_label"] = self.data["cluster_label"]
            self._embedding_cache["meta_label"] = self.data["meta_label"]
        return self._embedding_cache

    def plot_density(
        self,
        n: int = 1000,
        sample_id: Optional[str] = None,
        overwrite_cache: bool = False,
        method: str = "UMAP",
        dim_reduction_kwargs: Optional[Dict] = None,
        subset: Optional[str] = None,
        plot_n: Optional[int] = None,
        **plot_kwargs,
    ):
        dim_reduction_kwargs = dim_reduction_kwargs or {}
        data = self.dimension_reduction(
            n=n, sample_id=sample_id, overwrite_cache=overwrite_cache, method=method, **dim_reduction_kwargs
        )
        if subset:
            data = data.query(subset).copy()
        if plot_n and (data.shape[0] > plot_n):
            data = data.sample(plot_n)
        return single_cell_density(data=data, x=f"{method}1", y=f"{method}2", **plot_kwargs)

    def plot(
        self,
        label: str,
        discrete: bool = True,
        n: int = 1000,
        sample_id: Optional[str] = None,
        overwrite_cache: bool = False,
        method: str = "UMAP",
        dim_reduction_kwargs: Optional[Dict] = None,
        subset: Optional[str] = None,
        **plot_kwargs,
    ):
        dim_reduction_kwargs = dim_reduction_kwargs or {}
        data = self.dimension_reduction(
            n=n, sample_id=sample_id, overwrite_cache=overwrite_cache, method=method, **dim_reduction_kwargs
        )
        if subset:
            data = data.query(subset).copy()
        return single_cell_plot(
            data=data, x=f"{method}1", y=f"{method}2", label=label, discrete=discrete, **plot_kwargs
        )

    def plot_cluster_membership(
        self,
        n: int = 1000,
        sample_id: Optional[str] = None,
        overwrite_cache: bool = False,
        method: str = "UMAP",
        dim_reduction_kwargs: Optional[Dict] = None,
        subset: Optional[str] = None,
        **plot_kwargs,
    ):
        dim_reduction_kwargs = dim_reduction_kwargs or {}
        data = self.dimension_reduction(
            n=n, sample_id=sample_id, overwrite_cache=overwrite_cache, method=method, **dim_reduction_kwargs
        )
        data["cluster_label"] = self.data["cluster_label"]
        if subset:
            data = data.query(subset)
        return single_cell_plot(
            data=data, x=f"{method}1", y=f"{method}2", label="cluster_label", discrete=True, **plot_kwargs
        )

    def plot_meta_cluster_centroids(
        self,
        label: str = "meta_label",
        discrete: bool = True,
        method: str = "UMAP",
        dim_reduction_kwargs: Optional[Dict] = None,
        subset: Optional[str] = None,
        **kwargs,
    ):
        if "meta_label" not in self.data.columns:
            raise KeyError("Meta-clustering has not been performed")
        data = self.data
        if subset:
            data = data.query(subset)
        return plot_meta_clusters(
            data=data,
            features=self.features,
            colour_label=label,
            discrete=discrete,
            method=method,
            dim_reduction_kwargs=dim_reduction_kwargs,
            **kwargs,
        )

    def heatmap(
        self,
        features: Optional[str] = None,
        sample_id: Optional[str] = None,
        meta_label: bool = True,
        include_labels: Optional[List[str]] = None,
        subset: Optional[str] = None,
        plot_orientation="vertical",
        **kwargs,
    ):
        features = features or self.features
        data = self.data.copy()
        if subset:
            data = data.query(subset)
        if include_labels:
            if meta_label:
                data = data[data["meta_label"].isin(include_labels)]
            else:
                data = data[data["cluster_label"].isin(include_labels)]
        return clustered_heatmap(
            data=data,
            features=features,
            sample_id=sample_id,
            meta_label=meta_label,
            plot_orientation=plot_orientation,
            **kwargs,
        )

    @staticmethod
    def _count_to_proportion(df: pd.DataFrame):
        df["Percentage"] = (df["Count"] / df["Count"].sum()) * 100
        return df

    @staticmethod
    def _fill_null_clusters(data: pd.DataFrame, label: str):
        labels = data[label].unique()
        updated_data = []
        for sample_id, sample_df in data.groupby("sample_id"):
            missing_labels = [i for i in labels if i not in sample_df[label].unique()]
            updated_data.append(
                pd.concat(
                    [
                        sample_df,
                        pd.DataFrame(
                            {
                                "sample_id": [sample_id for _ in range(len(missing_labels))],
                                "Count": [0 for _ in range(len(missing_labels))],
                                label: missing_labels,
                            }
                        ),
                    ]
                )
            )
        return pd.concat(updated_data).reset_index(drop=True)

    def cluster_proportions(
        self,
        label: str = "cluster_label",
        filter_clusters: Optional[List] = None,
        hue: Optional[str] = None,
        plot_source_count: bool = False,
        log10_percentage: bool = False,
        replace_null_population: float = 0.01,
        y_label: str = "Percentage",
        subset: Optional[str] = None,
        **plot_kwargs,
    ):
        data = self.data.copy()
        if subset:
            data = data.query(subset).copy()
        if filter_clusters:
            data = data[data[label].isin(filter_clusters)]
        x = data.groupby("sample_id")[label].value_counts()
        x.name = "Count"
        x = x.reset_index()
        plot_data = x.groupby("sample_id").apply(self._count_to_proportion).reset_index()
        plot_data = self._fill_null_clusters(data=plot_data, label=label)
        plot_data.rename(columns={"Percentage": y_label}, inplace=True)

        if hue:
            colour_mapping = self.data[["sample_id", hue]].drop_duplicates()
            plot_data = plot_data.merge(colour_mapping, on="sample_id")

        if log10_percentage:
            plot_data[f"log10({y_label})"] = np.log10(
                plot_data[y_label].apply(lambda i: replace_null_population if i == 0 else i)
            )
            y_label = f"log10({y_label})"

        if plot_source_count:
            plot_data["n_sources"] = plot_data[label].map(self._n_sources)
            return boxswarm_and_source_count(plot_data=plot_data, x=label, y=y_label, hue=hue, **plot_kwargs)

        ax = box_swarm_plot(plot_df=plot_data, x=label, y=y_label, hue=hue, **plot_kwargs)
        return ax

    def cluster_proportion_stats(
        self, between_group: str, label: str = "cluster_label", subset: Optional[str] = None, **kwargs
    ):
        for c in [between_group, label]:
            if c not in self.data.columns:
                raise KeyError(f"No such column {c}")
        data = self.data[~self.data[between_group].isnull()]
        if subset:
            data = data.query(subset).copy()
        x = data.groupby("sample_id")[label].value_counts()
        x.name = "Count"
        x = x.reset_index()
        data = x.groupby("sample_id").apply(self._count_to_proportion).reset_index()
        group_mapping = self.data[["sample_id", between_group]].dropna().drop_duplicates()
        data = data.merge(group_mapping, on="sample_id")
        return hypothesis_test(
            data=data, dv="Percentage", between_group=between_group, independent_group=label, **kwargs
        )

    def performance(
        self,
        metrics: Optional[List[Union[str, InternalMetric]]] = None,
        sample_n: int = 10000,
        resamples: int = 10,
        features: Optional[List[str]] = None,
        labels: Union[Iterable, str] = "cluster_label",
        plot: bool = True,
        verbose: bool = True,
        col_wrap: int = 2,
        figure_kwargs: Optional[Dict] = None,
        **plot_kwargs,
    ):
        if sample_n > self.data.shape[0]:
            raise ValueError(f"sample_n is greater than the total number of events: {sample_n} > {self.data.shape[0]}")
        features = features or self.features
        metrics = init_internal_metrics(metrics=metrics)
        results = defaultdict(list)
        if isinstance(labels, list) or isinstance(labels, np.ndarray):
            self.data["tmp"] = labels
            labels = "tmp"

        for _ in progress_bar(range(resamples), verbose=verbose, total=resamples):
            df = self.data.sample(n=sample_n)
            for m in metrics:
                results[m.name].append(m(data=df, features=features, labels=df[labels].values))
        if "tmp" in self.data.columns.values:
            self.data.drop("tmp", axis=1, inplace=True)
        if plot:
            figure_kwargs = figure_kwargs or {}
            figure_kwargs["figsize"] = figure_kwargs.get("figure_size", (10, 10))
            fig = ColumnWrapFigure(n=len(metrics), col_wrap=col_wrap, **figure_kwargs)
            for i, (m, data) in enumerate(results.items()):
                box_swarm_plot(
                    plot_df=pd.DataFrame({"Method": [m] * len(data), "Score": data}),
                    x="Method",
                    y="Score",
                    ax=fig.add_wrapped_subplot(),
                    **plot_kwargs,
                )
            return results, fig
        return results

    def k_performance(
        self,
        max_k: int,
        cluster_n_param: str,
        method: Union[str, ClusterMethod],
        metric: InternalMetric,
        overwrite_features: Optional[List[str]] = None,
        sample_id: Optional[str] = None,
        reduce_dimensions: bool = False,
        dim_reduction_kwargs: Optional[Dict] = None,
        clustering_params: Optional[Dict] = None,
    ):
        clustering_params = clustering_params or {}

        overwrite_features = overwrite_features or self.features
        features = remove_null_features(self.data, features=overwrite_features)
        data = (
            self.data
            if not reduce_dimensions
            else dimension_reduction_with_sampling(data=self.data, features=features, **dim_reduction_kwargs)
        )
        if sample_id is not None:
            data = data[data.sample_id == sample_id].copy()

        ylabel = metric.name
        x = []
        y = []
        for k in progress_bar(np.arange(1, max_k + 1, 1)):
            df = data.copy()
            clustering_params[cluster_n_param] = k
            method = init_cluster_method(method=method, verbose=self.verbose, **clustering_params)
            df = method.cluster(data=df, features=features, evaluate=False)
            x.append(k)
            y.append(metric(data=df, features=features, labels=df["cluster_label"]))
        ax = sns.lineplot(x=x, y=y, markers=True)
        ax.set_xlabel("K")
        ax.set_ylabel(ylabel)
        return ax

    def silhouette_analysis(
        self,
        n: int = 5000,
        ax: Optional[plt.Axes] = None,
        figsize: Optional[Tuple[int, int]] = (7.5, 7.5),
        xlim: Tuple[int, int] = (-1, 1),
    ):
        data = self.data.sample(n=n)
        return silhouette_analysis(data=data, features=self.features, ax=ax, figsize=figsize, xlim=xlim)

    def merge_clusters(
        self,
        k_range: Optional[Iterable[int]] = None,
        summary: Union[str, Callable] = "median",
        cluster_method: Optional[ClusterMethod] = None,
        **kwargs,
    ):
        if summary == "median":
            data = self.data.groupby(["cluster_label"])[self.features].median()
        elif summary == "mean":
            data = self.data.groupby(["cluster_label"])[self.features].median()
        else:
            data = self.data.groupby(["cluster_label"])[self.features].apply(summary)

        cluster_method = cluster_method or AgglomerativeClustering()
        if k_range is None:
            k_range = [2, math.ceil(self.data.cluster_label.nunique() / 2)]
        kconsensus = KConsensusClustering(
            cluster=cluster_method, smallest_cluster_n=k_range[0], largest_cluster_n=k_range[1], **kwargs
        )
        data["cluster_label"] = kconsensus.fit_predict(data=data.values)
        data.index.name = "original_cluster_label"
        data.reset_index(drop=False, inplace=True)
        mapping = {o: n for o, n in data[["original_cluster_label", "cluster_label"]].values}
        self.data.cluster_label = self.data.cluster_label.replace(mapping)
        return self

    def _create_parent_populations(
        self,
        population_var: str,
        parent_populations: Dict,
        population_prefix: Optional[str] = None,
        verbose: bool = True,
    ):
        """
        Form parent populations from existing clusters

        Parameters
        ----------
        population_var: str
            Name of the cluster population variable i.e. cluster_label or meta_label
        parent_populations: Dict
            Dictionary of parent associations. Parent populations will be a merger of all child populations.
            Each child population intended to inherit from a parent that is not 'root' should be given as a
            key with the value being the parent to associate to.
        verbose: bool (default=True)
            Whether to provide feedback in the form of a progress bar

        Returns
        -------
        None
            Parent populations are saved to the FileGroup
        """
        logger.info("Creating parent populations from clustering results")
        parent_child_mappings = defaultdict(list)
        for child, parent in parent_populations.items():
            parent_child_mappings[parent].append(child)

        for sample_id in progress_bar(self.data.sample_id.unique(), verbose=verbose):
            fg = self.experiment.get_sample(sample_id)
            sample_data = self.data[self.data.sample_id == sample_id].copy()

            for parent, children in parent_child_mappings.items():
                cluster_data = sample_data[sample_data[population_var].isin(children)]
                if cluster_data.shape[0] == 0:
                    logger.warning(f"No clusters found for {sample_id} to generate requested parent {parent}")
                    continue
                parent_population_name = parent if population_prefix is None else f"{population_prefix}_{parent}"
                pop = Population(
                    population_name=parent_population_name,
                    n=cluster_data.shape[0],
                    parent=self.root_population,
                    source="cluster",
                )
                pop.index = cluster_data.original_index.to_list()
                fg.add_population(population=pop)
            fg.save()

    def _save(
        self,
        population_prefix: Optional[str] = None,
        verbose: bool = True,
        population_var: str = "cluster_label",
        parent_populations: Optional[Dict] = None,
    ):
        """
        Clusters are saved as new Populations in each FileGroup in the attached Experiment
        according to the sample_id in data.

        Parameters
        ----------
        verbose: bool (default=True)
        population_var: str (default='meta_label')
            Variable in data that should be used to identify individual Populations
        parent_populations: Dict
            Dictionary of parent associations. Parent populations will be a merger of all child populations.
            Each child population intended to inherit from a parent that is not 'root' should be given as a
            key with the value being the parent to associate to.

        Returns
        -------
        None

        Raises
        ------
        ValueError
            If population_var is 'meta_label' and meta clustering has not been previously performed
        """
        if self.experiment is None:
            raise ClusteringError("No experiment associated to clustering object")
        if population_var == "meta_label":
            if self.data.meta_label.isnull().all():
                raise ValueError("Meta clustering has not been performed")

        if parent_populations is not None:
            err = f"One or more cluster_labels are missing a parent definition"
            assert all([x in parent_populations.keys() for x in self.data.cluster_label.unique()]), err
            self._create_parent_populations(
                population_prefix=population_prefix,
                population_var=population_var,
                parent_populations=parent_populations,
            )
        parent_populations = parent_populations or {}

        for sample_id in progress_bar(self.data.sample_id.unique(), verbose=verbose):
            fg = self.experiment.get_sample(sample_id)
            sample_data = self.data[self.data.sample_id == sample_id].copy()

            for cluster_label, cluster in sample_data.groupby(population_var):
                population_name = (
                    str(cluster_label) if population_prefix is None else f"{population_prefix}_{cluster_label}"
                )
                parent = parent_populations.get(cluster_label, self.root_population)
                parent = (
                    parent
                    if population_prefix is None or parent == self.root_population
                    else f"{population_prefix}_{parent}"
                )
                pop = Population(
                    population_name=population_name,
                    n=cluster.shape[0],
                    parent=parent,
                    source="cluster",
                    n_sources=self._n_sources.get(cluster_label, 1),
                )
                pop.index = cluster.original_index.to_list()
                fg.add_population(population=pop)
            fg.save()


def clustering_statistics(
    experiment: Experiment, prefix: str, meta_vars: Optional[Dict] = None, additional_parent: Optional[str] = None
):
    return experiment.population_statistics(
        regex=f"{prefix}_.+",
        population_source="cluster",
        data_source="primary",
        meta_vars=meta_vars,
        additional_parent=additional_parent,
    )


def clustering_single_cell_data(experiment: Experiment, prefix: str, **kwargs):
    populations = experiment.list_populations(regex=f"{prefix}_.+", source="cluster", data_source="primary")
    return single_cell_dataframe(experiment=experiment, populations=populations, **kwargs).rename(
        columns={"population_label": "cluster_label"}
    )
