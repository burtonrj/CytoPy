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
from collections import defaultdict
from typing import Dict
from typing import List
from typing import Optional
from typing import Type
from typing import Union

import numpy as np
import pandas as pd
import phenograph

from .consensus_k import KConsensusClustering
from .flowsom import FlowSOM
from cytopy.data.experiment import Experiment
from cytopy.data.experiment import single_cell_dataframe
from cytopy.data.population import Population
from cytopy.data.subject import Subject
from cytopy.feedback import progress_bar
from cytopy.utils.dim_reduction import dimension_reduction_with_sampling
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
    return pd.merge(
        data,
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
        data = scaler(data=data, features=features)
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
            labels = self._cluster(data, features)
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
        metadata = summarise_clusters(
            data=data, features=features, summary_method=summary_method, scale=scale_method, scale_kwargs=scale_kwargs
        )
        metadata["meta_label"] = self._cluster(metadata, features)
        data = assign_metalabels(data, metadata)
        return data


class Clustering:
    def __init__(
        self,
        experiment: Experiment,
        features: list,
        sample_ids: list or None = None,
        root_population: str = "root",
        transform: str = "logicle",
        transform_kwargs: dict or None = None,
        verbose: bool = True,
        population_prefix: str = "cluster",
        data: Optional[pd.DataFrame] = None,
        random_state: int = 42,
    ):
        np.random.seed(random_state)
        self.experiment = experiment
        self.verbose = verbose
        self.features = features
        self.transform = transform
        self.transform_kwargs = transform_kwargs
        self.root_population = root_population
        self.sample_ids = sample_ids
        self.population_prefix = population_prefix

        if data is None:
            logger.info(f"Obtaining data for clustering for population {root_population}")
            self.data = single_cell_dataframe(
                experiment=experiment,
                sample_ids=sample_ids,
                transform=transform,
                transform_kwargs=transform_kwargs,
                populations=root_population,
            )
            self.data["meta_label"] = None
            self.data["cluster_label"] = None
            logger.info("Ready to cluster!")
        else:
            self.data = data

    def _init_cluster_method(
        self,
        method: Union[str, ClusterMethod],
        **kwargs,
    ) -> ClusterMethod:
        if method == "phenograph":
            method = ClusterMethod(klass=Phenograph, params=kwargs, verbose=self.verbose)
        elif method == "flowsom":
            method = ClusterMethod(klass=FlowSOM, params=kwargs, verbose=self.verbose)
        elif method == "consensus":
            method = ClusterMethod(klass=KConsensusClustering, params=kwargs, verbose=self.verbose)
        elif isinstance(method, str):
            raise ValueError("If a string is given must be either 'phenograph', 'consensus' or 'flowsom'")
        elif not isinstance(method, ClusterMethod):
            method = ClusterMethod(klass=method, params=kwargs, verbose=self.verbose)
        if not isinstance(method, ClusterMethod):
            raise ValueError(
                "Must provide a valid string, a ClusterMethod object, or a valid Scikit-Learn like "
                "clustering class (must have 'fit_predict' method)."
            )
        return method

    def scale_data(self, features: List[str], scale_method: Optional[str] = None, scale_kwargs: Optional[Dict] = None):
        scale_kwargs = scale_kwargs or {}
        scalar = None
        data = self.data.copy()
        if scale_method is not None:
            scalar = Scaler(scale_method, **scale_kwargs)
            data = scalar(data=self.data, features=features)
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

        scale_kwargs = scale_kwargs or {}
        data, _ = self.scale_data(features=features, scale_method=scale_method, scale_kwargs=scale_kwargs)
        if dim_reduction is not None:
            data, _ = dimension_reduction_with_sampling(
                data=self.data, features=features, method=dim_reduction, **dim_reduction_kwargs
            )
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

    def load_meta_variable(self, variable: str, verbose: bool = True, embedded: list or None = None):
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
        self.data[variable] = None
        for _id in progress_bar(self.data.subject_id.unique(), verbose=verbose):
            if _id is None:
                continue
            p = Subject.objects(subject_id=_id).get()
            try:
                if embedded is not None:
                    x = None
                    for key in embedded:
                        x = p[key]
                    self.data.loc[self.data.subject_id == _id, variable] = x[variable]
                else:
                    self.data.loc[self.data.subject_id == _id, variable] = p[variable]
            except KeyError:
                logger.warning(f"{_id} is missing meta-variable {variable}")
                self.data.loc[self.data.subject_id == _id, variable] = None

    def _create_parent_populations(self, population_var: str, parent_populations: Dict, verbose: bool = True):
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
                parent_population_name = (
                    parent if self.population_prefix is None else f"{self.population_prefix}_{parent}"
                )
                pop = Population(
                    population_name=parent_population_name,
                    n=cluster_data.shape[0],
                    parent=self.root_population,
                    source="cluster",
                    signature=cluster_data.mean().to_dict(),
                )
                pop.index = cluster_data.original_index.to_list()
                fg.add_population(population=pop)
            fg.save()

    def _save(
        self,
        verbose: bool = True,
        population_var: str = "meta_label",
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
        if population_var == "meta_label":
            if self.data.meta_label.isnull().all():
                raise ValueError("Meta clustering has not been performed")

        if parent_populations is not None:
            self._create_parent_populations(population_var=population_var, parent_populations=parent_populations)
        parent_populations = parent_populations or {}

        for sample_id in progress_bar(self.data.sample_id.unique(), verbose=verbose):
            fg = self.experiment.get_sample(sample_id)
            sample_data = self.data[self.data.sample_id == sample_id].copy()

            for cluster_label, cluster in sample_data.groupby(population_var):
                population_name = (
                    str(cluster_label)
                    if self.population_prefix is None
                    else f"{self.population_prefix}_{cluster_label}"
                )
                parent = parent_populations.get(cluster_label, self.root_population)
                parent = (
                    parent
                    if self.population_prefix is None or parent == self.root_population
                    else f"{self.population_prefix}_{parent}"
                )
                pop = Population(
                    population_name=population_name,
                    n=cluster.shape[0],
                    parent=parent,
                    source="cluster",
                    signature=cluster.mean().to_dict(),
                )
                pop.index = cluster.original_index.to_list()
                fg.add_population(population=pop)
            fg.save()
