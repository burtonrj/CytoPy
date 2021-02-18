#!/usr/bin.env/python
# -*- coding: utf-8 -*-
"""
For studies where the objective is the prediction of some endpoint and
characterisation of phenotypes that contribute to that prediction,
it is valuable to have tools for generating summaries of our cell
populations to serve as variables in differential analysis or modelling
tasks. This module provides the tools to summarise the populations
generated and has numerous utility functions for 'feature selection'.

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

from ..feedback import progress_bar, setup_standard_logger
from ..data.fcs import population_stats, Population, MissingPopulationError
from ..data.experiment import Experiment, fetch_subject_meta, fetch_subject, FileGroup
from collections import defaultdict
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from scipy import stats as scipy_stats
from functools import partial
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import re

__author__ = "Ross Burton"
__copyright__ = "Copyright 2020, CytoPy"
__credits__ = ["Ross Burton", "Simone Cuff", "Andreas Artemiou", "Matthias Eberl"]
__license__ = "MIT"
__version__ = "1.0.0"
__maintainer__ = "Ross Burton"
__email__ = "burtonrj@cardiff.ac.uk"
__status__ = "Production"

STATS = {"mean": np.mean,
         "SD": np.std,
         "median": np.median,
         "CV": scipy_stats.variation,
         "skew": scipy_stats.skew,
         "kurtosis": scipy_stats.kurtosis,
         "gmean": scipy_stats.gmean}


def _fetch_population_statistics(files: list,
                                 populations: set):
    return {f.primary_id: {p: f.population_stats(p) for p in populations}
            for f in files}


class FeatureSpace:
    def __init__(self,
                 experiment: Experiment,
                 sample_ids: list or None = None,
                 logging_level: int or None = None,
                 log: str or None = None):
        sample_ids = sample_ids or experiment.list_samples()
        self.logger = setup_standard_logger(name="FeatureSpace",
                                            default_level=logging_level,
                                            log=log)
        self._fcs_files = [x for x in experiment.fcs_files
                           if x.primary_id in sample_ids] or experiment.fcs_files
        populations = [x.list_populations() for x in self._fcs_files]
        self.populations = set([x for sl in populations for x in sl])
        self.population_statistics = _fetch_population_statistics(files=self._fcs_files,
                                                                  populations=self.populations)
        self.ratios = defaultdict(dict)
        self.channel_desc = defaultdict(dict)
        self.meta_labels = dict()

    def compute_ratios(self,
                       pop1: str,
                       pop2: str or None = None):
        for f in self._fcs_files:
            if pop1 not in f.list_populations():
                self.logger.warn(f"{f.primary_id} missing population {pop1}")
                if pop2 is None:
                    for p in [q for q in self.populations if q != pop1]:
                        self.ratios[f.primary_id][f"{pop1}:{p}"] = None
                else:
                    self.ratios[f.primary_id][f"{pop1}:{pop2}"] = None
            else:
                p1n = self.population_statistics[f.primary_id][pop1]["n"]
                if pop2 is None:
                    for p in [q for q in self.populations if q != pop1]:
                        pn = self.population_statistics[f.primary_id][p]["n"]
                        self.ratios[f.primary_id][f"{pop1}:{p}"] = p1n / pn
                else:
                    p2n = self.population_statistics[f.primary_id][pop2]["n"]
                    self.ratios[f.primary_id][f"{pop1}:{pop2}"] = p1n / p2n
        return self

    def channel_desc_stats(self,
                           channel: str,
                           stats: list or None = None,
                           transform: str or None = None,
                           transform_kwargs: dict or None = None,
                           populations: list or None = None,
                           verbose: bool = True):
        populations = populations or self.populations
        stats = stats or ["mean", "SD"]
        assert all([x in STATS.keys() for x in stats]), f"Invalid stats; valid stats are: {STATS.keys()}"
        for f in progress_bar(self._fcs_files, verbose=verbose):
            for p in populations:
                if p not in f.list_populations():
                    self.logger.warn(f"{f.primary_id} missing population {p}")
                    for s in stats:
                        self.channel_desc[f.primary_id][f"{p}_{s}"] = None
                else:
                    x = f.load_population_df(population=p,
                                             transform=transform,
                                             features_to_transform=[channel],
                                             transform_kwargs=transform_kwargs)[channel].values
                    for s in stats:
                        self.channel_desc[f.primary_id][f"{p}_{s}"] = STATS.get(s)(x)
        return self

    def add_meta_labels(self,
                        key: str or list):
        for f in self._fcs_files:
            subject = fetch_subject(f)
            if subject is None:
                continue
            try:
                if isinstance(key, str):
                    self.meta_labels[f.primary_id] = subject[key]
                else:
                    node = subject[key[0]]
                    for k in key[1:]:
                        node = node[k]
                    self.meta_labels[f.primary_id] = node
            except KeyError:
                self.logger.warn(f"{f.primary_id} missing meta variable {key} in Subject document")
                self.meta_labels[f.primary_id] = None
        return self

    def construct_dataframe(self):
        return [pd.DataFrame(self.ratios), pd.DataFrame(self.population_statistics),
                pd.DataFrame(self.meta_labels), pd.DataFrame(self.channel_desc)]


def meta_labelling(experiment: Experiment,
                   dataframe: pd.DataFrame,
                   meta_label: str):
    """
    Given a Pandas DataFrame containing a column of sample IDs from
    an Experiment (column should be named 'sample_id') search the
    related Subject of the samples and create a new column for
    the chosen 'meta_label' contained in the related Subject.
    If a sample does not have a related Subject or the 'meta_label'
    cannot be found, the row will be populated with None. Returns
    a mutated DataFrame.

    Parameters
    ----------
    experiment: Experiment
    dataframe: Pandas.DataFrame
    meta_label: str

    Returns
    -------
    Pandas.DataFrame
    """
    assert "sample_id" in dataframe.columns, "Expected column 'sample_id'"
    assert all([s in experiment.list_samples() for s in dataframe["sample_id"]]), \
        "One or more sample IDs not present in given Experiment"
    search_func = partial(fetch_subject_meta,
                          experiment=experiment,
                          meta_label=meta_label)
    df = dataframe.copy()
    df[meta_label] = df["sample_id"].apply(search_func)
    return df


def experiment_statistics(experiment: Experiment,
                          include_subject_id: bool = True):
    """
    Given an Experiment, generate a Pandas DataFrame detailing
    statistics for every population captured in all FileGroups
    contained within the Experiment.

    Parameters
    ----------
    experiment: Experiment
    include_subject_id: bool (default=True)

    Returns
    -------
    Pandas.DataFrame
    """
    data = list()
    for sample_id in experiment.list_samples():
        fg = experiment.get_sample(sample_id)
        df = population_stats(fg)
        df["sample_id"] = sample_id
        if include_subject_id:
            subject = fetch_subject(filegroup=fg)
            if subject is not None:
                df["subject_id"] = subject.subject_id
        data.append(df)
    return pd.concat(data)


def _population_cluster_statistics(pop: Population,
                                   meta_label: str or None,
                                   tag: str or None):
    data = defaultdict(list)
    clusters = pop.get_clusters(tag=tag, meta_labels=meta_label)
    for c in clusters:
        data["prop_of_population"].append(c.prop_of_events)
        data["cluster_id"].append(c.cluster_id)
        data["meta_label"].append(c.meta_label)
        data["tag"].append(c.tag)
        data["n"].append(c.n)
    data = pd.DataFrame(data)
    data["population"] = pop.population_name
    return data


def cluster_statistics(experiment: Experiment,
                       population: str or None = None,
                       meta_label: str or None = None,
                       tag: str or None = None,
                       include_subject_id: bool = True,
                       sample_ids: list or None = None):
    """
    Given an Experiment and the name of a Population known
    to contain clusters from some high-dimensional clustering
    algorithm, this function generates a dataframe of
    statistics. Details include the number of events
    within the cluster and what proportion of the total events
    in the Population this number represents.

    Parameters
    ----------
    experiment: Experiment
    population: str (optional)
        If no population is provided, will search all
        possible populations for clusters
    meta_label: str (optional)
        If given, will filter results to include only
        those clusters with this meta ID
    tag: str (optional)
        If given, will filter results to include only
        those clusters with this tag
    include_subject_id: bool (default=True)
        If True, includes a column for the subject ID in
        the resulting dataframe
    sample_ids: list, optional
        Samples to include, if None will include all samples
    Returns
    -------
    Pandas.DataFrame
    """
    sample_ids = sample_ids or list(experiment.list_samples())
    all_cluster_data = list()
    for sample_id in progress_bar(sample_ids, total=len(sample_ids)):
        fg = experiment.get_sample(sample_id)
        if population is not None:
            data = _population_cluster_statistics(pop=fg.get_population(population_name=population),
                                                  meta_label=meta_label,
                                                  tag=tag)
        else:
            data = list()
            for p in fg.list_populations():
                df = _population_cluster_statistics(pop=fg.get_population(population_name=p),
                                                    meta_label=meta_label,
                                                    tag=tag)
                data.append(df)
            data = pd.concat(data)
        data["sample_id"] = fg.primary_id
        if include_subject_id:
            subject = fetch_subject(filegroup=fg)
            if subject is not None:
                data["subject_id"] = subject.subject_id
        all_cluster_data.append(data)
    return pd.concat(all_cluster_data)


def sort_variance(summary: pd.DataFrame,
                  identifier_columns: list,
                  value_name: str = "summary_stat",
                  var_name: str = "population"):
    """
    Given a dataframe generated by one of the many
    functions in this module, sort that dataframe
    by variance.

    Parameters
    ----------
    summary: Pandas.DataFrame
        Dataframe of summary statistics
    identifier_columns: list
        Columns to use as identifier(s) e.g. sample_id
    value_name: str (default="summary_stat")
    var_name: str (default="population")

    Returns
    -------
    Pandas.DataFrame
    """
    x = summary.melt(id_vars=identifier_columns,
                     value_name=value_name,
                     var_name=var_name)
    return (x.groupby(var_name)
            .var()
            .reset_index()
            .sort_values(value_name, ascending=False)
            .rename(columns={value_name: 'variance'}))


def radar_plot(summary: pd.DataFrame,
               features: list,
               figsize: tuple = (10, 10)):
    """
    Given a Pandas DataFrame where columns are features and each row is a different subject
    (indexed by a column named 'subject_id'), generate a radar plot of all the features
    Parameters
    ----------
    summary: Pandas.DataFrame
    features: List
        Features to be included in the plot
    figsize: tuple, (default=(10,10))
    Returns
    -------
    Matplotlib.axes
    """
    summary = summary.melt(id_vars='subject_id',
                           value_name='stat',
                           var_name='feature')
    summary = summary[summary.feature.isin(features)]
    labels = summary.feature.values
    stats = summary.stat.values

    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False)
    stats = np.concatenate((stats, [stats[0]]))  # Closed
    angles = np.concatenate((angles, [angles[0]]))  # Closed

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, polar=True)  # Set polar axis
    ax.plot(angles, stats, 'o-', linewidth=2, c='blue')
    ax.set_thetagrids(angles * 180 / np.pi, labels)  # Set the label for each axis
    ax.tick_params(pad=30)
    ax.set_rlabel_position(145)
    ax.grid(True)
    return ax


def l1_feature_selection(feature_space: pd.DataFrame,
                         features: list,
                         label: str,
                         scale: bool = True,
                         search_space: tuple = (-2, 0, 50),
                         model: callable or None = None,
                         figsize: tuple = (10, 5)):
    """
    Perform L1 regularised classification over a defined search space for the L1 parameter and plot the
    coefficient of each feature in respect to the change in L1 parameter.
    Parameters
    ----------
    feature_space: Pandas.DataFrame
        A dataframe of features where each column is a feature, each row a subject, and a column whom's name is equal
        to the value of the label argument is the target label for prediction
    features: List
        List of features to include in model
    label: str
        The target label to predict
    scale: bool, (default=True)
        if True, features are scaled (standard scale) prior to analysis
    search_space: tuple, (default=(-2, 0, 50))
        Search range for L1 parameter
    model: callable, optional
        Must be a Scikit-Learn classifier that accepts an L1 regularisation parameter named 'C'. If left as None,
        a linear SVM is used
    figsize: tuple, (default=(10,5))
    Returns
    -------
    Matplotlib.axes
    """

    y = feature_space[label].values
    feature_space = feature_space[features].drop_na()
    if scale:
        scaler = StandardScaler()
        feature_space = pd.DataFrame(scaler.fit_transform(feature_space.values),
                                     columns=feature_space.columns,
                                     index=feature_space.index)
    x = feature_space.values
    cs = np.logspace(*search_space)
    coefs = []
    if model is None:
        model = LinearSVC(penalty='l1',
                          loss='squared_hinge',
                          dual=False,
                          tol=1e-3)
    for c in cs:
        model.set_params(C=c)
        model.fit(x, y)
        coefs.append(list(model.coef_[0]))
    coefs = np.array(coefs)

    # Plot result
    fig, ax = plt.subplots(figsize=figsize)
    for i, col in enumerate(range(len(features))):
        ax.plot(cs, coefs[:, col], label=features[i])
    ax.xscale('log')
    ax.title('L1 penalty')
    ax.xlabel('C')
    ax.ylabel('Coefficient value')
    ax.legend(bbox_to_anchor=(1, 1), loc='upper left', ncol=1)
    return ax


def _prop_of_parent(x: dict,
                    parent: str,
                    experiment: Experiment):
    parent_n = experiment.get_sample(x["sample_id"]).get_population(parent).n
    return x["sample_id"], x["n"] / parent_n


def _subset_and_summarise(data: pd.DataFrame,
                          search_terms: list,
                          parent: str,
                          group_var: str,
                          experiment: Experiment):
    subset_data = [data[data[group_var].str.contains(s, regex=False)] for s in search_terms]
    total_n = {i: x.groupby("sample_id")["n"].sum() for i, x in zip(search_terms, subset_data)}
    proportions = {k: v.reset_index().apply(lambda x: _prop_of_parent(x,
                                                                      parent=parent,
                                                                      experiment=experiment),
                                            axis=1).values
                   for k, v in total_n.items()}
    subsets = pd.DataFrame({subset: {sample_id: v for sample_id, v in x}
                            for subset, x in proportions.items()})
    return subsets


def cluster_subsets(experiment,
                    population,
                    tag,
                    search_terms,
                    sample_ids: list or None,
                    exclusion_terms: list or None):
    """

    Parameters
    ----------
    experiment
    population
    tag
    search_terms
    sample_ids
    exclusion_terms

    Returns
    -------

    """
    exp_data = cluster_statistics(experiment=experiment,
                                  population=population,
                                  tag=tag,
                                  sample_ids=sample_ids)
    if exclusion_terms is not None:
        for e in exclusion_terms:
            exp_data = exp_data[~exp_data.meta_label.str.contains(e)]
    return _subset_and_summarise(data=exp_data,
                                 search_terms=search_terms,
                                 parent=population,
                                 experiment=experiment,
                                 group_var="meta_label")


def population_subsets(experiment,
                       population,
                       search_terms,
                       sample_ids: list or None,
                       exclude: list or None):
    pop_stats = experiment_statistics(experiment)
    sample_ids = sample_ids or list(experiment.list_samples())
    pop_stats = pop_stats[pop_stats.sample_id.isin(sample_ids)]
    if exclude is not None:
        pop_stats = pop_stats[~pop_stats.population_name.isin(exclude)]
    return _subset_and_summarise(data=pop_stats,
                                 parent=population,
                                 search_terms=search_terms,
                                 experiment=experiment,
                                 group_var="population_name")
