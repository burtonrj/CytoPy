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

from ..data.fcs import population_stats, Population
from ..data.experiment import Experiment, fetch_subject_meta, fetch_subject
from collections import defaultdict
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from functools import partial
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

__author__ = "Ross Burton"
__copyright__ = "Copyright 2020, CytoPy"
__credits__ = ["Ross Burton", "Simone Cuff", "Andreas Artemiou", "Matthias Eberl"]
__license__ = "MIT"
__version__ = "1.0.0"
__maintainer__ = "Ross Burton"
__email__ = "burtonrj@cardiff.ac.uk"
__status__ = "Production"


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
    clusters = pop.get_clusters(tag=tag, meta_label=meta_label)
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
                       include_subject_id: bool = True):
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
        If not population is provided, will search all
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
    Returns
    -------
    Pandas.DataFrame
    """
    all_cluster_data = list()
    for sample_id in experiment.list_samples():
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
