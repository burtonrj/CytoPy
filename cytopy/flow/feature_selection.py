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
import logging
from collections import defaultdict
from warnings import warn

import graphviz
import matplotlib.pyplot as plt
import numpy as np
import pingouin
import polars as pl
import seaborn as sns
import shap
from imblearn.over_sampling import RandomOverSampler
from matplotlib.collections import EllipseCollection
from matplotlib.patches import Ellipse
from matplotlib.patches import Patch
from scipy import stats as scipy_stats
from sklearn.decomposition import PCA as SkPCA
from sklearn.inspection import permutation_importance
from sklearn.linear_model import Lasso
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import SGDRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import LinearSVC
from sklearn.svm import LinearSVR
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import DecisionTreeRegressor
from sklearn.tree import export_graphviz
from sklearn.tree import plot_tree
from yellowbrick.regressor import ResidualsPlot

from . import transform
from ..data.experiment import Experiment
from ..feedback import progress_bar
from .cell_classifier import utils as classifier_utils
from .plotting.embeddings_graphs import cont_scatterplot
from .plotting.embeddings_graphs import discrete_scatterplot

logger = logging.getLogger("feature_selection")

__author__ = "Ross Burton"
__copyright__ = "Copyright 2020, cytopy"
__credits__ = ["Ross Burton", "Simone Cuff", "Andreas Artemiou", "Matthias Eberl"]
__license__ = "MIT"
__version__ = "2.0.0"
__maintainer__ = "Ross Burton"
__email__ = "burtonrj@cardiff.ac.uk"
__status__ = "Production"

STATS = {
    "mean": np.mean,
    "SD": np.std,
    "median": np.median,
    "CV": scipy_stats.variation,
    "skew": scipy_stats.skew,
    "kurtosis": scipy_stats.kurtosis,
    "gmean": scipy_stats.gmean,
}

L1CLASSIFIERS = {
    "log": [LogisticRegression, dict(penalty="l1", solver="liblinear")],
    "SGD": [SGDClassifier, dict(penalty="l1")],
    "SVM": [LinearSVC, dict(penalty="l1", loss="squared_hinge", dual=False)],
}

L1REGRESSORS = {
    "lasso": [Lasso, dict()],
    "SGD": [SGDRegressor, dict(penalty="l1")],
    "SVM": [LinearSVR, dict(loss="epsilon_insensitive")],
}


def _fetch_population_statistics(files: list, populations: set):
    return {f.primary_id: {p: f.population_stats(p) for p in populations} for f in files}


def _fetch_subject(x) -> str or None:
    if x.subject is not None:
        return x.subject.subject_id
    return None


def fdr_correction():
    # https://github.com/mne-tools/mne-python/blob/678d333aa890da58689ab52dcf5b5f31d4e10af0/mne/stats/multi_comp.py#L11
    pass


class FeatureSpace:
    """
    Generate a DataFrame of features to use for visualisation, hypothesis testing, feature selection
    and much more. This class allows you to reach into an Experiment and summarise populations associated
    to individual samples. Summary statistics by default contain the following for each population:
    * N - the number of events within the population
    * FOP - the number of events as a fraction of events pertaining to the parent population that this
    population inherits from
    * FOR - the number of events as a fraction of all events in this sample

    Where a population is missing in a sample, these values will be 0. Additional methods in this class
    allow for the injection of:
    * Ratios of populations within a sample
    * Channel descriptive statistics such as mean fluorescent intensity, coefficient of variation, kurotisis,
    skew and many more
    * Meta labels associated to Subjects linked to samples can be used to populate additional columns in
    your resulting DataFrame

    Once the desired data is obtained, calling 'construct_dataframe' results in a polars DataFrame of
    the entire 'feature space'

    Parameters
    ----------
    experiment: Experiment
        The experiment to summarise
    sample_ids: list, optional
        List of sample IDs to be included

    Attributes
    ----------
    sample_ids: dict
    subject_ids: dict
    populations: list
    population_statistics: dict
    ratios: dict
    channel_desc: dict
    meta_labels: dict
    """

    def __init__(
        self,
        experiment: Experiment,
        sample_ids: list or None = None,
        filter_pops: str or None = None,
    ):
        sample_ids = sample_ids or experiment.list_samples()
        self._fcs_files = [x for x in experiment.fcs_files if x.primary_id in sample_ids] or experiment.fcs_files
        self.subject_ids = {x.primary_id: _fetch_subject(x) for x in self._fcs_files}
        self.subject_ids = {k: v for k, v in self.subject_ids.items() if v is not None}
        populations = [x.list_populations() for x in self._fcs_files]
        self.populations = set([x for sl in populations for x in sl])
        if filter_pops is not None:
            self.populations = set([p for p in self.populations if filter_pops in p])
        self.population_statistics = _fetch_population_statistics(files=self._fcs_files, populations=self.populations)
        self.ratios = defaultdict(dict)
        self.channel_desc = defaultdict(dict)
        self.meta_labels = defaultdict(dict)
        self.frac = defaultdict(dict)

    def frac_of(self, pop: str):
        for f in self._fcs_files:
            for pop2 in self.populations:
                try:
                    n = self.population_statistics[f.primary_id][pop]["n"]
                    child_n = self.population_statistics[f.primary_id][pop2]["n"]
                    self.frac[f.primary_id][f"{pop2}_frac_of_{pop}"] = child_n / n
                except KeyError:
                    self.frac[f.primary_id][f"{pop2}_frac_of_{pop}"] = None

    def compute_ratios(self, pop1: str, pop2: str or None = None):
        """
        For each sample compute the ratio of pop1 to pop2. If pop2 is not defined, will compute
        the ratio between pop1 and all other populations. Saved as dictionary to 'ratios' attribute.
        Call 'construct_dataframe' to output as polars.DataFrame.

        Parameters
        ----------
        pop1: str
        pop2: str, optional

        Returns
        -------
        self
        """
        for f in self._fcs_files:
            if pop1 not in f.list_populations():
                logger.warning(f"{f.primary_id} missing population {pop1}")
                if pop2 is None:
                    for p in [q for q in self.populations if q != pop1]:
                        self.ratios[f.primary_id][f"{pop1}:{p}"] = None
                else:
                    self.ratios[f.primary_id][f"{pop1}:{pop2}"] = None
            else:
                p1n = self.population_statistics[f.primary_id][pop1]["n"]
                if pop2 is None:
                    for p in [q for q in self.populations if q != pop1]:
                        if p in f.list_populations():
                            pn = self.population_statistics[f.primary_id][p]["n"]
                            self.ratios[f.primary_id][f"{pop1}:{p}"] = p1n / pn
                        else:
                            self.ratios[f.primary_id][f"{pop1}:{p}"] = None
                else:
                    p2n = self.population_statistics[f.primary_id][pop2]["n"]
                    self.ratios[f.primary_id][f"{pop1}:{pop2}"] = p1n / p2n
        return self

    def channel_desc_stats(
        self,
        channel: str,
        stats: list or None = None,
        channel_transform: str or None = None,
        transform_kwargs: dict or None = None,
        populations: list or None = None,
        verbose: bool = True,
    ):
        """
        For the given channel, generate the statistics given in 'stats', which should contain
        one or more of the following:

        * "mean": arithmetic average
        * "SD": standard deviation
        * "median": median
        * "CV": coefficient of variation
        * "skew": skew
        * "kurtosis": kurtosis
        * "gmean": geometric mean

        Statistics are calculated on a per sample, per population basis.
        Saved as dictionary to 'channel_desc' attribute. Call 'construct_dataframe' to output as polars.DataFrame.

        Parameters
        ----------
        channel: str
            Channel of interest.
        stats: list (default=['mean', 'SD'])
            Statistics to calculate
        channel_transform: str, optional
            Transform to apply to channel before computing stats
        transform_kwargs: dict, optional
            Additional keyword arguments to pass to Transformer
        populations: list, optional
            List of populations to calculate stats for. If not given, stats are computed for all
            available populations in a sample
        verbose: bool (default=True)
            Provide a progress bar

        Returns
        -------
        self
        """
        populations = populations or self.populations
        stats = stats or ["mean", "SD"]
        assert all([x in STATS.keys() for x in stats]), f"Invalid stats; valid stats are: {STATS.keys()}"
        for f in progress_bar(self._fcs_files, verbose=verbose):
            for p in populations:
                if p not in f.list_populations():
                    logger.debug(f"{f.primary_id} missing population {p}")
                    for s in stats:
                        self.channel_desc[f.primary_id][f"{p}_{channel}_{s}"] = None
                else:
                    x = f.load_population_df(
                        population=p,
                        transform=channel_transform,
                        features_to_transform=[channel],
                        transform_kwargs=transform_kwargs,
                    )[channel].values
                    for s in stats:
                        self.channel_desc[f.primary_id][f"{p}_{channel}_{s}"] = STATS.get(s)(x)
        return self

    def add_meta_labels(self, key: str or list, meta_label: str or None = None):
        """
        Search associated subjects for meta variables. You should provide a key as a string or a list of
        strings. If it is a string, this should be the name of an immediate field in the Subject document
        for which you want a column in your resulting DataFrame. If key is a list of strings, then this
        will be interpreted as a tree structure along which to navigate. So for example, if the
        key is ["disease", "category", "short_name"] then the value for the field "short_name", embedded
        in the field "category", embedded in the field "disease", will be used as the value to populate
        a new column. The column name will be the same as the last value in key or meta_label if defined.

        Parameters
        ----------
        key: str or List
        meta_label: str, optional

        Returns
        -------
        self
        """
        for f in self._fcs_files:
            subject = f.subject
            if subject is None:
                self.meta_labels[f.primary_id][meta_label] = None
                continue
            try:
                if isinstance(key, str):
                    meta_label = meta_label or key
                    self.meta_labels[f.primary_id][meta_label] = subject[key]
                else:
                    node = subject[key[0]]
                    for k in key[1:]:
                        node = node[k]
                    meta_label = meta_label or key[len(key) - 1]
                    self.meta_labels[f.primary_id][meta_label] = node
            except KeyError:
                logger.warning(f"{f.primary_id} missing meta variable {key} in Subject document")
                if meta_label is None:
                    meta_label = key if isinstance(key, str) else key[0]
                self.meta_labels[f.primary_id][meta_label] = None
        return self

    def construct_dataframe(
        self,
        n: bool = True,
        frac_of_root: bool = True,
        frac_of_parent: bool = True,
        pop_prefix: str = "",
    ):
        """
        Generate a DataFrame of the feature space collected within this FeatureSpace object, detailing
        populations of an experiment with the addition of ratios, channel stats, and meta labels.

        Returns
        -------
        polars.DataFrame
        """
        data = defaultdict(list)
        for sample_id, populations in self.population_statistics.items():
            data["sample_id"].append(sample_id)
            data["subject_id"].append(self.subject_ids.get(sample_id, None))
            for pop_name, pop_stats in populations.items():
                if n:
                    data[f"{pop_prefix}{pop_name}_N"].append(pop_stats["n"])
                if frac_of_root:
                    data[f"{pop_prefix}{pop_name}_FOR"].append(pop_stats["frac_of_root"])
                if frac_of_parent:
                    data[f"{pop_prefix}{pop_name}_FOP"].append(pop_stats["frac_of_parent"])

            if self.ratios:
                for n, r in self.ratios.get(sample_id).items():
                    data[n].append(r)
            if self.channel_desc:
                for n, s in self.channel_desc.get(sample_id).items():
                    data[n].append(s)
            if self.meta_labels:
                for m, v in self.meta_labels.get(sample_id).items():
                    data[m].append(v)
        return pl.DataFrame(data)


def clustered_heatmap(
    data: pl.DataFrame,
    features: list,
    index: str,
    row_colours: str or None = None,
    row_colours_cmap: str = "tab10",
    legend_kwargs: dict or None = None,
    **kwargs,
):
    """
    Generate a clustered heatmap using Seaborn's clustermap function. Has the additional
    option to colour rows using some specified column in data.

    Parameters
    ----------
    data: polars.DataFrame
        Target data. Must contain columns for features, index and row_colours (if given)
    features: list
        List of primary features to make up the columns of the heatmap
    index: str
        Name of the column to use as rows of the heatmap
    row_colours: str, optional#a848ab
        Column to use for an additional coloured label for row categories
    row_colours_cmap: str (default='tab10')
        Colour map to use for row categories
    kwargs:
        Additional keyword arguments passed to Seaborn clustermap call

    Returns
    -------
    Seaborn.ClusterGrid
    """
    legend_kwargs = legend_kwargs or {}
    df = data.set_index(index, drop=True)[features].copy()
    if row_colours is not None:
        row_colours_title = row_colours
        lut = dict(zip(data[row_colours].unique(), row_colours_cmap))
        row_colours = data[row_colours].map(lut)
        handles = [Patch(facecolor=lut[name]) for name in lut]
        g = sns.clustermap(df, row_colors=row_colours.values, **kwargs)
        plt.legend(
            handles,
            lut,
            title=row_colours_title,
            bbox_to_anchor=legend_kwargs.get("bbox_to_anchor", (1, 1)),
            bbox_transform=plt.gcf().transFigure,
            loc=legend_kwargs.get("log", "upper right"),
        )
    else:
        g = sns.clustermap(df, **kwargs)
    return g


def box_swarm_plot(
    plot_df: pl.DataFrame,
    x: str,
    y: str,
    hue: str or None = None,
    ax: plt.Axes or None = None,
    palette: str or None = None,
    overlay: bool = True,
    boxplot_kwargs: dict or None = None,
    overlay_kwargs: dict or None = None,
):
    """
    Convenience function for generating a boxplot with a swarmplot/stripplot overlaid showing
    individual datapoints (using tools from Seaborn library)

    Parameters
    ----------
    plot_df: polars.DataFrame
        Data to plot
    x: str
        Name of the column to use as x-axis variable
    y: str
        Name of the column to use as x-axis variable
    hue: str, optional
        Name of the column to use as factor to colour plot
    overlay: bool (default=True)
        Overlay swarm plot on boxplot
    ax: Matplotlib.Axes, optional
        Axis object to plot on. If None, will generate new axis of figure size (10,5)
    palette: str, optional
        Palette to use
    boxplot_kwargs: dict, optional
        Additional keyword arguments passed to Seaborn.boxplot
    overlay_kwargs: dict, optional
        Additional keyword arguments passed to Seaborn.swarmplot/stripplot

    Returns
    -------
    Matplotlib.Axes
    """
    boxplot_kwargs = boxplot_kwargs or {}
    overlay_kwargs = overlay_kwargs or {}
    ax = ax or plt.subplots(figsize=(10, 5))[1]
    sns.boxplot(
        data=plot_df,
        x=x,
        y=y,
        hue=hue,
        ax=ax,
        showfliers=False,
        boxprops=dict(alpha=0.3),
        palette=palette,
        **boxplot_kwargs,
    )
    if overlay:
        sns.stripplot(
            data=plot_df,
            x=x,
            y=y,
            hue=hue,
            ax=ax,
            dodge=True,
            palette=palette,
            **overlay_kwargs,
        )
    return ax


class InferenceTesting:
    """
    This class provides convenient functionality for common statistical inference tests.

    Parameters
    ----------
    data: polars.DataFrame
        Tabular data containing all dependent and independent variables
    scale: str, optional
        Scale data upon initiating object using one of the scaling methods provided
        by cytopy.flow.transform.Scaler
    scale_vars: List, optional
        Columns to scale. Must provide is scale is provided.
    scale_kwargs: dict, optional
        Additional keyword arguments passed to Scaler

    Attributes
    ----------
    data: polars.DataFrame
    scaler: CytoPy.flow.transform.Scaler
    """

    def __init__(
        self,
        data: pl.DataFrame,
        scale: str or None = None,
        scale_vars: list or None = None,
        scale_kwargs: dict or None = None,
    ):
        self.data = data.copy()
        self.scaler = None
        if scale is not None:
            scale_kwargs = scale_kwargs or {}
            self.scaler = transform.Scaler(method=scale, **scale_kwargs)
            assert scale_vars is not None, "Must provide variables to scale"
            self.data = self.scaler(data=self.data, features=scale_vars)

    def qq_plot(self, var: str, **kwargs):
        """
        Generate a QQ plot for the given variable

        Parameters
        ----------
        var: str
        kwargs:
            Additional keyword arguments passed to pingouin.qqplot

        Returns
        -------
        Matplotlib.Axes

        Raises
        ------
        AssertionError
            If var is not a valid column in data attribute
        """
        assert var in self.data.columns, "Invalid variable"
        return pingouin.qqplot(x=self.data[var].values, **kwargs)

    def normality(self, var: list, method: str = "shapiro", alpha: float = 0.05):
        """
        Check the normality of variables in associated data

        Parameters
        ----------
        var: list
            List of variables
        method: str (default='shapiro')
            See pingouin.normality for available methods
        alpha: float (default=0.05)
            Significance level

        Returns
        -------
        polars.DataFrame
            Contains two columns, one is the variable name the other is a boolean value as to
            whether it is normally distributed
        """
        results = {"Variable": list(), "Normal": list()}
        for i in var:
            results["Variable"].append(i)
            results["Normal"].append(
                pingouin.normality(self.data[i].values, method=method, alpha=alpha).iloc[0]["normal"]
            )
        return pl.DataFrame(results)

    def anova(
        self,
        dep_var: str,
        between: str,
        post_hoc: bool = True,
        post_hoc_kwargs: dict or None = None,
        **kwargs,
    ):
        """
        Classic one-way analysis of variance; performs welch anova if assumption of equal variance is broken.

        Parameters
        ----------
        dep_var: str
            Name of the column containing the dependent variable (what we're interested in measuring)
        between: str
            Name of the column containing grouping variable that divides our independent groups
        post_hoc: bool (default=True)
            If True, perform the suitable post-hoc test; for normal anova this is a pairwise Tukey
            test and for a welch anova it is a Games-Howell test
        post_hoc_kwargs: dict, optional
            Keyword arguments passed to post-hoc test
        kwargs:
            Additional keyword arguments passed to the respective pingouin anova function

        Returns
        -------
        polars.DataFrame, polars.DataFrame or None
            DataFrame of ANOVA results and DataFrame of post-hoc test results if post_hoc is True

        Raises
        ------
        AssertionError
            If assumption of normality is broken
        """
        post_hoc_kwargs = post_hoc_kwargs or {}
        err = "Chosen dependent variable must be normally distributed"
        assert all(
            [pingouin.normality(df[dep_var].values).iloc[0]["normal"] for _, df in self.data.groupby(between)]
        ), err
        eq_var = all(
            [
                pingouin.homoscedasticity(df[dep_var].values).iloc[0]["equal_var"]
                for _, df in self.data.groupby(between)
            ]
        )
        if eq_var:
            aov = pingouin.anova(data=self.data, dv=dep_var, between=between, **kwargs)
            if post_hoc:
                return aov, pingouin.pairwise_tukey(data=self.data, dv=dep_var, between=between, **post_hoc_kwargs)
            return aov, None
        aov = pingouin.welch_anova(data=self.data, dv=dep_var, between=between, **kwargs)
        if post_hoc:
            return aov, pingouin.pairwise_gameshowell(data=self.data, dv=dep_var, between=between, **post_hoc_kwargs)
        return aov, None

    def ttest(
        self,
        between: str,
        dep_var: list,
        paired: bool = False,
        multicomp: str = "holm",
        multicomp_alpha: float = 0.05,
        **kwargs,
    ):
        """
        Performs a classic T-test; Welch T-test performed if assumption of equal variance is broken.

        Parameters
        ----------
        between: str
            Name of the column containing grouping variable that divides our independent groups
        dep_var: str
            Name of the column containing the dependent variable (what we're interested in measuring).
            More than one variable can be provided as a list and correction for multiple comparisons
            made according to the method specified in 'multicomp'
        paired: bool (default=False)
            Perform paired T-test (i.e. samples are paired)
        multicomp: str (default='holm')
            Method to perform for multiple comparison correction if length of dep_var is greater than 1
        multicomp_alpha: float (default=0.05)
            Significance level for multiple comparison correction
        kwargs:
            Additional keyword arguments passed to pingouin.ttest

        Returns
        -------
        polars.DataFrame
            DataFrame of T-test results

        Raises
        ------
        AssertionError
            If assumption of normality is broken

        ValueError
            More than two unique groups in the 'between' column
        """
        if self.data[between].nunique() > 2:
            raise ValueError("More than two groups, consider using 'anova' method")
        x, y = self.data[between].unique()
        x, y = (
            self.data[self.data[between] == x].copy(),
            self.data[self.data[between] == y].copy(),
        )
        results = list()
        for i in dep_var:
            assert all(
                [pingouin.normality(df[i].values).iloc[0]["normal"] for _, df in self.data.groupby(between)]
            ), f"Groups for {i} are not normally distributed"
            eq_var = all(
                [pingouin.homoscedasticity(df[i].values).iloc[0]["equal_var"] for _, df in self.data.groupby(between)]
            )
            if eq_var:
                tstats = pingouin.ttest(
                    x=x[i].values,
                    y=y[i].values,
                    paired=paired,
                    correction=False,
                    **kwargs,
                )
            else:
                tstats = pingouin.ttest(
                    x=x[i].values,
                    y=y[i].values,
                    paired=paired,
                    correction=True,
                    **kwargs,
                )
            tstats["Variable"] = i
            results.append(tstats)
        results = pl.concat(results)
        if len(dep_var) > 1:
            results["p-val"] = pingouin.multicomp(results["p-val"].values, alpha=multicomp_alpha, method=multicomp)
        return results

    def non_parametric(
        self,
        between: str,
        dep_var: list,
        paired: bool = False,
        multicomp: str = "holm",
        multicomp_alpha: float = 0.05,
        **kwargs,
    ):
        """
        Non-parametric tests for paired and un-paired samples:
        * If more than two unique groups, performs Friedman test for paired samples or Kruskal–Wallis
        for unpaired
        * If only two unique groups, performs Wilcoxon signed-rank test for paired samples or
        Mann-Whitney U test for unpaired

        Parameters
        ----------
        between: str
            Name of the column containing grouping variable that divides our independent groups
        dep_var: str
            Name of the column containing the dependent variable (what we're interested in measuring).
            More than one variable can be provided as a list and correction for multiple comparisons
            made according to the method specified in 'multicomp'
        paired: bool (default=False)
            Perform paired testing (i.e. samples are paired)
        multicomp: str (default='holm')
            Method to perform for multiple comparison correction if length of dep_var is greater than 1
        multicomp_alpha: float (default=0.05)
            Significance level for multiple comparison correction
        kwargs:
            Additional keyword arguments passed to respective pingouin function

        Returns
        -------
        polars.DataFrame
        """
        results = list()
        if self.data[between].nunique() > 2:
            if paired:
                for i in dep_var:
                    np_stats = pingouin.friedman(data=self.data, dv=i, within=between, **kwargs)
                    np_stats["Variable"] = i
                    results.append(np_stats)
            else:
                for i in dep_var:
                    np_stats = pingouin.kruskal(data=self.data, dv=i, between=between, **kwargs)
                    np_stats["Variable"] = i
                    results.append(np_stats)
        else:
            x, y = self.data[between].unique()
            x, y = (
                self.data[self.data[between] == x].copy(),
                self.data[self.data[between] == y].copy(),
            )
            if paired:
                for i in dep_var:
                    np_stats = pingouin.wilcoxon(x[i].values, y[i].values, **kwargs)
                    np_stats["Variable"] = i
                    results.append(np_stats)
            else:
                for i in dep_var:
                    np_stats = pingouin.mwu(x[i].values, y[i].values, **kwargs)
                    np_stats["Variable"] = i
                    results.append(np_stats)
        results = pl.concat(results)
        if len(dep_var) > 1:
            results["p-val"] = pingouin.multicomp(results["p-val"].values, alpha=multicomp_alpha, method=multicomp)[1]
        return results


def plot_multicollinearity(
    data: pl.DataFrame,
    features: list,
    method: str = "spearman",
    ax: plt.Axes or None = None,
    plot_type: str = "ellipse",
    **kwargs,
):
    """
    Generate a pairwise correlation matrix to help detect multicollinearity between
    independent variables.

    Parameters
    ----------
    data: polars.DataFrame
        DataFrame of variables to test; must contain the variables as columns in 'features'
    features: list
        List of columns to use for correlations
    method: str (default="spearman")
        Correlation coefficient; must be either 'pearson', 'spearman' or 'kendall'
    plot_type: str, (default="ellipse")
        Specifies the type of plot to generate:
        * 'ellipse' - this generates a matrix of ellipses (similar to the plotcorr library in R). Each
        ellipse is coloured by the intensity of the correlation and the angle of the ellipse demonstrates
        the relationship between variables
        * 'matrix' - clustered correlation matrix using the Seaborn.clustermap function
    ax: Matplotlib.Axes, optional
    kwargs:
        Additional keyword arguments; passed to Matplotlib.patches.EllipseCollection in the case of
        method = 'ellipse' or passed to seaborn.clustermap in the case of method = 'matrix'

    Returns
    -------
    (Matplotlib.Axes, Matplotlib.collections.Collection) or (Seaborn.Clustergrid, None)
    """
    corr = data[features].corr(method=method)
    if plot_type == "ellipse":
        ax = ax or plt.subplots(figsize=(8, 8), subplot_kw={"aspect": "equal"})[1]
        ax.set_xlim(-0.5, corr.shape[1] - 0.5)
        ax.set_ylim(-0.5, corr.shape[0] - 0.5)
        xy = np.indices(corr.shape)[::-1].reshape(2, -1).T
        w = np.ones_like(corr.values).ravel()
        h = 1 - np.abs(corr.values).ravel()
        a = 45 * np.sign(corr.values).ravel()
        ec = EllipseCollection(
            widths=w,
            heights=h,
            angles=a,
            units="x",
            offsets=xy,
            transOffset=ax.transData,
            array=corr.values.ravel(),
            **kwargs,
        )
        ax.add_collection(ec)
        ax.set_xticks(np.arange(corr.shape[1]))
        ax.set_xticklabels(corr.columns, rotation=90)
        ax.set_yticks(np.arange(corr.shape[0]))
        ax.set_yticklabels(corr.index)
        return ax, ec
    return sns.clustermap(data=corr, **kwargs), None


class PCA:
    """
    This class provides convenient functionality for principle component analysis (PCA) with
    plotting methods and tools for inspecting this model. PCA is an easily interpreted model
    for dimension reduction through the linear combination of your independent variables.

    Parameters
    ----------
    data: polars.DataFrame
        Tabular data to investigate, must contain variables given in 'features'. Additional columns
        can be included to colour data points in plots (see 'plot' method)
    features: list
        List of features used in PCA model
    scale: str, optional (default='standard')
        How data should be scaled prior to generating PCA. See cytopy.flow.transform.Scaler for
        available methods.
    scale_kwargs: dict, optional
        Additional keyword arguments passed to Scaler
    kwargs:
        Additional keyword arguments passed to sklearn.decomposition.PCA

    Attributes
    ----------
    data: polars.DataFrame
    features: list
    scaler: Scaler
    pca: sklearn.decomposition.PCA
    embeddings: numpy.ndarray
        Principle components of shape (n_samples, n_components). Populated upon call to 'fit', otherwise None.
    """

    def __init__(
        self,
        data: pl.DataFrame,
        features: list,
        scale: str or None = "standard",
        scale_kwargs: dict or None = None,
        **kwargs,
    ):
        self.scaler = None
        self.data = data.dropna(axis=0).reset_index(drop=True)
        self.features = features
        if scale is None:
            warn("PCA requires that input variables have unit variance and therefore scaling is recommended")
        else:
            scale_kwargs = scale_kwargs or {}
            self.scaler = transform.Scaler(method=scale, **scale_kwargs)
            self.data = self.scaler(self.data, self.features)
        kwargs = kwargs or dict()
        kwargs["random_state"] = kwargs.get("random_state", 42)
        self.pca = SkPCA(**kwargs)
        self.embeddings = None

    def fit(self):
        """
        Fit model and populate embeddings

        Returns
        -------
        self
        """
        self.embeddings = self.pca.fit_transform(self.data[self.features])
        return self

    def scree_plot(self, **kwargs):
        """
        Generate a scree plot of the explained variance of each component; useful
        to assess the explained variance of the PCA model and which components
        to plot.

        Parameters
        ----------
        kwargs:
            Additional keyword argument passed to Seaborn.barplot call

        Returns
        -------
        Matplotlib.Axes

        Raises
        ------
        AssertionError
            If function called prior to calling 'fit'
        """
        assert self.embeddings is not None, "Call fit first"
        var = pl.DataFrame(
            {
                "Variance Explained": self.pca.explained_variance_ratio_,
                "PC": [f"PC{i + 1}" for i in range(len(self.pca.explained_variance_ratio_))],
            }
        )
        return sns.barplot(data=var, x="PC", y="Variance Explained", ci=None, **kwargs)

    def loadings(self, component: int = 0):
        """
        The loadings of a component are the coefficients of the linear combination of
        the original variables from which the principle component was constructed. They
        give some indication of the contribution of a variable to the explained variance
        of a component.

        Parameters
        ----------
        component: int (default=0)
            The component to inspect; by default the first component is chosen (indexed at 0)
            as this component maintains the maximum variance of the data.

        Returns
        -------
        polars.DataFrame
            Columns: Feature (listing the variable names) and EV Magnitude (the coefficient of each
            feature within this component)
        """
        assert self.embeddings is not None, "Call fit first"
        return pl.DataFrame(
            {
                "Feature": self.features,
                "EV Magnitude": abs(self.pca.components_)[component],
            }
        )

    def plot(
        self,
        label: str,
        size: int = 5,
        components: list or None = None,
        discrete: bool = True,
        cmap: str = "tab10",
        loadings: bool = False,
        limit_loadings: list or None = None,
        arrow_kwargs: dict or None = None,
        ellipse: bool = False,
        ellipse_kwargs: dict or None = None,
        figsize: tuple or None = (5, 5),
        cbar_kwargs: dict or None = None,
        **kwargs,
    ):
        """
        Generate a plot of either 2 or 3 components (the latter generates a 3D plot). Data
        point are coloured using an existing column in 'data' attribute.

        Parameters
        ----------
        label: str
            Column to use to colour data points
        size: int (default=5)
            Data point size
        components: list (default=(0, 1))
            The index of components to plot. Components index starts at 0 and this list must
            be of length 2 or 3.
        discrete: bool (default=True)
            If the label should be treated as a discrete variable or continous
        cmap: str (default='tab10')
            Colour mapping to use for label. Choose an appropriate colour map depending on whether
            the label is discrete or continuous; we recommend 'tab10' for discrete and 'coolwarm'
            for continuous.
        loadings: bool (default=False)
            If True, loadings are plotted as labelled arrows showing the direction and magnitude
            of coefficients
        limit_loadings: list, optional
            If given, loadings are limited to include only the given features
        arrow_kwargs: dict, optional
            Additional keyword arguments passed to Matplotlib.Axes.arrow
        ellipse: bool (default=False)
            Whether to plot a confidence ellipse for the distribution of each label
        ellipse_kwargs:
            Additional keyword arguments passed to Matplotlib.patches.Ellipse. Can also
            include an additional argument 's' (of type int) which specifies the number of standard
            deviations to use for confidence ellipse; defaults to 3 standard deviations
        figsize: tuple (default=(5, 5))
            Figure size
        cbar_kwargs:
            Additional keyword arguments passed to colourbar
        kwargs:
            Additional keyword argument passed to Matplotlib.Axes.scatter

        Returns
        -------
        Matplotlib.Figure, Matplotlib.Axes

        Raises
        ------
        AssertionError
            If function called prior to calling fit

        ValueError
            Invalid number of components provided

        TypeError
            Ellipse requested for non-discrete label

        IndexError
            Chosen colourmap specifies less unique colours than the number of unique values
            in label
        """
        components = components or [0, 1]
        if not 2 <= len(components) <= 3:
            raise ValueError("Components should be of length 2 or 3")
        assert self.embeddings is not None, "Call fit first"
        plot_df = pl.DataFrame({f"PC{i + 1}": self.embeddings[:, i] for i in components})
        plot_df[label] = self.data[label]
        fig = plt.figure(figsize=figsize)
        z = None
        if len(components) == 3:
            z = f"PC{components[2] + 1}"
        if discrete:
            ax = discrete_scatterplot(
                data=plot_df,
                x=f"PC{components[0] + 1}",
                y=f"PC{components[1] + 1}",
                z=z,
                label=label,
                cmap=cmap,
                size=size,
                fig=fig,
                **kwargs,
            )
        else:
            cbar_kwargs = cbar_kwargs or {}
            ax = cont_scatterplot(
                data=plot_df,
                x=f"PC{components[0] + 1}",
                y=f"PC{components[1] + 1}",
                z=z,
                label=label,
                cmap=cmap,
                size=size,
                fig=fig,
                cbar_kwargs=cbar_kwargs,
                **kwargs,
            )
        if loadings:
            if len(components) != 2:
                ValueError("cytopy only supports 2D byplots")
            arrow_kwargs = arrow_kwargs or {}
            arrow_kwargs["color"] = arrow_kwargs.get("color", "r")
            arrow_kwargs["alpha"] = arrow_kwargs.get("alpha", 0.5)
            features_i = list(range(len(self.features)))
            if limit_loadings:
                features_i = [i for i, x in enumerate(self.features) if x in limit_loadings]
            ax = self._add_loadings(components=components, ax=ax, features_i=features_i, **arrow_kwargs)
        if ellipse:
            if len(components) != 2:
                ValueError("cytopy only supports confidence ellipse for 2D plots")
            if not discrete:
                TypeError("Ellipse only value for discrete label")
            ellipse_kwargs = ellipse_kwargs or {}
            ax = self._add_ellipse(components=components, label=label, cmap=cmap, ax=ax, **ellipse_kwargs)
        return fig, ax

    def _add_loadings(self, components: list, ax: plt.Axes, features_i: list, **kwargs):
        coeffs = np.transpose(self.pca.components_[np.array(components), :])
        for i in features_i:
            ax.arrow(0, 0, coeffs[i, 0], coeffs[i, 1], **kwargs)
            ax.text(
                coeffs[i, 0] * 1.15,
                coeffs[i, 1] * 1.15,
                self.features[i],
                color="b",
                ha="center",
                va="center",
            )
        return ax

    def _add_ellipse(self, components: list, label: str, cmap: str, ax: plt.Axes, **kwargs):
        kwargs = kwargs or {}
        s = kwargs.pop("s", 3)
        kwargs["linewidth"] = kwargs.get("linewidth", 2)
        kwargs["edgecolor"] = kwargs.get("edgecolor", "#383838")
        kwargs["alpha"] = kwargs.get("alpha", 0.2)
        colours = plt.get_cmap(cmap).colors
        if len(colours) < self.data[label].nunique():
            raise IndexError("Chosen cmap doesn't contain enough unique colours")
        for l, c in zip(self.data[label].unique(), colours):
            idx = self.data[self.data[label] == l].index.values
            x, y = (
                self.embeddings[idx, components[0]],
                self.embeddings[idx, components[1]],
            )
            cov = np.cov(x, y)
            v, w = np.linalg.eig(cov)
            v = np.sqrt(v)
            ellipse = Ellipse(
                xy=(np.mean(x), np.mean(y)),
                width=v[0] * s * 2,
                height=v[1] * s * 2,
                angle=np.rad2deg(np.arccos(w[0, 0])),
                facecolor=c,
                **kwargs,
            )
            ax.add_artist(ellipse)
        return ax


class L1Selection:
    """
    A method for eliminating redundant variables is to apply an L1 regularisation penalty to linear
    models; in linear regression this is referred to as 'lasso' regression. The l1 norm of the weight vector
    is added to the cost function and results in the weights of less important features (i.e. those with
    small coefficients that do not contribute as much to predictions) being eliminated; it produces a sparse
    model that only includes the features that matter most.

    You must always ensure that the assumptions of the model used are upheld. Make sure to investigate these
    assumptions before proceeding. Common assumptions are:
    * Features are independent of one another; you should try to eliminate as much
    multicollinearity as possible prior to performing L1 selection
    * Linear models such as lasso regression assume that the residuals are normally distrbuted and
    have equal variance. You can plot the residuals using the 'residuals_plot' function to test
    this assumption

    Parameters
    ----------
    data: polars.DataFrame
        Feature space for classification/regression; must contain columns for features and target.
    target: str
        Endpoint for regression/classification; must be a column in 'data'
    features: list
        List of columns in 'data' to use as feature space
    model: str
        Model to use. If performing classification (i.e. target is discrete):
        * 'log' - Logistic regression (sklearn.linear_model.LogisticRegression) with set parameters
        penalty='l1' and solver='liblinear'
        * SGD - stochastic gradient descent (sklearn.linear_model.SGDClassifier) with L1 penalty.
        Defaults to linear support vector machine ('hinge' loss function) but can be controlled by
        changing the loss parameter (https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.SGDClassifier.html)
        * SVM - linear support vector machine (sklearn.svm.LinearSVC) with set parameters penalty = 'l1',
        loss = 'squared_hinge' and dual = False.

        If performing regression (i.e. target is continuous):
        * lasso - lasso regression (sklear.linear_model.Lasso)
        * stochastic gradient descent (sklearn.linear_model.SGDClassifier) with L1 penalty.
        Defaults to ordinary least squares ('squared_loss' loss function) but can be controlled by
        changing the loss parameter (https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.SGDRegressor.html)
        * SVM - linear support vector machine (sklearn.svm.LinearSVR) with set parameter loss="epsilon_insensitive"
    category: str (default='classification')
        Specifies if the task is one of classification (discrete target variable) or regression
        (continuous target variable)
    scale: str, optional (default='standard')
        Whether to scale data prior to fitting model; if given, indicates which method to use, see
        cytopy.flow.transform.Scaler for valid methods
    scale_kwargs: dict, optional
        Keyword arguments to pass to Scaler
    kwargs:
        Additional keyword arguments passed to construction of Scikit-Learn classifier/regressor

    Attributes
    ----------
    model: Scikit-Learn classifier/regressor
    scaler: CytoPy.flow.transform.Scaler
    features: list
    x: polars.DataFrame
        Feature space
    y: numpy.ndarry
        Target
    scores: polars.DataFrame
        Feature coefficients under a given value for the regularisation penalty; populated
        upon calling 'fit'
    """

    def __init__(
        self,
        data: pl.DataFrame,
        target: str,
        features: list,
        model: str,
        category: str = "classification",
        scale: str or None = "standard",
        scale_kwargs: dict or None = None,
        **kwargs,
    ):
        scale_kwargs = scale_kwargs or {}

        if category == "classification":
            self._category = "classification"
            assert model in L1CLASSIFIERS.keys(), f"Invalid model must be one of: {L1CLASSIFIERS.keys()}"
            assert data[target].nunique() == 2, "L1Selection only supports binary classification"
            klass, req_kwargs = L1CLASSIFIERS.get(model)
        elif category == "regression":
            self._category = "regression"
            assert model in L1REGRESSORS.keys(), f"Invalid model must be one of: {L1REGRESSORS.keys()}"
            klass, req_kwargs = L1REGRESSORS.get(model)
        else:
            raise ValueError("Category should be 'classification' or 'regression'")

        for k, v in req_kwargs.items():
            kwargs[k] = v

        self.model = klass(**kwargs)
        self._reg_param = "C"
        if "alpha" in self.model.get_params().keys():
            self._reg_param = "alpha"

        data = data.dropna(axis=0).reset_index(drop=True)
        self.scaler = None

        if scale:
            self.scaler = transform.Scaler(method=scale, **scale_kwargs)
            data = self.scaler(data=data, features=features)
        self.x, self.y = data[features], data[target].values
        self.features = features

        self.scores = None

    def fit(self, search_space: tuple = (-2, 0, 50), **kwargs):
        """
        Given a range of L1 penalties (search_space) fit the model and store the
        coefficients of each feature in the 'scores' attribute.

        Parameters
        ----------
        search_space: tuple (default=-2, 0, 50)
            Used to generate a search space for L1 penalty using the Numpy logspace function.
            By default, generates a range of length 50 between 0.01 (10^-2) and 1 (10^0).
        kwargs:
            Additional keyword arguments passed to the 'fit' call of 'model'

        Returns
        -------
        self
        """
        search_space = np.logspace(*search_space)
        coefs = list()
        for r in search_space:
            self.model.set_params(**{self._reg_param: r})
            self.model.fit(self.x, self.y, **kwargs)
            if self._category == "classification":
                coefs.append(list(self.model.coef_[0]))
            else:
                coefs.append(list(self.model.coef_))
        self.scores = pl.DataFrame(np.array(coefs), columns=self.features)
        self.scores[self._reg_param] = search_space
        return self

    def plot(
        self,
        ax: plt.Axes or None = None,
        title: str = "L1 Penalty",
        xlabel: str = "Regularisation parameter",
        ylabel: str = "Coefficient",
        cmap: str = "tab10",
        **kwargs,
    ):
        """
        Plot the coefficients of each feature against L1 penalty. Assumes 'fit' has been called
        prior.

        Parameters
        ----------
        ax: Matplotlig.Axes, optional
        title: str (default="L1 Penalty")
        xlabel: str (default="Regularisation parameter")
        ylabel: str (default="Coefficient")
        cmap: str (default="tab10")
        kwargs:
            Additional keyword argument pased to Matplotlib.Axes.plot

        Returns
        -------
        Matplotlib.Axes

        Raises
        ------
        AssertionError
            'fit' not called prior to calling 'plot'
        """
        ax = ax or plt.subplots(figsize=(10, 5))[1]
        assert self.scores is not None, "Call fit prior to plot"
        colours = plt.get_cmap(cmap).colors
        for i, feature in enumerate(self.features):
            ax.plot(
                self.scores[self._reg_param],
                self.scores[feature],
                label=feature,
                color=colours[i],
                **kwargs,
            )
        ax.set_xscale("log")
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend(bbox_to_anchor=(1.1, 1))
        return ax

    def plot_residuals(self, val_frac: float = 0.5, **kwargs):
        """

        Parameters
        ----------
        val_frac: float (default=0.5)
            Fraction of data to use for validation
        kwargs:
            Additional keyword arguments passed to yellowbrick.regressor.ResidualsPlot
            (see https://www.scikit-yb.org/en/latest/api/regressor/residuals.html)

        Returns
        -------
        Matplotlib.Figure

        Raises
        ------
        AssertionError
            plot_residuals only valid for regression models
        """
        assert self._category == "regression", "plot_residuals only valid for regression models"
        x_train, x_test, y_train, y_test = train_test_split(self.x, self.y, test_size=val_frac, random_state=42)
        viz = ResidualsPlot(self.model, **kwargs)
        viz.fit(x_train, y_train)
        viz.score(x_test, y_test)
        return viz.show()


class DecisionTree:
    """
    Decision tree's offer non-linear modelling for both regression and classification tasks, whilst
    also being simple to interpret and offers information regarding feature interactions. Their simplicity
    comes with a trade-off as they are prone to overfitting and can therefore be misinterpreted.
    Therefore the DecisionTree class offers validation methods and pruning to improve the reliability
    of results. Despite this, we recommend that care be taking when constructing decision trees and
    that the number of features limited.

    Parameters
    ----------
    data: polars.DataFrame
        Feature space for classification/regression; must contain columns for features and target.
    target: str
        Endpoint for regression/classification; must be a column in 'data'
    features: list
        List of columns in 'data' to use as feature space
    tree_type: str (default='classification')
        Should either be 'classification' or 'regression'
    balance_classes: str (default='sample')
        Class imbalance is a significant issue in decision tree classifier and should be addressed
        prior to fitting the model. This parameter specifies how to address this issue. Should either
        be 'balanced' which will result in class weights being included in the cost function or 'sample'
        to perform random over sampling (with replacement) of the under represented class
    sampling_kwargs: dict, optional
        Additional keyword arguments passed to RandomOverSampler class of imbalance-learn
    kwargs:
        Additional keyword arguments passed to DecisionTreeClassifier/DecisionTreeRegressor

    Attributes
    ----------
    x: polars.DataFrame
        Feature space
    y: numpy.ndarray
        Target array
    features: list
    tree_builder: sklearn.tree.DecisionTreeClassifier/sklearn.tree.DecisionTreeRegressor
    """

    def __init__(
        self,
        data: pl.DataFrame,
        target: str,
        features: list,
        tree_type: str = "classification",
        balance_classes: str = "sample",
        sampling_kwargs: dict or None = None,
        **kwargs,
    ):
        data = data.dropna(axis=0).reset_index(drop=True)
        self.x, self.y = data[features], data[target].values
        self.features = features
        self._balance = None
        if data.shape[0] < len(features):
            warn(
                "Decision trees tend to overfit when the feature space is large but there are a limited "
                "number of samples. Consider limiting the number of features or setting max_depth accordingly"
            )
        if tree_type == "classification":
            self.tree_builder = DecisionTreeClassifier(**kwargs)
            if balance_classes == "balanced":
                self._balance = "balanced"
            else:
                sampling_kwargs = sampling_kwargs or {}
                sampler = RandomOverSampler(**sampling_kwargs)
                self.x, self.y = sampler.fit_resample(self.x, self.y)
        else:
            self.tree_builder = DecisionTreeRegressor(**kwargs)

    def _fit(self, x: np.ndarray, y: np.ndarray, params: dict or None = None, **kwargs):
        params = params or {}
        if isinstance(self.tree_builder, DecisionTreeClassifier):
            if self._balance == "balanced":
                params["class_weight"] = "balanced"
        self.tree_builder.set_params(**params)
        self.tree_builder.fit(x, y, **kwargs)

    def validate_tree(
        self,
        validation_frac: float = 0.5,
        params: dict or None = None,
        performance_metrics: list or None = None,
        **kwargs,
    ):
        """
        Fit decision tree to data and evaluate on holdout data

        Parameters
        ----------
        validation_frac: float (default=0.5)
            Fraction of data to keep as holdout
        params: dict, optional
            Overwrite decision tree parameters prior to fit
        performance_metrics: list, optional
            List of performance metrics to use. Must be the name of a valid Scikit-Learn metric
            function or callable. See cytopy.flow.cell_classifier.uitls.calc_metrics
        kwargs:
            Additional keyword arguments passed to fit

        Returns
        -------
        polars.DataFrame
            Training and testing results
        """
        performance_metrics = performance_metrics or ["accuracy_score"]
        x_train, x_test, y_train, y_test = train_test_split(
            self.x.values, self.y, test_size=validation_frac, random_state=42
        )
        self._fit(x_train, y_train, params, **kwargs)
        y_pred_train = self.tree_builder.predict(x_train)
        y_pred_test = self.tree_builder.predict(x_test)
        y_score_train, y_score_test = None, None
        if isinstance(self.tree_builder, DecisionTreeClassifier):
            y_score_train = self.tree_builder.predict_proba(x_train)
            y_score_test = self.tree_builder.predict_proba(x_test)
        train_score = classifier_utils.calc_metrics(
            metrics=performance_metrics,
            y_true=y_train,
            y_pred=y_pred_train,
            y_score=y_score_train,
        )
        train_score["Dataset"] = "Training"
        train_score = pl.DataFrame(train_score, index=[0])
        test_score = classifier_utils.calc_metrics(
            metrics=performance_metrics,
            y_true=y_test,
            y_pred=y_pred_test,
            y_score=y_score_test,
        )
        test_score["Dataset"] = "Testing"
        test_score = pl.DataFrame(test_score, index=[1])
        return pl.concat([train_score, test_score])

    def prune(
        self,
        depth: tuple = (3,),
        verbose: bool = True,
        metric: str = "accuracy_score",
        validation_frac: float = 0.5,
        ax: plt.Axes or None = None,
        fit_kwargs: dict or None = None,
        **kwargs,
    ):
        """
        Iterate over a range of values for the 'depth' of a decision tree and plot
        the validation performance. This will highlight overfitting and inform on
        the maximum depth to achieve a suitable variability/bias trade-off

        Parameters
        ----------
        depth: tuple (default=(3,))
            Range of values to search for depth; (start, end). If length of depth is 1 (only
            start value is given), then maximum depth will equal the total number of features
        verbose: bool (default=True)
            Provide a progress bar
        metric: str (default='accuracy_score')
            Metric to assess validation score; should be the name of a valid Scikit-learn metric function
        validation_frac: float (default=0.5)
            Fraction of data to holdout for validation
        ax: Matplotlig.Axes, optional
        fit_kwargs: dict, optional
        kwargs:
            Additional keyword arguments passed to Seaborn.lineplot

        Returns
        -------
        Matplotlib.Axes
        """
        fit_kwargs = fit_kwargs or {}
        if len(depth) == 1:
            depth = np.arange(depth[0], len(self.x.shape[1]), 1)
        else:
            depth = np.arange(depth[0], depth[1], 1)
        depth_performance = list()
        for d in progress_bar(depth, verbose=verbose):
            performance = self.validate_tree(
                validation_frac=validation_frac,
                params={"max_depth": d, "random_state": 42},
                performance_metrics=[metric],
                **fit_kwargs,
            )
            performance["Max depth"] = d
            depth_performance.append(performance)
        depth_performance = pl.concat(depth_performance)
        return sns.lineplot(
            data=depth_performance,
            x="Max depth",
            y=metric,
            ax=ax,
            hue="Dataset",
            **kwargs,
        )

    def plot_tree(
        self,
        plot_type: str = "graphviz",
        ax: plt.Axes or None = None,
        graphviz_outfile: str or None = None,
        fit_kwargs: dict or None = None,
        **kwargs,
    ):
        """
        Plot the decision tree. Will call fit on all available data prior to generating tree.

        Parameters
        ----------
        plot_type: str (default='graphviz')
            What library to use for generating tree; should be 'graphviz' or 'matplotlib'
        ax: Matplotlib.Axes, optional
        graphviz_outfile: str, optional
            Path to save graphviz binary to
        fit_kwargs: dict, optional
        kwargs:
            Additional keyword arguments passed to sklearn.tree.plot_tree call (if plot_type =
            'matplotlib') or sklearn.tree.export_graphviz (if plot_type = 'graphviz')

        Returns
        -------
        Matplotlib.Axes or graphviz.Source
        """
        fit_kwargs = fit_kwargs or {}
        self._fit(x=self.x, y=self.y, **fit_kwargs)
        if plot_type == "graphviz":
            kwargs["feature_names"] = kwargs.get("feature_names", self.features)
            kwargs["filled"] = kwargs.get("filled", True)
            kwargs["rounded"] = kwargs.get("rounded", True)
            kwargs["special_characters"] = kwargs.get("special_characters", True)
            graph = export_graphviz(self.tree_builder, out_file=graphviz_outfile, **kwargs)
            graph = graphviz.Source(graph)
            return graph
        else:
            return plot_tree(
                decision_tree=self.tree_builder,
                feature_names=self.features,
                ax=ax,
                **kwargs,
            )

    def plot_importance(
        self,
        ax: plt.Axes or None = None,
        params: dict or None = None,
        fit_kwargs: dict or None = None,
        **kwargs,
    ):
        """
        Plot, as a bar chart, the feature importance for each of the variables in the feature space

        Warnings:
        Parameters
        ----------
        ax: Matplotlib.Axes
        params: dict, optional
            Overwrite existing tree parameters prior to fit
        fit_kwargs: dict, optional
            Additional keyword arguments passed to fit call
        kwargs:
            Additional keyword arguments passed to Seaborn.barplot call

        Returns
        -------
        Matplotlib.Axes
        """
        warn(
            "Impurity-based feature importance can be misleading for high cardinality features "
            "(many unique values). Consider FeatureImportance class to perform more robust permutation "
            "feature importance"
        )
        fit_kwargs = fit_kwargs or {}
        kwargs["color"] = kwargs.get("color", "#688bc4")
        self._fit(self.x, self.y, params=params, **fit_kwargs)
        tree_importance_sorted_idx = np.argsort(self.tree_builder.feature_importances_)
        features = np.array(self.features)[tree_importance_sorted_idx]
        return sns.barplot(
            y=features,
            x=self.tree_builder.feature_importances_[tree_importance_sorted_idx],
            ax=ax,
            **kwargs,
        )


class FeatureImportance:
    """
    This class provides convenient functionality for assessing the importance of features
    in Scikit-learn classifiers or equivalent models that follow the Scikit-Learn signatures
    and contain an attribute 'feature_importances_'.

    This includes permutation feature importance, whereby the model performance is observed
    upon after randomly shuffling a single feature; breaking the relationship between the
    feature and the target, therefore a reduction in performance corresponds to the value of a
    feature in the classification task.

    Classifier is automatically fitted to the available data, but a test/train subset is generated
    and the 'validation_performance' method allows you to observe holdout performance before continuing.
    Despite this, it is worth checking the performance of the model prior to assessing feature
    importance using cross-validation methods.

    Parameters
    ----------
    classifier: Scikit-Learn classifier
        Must contain the attribute 'feature_importances_'
    data: polars.DataFrame
        Feature space for classification/regression; must contain columns for features and target.
    target: str
        Endpoint for regression/classification; must be a column in 'data'
    features: list
        List of columns in 'data' to use as feature space
    validation_frac: float (default=0.5)
        Fraction of data to keep as holdout data
    balance_by_resampling: bool (default=False)
        If True, under represented class in data is sampled with replacement to account
        for class imbalance
    sampling_kwargs: dict, optional
        Additional keyword arguments passed to RandomOverSampler class of imbalance-learn
    kwargs:
        Additional keyword arguments passed to fit call on classifier

    Attributes
    ----------
    classifier: Scikit-Learn classifier
    features: list
    x: polars.DataFrame
        Feature space
    y: numpy.ndarray
        Target array
    x_train: numpy.ndarray
    x_test: numpy.ndarray
    y_train: numpy.ndarray
    y_test: numpy.ndarray
    """

    def __init__(
        self,
        classifier,
        data: pl.DataFrame,
        features: list,
        target: str,
        validation_frac: float = 0.5,
        balance_by_resampling: bool = False,
        sampling_kwargs: dict or None = None,
        **kwargs,
    ):
        self.classifier = classifier
        self.features = features
        data = data.dropna(axis=0).reset_index(drop=True)
        self.x, self.y = data[features], data[target].values
        if balance_by_resampling:
            sampling_kwargs = sampling_kwargs or {}
            sampler = RandomOverSampler(**sampling_kwargs)
            self.x, self.y = sampler.fit_resample(self.x, self.y)
        tt = train_test_split(self.x.values, self.y, test_size=validation_frac, random_state=42)
        self.x_train, self.x_test, self.y_train, self.y_test = tt
        self.classifier.fit(self.x_train, self.y_train, **kwargs)

    def validation_performance(self, performance_metrics: list or None = None, **kwargs):
        """
        Generate a DataFrame of test/train performance of given classifier

        Parameters
        ----------
        performance_metrics: list, optional
            List of performance metrics to use. Must be the name of a valid Scikit-Learn metric
            function or callable. See cytopy.flow.cell_classifier.uitls.calc_metrics
        kwargs:
            Additional keyword arguments passed to predict method of classifier

        Returns
        -------
        polars.DataFrame
        """
        performance_metrics = performance_metrics or ["accuracy_score"]
        y_pred_train = self.classifier.predict(self.x_train, self.y_train, **kwargs)
        y_pred_test = self.classifier.predict(self.x_test, self.y_test, **kwargs)
        train_score = pl.DataFrame(
            classifier_utils.calc_metrics(metrics=performance_metrics, y_true=self.y_train, y_pred=y_pred_train)
        )
        train_score["Dataset"] = "Training"
        test_score = pl.DataFrame(
            classifier_utils.calc_metrics(metrics=performance_metrics, y_true=self.y_test, y_pred=y_pred_test)
        )
        test_score["Dataset"] = "Testing"
        return pl.concat([train_score, test_score])

    def importance(self, ax: plt.Axes or None = None, **kwargs):
        """
        Generate a barplot of feature importance.

        Parameters
        ----------
        ax: Matplotlib.Axes, optional
        kwargs:
            Additional keyword arguments passed to Seaborn.barplot function

        Returns
        -------
        Matplotlib.Axes
        """
        warn(
            "Impurity-based feature importance can be misleading for high cardinality features "
            "(many unique values). Consider permutation_importance function."
        )
        tree_importance_sorted_idx = np.argsort(self.classifier.feature_importances_)
        features = np.array(self.features)[tree_importance_sorted_idx]
        return sns.barplot(
            y=features,
            x=self.classifier.feature_importances_[tree_importance_sorted_idx],
            ax=ax,
            **kwargs,
        )

    def permutation_importance(
        self,
        use_validation: bool = True,
        permutation_kwargs: dict or None = None,
        boxplot_kwargs: dict or None = None,
        overlay_kwargs: dict or None = None,
    ):
        """
        Assess feature importance using permutations
        (See https://scikit-learn.org/stable/modules/permutation_importance.html for indepth discussion and
        comparison to feature importance)

        Parameters
        ----------
        use_validation: bool (default=True)
            Use holdout data when assessing feature importance
        permutation_kwargs: dict, optional
            Additional keyword arguments passed to sklearn.inspection.permutation_importance call
        boxplot_kwargs: dict, optional
            See cytopy.flow.feature_selection.box_swarm_plot
        overlay_kwargs: dict, optional
            See cytopy.flow.feature_selection.box_swarm_plot

        Returns
        -------
        Matplotlib.Axes
        """
        permutation_kwargs = permutation_kwargs or {}
        if use_validation:
            result = permutation_importance(self.classifier, self.x_test, self.y_test, **permutation_kwargs)
        else:
            result = permutation_importance(self.classifier, self.x_train, self.y_train, **permutation_kwargs)
        result = pl.DataFrame(result.importances, columns=self.features)
        result = result.melt(var_name="Feature", value_name="Permutation importance")
        perm_sorted_idx = result.importances_mean.argsort()
        boxplot_kwargs = boxplot_kwargs or {}
        overlay_kwargs = overlay_kwargs or {}
        boxplot_kwargs["order"] = list(np.array(self.features)[perm_sorted_idx])
        overlay_kwargs["order"] = list(np.array(self.features)[perm_sorted_idx])
        return box_swarm_plot(
            plot_df=result,
            x="Permutation importance",
            y="Feature",
            boxplot_kwargs=boxplot_kwargs,
            overlay_kwargs=overlay_kwargs,
        )


class SHAP:
    """
    Game theoretic approach to non-linear model explanations (https://github.com/slundberg/shap)
    Currently this class supports tree model explanations and KernelSHAP. Future versions of cytopy
    will include Deep learning explanations.
    """

    def __init__(
        self,
        model,
        data: pl.DataFrame,
        features: list,
        target: str,
        explainer: str = "tree",
        link: str = "logit",
        js_backend: bool = True,
    ):
        if js_backend:
            shap.initjs()
        assert explainer in [
            "tree",
            "kernel",
        ], "explainer must be one of: 'tree', 'kernel'"
        self.x, self.y = data[features], data[target].values
        self.link = link
        if explainer == "tree":
            self.explainer = shap.TreeExplainer(model)
            self.shap_values = self.explainer.shap_values(self.x)
        else:
            self.explainer = shap.KernelExplainer(model, self.x, link=link)

    def force_plot(self, **kwargs):
        return shap.force_plot(self.explainer.expected_value, self.shap_values, self.x, **kwargs)

    def dependency_plot(self, feature: str, **kwargs):
        return shap.dependence_plot(feature, self.shap_values, self.x, **kwargs)

    def summary_plot(self, **kwargs):
        return shap.summary_plot(self.shap_values, self.x, **kwargs)


data = pl.DataFrame(
    {
        "x": [1, 4, 6, 2, 2, 5, 2, 3],
        "y": [5, 2, 8, 12, 3, 5, 4, 5],
        "z": [9, 7, 16, 4, 8, 5, 2, 3],
        "i": ["1", "1", "2", "2", "2", "3", "3", "3"],
    }
)
