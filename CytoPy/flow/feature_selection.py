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
from ..data.experiment import Experiment
from .plotting.embeddings_graphs import discrete_scatterplot, cont_scatterplot
from .cell_classifier import utils as classifier_utils
from . import transform
from sklearn.linear_model import Lasso, LogisticRegression, SGDClassifier, SGDRegressor
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor, plot_tree, export_graphviz
from sklearn.inspection import permutation_importance
from sklearn.decomposition import PCA as SkPCA
from sklearn.svm import LinearSVC, LinearSVR
from imblearn.over_sampling import RandomOverSampler
from collections import defaultdict
from scipy import stats as scipy_stats
from matplotlib.collections import EllipseCollection
from matplotlib.patches import Patch, Ellipse
from warnings import warn
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import graphviz
import pingouin

__author__ = "Ross Burton"
__copyright__ = "Copyright 2020, CytoPy"
__credits__ = ["Ross Burton", "Simone Cuff", "Andreas Artemiou", "Matthias Eberl"]
__license__ = "MIT"
__version__ = "2.0.0"
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

L1CLASSIFIERS = {"log": [LogisticRegression, dict(penalty="l1", solver="liblinear")],
                 "SGD": [SGDClassifier, dict(penalty="l1")],
                 "SVM": [LinearSVC, dict(penalty="l1", loss="squared_hinge", dual=False)]}

L1REGRESSORS = {"lasso": [Lasso, dict()],
                "SGD": [SGDRegressor, dict(penalty="l1")],
                "SVM": [LinearSVR, dict(loss="epsilon_insensitive")]}


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
        self.subject_ids = {x.primary_id: fetch_subject(x) for x in self._fcs_files}
        self.subject_ids = {k: v.subject_id for k, v in self.subject_ids.items() if v is not None}
        populations = [x.list_populations() for x in self._fcs_files]
        self.populations = set([x for sl in populations for x in sl])
        self.population_statistics = _fetch_population_statistics(files=self._fcs_files, populations=self.populations)
        self.ratios = defaultdict(dict)
        self.channel_desc = defaultdict(dict)
        self.meta_labels = defaultdict(dict)

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
                        if p in f.list_populations():
                            pn = self.population_statistics[f.primary_id][p]["n"]
                            self.ratios[f.primary_id][f"{pop1}:{p}"] = p1n / pn
                        else:
                            self.ratios[f.primary_id][f"{pop1}:{p}"] = None
                else:
                    p2n = self.population_statistics[f.primary_id][pop2]["n"]
                    self.ratios[f.primary_id][f"{pop1}:{pop2}"] = p1n / p2n
        return self

    def channel_desc_stats(self,
                           channel: str,
                           stats: list or None = None,
                           channel_transform: str or None = None,
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
                        self.channel_desc[f.primary_id][f"{p}_{channel}_{s}"] = None
                else:
                    x = f.load_population_df(population=p,
                                             transform=channel_transform,
                                             features_to_transform=[channel],
                                             transform_kwargs=transform_kwargs)[channel].values
                    for s in stats:
                        self.channel_desc[f.primary_id][f"{p}_{channel}_{s}"] = STATS.get(s)(x)
        return self

    def add_meta_labels(self,
                        key: str or list,
                        meta_label: str or None = None):
        for f in self._fcs_files:
            subject = fetch_subject(f)
            if subject is None:
                continue
            try:
                if isinstance(key, str):
                    self.meta_labels[f.primary_id][key] = subject[key]
                else:
                    node = subject[key[0]]
                    for k in key[1:]:
                        node = node[k]
                    meta_label = meta_label or k
                    self.meta_labels[f.primary_id][meta_label] = node
            except KeyError:
                self.logger.warn(f"{f.primary_id} missing meta variable {key} in Subject document")
                self.meta_labels[f.primary_id][key] = None
        return self

    def construct_dataframe(self):
        data = defaultdict(list)
        for sample_id, populations in self.population_statistics.items():
            data["sample_id"].append(sample_id)
            data["subject_id"].append(self.subject_ids.get(sample_id, None))
            for pop_name, pop_stats in populations.items():
                data[f"{pop_name}_N"].append(pop_stats["n"])
                data[f"{pop_name}_FOR"].append(pop_stats["frac_of_root"])
                data[f"{pop_name}_FOP"].append(pop_stats["frac_of_parent"])
            if self.ratios:
                for n, r in self.ratios.get(sample_id).items():
                    data[n].append(r)
            if self.channel_desc:
                for n, s in self.channel_desc.get(sample_id).items():
                    data[n].append(s)
            if self.meta_labels:
                for m, v in self.meta_labels.get(sample_id).items():
                    data[m].append(v)
        return pd.DataFrame(data)


def sort_variance(summary: pd.DataFrame,
                  identifier_columns: list,
                  value_name: str = "value",
                  var_name: str = "var"):
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


def clustered_heatmap(data: pd.DataFrame,
                      features: list,
                      index: str,
                      row_colours: str or None = None,
                      row_colours_cmap: str = "tab10",
                      **kwargs):
    df = data.set_index(index, drop=True)[features].copy()
    if row_colours is not None:
        row_colours_title = row_colours
        lut = dict(zip(data[row_colours].unique(), row_colours_cmap))
        row_colours = data[row_colours].map(lut)
        handles = [Patch(facecolor=lut[name]) for name in lut]
        g = sns.clustermap(df, row_colors=row_colours.values, **kwargs)
        plt.legend(handles, lut,
                   title=row_colours_title,
                   bbox_to_anchor=(1, 1),
                   bbox_transform=plt.gcf().transFigure,
                   loc='upper right')
    else:
        g = sns.clustermap(df, **kwargs)
    return g


def box_swarm_plot(plot_df: pd.DataFrame,
                   x: str,
                   y: str,
                   hue: str or None = None,
                   ax: plt.Axes or None = None,
                   palette: str or None = None,
                   boxplot_kwargs: dict or None = None,
                   overlay_kwargs: dict or None = None):
    """
    Convenience function for generating a boxplot with a swarmplot/stripplot overlaid showing
    individual datapoints (using tools from Seaborn library)

    Parameters
    ----------
    plot_df: Pandas.DataFrame
        Data to plot
    x: str
        Name of the column to use as x-axis variable
    y: str
        Name of the column to use as x-axis variable
    hue: str, optional
        Name of the column to use as factor to colour plot
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
    sns.boxplot(data=plot_df,
                x=x,
                y=y,
                hue=hue,
                ax=ax,
                showfliers=False,
                boxprops=dict(alpha=.3),
                palette=palette,
                **boxplot_kwargs)
    sns.swarmplot(data=plot_df,
                  x=x,
                  y=y,
                  hue=hue,
                  ax=ax,
                  dodge=True,
                  palette=palette,
                  **overlay_kwargs)
    return ax


class InferenceTesting:
    def __init__(self,
                 data: pd.DataFrame,
                 scale: str or None = None,
                 scale_vars: list or None = None,
                 scale_kwargs: dict or None = None):
        self.data = data.copy()
        self.scaler = None
        if scale is not None:
            scale_kwargs = scale_kwargs or {}
            self.scaler = transform.Scaler(method=scale, **scale_kwargs)
            assert scale_vars is not None, "Must provide variables to scale"
            self.data = self.scaler(data=self.data, features=scale_vars)

    def qq_plot(self,
                var: str,
                **kwargs):
        return pingouin.qqplot(x=self.data[var].values, **kwargs)

    def normality(self, var: list, method: str = "shapiro", alpha: float = 0.05):
        results = {"Variable": list(), "Normal": list()}
        for i in var:
            results["Variable"].append(i)
            results["Normal"].append(pingouin.normality(self.data[i].values, method=method, alpha=alpha)
                                     .iloc[0]["normal"])
        return pd.DataFrame(results)

    def anova(self,
              dep_var: str,
              between: str,
              post_hoc: bool = True,
              post_hoc_kwargs: dict or None = None,
              **kwargs):
        post_hoc_kwargs = post_hoc_kwargs or {}
        err = "Chosen dependent variable must be normally distributed"
        assert all([pingouin.normality(df[dep_var].values).iloc[0]["normal"]
                    for _, df in self.data.groupby(between)]), err
        eq_var = all([pingouin.homoscedasticity(df[dep_var].values).iloc[0]["equal_var"]
                      for _, df in self.data.groupby(between)])
        if eq_var:
            aov = pingouin.anova(data=self.data, dv=dep_var, between=between, **kwargs)
            if post_hoc:
                return aov, pingouin.pairwise_tukey(data=self.data, dv=dep_var, between=between, **post_hoc_kwargs)
            return aov, None
        aov = pingouin.welch_anova(data=self.data, dv=dep_var, between=between, **kwargs)
        if post_hoc:
            return aov, pingouin.pairwise_gameshowell(data=self.data, dv=dep_var, between=between, **post_hoc_kwargs)
        return aov, None

    def ttest(self,
              between: str,
              ind_var: list,
              paried: bool = False,
              multicomp: str = "holm",
              multicomp_alpha: float = 0.05,
              **kwargs):
        assert self.data[between].nunique() == 2, "More than two groups, consider using 'anova' method"
        x, y = self.data[between].unique()
        x, y = self.data[self.data[between] == x].copy(), self.data[self.data[between] == y].copy()
        results = list()
        for i in ind_var:
            assert all([pingouin.normality(df[i].values).iloc[0]["normal"]
                        for _, df in self.data.groupby(between)]), f"Groups for {i} are not normally distributed"
            eq_var = all([pingouin.homoscedasticity(df[i].values).iloc[0]["equal_var"]
                          for _, df in self.data.groupby(between)])
            if eq_var:
                tstats = pingouin.ttest(x=x[i].values,
                                        y=y[i].values,
                                        paired=paried,
                                        correction=False,
                                        **kwargs)
            else:
                tstats = pingouin.ttest(x=x[i].values,
                                        y=y[i].values,
                                        paired=paried,
                                        correction=True,
                                        **kwargs)
            tstats["Variable"] = i
            results.append(tstats)
        results = pd.concat(results)
        results["p-val"] = pingouin.multicomp(results["p-val"].values, alpha=multicomp_alpha, method=multicomp)
        return results

    def non_parametric(self,
                       between: str,
                       ind_var: list,
                       paired: bool = False,
                       multicomp: str = "holm",
                       multicomp_alpha: float = 0.05,
                       **kwargs):
        results = list()
        if self.data[between].nunique() > 2:
            if paired:
                for i in ind_var:
                    np_stats = pingouin.friedman(data=self.data,
                                                 dv=i,
                                                 within=between,
                                                 **kwargs)
                    np_stats["Variable"] = i
                    results.append(np_stats)
            else:
                for i in ind_var:
                    np_stats = pingouin.kruskal(data=self.data,
                                                dv=i,
                                                between=between,
                                                **kwargs)
                    np_stats["Variable"] = i
                    results.append(np_stats)
        else:
            x, y = self.data[between].unique()
            x, y = self.data[self.data[between] == x].copy(), self.data[self.data[between] == y].copy()
            if paired:
                for i in ind_var:
                    np_stats = pingouin.wilcoxon(x[i].values, y[i].values, **kwargs)
                    np_stats["Variable"] = i
                    results.append(np_stats)
            else:
                for i in ind_var:
                    np_stats = pingouin.mwu(x[i].values, y[i].values, **kwargs)
                    np_stats["Variable"] = i
                    results.append(np_stats)
        results = pd.concat(results)
        results["p-val"] = pingouin.multicomp(results["p-val"].values, alpha=multicomp_alpha, method=multicomp)[1]
        return results


def plot_multicolinearity(data: pd.DataFrame,
                          features: list,
                          method: str = "spearman",
                          ax: plt.Axes or None = None,
                          plot_type: str = "ellipse",
                          **kwargs):
    corr = data[features].corr(method=method)
    if plot_type == "ellipse":
        ax = ax or plt.subplots(figsize=(8, 8), subplot_kw={'aspect': 'equal'})[1]
        ax.set_xlim(-0.5, corr.shape[1] - 0.5)
        ax.set_ylim(-0.5, corr.shape[0] - 0.5)
        xy = np.indices(corr.shape)[::-1].reshape(2, -1).T
        w = np.ones_like(corr.values).ravel()
        h = 1 - np.abs(corr.values).ravel()
        a = 45 * np.sign(corr.values).ravel()
        ec = EllipseCollection(widths=w, heights=h, angles=a, units='x', offsets=xy,
                               transOffset=ax.transData, array=corr.values.ravel(), **kwargs)
        ax.add_collection(ec)
        ax.set_xticks(np.arange(corr.shape[1]))
        ax.set_xticklabels(corr.columns, rotation=90)
        ax.set_yticks(np.arange(corr.shape[0]))
        ax.set_yticklabels(corr.index)
        return ax, ec
    return sns.clustermap(data=corr, **kwargs), None


class PCA:
    def __init__(self,
                 data: pd.DataFrame,
                 features: list,
                 scale: str or None = "standard",
                 scale_kwargs: dict or None = None,
                 **kwargs):
        self.scaler = None
        self.data = data.dropna(axis=0).reset_index(drop=True)
        self.features = features
        if scale is None:
            warn("PCA requires that input variables have unit variance and therefore scaling is recommended",
                 stacklevel=2)
        else:
            scale_kwargs = scale_kwargs or {}
            self.scaler = transform.Scaler(method=scale, **scale_kwargs)
            self.data = self.scaler(self.data, self.features)
        kwargs = kwargs or dict()
        kwargs["random_state"] = kwargs.get("random_state", 42)
        self.pca = SkPCA(**kwargs)
        self.embeddings = None

    def fit(self):
        self.embeddings = self.pca.fit_transform(self.data[self.features])
        return self

    def scree_plot(self, **kwargs):
        assert self.embeddings is not None, "Call fit first"
        var = pd.DataFrame({"Variance Explained": self.pca.explained_variance_ratio_,
                            "PC": [f"PC{i + 1}" for i in range(len(self.pca.explained_variance_ratio_))]})
        return sns.barplot(data=var, x="PC", y="Variance Explained", ci=None, **kwargs)

    def loadings(self,
                 component: int = 0):
        assert self.embeddings is not None, "Call fit first"
        return pd.DataFrame({"Feature": self.features,
                             "EV Magnitude": abs(self.pca.components_)[component]})

    def plot(self,
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
             **kwargs):
        components = components or [0, 1]
        assert 2 <= len(components) <= 3, "Components should be of length 2 or 3"
        assert self.embeddings is not None, "Call fit first"
        plot_df = pd.DataFrame({f"PC{i + 1}": self.embeddings[:, i] for i in components})
        plot_df[label] = self.data[label]
        fig = plt.figure(figsize=figsize)
        z = None
        if len(components) == 3:
            z = f"PC{components[2] + 1}"
        if discrete:
            ax = discrete_scatterplot(data=plot_df,
                                      x=f"PC{components[0] + 1}",
                                      y=f"PC{components[1] + 1}",
                                      z=z,
                                      label=label,
                                      cmap=cmap,
                                      size=size,
                                      fig=fig,
                                      **kwargs)
        else:
            cbar_kwargs = cbar_kwargs or {}
            ax = cont_scatterplot(data=plot_df,
                                  x=f"PC{components[0] + 1}",
                                  y=f"PC{components[1] + 1}",
                                  z=z,
                                  label=label,
                                  cmap=cmap,
                                  size=size,
                                  fig=fig,
                                  cbar_kwargs=cbar_kwargs,
                                  **kwargs)
        if loadings:
            assert len(components) == 2, "CytoPy only supports 2D byplots"
            arrow_kwargs = arrow_kwargs or {}
            arrow_kwargs["color"] = arrow_kwargs.get("color", "r")
            arrow_kwargs["alpha"] = arrow_kwargs.get("alpha", 0.5)
            features_i = list(range(len(self.features)))
            if limit_loadings:
                features_i = [i for i, x in enumerate(self.features) if x in limit_loadings]
            ax = self._add_loadings(components=components,
                                    ax=ax,
                                    features_i=features_i,
                                    **arrow_kwargs)
        if ellipse:
            assert len(components) == 2, "CytoPy only supports confidence ellipse for 2D plots"
            assert discrete, "Ellipse only value for discrete label"
            ellipse_kwargs = ellipse_kwargs or {}
            ax = self._add_ellipse(components=components,
                                   label=label,
                                   cmap=cmap,
                                   ax=ax,
                                   **ellipse_kwargs)
        return fig, ax

    def _add_loadings(self,
                      components: list,
                      ax: plt.Axes,
                      features_i: list,
                      **kwargs):
        coeffs = np.transpose(self.pca.components_[np.array(components), :])
        for i in features_i:
            ax.arrow(0, 0, coeffs[i, 0], coeffs[i, 1], **kwargs)
            ax.text(coeffs[i, 0] * 1.15, coeffs[i, 1] * 1.15, self.features[i], color='b', ha='center', va='center')
        return ax

    def _add_ellipse(self,
                     components: list,
                     label: str,
                     cmap: str,
                     ax: plt.Axes,
                     **kwargs):
        kwargs = kwargs or {}
        s = kwargs.pop("s", 3)
        kwargs["linewidth"] = kwargs.get("linewidth", 2)
        kwargs["edgecolor"] = kwargs.get("edgecolor", "#383838")
        kwargs["alpha"] = kwargs.get("alpha", 0.2)
        colours = plt.get_cmap(cmap).colors
        assert len(colours) >= self.data[label].nunique(), "Chosen cmap doesn't contain enough unique colours"
        for l, c in zip(self.data[label].unique(), colours):
            idx = self.data[self.data[label] == l].index.values
            x, y = self.embeddings[idx, components[0]], self.embeddings[idx, components[1]]
            cov = np.cov(x, y)
            v, w = np.linalg.eig(cov)
            v = np.sqrt(v)
            ellipse = Ellipse(xy=(np.mean(x), np.mean(y)),
                              width=v[0] * s * 2,
                              height=v[1] * s * 2,
                              angle=np.rad2deg(np.arccos(w[0, 0])),
                              facecolor=c,
                              **kwargs)
            ax.add_artist(ellipse)
        return ax


class L1Selection:
    def __init__(self,
                 data: pd.DataFrame,
                 target: str,
                 features: list,
                 model: str,
                 category: str = "classification",
                 scale: str or None = "standard",
                 scale_kwargs: dict or None = None,
                 **kwargs):
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

    def fit(self,
            search_space: tuple = (-2, 0, 50),
            **kwargs):
        search_space = np.logspace(*search_space)
        coefs = list()
        for r in search_space:
            self.model.set_params(**{self._reg_param: r})
            self.model.fit(self.x, self.y, **kwargs)
            if self._category == "classification":
                coefs.append(list(self.model.coef_[0]))
            else:
                coefs.append(list(self.model.coef_))
        self.scores = pd.DataFrame(np.array(coefs), columns=self.features)
        self.scores[self._reg_param] = search_space
        return self

    def plot(self,
             ax: plt.Axes or None = None,
             title: str = "L1 Penalty",
             xlabel: str = "Regularisation parameter",
             ylabel: str = "Coefficient",
             cmap: str = "tab10",
             **kwargs):
        ax = ax or plt.subplots(figsize=(10, 5))[1]
        assert self.scores is not None, "Call fit prior to plot"
        colours = plt.get_cmap(cmap).colors
        for i, feature in enumerate(self.features):
            ax.plot(self.scores[self._reg_param], self.scores[feature], label=feature,
                    color=colours[i], **kwargs)
        ax.set_xscale("log")
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend(bbox_to_anchor=(1.1, 1))
        return ax


class DecisionTree:
    def __init__(self,
                 data: pd.DataFrame,
                 target: str,
                 features: list,
                 tree_type: str = "classification",
                 balance_classes: str = "sample",
                 sampling_kwargs: dict or None = None,
                 **kwargs):
        data = data.dropna(axis=0).reset_index(drop=True)
        self.x, self.y = data[features], data[target].values
        self.features = features
        self._balance = None
        if data.shape[0] < len(features):
            warn("Decision trees tend to overfit when the feature space is large but there are a limited "
                 "number of samples. Consider limiting the number of features or setting max_depth accordingly",
                 stacklevel=2)
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

    def _fit(self,
             x: np.ndarray,
             y: np.ndarray,
             params: dict or None = None,
             **kwargs):
        params = params or {}
        if isinstance(self.tree_builder, DecisionTreeClassifier):
            if self._balance == "balanced":
                params["class_weight"] = "balanced"
        self.tree_builder.set_params(**params)
        self.tree_builder.fit(x, y, **kwargs)

    def validate_tree(self,
                      validation_frac: float = 0.5,
                      params: dict or None = None,
                      performance_metrics: list or None = None,
                      **kwargs):
        performance_metrics = performance_metrics or ["accuracy_score"]
        x_train, x_test, y_train, y_test = train_test_split(self.x.values, self.y,
                                                            test_size=validation_frac,
                                                            random_state=42)
        self._fit(x_train, y_train, params, **kwargs)
        y_pred_train = self.tree_builder.predict(x_train)
        y_pred_test = self.tree_builder.predict(x_test)
        y_score_train, y_score_test = None, None
        if isinstance(self.tree_builder, DecisionTreeClassifier):
            y_score_train = self.tree_builder.predict_proba(x_train)
            y_score_test = self.tree_builder.predict_proba(x_test)
        train_score = classifier_utils.calc_metrics(metrics=performance_metrics,
                                                    y_true=y_train,
                                                    y_pred=y_pred_train,
                                                    y_score=y_score_train)
        train_score["Dataset"] = "Training"
        train_score = pd.DataFrame(train_score, index=[0])
        test_score = classifier_utils.calc_metrics(metrics=performance_metrics,
                                                   y_true=y_test,
                                                   y_pred=y_pred_test,
                                                   y_score=y_score_test)
        test_score["Dataset"] = "Testing"
        test_score = pd.DataFrame(test_score, index=[1])
        return pd.concat([train_score, test_score])

    def prune(self,
              depth: tuple = (3,),
              verbose: bool = True,
              metric: str = "accuracy_score",
              validation_frac: float = 0.5,
              ax: plt.Axes or None = None,
              fit_kwargs: dict or None = None,
              **kwargs):
        fit_kwargs = fit_kwargs or {}
        if len(depth) == 1:
            depth = np.arange(depth[0], len(self.x.shape[1]), 1)
        else:
            depth = np.arange(depth[0], depth[1], 1)
        depth_performance = list()
        for d in progress_bar(depth, verbose=verbose):
            performance = self.validate_tree(validation_frac=validation_frac,
                                             params={"max_depth": d, "random_state": 42},
                                             performance_metrics=[metric],
                                             **fit_kwargs)
            performance["Max depth"] = d
            depth_performance.append(performance)
        depth_performance = pd.concat(depth_performance)
        return sns.lineplot(data=depth_performance,
                            x="Max depth",
                            y=metric,
                            ax=ax,
                            hue="Dataset",
                            **kwargs)

    def plot_tree(self,
                  plot_type: str = "graphviz",
                  ax: plt.Axes or None = None,
                  graphviz_outfile: str or None = None,
                  fit_kwargs: dict or None = None,
                  **kwargs):
        fit_kwargs = fit_kwargs or {}
        self._fit(x=self.x, y=self.y, **fit_kwargs)
        if plot_type == "graphviz":
            kwargs["feature_names"] = kwargs.get("feature_names", self.features)
            kwargs["filled"] = kwargs.get("filled", True)
            kwargs["rounded"] = kwargs.get("rounded", True)
            kwargs["special_characters"] = kwargs.get("special_characters", True)
            graph = export_graphviz(self.tree_builder,
                                    out_file=graphviz_outfile,
                                    **kwargs)
            graph = graphviz.Source(graph)
            return graph
        else:
            return plot_tree(decision_tree=self.tree_builder,
                             feature_names=self.features,
                             ax=ax,
                             **kwargs)

    def plot_importance(self,
                        ax: plt.Axes or None = None,
                        params: dict or None = None,
                        fit_kwargs: dict or None = None,
                        **kwargs):
        warn("Impurity-based feature importances can be misleading for high cardinality features "
             "(many unique values). Consider FeatureImportance class to perform more robust permutation "
             "feature importance")
        fit_kwargs = fit_kwargs or {}
        kwargs["color"] = kwargs.get("color", "#688bc4")
        self._fit(self.x, self.y, params=params, **fit_kwargs)
        tree_importance_sorted_idx = np.argsort(self.tree_builder.feature_importances_)
        features = np.array(self.features)[tree_importance_sorted_idx]
        return sns.barplot(y=features,
                           x=self.tree_builder.feature_importances_[tree_importance_sorted_idx],
                           ax=ax,
                           **kwargs)


class FeatureImportance:
    def __init__(self,
                 classifier,
                 data: pd.DataFrame,
                 features: list,
                 target: str,
                 validation_frac: float = 0.5,
                 balance_by_resampling: bool = False,
                 sampling_kwargs: dict or None = None,
                 **kwargs):
        self.classifier = classifier
        self.features = features
        data = data.dropna(axis=0).reset_index(drop=True)
        self.x, self.y = data[features], data[target].values
        if balance_by_resampling:
            sampling_kwargs = sampling_kwargs or {}
            sampler = RandomOverSampler(**sampling_kwargs)
            self.x, self.y = sampler.fit_resample(self.x, self.y)
        tt = train_test_split(self.x.values, self.y,
                              test_size=validation_frac,
                              random_state=42)
        self.x_train, self.x_test, self.y_train, self.y_test = tt
        self.classifier.fit(self.x_train, self.y_train, **kwargs)

    def validation_performance(self,
                               performance_metrics: list or None = None,
                               **kwargs):
        performance_metrics = performance_metrics or ["accuracy_score"]
        y_pred_train = self.classifier.predict(self.x_train, self.y_train, **kwargs)
        y_pred_test = self.classifier.predict(self.x_test, self.y_test, **kwargs)
        train_score = pd.DataFrame(classifier_utils.calc_metrics(metrics=performance_metrics,
                                                                 y_true=self.y_train,
                                                                 y_pred=y_pred_train))
        train_score["Dataset"] = "Training"
        test_score = pd.DataFrame(classifier_utils.calc_metrics(metrics=performance_metrics,
                                                                y_true=self.y_test,
                                                                y_pred=y_pred_test))
        test_score["Dataset"] = "Testing"
        return pd.concat([train_score, test_score])

    def importance(self,
                   ax: plt.Axes or None = None,
                   **kwargs):
        tree_importance_sorted_idx = np.argsort(self.classifier.feature_importances_)
        features = np.array(self.features)[tree_importance_sorted_idx]
        return sns.barplot(y=features,
                           x=self.classifier.feature_importances_[tree_importance_sorted_idx],
                           ax=ax,
                           **kwargs)

    def permutation_importance(self,
                               use_validation: bool = True,
                               permutation_kwargs: dict or None = None,
                               boxplot_kwargs: dict or None = None,
                               overlay_kwargs: dict or None = None):
        permutation_kwargs = permutation_kwargs or {}
        if use_validation:
            result = permutation_importance(self.classifier,
                                            self.x_test,
                                            self.y_test,
                                            **permutation_kwargs)
        else:
            result = permutation_importance(self.classifier,
                                            self.x_train,
                                            self.y_train,
                                            **permutation_kwargs)
        result = pd.DataFrame(result.importances, columns=self.features)
        result = result.melt(var_name="Feature", value_name="Permutation importance")
        perm_sorted_idx = result.importances_mean.argsort()
        boxplot_kwargs = boxplot_kwargs or {}
        overlay_kwargs = overlay_kwargs or {}
        boxplot_kwargs["order"] = list(np.array(self.features)[perm_sorted_idx])
        overlay_kwargs["order"] = list(np.array(self.features)[perm_sorted_idx])
        return box_swarm_plot(plot_df=result,
                              x="Permutation importance",
                              y="Feature",
                              boxplot_kwargs=boxplot_kwargs,
                              overlay_kwargs=overlay_kwargs)
