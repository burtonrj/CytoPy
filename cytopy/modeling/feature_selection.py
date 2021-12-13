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
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple
from warnings import warn

import graphviz
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import shap
from imblearn.over_sampling import RandomOverSampler
from matplotlib.patches import Ellipse
from scipy import stats as scipy_stats
from sklearn.decomposition import PCA as SkPCA
from sklearn.inspection import permutation_importance
from sklearn.linear_model import Lasso
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import SGDRegressor
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.svm import LinearSVR
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import DecisionTreeRegressor
from sklearn.tree import export_graphviz
from sklearn.tree import plot_tree
from yellowbrick.regressor import ResidualsPlot

from cytopy.classification import utils as classifier_utils
from cytopy.feedback import progress_bar
from cytopy.plotting.general import box_swarm_plot
from cytopy.utils import transform

logger = logging.getLogger("modeling")

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
        How data should be scaled prior to generating PCA. See cytopy.utils.transform.Scaler for
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
        data: pd.DataFrame,
        features: List[str],
        scale: Optional[str] = "standard",
        scale_kwargs: Optional[Dict] = None,
        **kwargs,
    ):
        self.scaler = None
        self.data = data.dropna(axis=0).reset_index(drop=True)
        self.features = features
        if scale is None:
            logger.warning("PCA requires that input variables have unit variance and therefore scaling is recommended")
        else:
            scale_kwargs = scale_kwargs or {}
            self.scaler = transform.Scaler(method=scale, **scale_kwargs)
            self.data = self.scaler.fit_transform(self.data, self.features)
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
        var = pd.DataFrame(
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
        return pd.DataFrame(
            {
                "Feature": self.features,
                "EV Magnitude": abs(self.pca.components_)[component],
            }
        )

    def plot(
        self,
        label: str,
        discrete: bool = True,
        cmap: str = "tab10",
        loadings: bool = False,
        ax: Optional[plt.Axes] = None,
        limit_loadings: Optional[Dict] = None,
        arrow_kwargs: Optional[Dict] = None,
        ellipse: bool = False,
        ellipse_kwargs: Optional[Dict] = None,
        figsize: Optional[Tuple] = (5, 5),
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
        assert self.embeddings is not None, "Call fit first"
        plot_df = pd.DataFrame({f"PC{i + 1}": self.embeddings[:, i] for i in range(2)})
        plot_df[label] = self.data[label]
        ax = ax if ax is not None else plt.subplots(figsize=figsize)[1]
        if discrete:
            plot_df[label] = plot_df[label].astype(str)
        sns.scatterplot(data=plot_df, x="PC1", y="PC2", hue=label, **kwargs)
        if loadings:
            arrow_kwargs = arrow_kwargs or {}
            arrow_kwargs["color"] = arrow_kwargs.get("color", "r")
            arrow_kwargs["alpha"] = arrow_kwargs.get("alpha", 0.5)
            features_i = list(range(len(self.features)))
            if limit_loadings:
                features_i = [i for i, x in enumerate(self.features) if x in limit_loadings]
            ax = self._add_loadings(components=[0, 1], ax=ax, features_i=features_i, **arrow_kwargs)
        if ellipse:
            if not discrete:
                TypeError("Ellipse only valid for discrete label")
            ellipse_kwargs = ellipse_kwargs or {}
            ax = self._add_ellipse(components=[0, 1], label=label, cmap=cmap, ax=ax, **ellipse_kwargs)
        return ax

    def _add_loadings(self, components: List[int], ax: plt.Axes, features_i: List[int], **kwargs):
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

    def _add_ellipse(self, components: List[int], label: str, cmap: str, ax: plt.Axes, **kwargs):
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
        cytopy.utils.transform.Scaler for valid methods
    scale_kwargs: dict, optional
        Keyword arguments to pass to Scaler
    kwargs:
        Additional keyword arguments passed to construction of Scikit-Learn classifier/regressor

    Attributes
    ----------
    model: Scikit-Learn classifier/regressor
    scaler: CytoPy.utils.transform.Scaler
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
        data: pd.DataFrame,
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
            data = self.scaler.fit_transform(data=data, features=features)
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
        coefs = []
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
        data: pd.DataFrame,
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
            function or callable. See cytopy.utils.classification.uitls.calc_metrics
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
        train_score = pd.DataFrame(train_score, index=[0])
        test_score = classifier_utils.calc_metrics(
            metrics=performance_metrics,
            y_true=y_test,
            y_pred=y_pred_test,
            y_score=y_score_test,
        )
        test_score["Dataset"] = "Testing"
        test_score = pd.DataFrame(test_score, index=[1])
        return pd.concat([train_score, test_score])

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
        depth_performance = []
        for d in progress_bar(depth, verbose=verbose):
            performance = self.validate_tree(
                validation_frac=validation_frac,
                params={"max_depth": d, "random_state": 42},
                performance_metrics=[metric],
                **fit_kwargs,
            )
            performance["Max depth"] = d
            depth_performance.append(performance)
        depth_performance = pd.concat(depth_performance)
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
        data: pd.DataFrame,
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
            function or callable. See cytopy.utils.classification.uitls.calc_metrics
        kwargs:
            Additional keyword arguments passed to predict method of classifier

        Returns
        -------
        polars.DataFrame
        """
        performance_metrics = performance_metrics or ["accuracy_score"]
        y_pred_train = self.classifier.predict(self.x_train, self.y_train, **kwargs)
        y_pred_test = self.classifier.predict(self.x_test, self.y_test, **kwargs)
        train_score = pd.DataFrame(
            classifier_utils.calc_metrics(metrics=performance_metrics, y_true=self.y_train, y_pred=y_pred_train)
        )
        train_score["Dataset"] = "Training"
        test_score = pd.DataFrame(
            classifier_utils.calc_metrics(metrics=performance_metrics, y_true=self.y_test, y_pred=y_pred_test)
        )
        test_score["Dataset"] = "Testing"
        return pd.concat([train_score, test_score])

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
            See cytopy.utils.modeling.box_swarm_plot
        overlay_kwargs: dict, optional
            See cytopy.utils.modeling.box_swarm_plot

        Returns
        -------
        Matplotlib.Axes
        """
        permutation_kwargs = permutation_kwargs or {}
        if use_validation:
            result = permutation_importance(self.classifier, self.x_test, self.y_test, **permutation_kwargs)
        else:
            result = permutation_importance(self.classifier, self.x_train, self.y_train, **permutation_kwargs)
        result = pd.DataFrame(result.importances, columns=self.features)
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
        data: pd.DataFrame,
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
