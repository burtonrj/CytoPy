#!/usr/bin.env/python
# -*- coding: utf-8 -*-
"""
Before we perform any detailed analysis and/or classification of our
single cell data, it is valuable to assess the inter-sample variation
that could be arising from biological differences, but also technical
variation introduced by batch effects. This module contains multiple functions
for visualising univariate and multivatiate differences between
FileGroups in the same experiment. Additionally we provide a convenient class
for correcting 'global' batch effects using the Harmony algorithm.


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
from typing import *
from warnings import warn

import harmonypy
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import seaborn as sns
from KDEpy import FFTKDE
from matplotlib import cm
from scipy.cluster import hierarchy
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KernelDensity

from ..data.experiment import Experiment
from ..feedback import progress_bar
from ..feedback import vprint
from ..flow import transform as transform_module
from .dim_reduction import DimensionReduction
from cytopy.flow.transform import TRANSFORMERS

np.random.seed(42)

COLOURS = list(cm.get_cmap("tab20").colors) + list(cm.get_cmap("tab20b").colors) + list(cm.get_cmap("tab20c").colors)

__author__ = "Ross Burton"
__copyright__ = "Copyright 2020, cytopy"
__credits__ = ["Ross Burton", "Simone Cuff", "Andreas Artemiou", "Matthias Eberl"]
__license__ = "MIT"
__version__ = "2.0.0"
__maintainer__ = "Ross Burton"
__email__ = "burtonrj@cardiff.ac.uk"
__status__ = "Production"
logger = logging.getLogger(__name__)


def bw_optimisation(
    data: pd.DataFrame,
    features: List[str],
    kernel: str = "gaussian",
    bandwidth: Tuple[float] = (0.01, 0.1, 10),
    cv: int = 10,
    verbose: int = 0,
) -> float:
    """
    Using GridSearchCV and the Scikit-Learn implementation of KDE, find the optimal
    bandwidth for the given data using grid search cross-validation

    Parameters
    ----------
    data: pd.DataFrame
    features: features
    kernel: str (default="gaussian")
    bandwidth: tuple (default=(0.01, 0.1, 20))
        Linear search space for bandwidth (min, max, increments)
    cv: int (default=10)
        Number of k-folds
    verbose: int (default=0)

    Returns
    -------
    float
    """
    bandwidth = np.linspace(*bandwidth)
    kde = KernelDensity(kernel=kernel)
    grid = GridSearchCV(
        estimator=kde,
        param_grid={"bandwidth": bandwidth},
        cv=cv,
        n_jobs=-1,
        verbose=verbose,
    )
    grid.fit(data[features])
    return grid.best_params_.get("bandwidth")


def calculate_ref_sample(data: pd.DataFrame, features: Union[List[str], None] = None, verbose: bool = True) -> str:
    """

    This is performed as described in Li et al paper (https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5860171/) on
    DeepCyTOF: for every 2 samples i, j compute the euclidean norm of the difference between their covariance matrics
    and then select the sample with the smallest average distance to all other samples.

    This is an optimised version of supervised.ref.calculate_red_sample that leverages the multi-processing library
    to speed up operations

    Parameters
    ----------
    data: dict
    features: list, optional
    verbose: bool, (default=True)
        Feedback

    Returns
    -------
    str
        Sample ID of reference sample
    """
    feedback = vprint(verbose)
    feedback("Calculate covariance matrix for each sample...")
    # Calculate covar for each
    data = data.dropna(axis=1, how="any")
    features = features or list(data.columns)
    covar = {k: np.cov(v[features].astype(np.float32), rowvar=False) for k, v in data.groupby(by="sample_id")}
    feedback("Search for sample with smallest average euclidean distance to all other samples...")
    # Make comparisons
    sample_ids = list(covar.keys())
    n = len(sample_ids)
    norms = np.zeros(shape=[n, n])
    ref_ind = None
    for i, sample_i in enumerate(sample_ids):
        for j, sample_j in enumerate(sample_ids):
            cov_diff = covar.get(sample_i) - covar.get(sample_j)
            norms[i, j] = np.linalg.norm(cov_diff, ord="fro")
            norms[j, i] = norms[i, j]
            avg = np.mean(norms, axis=1)
            ref_ind = np.argmin(avg)
    return sample_ids[int(ref_ind)]


def marker_variance(
    data: pd.DataFrame,
    reference: str,
    comparison_samples: Union[List[str], None] = None,
    markers: Union[List[str], None] = None,
    figsize: tuple = (10, 10),
    xlim: Union[Tuple[float], None] = None,
    verbose: bool = True,
    kernel: str = "gaussian",
    kde_bw: Union[str, float] = "silverman",
    **kwargs,
):
    """
    Compare the kernel density estimates for each marker in the associated experiment for the given
    comparison samples. The estimated distributions of the comparison samples will be plotted against
    the reference sample.

    Parameters
    ----------
    data: Pandas.DataFrame
        DataFrame as generated from load_and_sample
    reference: str
        Reference sample to plot in the background
    comparison_samples: list
        List of valid sample IDs for the associated experiment
    markers: list (optional)
        List of markers to include (defaults to all available markers)
    figsize: figsize (default=(10,10))
    xlim: tuple (optional)
        x-axis limits
    verbose: bool (default=True)
    kernel: str (default="gaussian")
    kde_bw: str or float (default="silverman")
    kwargs: dict
        Additional kwargs passed to Matplotlib.Axes.plot call

    Returns
    -------
    matplotlib.Figure

    Raises
    ------
    ValueError
        Reference absent from data
    """
    if reference not in data.sample_id.unique():
        raise ValueError("Reference absent from given data")

    comparison_samples = comparison_samples or [x for x in data.sample_id.unique() if x != reference]
    fig = plt.figure(figsize=figsize)
    markers = markers or data.get(reference).columns.tolist()
    i = 0
    nrows = math.ceil(len(markers) / 3)
    fig.suptitle(f"Per-channel KDE, Reference: {reference}", y=1.02)
    for marker in progress_bar(markers, verbose=verbose):
        i += 1
        ax = fig.add_subplot(nrows, 3, i)
        x, y = FFTKDE(kernel=kernel, bw=kde_bw).fit(data[data.sample_id == reference][marker].values).evaluate()
        ax.plot(x, y, color="b", **kwargs)
        ax.fill_between(x, 0, y, facecolor="b", alpha=0.2)
        ax.set_title(marker)
        if xlim:
            ax.set_xlim(xlim)
        for comparison_sample_id in comparison_samples:
            df = data[data.sample_id == comparison_sample_id]
            if marker not in df.columns:
                warn(f"{marker} missing from {comparison_sample_id}, this marker will be ignored")
            else:
                x, y = FFTKDE(kernel=kernel, bw=kde_bw).fit(df[marker].values).evaluate()
                ax.plot(x, y, color="r", **kwargs)
                if ax.get_legend() is not None:
                    ax.get_legend().remove()
        ax.set(aspect="auto")
    fig.tight_layout()
    return fig


def dim_reduction_grid(
    data: pd.DataFrame,
    reference: str,
    features: List[str],
    comparison_samples: Union[List[str], None] = None,
    figsize: Tuple[int] = (10, 10),
    method: str = "PCA",
    kde: bool = False,
    verbose: bool = True,
    dim_reduction_kwargs: Optional[Dict] = None,
):
    """
    Generate a grid of embeddings using a valid dimensionality reduction technique, in each plot a reference sample
    is shown in blue and a comparison sample in red. The reference sample is conserved across all plots.

    Parameters
    ------------
    data: Pandas.DataFrame
        DataFrame as generated from load_and_sample
    reference: str
        Reference sample to plot in the background
    comparison_samples: list
        List of samples to compare to reference (blue)
    features: list
        List of features to use for dimensionality reduction
    figsize: tuple, (default=(10,10))
        Size of figure
    method: str, (default='PCA')
        Method to use for dimensionality reduction (see flow.dim_reduction)
    dim_reduction_kwargs: dict
        Additional keyword arguments passed to cytopy.dim_reduction.dimensionality_reduction
    kde: bool, (default=False)
        If True, overlay with two-dimensional PDF estimated by KDE
    verbose: bool (default=True)

    Returns
    -------
    None
        Plot printed to stdout

    Raises
    ------
    AssertionError
        Reference absent from data

    ValueError
        Invalid features provided
    """
    assert reference in data.sample_id.unique(), "Reference absent from given data"
    data = data.dropna(axis=1, how="any")
    comparison_samples = comparison_samples or [x for x in data.sample_id.unique() if x != reference]
    dim_reduction_kwargs = dim_reduction_kwargs or {}
    fig = plt.figure(figsize=figsize)
    nrows = math.ceil(len(comparison_samples) / 3)
    reference_df = data[data.sample_id == reference].copy()
    if not all([f in reference_df.columns for f in features]):
        raise ValueError(f"Invalid features; valid are: {reference_df.columns}")
    reducer = DimensionReduction(method=method, n_components=2, **dim_reduction_kwargs)
    reference_df = reducer.fit_transform(data=reference_df.reset_index(), features=features)
    i = 0
    fig.suptitle(f"{method}, Reference: {reference}", y=1.05)
    for sample_id in progress_bar(comparison_samples, verbose=verbose):
        i += 1
        ax = fig.add_subplot(nrows, 3, i)
        embeddings = reducer.transform(data[data.sample_id == sample_id].reset_index(), features=features)
        x = f"{method}1"
        y = f"{method}2"
        ax.scatter(reference_df[x], reference_df[y], c="blue", s=4, alpha=0.2)
        if kde:
            sns.kdeplot(
                reference_df[x],
                reference_df[y],
                c="blue",
                n_levels=100,
                ax=ax,
                shade=False,
            )
        ax.scatter(embeddings[:, 0], embeddings[:, 1], c="red", s=4, alpha=0.1)
        if kde:
            sns.kdeplot(
                embeddings[:, 0],
                embeddings[:, 1],
                c="red",
                n_levels=100,
                ax=ax,
                shade=False,
            )
        ax.set_title(sample_id)
        ax.set_yticklabels([])
        ax.set_xticklabels([])
        ax.set(aspect="auto")
    fig.tight_layout()
    return fig


def generate_groups(linkage_matrix: np.ndarray, sample_ids: Union[List[str], np.ndarray], n_groups: int):
    """
    Given the output of SimilarityMatrix (that is the linkage matrix and ordered list of sample
    IDs) and a desired number of groups, return a Pandas DataFrame of sample IDs and assigned group ID, generated by
    cutting the linkage matrix in such a way that the desired number of groups are generated.
    Parameters
    ----------
    linkage_matrix: np.array
        Linkage matrix generated from EvaluateBatchEffects.similarity_matrix (using SciPy.cluster.hierarchy.linkage)
    sample_ids: list or np.array
        Ordered list of sample IDs generated from EvaluateBatchEffects.similarity_matrix
    n_groups: int
        Desired number of groups
    Returns
    -------
    Pandas.DataFrame
    """
    groups = pd.DataFrame(
        {
            "sample_id": sample_ids,
            "group": list(
                map(
                    lambda x: x + 1,
                    hierarchy.cut_tree(linkage_matrix, n_groups).flatten(),
                )
            ),
        }
    )
    groups = groups.sort_values("group")
    return groups


class Harmony:
    """
    Perform batch-effect correction using the Harmony algorithm, first described by Korsunsky et al [1] and
    implemented in Python by Kamil Slowikowski [2].

    [1] Korsunsky, I., Millard, N., Fan, J. et al. Fast, sensitive and accurate integration of single-cell data
    with Harmony. Nat Methods 16, 1289–1296 (2019). https://doi.org/10.1038/s41592-019-0619-0
    [2] https://github.com/slowkow/harmonypy

    Parameters
    ----------
    experiment: Experiment
        Experiment to load data from for batch effect correction
    population: str
        Starting population to load data from e.g. 'root' for originald ata
    features: list
        List of features to include in batch correction; only these features will appear
        in saved data, all other columns will be removed
    sample_size: int or float
        Number of events to sample from original data for batch correction; it is recommended that
        the number of data points (after summing all events from all FileGroups) does not exceed
        2*10^6, however this can vary depending on your time constraints and computational resources
    sampling_method: str (default='uniform')
        How to downsample events; should be one of: 'uniform', 'density' or 'faithful'
        (see cytopy.flow.sampling)
    transform: str (default='logicle')
        How to transform data prior to batch correction. For valid methods see cytopy.flow.transform
    transform_kwargs: dict, optional
        Additional keyword arguments passed to Transformer
    scale: str, optional (default='standard')
        How to scale data prior to batch correction. For valid methods see cytopy.flow.transform.Scaler
    scale_kwargs: dict, optional
        Additional keyword arguments passed to Scaler
    sampling_kwargs: dict, optional
        Additional keyword arguments to pass to sampling method

    Attributes
    ----------
    data: Pandas.DataFrame
        Downsampled data from Experiment as generated by cytopy.flow.variance.load_and_sample
    transformer: Transformer
        Transfomation object; used to inverse transform when saving data back to the database
        after correction
    features: list
        List of features
    meta: Pandas.DataFrame
        Meta DataFrame; by default it has one column that identifies 'batches' and is always 'sample_id'
    scaler: Scaler
        Scaler object; used to inverse the scale when saving data back to the database after correction
    """

    def __init__(
        self,
        data: pd.DataFrame,
        features: List[str],
        transform: Optional[str] = "logicle",
        transform_kwargs: Union[Dict[str, str], None] = None,
        scale: Optional[str] = "standard",
        scale_kwargs: Optional[Dict] = None,
    ):
        transform_kwargs = transform_kwargs or {}
        self.transformer = None if transform is None else TRANSFORMERS[transform](**transform_kwargs)
        if self.transformer is not None:
            self.data = self.transformer.scale(data=data, features=features)
        else:
            self.data = data
        self.data = self.data.dropna(axis=1, how="any")
        self.features = [x for x in features if x in self.data.columns]
        self.meta = self.data[["sample_id"]].copy()
        self.harmony = None
        self.scaler = None
        if scale is not None:
            scale_kwargs = scale_kwargs or {}
            scale = transform_module.Scaler(method=scale, **scale_kwargs)
            self.data = scale(data=self.data, features=self.features)
            self.scaler = scale

    def run(self, var_use: str = "sample_id", **kwargs):
        """
        Run the harmony algorithm (see https://github.com/slowkow/harmonypy for details). Resulting object
        is stored in 'harmony' attribute

        Parameters
        ----------
        kwargs:
            Additional keyword arguments passed to harmonypy.run_harmony

        Returns
        -------
        Harmony
        """
        logger.info("Running harmony")
        data = self.data[self.features].astype(float)
        self.harmony = harmonypy.run_harmony(data_mat=data.values, meta_data=self.meta, vars_use=var_use, **kwargs)
        return

    def plot_kde(self, var: Union[str, List[str]]):
        """
        Utility function; generates a KDE plot for a single variable in 'data' attribute.
        Uses gaussian kernel and Silverman's method for bandwidth estimation.

        Parameters
        ----------
        var: str

        Returns
        -------
        Matplotlib.Axes
        """
        v = self.data[var].values
        if isinstance(var, list):
            v = np.sum([self.data[x].values for x in var], axis=0)
        x, y = FFTKDE(kernel="gaussian", bw="silverman").fit(v).evaluate()
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.plot(x, y)
        ax.set_xlabel(var)
        return fig, ax

    def add_meta_var(self, mask: pd.DataFrame, meta_var_name: str):
        """
        Add a binary meta variable. Should provide a mask to the 'data' attribute; rows that
        return True for this mask will have a value of 'P' in 'meta_var_name' column of 'meta',
        all other rows will have a value of 'N'.

        Parameters
        ----------
        mask: Pandas.DataFrame
        meta_var_name: str

        Returns
        -------
        self
        """
        self.data[meta_var_name] = "N"
        self.data.loc[mask, meta_var_name] = "P"
        self.meta[meta_var_name] = self.data[meta_var_name]
        self.data.drop(meta_var_name, axis=1, inplace=True)
        return self

    def batch_lisi(self, meta_var: str = "sample_id", sample: float = 1.0):
        """
        Compute LISI using the given meta_var as label

        Parameters
        ----------
        meta_var: str (default="sample_id")
        sample: float (default=1.)
            Fraction to downsample data to prior to calculating LISI. Downsampling is recommended
            for large datasets, since LISI is computationally expensive

        Returns
        -------
        numpy.ndarray

        Raises
        ------
        ValueError
            meta_var is not present in 'meta' attribute
        """
        if meta_var not in self.meta.columns:
            logger.error(f"{meta_var} missing from meta attribute")
            raise ValueError(f"{meta_var} missing from meta attribute")
        idx = np.random.randint(self.data.shape[0], size=int(self.data.shape[0] * sample))
        return harmonypy.lisi.compute_lisi(
            self.batch_corrected()[self.features].values[idx],
            metadata=self.meta.iloc[idx],
            label_colnames=[meta_var],
        )

    def plot_overlay(
        self,
        n: int = 10000,
        dim_reduction_method: str = "UMAP",
        dim_reduction_kwargs: Optional[Dict] = None,
        figsize: Tuple[int] = (12, 7),
        legend: bool = False,
        **plot_kwargs,
    ):
        plot_kwargs["hue"] = plot_kwargs.get("hue", "sample_id")
        plot_kwargs["linewidth"] = plot_kwargs.get("linewidth", 0)
        plot_kwargs["s"] = plot_kwargs.get("s", 1)
        dim_reduction_kwargs = dim_reduction_kwargs or {}
        reducer = DimensionReduction(method=dim_reduction_method, **dim_reduction_kwargs)
        logger.info("Performing dimension reduction on original data")
        before = reducer.fit_transform(data=self.data.sample(n), features=self.features)
        logger.info("Performing dimension reduction on batch corrected data")
        after = reducer.transform(data=self.batch_corrected().loc[before.index], features=self.features)
        logger.info("Plotting comparison")
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        sns.scatterplot(data=before, x="UMAP1", y="UMAP2", ax=axes[0], **plot_kwargs)
        sns.scatterplot(data=after, x="UMAP1", y="UMAP2", ax=axes[1], **plot_kwargs)
        axes[0].set_title("Before")
        axes[1].set_title("After")
        if not legend:
            for ax in axes:
                ax.legend().remove()
        return fig

    def batch_lisi_distribution(self, meta_var: str = "sample_id", sample: Union[float, None] = 0.1, **kwargs):
        """
        Plot the distribution of LISI using the given meta_var as label

        Parameters
        ----------
        meta_var: str (default="sample_id")
        sample: float (default=1.)
            Fraction to downsample data to prior to calculating LISI. Downsampling is recommended
            for large datasets, since LISI is computationally expensive
        kwargs:
            Additional keyword arguments passed to Seaborn.histplot

        Returns
        -------
        Matplotlib.Axes
        """
        idx = np.random.randint(self.data.shape[0], size=int(self.data.shape[0] * sample))
        before = harmonypy.lisi.compute_lisi(
            self.data[self.features].values[idx],
            metadata=self.meta.iloc[idx],
            label_colnames=[meta_var],
        )
        data = pd.DataFrame(
            {
                "Before": before.reshape(-1),
                "After": self.batch_lisi(meta_var=meta_var, sample=sample).reshape(-1),
            }
        )
        data = data.melt(var_name="Data", value_name="LISI")
        kwargs = kwargs or {}
        kwargs["ax"] = kwargs.get("ax", plt.subplots(figsize=(5, 5))[1])
        return sns.histplot(data=data, x="LISI", hue="Data", **kwargs)

    def batch_corrected(self):
        """
        Generates a Pandas DataFrame of batch corrected values. If L2 normalisation was performed prior to
        this, it is reversed. Additional column 'batch_id' identifies rows.

        Returns
        -------
        Pandas.DataFrame

        Raises
        ------
        AssertionError
            Run not called prior to calling 'batch_corrected'
        """
        assert self.harmony is not None, "Call 'run' first"
        corrected = pd.DataFrame(self.harmony.Z_corr.T, columns=self.features)
        corrected["sample_id"] = self.meta.sample_id.values
        corrected["original_index"] = self.data["original_index"]
        return corrected

    def save(
        self,
        experiment: Experiment,
        prefix: str = "Corrected_",
        subject_mappings: Union[Dict[str, str], None] = None,
    ):
        """
        Saved the batch corrected data to an Experiment with each biological specimen (batch) saved
        to an individual FileGroup.

        Parameters
        ----------
        experiment: Experiment
        prefix: str (default="Corrected_")
            Prefix added to sample ID when creating new FileGroup
        subject_mappings: dict, optional
            If provided, key values should match batch_id and value the Subject to associate the new
            FileGroup to

        Returns
        -------
        None

        Raises
        ------
        AssertionError
            Save called before running the Harmony algorithm
        """
        assert self.harmony is not None, "Call 'run' first"
        subject_mappings = subject_mappings or {}
        for sample_id, df in progress_bar(
            self.batch_corrected().groupby("sample_id"),
            verbose=True,
            total=self.meta.sample_id.nunique(),
        ):
            if self.scaler is not None:
                df = self.scaler.inverse(data=df, features=self.features)
            if self.transformer is not None:
                df = self.transformer.inverse_scale(data=df, features=self.features)
            experiment.add_dataframes(
                sample_id=str(prefix) + str(sample_id),
                primary_data=df[self.features],
                mappings=[{"channel": x, "marker": x} for x in self.features],
                subject_id=subject_mappings.get(sample_id, None),
            )


def create_experiment(project, features: List[str], experiment_name: str) -> Experiment:
    """
    Utility function for creating an experiment with FileGroups that contain
    the given features. Useful for creating an Experiment to house batch corrected
    data.

    Parameters
    ----------
    project: Project
    features: list
    experiment_name: str

    Returns
    -------
    Experiment
    """
    markers = [{"name": x, "regex": f"^{x}$", "case": 0, "permutations": ""} for x in features]
    channels = [{"name": x, "regex": f"^{x}$", "case": 0, "permutations": ""} for x in features]
    mappings = [(x, x) for x in features]
    panel_definition = {"markers": markers, "channels": channels, "mappings": mappings}
    return project.add_experiment(experiment_id=experiment_name, panel_definition=panel_definition)
