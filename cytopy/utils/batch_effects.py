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
import os
import pickle
from typing import *

import flowio
import harmonypy
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from KDEpy import FFTKDE
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KernelDensity

from ..data.experiment import Experiment
from ..data.panel import Channel
from ..data.panel import Panel
from ..data.project import Project
from ..feedback import progress_bar
from ..feedback import vprint
from ..utils import transform as transform_module
from .dim_reduction import DimensionReduction
from cytopy.utils.transform import TRANSFORMERS

np.random.seed(42)
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
    grid.fit(data[features].to_numpy())
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
    data: pd.DataFrame
    features: list, optional
    verbose: bool, (default=True)

    Returns
    -------
    str
        Sample ID of reference sample
    """
    feedback = vprint(verbose)
    feedback("Calculate covariance matrix for each sample...")
    # Calculate covar for each
    data = data.dropna()
    features = features or [x for x in data.columns if x != "sample_id"]
    covar = {k: np.cov(v[features].to_numpy(), rowvar=False) for k, v in data.groupby(by="sample_id")}
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
    data: pandas.DataFrame
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
    markers = markers or data.columns.tolist()
    i = 0
    nrows = math.ceil(len(markers) / 3)
    fig.suptitle(f"Per-channel KDE, Reference: {reference}", y=1.02)
    for marker in progress_bar(markers, verbose=verbose):
        i += 1
        ax = fig.add_subplot(nrows, 3, i)
        x, y = FFTKDE(kernel=kernel, bw=kde_bw).fit(data[data.sample_id == reference][marker].to_numpy()).evaluate()
        ax.plot(x, y, color="b", **kwargs)
        ax.fill_between(x, 0, y, facecolor="b", alpha=0.2)
        ax.set_title(marker)
        if xlim:
            ax.set_xlim(xlim)
        for comparison_sample_id in comparison_samples:
            df = data[data.sample_id == comparison_sample_id]
            if marker not in df.columns:
                logger.warning(f"{marker} missing from {comparison_sample_id}, this marker will be ignored")
            else:
                x, y = FFTKDE(kernel=kernel, bw=kde_bw).fit(df[marker].to_numpy()).evaluate()
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
) -> plt.Figure:
    """
    Generate a grid of embeddings using a valid dimensionality reduction technique, in each plot a reference sample
    is shown in blue and a comparison sample in red. The reference sample is conserved across all plots.

    Parameters
    ------------
    data: pandas.DataFrame
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
        Method to use for dimensionality reduction (see utils.dim_reduction)
    dim_reduction_kwargs: dict
        Additional keyword arguments passed to cytopy.dim_reduction.dimensionality_reduction
    kde: bool, (default=False)
        If True, overlay with two-dimensional PDF estimated by KDE
    verbose: bool (default=True)

    Returns
    -------
    Matplotlib.Figure
        Plot printed to stdout

    Raises
    ------
    AssertionError
        Reference absent from data

    ValueError
        Invalid features provided
    """
    assert reference in data.sample_id.unique(), "Reference absent from given data"
    data = data.dropna()
    comparison_samples = comparison_samples or [x for x in data.sample_id.unique() if x != reference]
    dim_reduction_kwargs = dim_reduction_kwargs or {}
    fig = plt.figure(figsize=figsize)
    nrows = math.ceil(len(comparison_samples) / 3)
    reference_df = data[data.sample_id == reference]
    if not all([f in reference_df.columns for f in features]):
        raise ValueError(f"Invalid features; valid are: {reference_df.columns}")
    reducer = DimensionReduction(method=method, n_components=2, **dim_reduction_kwargs)
    reference_df = reducer.fit_transform(data=reference_df, features=features)
    i = 0
    fig.suptitle(f"{method}, Reference: {reference}", y=1.05)
    for sample_id in progress_bar(comparison_samples, verbose=verbose):
        i += 1
        ax = fig.add_subplot(nrows, 3, i)
        embeddings = reducer.transform(data[data.sample_id == sample_id].copy(), features=features)
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
        ax.scatter(embeddings[x], embeddings[y], c="red", s=4, alpha=0.1)
        if kde:
            sns.kdeplot(
                embeddings[x],
                embeddings[y],
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


class Harmony:
    """
    Perform batch-effect correction using the Harmony algorithm, first described by Korsunsky et al [1] and
    implemented in Python by Kamil Slowikowski [2].

    [1] Korsunsky, I., Millard, N., Fan, J. et al. Fast, sensitive and accurate integration of single-cell data
    with Harmony. Nat Methods 16, 1289–1296 (2019). https://doi.org/10.1038/s41592-019-0619-0
    [2] https://github.com/slowkow/harmonypy

    Parameters
    ----------
    data: Pandas.DataFrame
        Expects a Pandas DataFrame containing data from multiple FileGroups; you should generate this data
        using the cytopy.data.experiment.single_cell_dataframe function. It is recommended that you downsample
        at the file level using this function as Harmony is computationally expensive.
    features: list
        List of features to include in batch correction; only these features will appear
        in saved data, all other columns will be removed
    transform: str (default='logicle')
        How to transform data prior to batch correction. For valid methods see cytopy.utils.transform
    transform_kwargs: dict, optional
        Additional keyword arguments passed to Transformer
    scale: str, optional (default='standard')
        How to scale data prior to batch correction. For valid methods see cytopy.utils.transform.Scaler
    scale_kwargs: dict, optional
        Additional keyword arguments passed to Scaler

    Attributes
    ----------
    data: pandas.DataFrame
        Downsampled data from Experiment as generated by cytopy.utils.variance.load_and_sample
    transformer: Transformer
        Transfomation object; used to inverse transform when saving data back to the database
        after correction
    features: list
        List of features
    meta: pandas.DataFrame
        Meta DataFrame; by default it has one column that identifies 'batches' and is always 'sample_id'
    scaler: Scaler
        Scaler object; used to inverse the scale when saving data back to the database after correction
    """

    def __init__(
        self,
        data: Union[pd.DataFrame, str],
        features: List[str],
        transform: Optional[str] = "logicle",
        transform_kwargs: Union[Dict[str, str], None] = None,
        scale: Optional[str] = "standard",
        scale_kwargs: Optional[Dict] = None,
        harmony_cache: Optional[str] = None,
    ):
        if isinstance(data, str):
            data = pd.read_csv(data)

        transform_kwargs = transform_kwargs or {}
        self.transformer = None if transform is None else TRANSFORMERS[transform](**transform_kwargs)
        if self.transformer is not None:
            self.data = self.transformer.scale(data=data, features=features)
        else:
            self.data = data
        self.data = self.data[~self.data[features].isnull().any(axis=1)].reset_index(drop=True)
        self.features = [x for x in features if x in self.data.columns]
        self.meta = self.data[["sample_id"]]
        self.harmony = None
        self.scaler = None
        if scale is not None:
            scale_kwargs = scale_kwargs or {}
            scale = transform_module.Scaler(method=scale, **scale_kwargs)
            self.data = scale(data=self.data, features=self.features)
            self.scaler = scale
        if harmony_cache is not None:
            logger.warning(
                "Loading harmony object from disk. It is the users responsibility to ensure that the "
                "cached harmony object matches the loaded dataframe!"
            )
            with open(harmony_cache, "rb") as f:
                self.harmony = pickle.load(f)

    def cache(self, data_path: str = "harmony_data.csv", harmony_path: str = "harmony.pkl"):
        self.data.to_csv(data_path, index=False)
        with open(harmony_path, "wb") as f:
            pickle.dump(self.harmony, f)

    def run(self, var_use: str = "sample_id", **kwargs):
        """
        Run the harmony algorithm (see https://github.com/slowkow/harmonypy for details). Resulting object
        is stored in 'harmony' attribute

        Parameters
        ----------
        var_use: str (default="sample_id")
            Name of the meta variable to use to match batches
        kwargs:
            Additional keyword arguments passed to harmonypy.run_harmony

        Returns
        -------
        Harmony
        """
        logger.info("Running harmony")
        data = self.data[self.features]
        self.harmony = harmonypy.run_harmony(data_mat=data.to_numpy(), meta_data=self.meta, vars_use=var_use, **kwargs)
        return self

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
        v = self.data[var].to_numpy()
        if isinstance(var, list):
            v = np.sum([self.data[x].to_numpy() for x in var], axis=0)
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
        mask: pandas.DataFrame
        meta_var_name: str

        Returns
        -------
        self
        """
        self.data[meta_var_name] = "N"
        self.data[mask, meta_var_name] = "P"
        self.meta[meta_var_name] = self.data[meta_var_name]
        self.data = self.data.drop(meta_var_name)
        return self

    def batch_lisi(
        self, meta_var: str = "sample_id", sample: Optional[int] = 1000, idx: Optional[Iterable[int]] = None
    ):
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
        corrected = self.batch_corrected()
        corrected["meta"] = self.meta[meta_var]

        if idx is not None:
            corrected = corrected.iloc[idx]
        elif sample is not None:
            corrected = corrected.sample(n=sample)

        return harmonypy.lisi.compute_lisi(
            corrected[self.features].values,
            metadata=corrected[["meta"]],
            label_colnames=["meta"],
        )

    def plot_correction(
        self,
        n: int = 10000,
        dim_reduction_method: str = "UMAP",
        dim_reduction_kwargs: Optional[Dict] = None,
        figsize: Tuple[int] = (12, 7),
        legend: bool = False,
        **plot_kwargs,
    ):
        """
        Using a chosen dimension reduction technique, embed the original and corrected data in the same
        space and plot as scatterplots for comparison.

        Parameters
        ----------
        n: int (default=10000)
        dim_reduction_method: str (default="UMAP")
        dim_reduction_kwargs: Dict, optional
        figsize: Tuple[int] (default=(12, 7)
        legend: bool (default=False)
        plot_kwargs: Dict
            Additional plotting kwargs passed to Seaborn.scatterplot call

        Returns
        -------
        Matplotlib.Figure
        """
        plot_kwargs["hue"] = plot_kwargs.get("hue", "sample_id")
        plot_kwargs["linewidth"] = plot_kwargs.get("linewidth", 0)
        plot_kwargs["s"] = plot_kwargs.get("s", 1)
        dim_reduction_kwargs = dim_reduction_kwargs or {}
        reducer = DimensionReduction(method=dim_reduction_method, **dim_reduction_kwargs)
        logger.info("Performing dimension reduction on original data")
        before = reducer.fit_transform(data=self.data.sample(n), features=self.features)
        logger.info("Performing dimension reduction on batch corrected data")
        after = reducer.transform(data=self.batch_corrected().iloc[before.index], features=self.features)
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

    def batch_lisi_distribution(self, meta_var: str = "sample_id", sample: Optional[int] = 1000, **kwargs):
        """
        Plot the distribution of LISI using the given meta_var as label

        Parameters
        ----------
        meta_var: str (default="sample_id")
        sample: float (default=1.)
            Number of events to sample prior to calculating LISI. Downsampling is recommended
            for large datasets, since LISI is computationally expensive
        kwargs:
            Additional keyword arguments passed to Seaborn.histplot

        Returns
        -------
        Matplotlib.Axes
        """
        data = self.data[self.features].copy()
        data["meta"] = self.meta[meta_var]
        idx = None
        if sample is not None:
            data = data.sample(n=sample)
            idx = data.index.values
        before = harmonypy.lisi.compute_lisi(
            data[self.features].values,
            metadata=data[["meta"]],
            label_colnames=["meta"],
        )
        data = pd.DataFrame(
            {
                "Before": before.reshape(-1),
                "After": self.batch_lisi(meta_var=meta_var, idx=idx).reshape(-1),
            }
        )
        data = data.melt(var_name="Data", value_name="LISI")
        kwargs = kwargs or {}
        kwargs["ax"] = kwargs.get("ax", plt.subplots(figsize=(5, 5))[1])
        return sns.histplot(data=data, x="LISI", hue="Data", **kwargs)

    def batch_corrected(self):
        """
        Generates a pandas DataFrame of batch corrected values. If L2 normalisation was performed prior to
        this, it is reversed. Additional column 'batch_id' identifies rows.

        Returns
        -------
        pandas.DataFrame

        Raises
        ------
        AssertionError
            Run not called prior to calling 'batch_corrected'
        """
        assert self.harmony is not None, "Call 'run' first"
        corrected = pd.DataFrame(self.harmony.Z_corr.T, columns=self.features)
        corrected["sample_id"] = self.meta.sample_id
        corrected["original_index"] = self.data["original_index"]
        return corrected

    def save(
        self,
        project: Project,
        fcs_dir: str,
        experiment_id: str,
        prefix: str = "HarmonyCorrected_",
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
            FileGroup to. If None and the 'data' attribute has 'subject_id' column, associations will
            be inferred from here.

        Returns
        -------
        None

        Raises
        ------
        AssertionError
            Save called before running the Harmony algorithm
        """
        assert self.harmony is not None, "Call 'run' first"
        assert experiment_id not in project.list_experiments(), f"Experiment with ID {experiment_id} already exists!"
        if not os.path.isdir(fcs_dir):
            raise FileNotFoundError(f"Directory {fcs_dir} not found.")

        logger.info(f"Creating {experiment_id}...")
        exp = Experiment(experiment_id=experiment_id)
        exp.panel = Panel()
        for channel in self.features:
            exp.panel.channels.append(Channel(channel=channel, name=channel))
        exp.save()
        project.experiments.append(exp)
        project.save()

        subject_mappings = subject_mappings or {}
        if len(subject_mappings) == 0 and "subject_id" in self.data.columns:
            logger.info("Inferring subject mappings from data attribute")
            subject_mappings = self.data[["sample_id", "subject_id"]]
            subject_mappings = dict(zip(subject_mappings.sample_id, subject_mappings.subject_id))

        logger.info(f"Saving corrected data to disk and associating to {experiment_id}")
        try:
            for sample_id, df in progress_bar(
                self.batch_corrected().groupby("sample_id"),
                verbose=True,
                total=self.meta.sample_id.nunique(),
            ):
                df = df[self.features]
                if self.scaler is not None:
                    df = self.scaler.inverse(data=df, features=self.features)
                if self.transformer is not None:
                    df = self.transformer.inverse_scale(data=df, features=self.features)
                filepath = os.path.join(fcs_dir, f"{prefix}_{sample_id}.fcs")
                with open(filepath, "wb") as f:
                    flowio.create_fcs(
                        event_data=df.to_numpy().flatten(),
                        file_handle=f,
                        channel_names=self.features,
                        opt_channel_names=self.features,
                    )
                exp.add_filegroup(
                    sample_id=sample_id,
                    paths={"primary": filepath},
                    compensate=False,
                    subject_id=subject_mappings.get(sample_id, None),
                )
            project.save()
        except (TypeError, ValueError, AttributeError, AssertionError) as e:
            logger.error("Failed to save data. Rolling back changes.")
            logger.exception(e)
            project.delete_experiment(experiment_id=experiment_id)
            project.save()
