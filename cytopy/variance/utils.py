import logging
import math
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union

import numpy as np
import pandas as pd
import seaborn as sns
from KDEpy import FFTKDE
from matplotlib import pyplot as plt

from cytopy.feedback import progress_bar
from cytopy.utils import DimensionReduction

np.random.seed(42)
logger = logging.getLogger(__name__)


def calculate_ref_sample(data: pd.DataFrame, features: Union[List[str], None] = None) -> str:
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

    Returns
    -------
    str
        Sample ID of reference sample
    """
    data = data.dropna()
    features = features or [x for x in data.columns if x != "sample_id"]
    covar = {k: np.cov(v[features].to_numpy(), rowvar=False) for k, v in data.groupby(by="sample_id")}
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
