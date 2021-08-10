#!/usr/bin.env/python
# -*- coding: utf-8 -*-
"""
For manageable analysis sampling is unavoidable. This module contains all
the functionality for downsampling and subsequent upsampling in cytopy.
cytopy supports uniform sampling that wraps the Pandas DataFrame sample
method. In addition we provide support for density dependent downsampling
(adapted from SPADE; https://www.nature.com/articles/nbt.1991) and faithful
downsampling (adapted from SamSPECTRAL; https://bmcbioinformatics.biomedcentral.com/articles/10.1186/1471-2105-11-403).

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
from functools import partial
from multiprocessing import cpu_count
from multiprocessing import Pool
from typing import *

import numpy as np
import pandas as pd
from sklearn.neighbors import BallTree
from sklearn.neighbors import KDTree

from .neighbours import calculate_optimal_neighbours
from .neighbours import knn

np.random.seed(42)

logger = logging.getLogger(__name__)


class SamplingError(Exception):
    def __init__(self, message: str):
        logger.error(message)
        super().__init__(message)


def uniform_downsampling(data: pd.DataFrame, sample_size: Union[int, float], **kwargs):
    """
    Uniform downsampling. Wraps the Pandas DataFrame sample method
    with some additional error handling for when the requested sample
    size is invalid.

    Parameters
    ----------
    data: Pandas.DataFrame
    sample_size: int or float
        Size of sample required. If a float is given will return a sample
        of this proportion.
    kwargs:
        Additional keyword arguments passed to Pandas.DataFrame.sample

    Returns
    -------
    Pandas.DataFrame

    Raises
    ------
    SamplingError
        Sample size type is invalid; should be either int or float
    """
    if isinstance(sample_size, int):
        if sample_size >= data.shape[0]:
            logger.warning(
                f"Number of observations larger than or equal requested sample size {sample_size}, "
                f"returning complete data (n={data.shape[0]})"
            )
            return data
        return data.sample(n=sample_size, **kwargs)
    if isinstance(sample_size, float):
        return data.sample(frac=sample_size, **kwargs)
    raise SamplingError("sample_size should be an int or float value")


def faithful_downsampling(data: np.array, h: float = 0.1):
    """
    An implementation of faithful downsampling as described in:  Zare H, Shooshtari P, Gupta A, Brinkman R.
    Data reduction for spectral clustering to analyze high throughput flow cytometry data.
    BMC Bioinformatics 2010;11:403

    Parameters
    -----------
    data: numpy.ndarray
        numpy array to be down-sampled
    h: float
        radius for nearest neighbours search

    Returns
    --------
    numpy.ndarray
        Down-sampled array
    """
    communities = None
    registered = np.zeros(data.shape[0])
    tree = BallTree(data)
    while not all([x == 1 for x in registered]):
        i_ = np.random.choice(np.where(registered == 0)[0])
        registered[i_] = 1
        registering_idx = tree.query_radius(data[i_].reshape(1, -1), r=h)[0]
        registering_idx = [t for t in registering_idx if t != i_]
        registered[registering_idx] = 1
        if communities is None:
            communities = data[registering_idx]
        else:
            communities = np.unique(np.concatenate((communities, data[registering_idx]), 0), axis=0)
    return communities


def prob_downsample(local_d: int, target_d: int, outlier_d: int):
    """
    Given local, target and outlier density (as estimated by KNN) calculate
    the probability of retaining the event. If local density is less than or
    equal to the outlier density, returns a probability of 0 (event will be
    discarded). If the local density is greater than the outlier density
    but less than the target density, return a value of 1 (absolutely keep this
    event). If the local density is greater than the target density, then
    the probability of retention is the ratio between the target and local
    density.

    Parameters
    ----------
    local_d: int
    target_d: int
    outlier_d: int

    Returns
    -------
    float
        Value between 0 and 1
    """
    if local_d <= outlier_d:
        return 0
    if outlier_d < local_d <= target_d:
        return 1
    if local_d > target_d:
        return target_d / local_d


def density_dependent_downsampling(
    data: pd.DataFrame,
    features: Optional[List[str]] = None,
    sample_size: Union[int, float] = 0.1,
    alpha: int = 5,
    distance_metric: str = "manhattan",
    tree_sample: Union[int, float] = 0.1,
    outlier_dens: int = 1,
    target_dens: int = 5,
    njobs: int = -1,
):
    """
    Perform density dependent down-sampling to remove risk of under-sampling rare populations;
    adapted from SPADE*

    * Extracting a cellular hierarchy from high-dimensional cytometry data with SPADE
    Peng Qiu-Erin Simonds-Sean Bendall-Kenneth Gibbs-Robert
    Bruggner-Michael Linderman-Karen Sachs-Garry Nolan-Sylvia Plevritis - Nature Biotechnology - 2011

    Parameters
    -----------
    data: Pandas.DataFrame
        Data to sample
    features: list (defaults to all columns)
        Name of columns to be used as features in down-sampling algorithm
    sample_size: int or float (default=0.1)
        number of events to return in sample, either as an integer of fraction of original
        sample size
    alpha: int, (default=5)
        used for estimating distance threshold between cell and nearest neighbour (default = 5 used in
        original paper)
    distance_metric: str (default="manhattan")
        Metric used for neighbour assignment
    tree_sample: float or int, (default=0.1)
        proportion/number of cells to sample for generation of KD tree
    outlier_dens: float, (default=1)
        used to exclude cells with the lowest local densities; int value as a percentile of the
        lowest local densities e.g. 1 (the default value) means the bottom 1% of cells with lowest local densities
        are regarded as noise
    target_dens: float, (default=5)
        determines how many cells will survive the down-sampling process; int value as a
        percentile of the lowest local densities e.g. 5 (the default value) means the density of bottom 5% of cells
        will serve as the density threshold for rare cell populations
    njobs: int (default=-1)
        Number of jobs to run in unison when calculating weights (defaults to all available cores)
    Returns
    -------
    Pandas.DataFrame
        Down-sampled pandas dataframe
    """
    if isinstance(sample_size, int) and sample_size >= data.shape[0]:
        logger.warning("Requested sample size >= size of dataframe")
        return data
    df = data.copy()
    features = features or df.columns.tolist()
    tree_sample = uniform_downsampling(data=df, sample_size=tree_sample)
    prob = density_probability_assignment(
        sample=tree_sample[features],
        data=df[features],
        distance_metric=distance_metric,
        alpha=alpha,
        outlier_dens=outlier_dens,
        target_dens=target_dens,
        njobs=njobs,
    )
    if sum(prob) == 0:
        logger.warning(
            "Error: density dependendent downsampling failed; weights sum to zero. " "Defaulting to uniform sampling"
        )
        return uniform_downsampling(data=data, sample_size=sample_size)
    if isinstance(sample_size, int):
        return df.sample(n=sample_size, weights=prob)
    return df.sample(frac=sample_size, weights=prob)


def density_probability_assignment(
    sample: pd.DataFrame,
    data: pd.DataFrame,
    distance_metric: str = "manhattan",
    alpha: int = 5,
    outlier_dens: int = 1,
    target_dens: int = 5,
    njobs: int = -1,
):
    """
    Generate an estimation of local density amongst single cell population
    using the KDTree algorithm from Scikit-Learn. Using this representation
    return the probability assignment for retention of each event using
    prob_downsample. adapted from SPADE*

    * Extracting a cellular hierarchy from high-dimensional cytometry data with SPADE
    Peng Qiu-Erin Simonds-Sean Bendall-Kenneth Gibbs-Robert
    Bruggner-Michael Linderman-Karen Sachs-Garry Nolan-Sylvia Plevritis - Nature Biotechnology - 2011

    Parameters
    ----------
    sample: Pandas.DataFrame
        Downsampled data to use for generating nearest neighbours tree graph
    data: Pandas.DataFrame
        Original dataframe
    distance_metric: str (default="manhattan")
        Metric used for neighbour assignment
    alpha: int
        Used for estimating distance threshold between cell and nearest neighbour (default = 5 used in
        original paper)
    outlier_dens: int, (default=1)
        used to exclude cells with the lowest local densities; float value as a percentile of the
        lowest local densities e.g. 1 (the default value) means the bottom 1% of cells with lowest local densities
        are regarded as noise
    target_dens: int, (default=5)
        determines how many cells will receive a probability > 0; int value as a
        percentile of the lowest local densities e.g. 5 (the default value)
        means the density of bottom 5% of cells will serve as the density threshold
        for rare cell populations
    njobs: int (default=-1)
        Controls how many parallel processed to run in KDTree search. Default is -1, which
        will use all available cores.

    Returns
    -------
    numpy.ndarray
    """
    if njobs < 0:
        njobs = cpu_count()
    tree = KDTree(sample, metric=distance_metric, leaf_size=100)
    dist, _ = tree.query(data, k=2)
    dist = np.median([x[1] for x in dist])
    dist_threshold = dist * alpha
    ld = tree.query_radius(data, r=dist_threshold, count_only=True)
    od = np.percentile(ld, q=outlier_dens)
    td = np.percentile(ld, q=target_dens)
    prob_f = partial(prob_downsample, target_d=td, outlier_d=od)
    with Pool(njobs) as pool:
        prob = np.array(list(pool.map(prob_f, ld)))
    return np.array(prob)


def upsample_density(
    data: pd.DataFrame,
    features: list or None = None,
    upsample_factor: int = 2,
    sample_size: int or None = None,
    tree_sample: int or float = 0.1,
    distance_metric: str = "manhattan",
    alpha: int = 5,
    outlier_dens: int = 1,
    target_dens: int = 5,
    njobs: int = -1,
):
    """
    Perform upsampling in a density dependent manner; neighbourhoods of cells of low
    density will have a high probability of being upsampled versus dense neighbourhoods.
    Ignores outliers. adapted from SPADE*

    * Extracting a cellular hierarchy from high-dimensional cytometry data with SPADE
    Peng Qiu-Erin Simonds-Sean Bendall-Kenneth Gibbs-Robert
    Bruggner-Michael Linderman-Karen Sachs-Garry Nolan-Sylvia Plevritis - Nature Biotechnology - 2011

    Parameters
    ----------
    data: Pandas.DataFrame
        Data to sample
    features: list (defaults to all columns)
        Name of columns to be used as features in down-sampling algorithm
    sample_size: int or float (default=0.1)
        number of events to return in sample, either as an integer of fraction of original
        sample size
    alpha: int, (default=5)
        used for estimating distance threshold between cell and nearest neighbour (default = 5 used in
        original paper)
    distance_metric: str (default="manhattan")
        Metric used for neighbour assignment
    upsample_factor: int (default=2)
        Factor to upsample by (e.g. default=2 would double the observations)
    tree_sample: float or int, (default=0.1)
        proportion/number of cells to sample for generation of KD tree
    outlier_dens: float, (default=1)
        used to exclude cells with the lowest local densities; int value as a percentile of the
        lowest local densities e.g. 1 (the default value) means the bottom 1% of cells with lowest local densities
        are regarded as noise
    target_dens: float, (default=5)
        determines how many cells will survive the down-sampling process; int value as a
        percentile of the lowest local densities e.g. 5 (the default value) means the density of bottom 5% of cells
        will serve as the density threshold for rare cell populations
    njobs: int (default=-1)
        Number of jobs to run in unison when calculating weights (defaults to all available cores)

    Returns
    -------

    """
    features = features or data.columns.tolist()
    tree_sample = uniform_downsampling(data=data, sample_size=tree_sample)
    prob = density_probability_assignment(
        sample=tree_sample[features],
        data=data[features],
        distance_metric=distance_metric,
        alpha=alpha,
        outlier_dens=outlier_dens,
        target_dens=target_dens,
        njobs=njobs,
    )
    low_dens_idx = np.where(prob > 1.0)
    low_dens_regions = data.iloc[low_dens_idx]
    upsampled_data = [low_dens_regions for _ in range(upsample_factor)]
    data = pd.concat([data] + upsampled_data)
    if sample_size is None:
        return data
    return uniform_downsampling(data=data, sample_size=sample_size)


def upsample_knn(
    sample: pd.DataFrame,
    original_data: pd.DataFrame,
    labels: list,
    features: list,
    scoring: str = "balanced_accuracy",
    **kwargs,
):
    """
    Given some sampled dataframe and the original dataframe from which it was derived, use the
    given labels (which should correspond to the sampled dataframe row index) to fit a nearest
    neighbours model to the sampled data and predict the assignment of labels in the original data.
    Uses sklearn.neighbors.KNeighborsClassifier for KNN implementation. If n_neighbors parameter
    is not provided, will estimate using grid search cross validation. The scoring parameter
    can be tuned by changing the `scoring` input (default="balanced_accuracy")

    Parameters
    ----------
    sample: Pandas.DataFrame
        Sampled dataframe that has been classified/gated/etc
    original_data: Pandas.DataFrame
        Original dataframe prior to sampling (unlabeled)
    labels: list
        List of labels (should correspond to the label for each row)
    features: list
        List of features (column names)
    scoring: str (default="balanced_accuracy")
        Scoring parameter to use for GridSearchCV. Only relevant is n_neighbors parameter is not provided
    kwargs: dict
        Additional keyword arguments passed to Scikit-Learn's KNeighborsClassifier

    Returns
    -------
    numpy.ndarray
        Array of labels for original data
    """
    logger.info("Upsampling...")
    n = kwargs.get("n_neighbors", None)
    if n is None:
        logger.info("Calculating optimal n_neighbours by grid search CV...")
        n, score = calculate_optimal_neighbours(x=sample[features].values, y=labels, scoring=scoring, **kwargs)
        logger.info(f"Continuing with n={n}; chosen with balanced accuracy of {round(score, 3)}...")
    logger.info("Training...")
    train_acc, val_acc, model = knn(
        data=sample,
        features=features,
        labels=np.array(labels),
        n_neighbours=n,
        holdout_size=0.2,
        random_state=42,
        return_model=True,
        **kwargs,
    )
    logger.info(f"...training balanced accuracy score: {train_acc}")
    logger.info(f"...validation balanced accuracy score: {val_acc}")
    logger.info("Predicting labels in original data...")
    new_labels = model.predict(original_data[features].values)
    logger.info("Complete!")
    return new_labels


def sample_dataframe(
    data: pd.DataFrame,
    sample_size: Union[int, float] = 0.1,
    method: str = "uniform",
    **kwargs,
) -> pd.DataFrame:
    """
    Convenient wrapper function for common sampling methods.

    Parameters
    ----------
    data: Pandas.DataFrame
    sample_size: float or int (default=0.1)
    method: str
        One of 'uniform', 'density' or 'faithful'
    kwargs:
        Additional keyword arguments passed to chosen method. See cytopy.flow.sampling for details

    Returns
    -------
    Pandas.DataFrame

    Raises
    ------
    SamplingError
        Invalid method
    """
    if method == "uniform":
        return uniform_downsampling(data=data, sample_size=sample_size, **kwargs)
    elif method == "density":
        return density_dependent_downsampling(data=data, sample_size=sample_size, **kwargs)
    elif method == "faithful":
        return pd.DataFrame(faithful_downsampling(data=data.values, **kwargs), columns=data.columns)
    else:
        valid = ["uniform", "density", "faithful"]
        raise SamplingError(f"Invalid method, must be one of {valid}")


def sample_dataframe_uniform_groups(data: pd.DataFrame, group_id: str, sample_size: int):
    sample_data = list()
    n = int(sample_size / data[group_id].nunique())
    for _, df in data.groupby(group_id):
        if n >= df.shape[0]:
            sample_data.append(df)
        else:
            sample_data.append(df.sample(n))
    return pd.concat(sample_data)
