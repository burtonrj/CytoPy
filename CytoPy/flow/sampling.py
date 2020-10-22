from .neighbours import calculate_optimal_neighbours, knn
from ..feedback import vprint
from sklearn.neighbors import BallTree, KDTree
from multiprocessing import Pool, cpu_count
from functools import partial
from warnings import warn
import pandas as pd
import numpy as np


def uniform_downsampling(data: pd.DataFrame,
                         sample_size: int or float):
    if isinstance(sample_size, int):
        if sample_size >= data.shape[0]:
            warn(f"Number of observations larger than requested sample size {sample_size}, "
                 f"returning complete data (n={data.shape[0]})")
            return data
        return data.sample(n=sample_size)
    if isinstance(sample_size, float):
        return data.sample(frac=sample_size)
    raise ValueError("sample_size should be an int or float value")


def faithful_downsampling(data: np.array,
                          h: float):
    """
    An implementation of faithful downsampling as described in:  Zare H, Shooshtari P, Gupta A, Brinkman R.
    Data reduction for spectral clustering to analyze high throughput flow cytometry data.
    BMC Bioinformatics 2010;11:403

    Parameters
    -----------
    data: Numpy.array
        numpy array to be down-sampled
    h: float
        radius for nearest neighbours search

    Returns
    --------
    Numpy.array
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


def prob_downsample(local_d, target_d, outlier_d):
    if local_d <= outlier_d:
        return 0
    if outlier_d < local_d <= target_d:
        return 1
    if local_d > target_d:
        return target_d / local_d


def density_dependent_downsampling(data: pd.DataFrame,
                                   features: list or None = None,
                                   sample_size: int or float = 0.1,
                                   alpha: int = 5,
                                   distance_metric: str = "manhattan",
                                   tree_sample: float or int = 0.1,
                                   outlier_dens: float = 1,
                                   target_dens: float = 5,
                                   njobs: int = -1):
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
        warn("Requested sample size >= size of dataframe")
        return data
    df = data.copy()
    features = features or df.columns.tolist()
    tree_sample = uniform_downsampling(data=df, sample_size=tree_sample)
    prob = density_probability_assignment(sample=tree_sample[features],
                                          data=df[features],
                                          distance_metric=distance_metric,
                                          alpha=alpha,
                                          outlier_dens=outlier_dens,
                                          target_dens=target_dens,
                                          njobs=njobs)
    if sum(prob) == 0:
        warn('Error: density dependendent downsampling failed; weights sum to zero. '
             'Defaulting to uniform sampling')
        return uniform_downsampling(data=data, sample_size=sample_size)
    if isinstance(sample_size, int):
        return df.sample(n=sample_size, weights=prob)
    return df.sample(frac=sample_size, weights=prob)


def density_probability_assignment(sample: pd.DataFrame,
                                   data: pd.DataFrame,
                                   distance_metric: str = "manhattan",
                                   alpha: int = 5,
                                   outlier_dens: float = 1,
                                   target_dens: float = 5,
                                   njobs: int = -1):
    if njobs < 0:
        njobs = cpu_count()
    tree = KDTree(sample, metric=distance_metric)
    dist, _ = tree.query(data, k=2)
    dist = np.median([x[1] for x in dist])
    dist_threshold = dist * alpha
    ld = tree.query_radius(data, r=dist_threshold, count_only=True)
    od = np.percentile(ld, q=outlier_dens)
    td = np.percentile(ld, q=target_dens)
    prob_f = partial(prob_downsample, target_d=td, outlier_d=od)
    with Pool(njobs) as pool:
        prob = list(pool.map(prob_f, ld))
    return np.array(prob)


def upsample_density(data: pd.DataFrame,
                     features: list or None = None,
                     upsample_factor: int = 2,
                     sample_size: int or None = None,
                     tree_sample: int or float = 0.1,
                     distance_metric: str = "manhattan",
                     alpha: int = 5,
                     outlier_dens: float = 1,
                     target_dens: float = 5,
                     njobs: int = -1):
    features = features or data.columns.tolist()
    tree_sample = uniform_downsampling(data=data, sample_size=tree_sample)
    prob = density_probability_assignment(sample=tree_sample[features],
                                          data=data[features],
                                          distance_metric=distance_metric,
                                          alpha=alpha,
                                          outlier_dens=outlier_dens,
                                          target_dens=target_dens,
                                          njobs=njobs)
    low_dens_idx = np.where(prob > 1.)
    low_dens_regions = data.iloc[low_dens_idx]
    upsampled_data = [low_dens_regions for _ in range(upsample_factor)]
    data = pd.concat([data] + upsampled_data)
    if sample_size is None:
        return data
    return uniform_downsampling(data=data, sample_size=sample_size)


def upsample_knn(sample: pd.DataFrame,
                 original_data: pd.DataFrame,
                 labels: list,
                 features: list,
                 verbose: bool = True,
                 scoring: str = "balanced_accuracy",
                 **kwargs):
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
    verbose: bool (default=True)
        If True, will provide feedback to stdout
    scoring: str (default="balanced_accuracy")
        Scoring parameter to use for GridSearchCV. Only relevant is n_neighbors parameter is not provided
    kwargs: dict
        Additional keyword arguments passed to Scikit-Learn's KNeighborsClassifier

    Returns
    -------
    numpy.Array
        Array of labels for original data
    """
    feedback = vprint(verbose)
    feedback("Upsampling...")
    n = kwargs.get("n_neighbors", None)
    if n is None:
        feedback("Calculating optimal n_neighbours by grid search CV...")
        n, score = calculate_optimal_neighbours(x=sample[features].values,
                                                y=labels,
                                                scoring=scoring,
                                                **kwargs)
        feedback(f"Continuing with n={n}; chosen with balanced accuracy of {round(score, 3)}...")
    feedback("Training...")
    train_acc, val_acc, model = knn(data=sample,
                                    features=features,
                                    labels=labels,
                                    n_neighbours=n,
                                    holdout_size=0.2,
                                    random_state=42,
                                    return_model=True,
                                    **kwargs)
    feedback(f"...training balanced accuracy score: {train_acc}")
    feedback(f"...validation balanced accuracy score: {val_acc}")
    feedback("Predicting labels in original data...")
    new_labels = model.predict(original_data[features].values)
    feedback("Complete!")
    return new_labels



