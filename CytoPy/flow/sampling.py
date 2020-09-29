from ..data.populations import Population
from ..feedback import vprint
from sklearn.neighbors import BallTree, KDTree, KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, train_test_split, RandomizedSearchCV
from sklearn.metrics import balanced_accuracy_score
from functools import partial
from typing import List
import pandas as pd
import numpy as np


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


def density_dependent_downsampling(data: pd.DataFrame,
                                   features: list,
                                   frac: float = 0.1,
                                   sample_n: int or None = None,
                                   alpha: int = 5,
                                   mmd_sample_n: int = 2000,
                                   outlier_dens: float = 1,
                                   target_dens: float = 5):
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
    features: list
        Name of columns to be used as features in down-sampling algorithm
    frac: float, (default=0.1)
        fraction of dataset to return as a sample
    sample_n: int, optional
        number of events to return in sample (used as alternative to frac)
    alpha: int, (default=5)
        used for estimating distance threshold between cell and nearest neighbour (default = 5 used in
        original paper)
    mmd_sample_n: int, (default=2000)
        number of cells to sample for generation of KD tree
    outlier_dens: float, (default=1)
        used to exclude cells with the lowest local densities; int value as a percentile of the
        lowest local densities e.g. 1 (the default value) means the bottom 1% of cells with lowest local densities
        are regarded as noise
    target_dens: float, (default=5)
        determines how many cells will survive the down-sampling process; int value as a
        percentile of the lowest local densities e.g. 5 (the default value) means the density of bottom 5% of cells
        will serve as the density threshold for rare cell populations

    Returns
    -------
    Pandas.DataFrame
        Down-sampled pandas dataframe
    """

    def prob_downsample(local_d, target_d, outlier_d):
        if local_d <= outlier_d:
            return 0
        if outlier_d < local_d <= target_d:
            return 1
        if local_d > target_d:
            return target_d / local_d

    df = data.copy()
    mmd_sample = df.sample(mmd_sample_n)
    tree = KDTree(mmd_sample[features], metric='manhattan')
    dist, _ = tree.query(mmd_sample[features], k=2)
    dist = np.median([x[1] for x in dist])
    dist_threshold = dist * alpha
    ld = tree.query_radius(df[features], r=dist_threshold, count_only=True)
    od = np.percentile(ld, q=outlier_dens)
    td = np.percentile(ld, q=target_dens)
    prob_f = partial(prob_downsample, target_d=td, outlier_d=od)
    prob = list(map(lambda x: prob_f(x), ld))
    if sum(prob) == 0:
        print('Error: density dependendent downsampling failed; weights sum to zero. Defaulting to uniform '
              'samplings')
        if sample_n is not None:
            return df.sample(n=sample_n)
        return df.sample(frac=frac)
    if sample_n is not None:
        return df.sample(n=sample_n, weights=prob)
    return df.sample(frac=frac, weights=prob)


def upsample_knn(sample: pd.DataFrame,
                 populations: List[Population],
                 original: pd.DataFrame,
                 features: list,
                 n: int or None = None,
                 verbose: bool = True,
                 **kwargs):
    feedback = vprint(verbose)
    feedback("Upsampling...")
    if "n" in kwargs.keys():
        kwargs.pop("n")
    sample = sample.copy()
    sample["label"] = None
    for i, p in enumerate(populations):
        sample.loc[p.index, "label"] = i
    sample["label"].fillna(-1, inplace=True)
    if n is None:
        feedback("Calculating optimal n by cross-validation...")
        n = np.arange(int(sample.shape[0]*0.01),
                      int(sample.shape[0]*0.05),
                      int(sample.shape[0]*0.01)/2, dtype=np.int)
        knn = KNeighborsClassifier(**kwargs)
        grid_cv = GridSearchCV(knn, {"n_neighbors": n}, scoring="balanced_accuracy", n_jobs=-1, cv=10)
        grid_cv.fit(sample[features].values, sample["label"].values)
        n = grid_cv.best_params_.get("n_neighbors")
        feedback(f"Continuing with n={n}; chosen with balanced accuracy of {round(grid_cv.best_score_, 3)}...")
    feedback("Training...")
    X_train, X_test, y_train, y_test = train_test_split(sample[features].values,
                                                        sample["label"].values,
                                                        test_size=0.2,
                                                        random_state=42)
    knn = KNeighborsClassifier(n_neighbors=n, **kwargs)
    knn.fit(X_train, y_train)
    train_acc = balanced_accuracy_score(y_pred=knn.predict(X_train), y_true=y_train)
    val_acc = balanced_accuracy_score(y_pred=knn.predict(X_test), y_true=y_test)
    feedback(f"...training balanced accuracy score: {train_acc}")
    feedback(f"...validation balanced accuracy score: {val_acc}")
    feedback("Predicting labels in original data...")
    original["label"] = knn.predict(original[features].values)
    for i, p in enumerate(populations):
        p.index = original[original["label"] == i].index.values
    return populations


def upsample_svm(sample: pd.DataFrame,
                 populations: List[Population],
                 original: pd.DataFrame,
                 features: list,
                 verbose: bool = True,
                 **kwargs):
    feedback = vprint(verbose)
    feedback("Upsampling...")
    sample = sample.copy()
    sample["label"] = None
    for i, p in enumerate(populations):
        sample.loc[p.index, "label"] = i
    sample["label"].fillna(-1, inplace=True)
    if not kwargs:
        feedback("Calculating optimal parameters by cross-validation...")
        svm = SVC()
        params = {"C": [0.5, 0.75, 1.0],
                  "kernel": ["linear", "poly", "rbf", "sigmoid"],
                  "degree": [3, 4, 5],
                  "class_weight": [None, "balanced"]}
        grid_cv = RandomizedSearchCV(svm, params, scoring="balanced_accuracy", n_jobs=-1, n_iter=10, cv=10)
        grid_cv.fit(sample[features].values, sample["label"].values)
        feedback(f"Continuing with params: {grid_cv.best_params_}; "
                 f"chosen with balanced accuracy of {round(grid_cv.best_score_, 3)}...")
        svm = grid_cv.best_estimator_
    else:
        svm = SVC(**kwargs)
    feedback("Training...")
    X_train, X_test, y_train, y_test = train_test_split(sample[features].values,
                                                        sample["label"].values,
                                                        test_size=0.2,
                                                        random_state=42)
    svm.fit(X_train, y_train)
    train_acc = balanced_accuracy_score(y_pred=svm.predict(X_train), y_true=y_train)
    val_acc = balanced_accuracy_score(y_pred=svm.predict(X_test), y_true=y_test)
    feedback(f"...training balanced accuracy score: {train_acc}")
    feedback(f"...validation balanced accuracy score: {val_acc}")
    feedback("Predicting labels in original data...")
    original["label"] = svm.predict(original[features].values)
    for i, p in enumerate(populations):
        p.index = original[original["label"] == i].index.values
    return populations