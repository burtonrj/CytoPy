from scipy.spatial.distance import euclidean
from scipy.linalg import norm
import mongoengine
import numpy as np
import os


def valid_directory(val: str):
    if not os.path.isdir(val):
        raise mongoengine.errors.ValidationError(f"{val} is not a valid directory")


def indexed_parallel_func(x: tuple,
                          func: callable,
                          **kwargs):
    x[0], func(x[1].values, **kwargs)


def hellinger_dist(p: np.array, q: np.array):
    """
    https://gist.github.com/larsmans/3116927
    Parameters
    ----------
    p
    q

    Returns
    -------

    """
    return np.sqrt(np.sum((np.sqrt(p) - np.sqrt(q)) ** 2)) / np.sqrt(2)