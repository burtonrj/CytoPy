from shapely.geometry import Polygon, Point
import pandas as pd
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


def inside_polygon(df: pd.DataFrame,
                   x: str,
                   y: str,
                   poly: Polygon):
    """
    Return rows in dataframe who's values for x and y are contained in some polygon coordinate shape

    Parameters
    ----------
    df: Pandas.DataFrame
        Data to query
    x: str
        name of x-axis plane
    y: str
        name of y-axis plane
    poly: shapely.geometry.Polygon
        Polygon object to search

    Returns
    --------
    Pandas.DataFrame
        Masked DataFrame containing only those rows that fall within the Polygon
    """
    xy = df[[x, y]].values
    pos_idx = list(map(lambda i: poly.contains(Point(i)), xy))
    return df.iloc[pos_idx]