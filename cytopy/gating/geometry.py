#!/usr/bin.env/python
# -*- coding: utf-8 -*-
"""
For the purpose of cytometry analysis we often think of a population
of cells as having a particular phenotype that can be identified by
sub-setting cells in one or two dimensional space. This results in
geometric objects that define a population. This module houses the
functionality around those geometric objects.

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
from functools import partial
from typing import List
from typing import Tuple
from typing import Union

import alphashape
import mongoengine
import numpy as np
import pandas as pd
import polars as pl
from matplotlib.patches import Ellipse
from numba import jit
from scipy import linalg
from scipy import stats
from shapely.geometry import Point
from shapely.geometry import Polygon

from cytopy.data.read_write import pandas_to_polars
from cytopy.data.read_write import polars_to_pandas
from cytopy.utils import transform


class GeometryError(Exception):
    pass


class PopulationGeometry(mongoengine.EmbeddedDocument):
    """
    Geometric shape generated by non-threshold generating Gate

    Attributes
    -----------
    x: str
        Name of the X-dimension e.g. CD3, FSC-A etc
    y: str
        Name of the Y-dimension e.g. CD3, FSC-A etc
    transform_x: str
        Transformation method applied to the x-axis
    transform_y: str
        Transformation method applied to the y-axis
    transform_x_kwargs: dict
        Transformation keyword arguments for transform method applied to the x-axis
    transform_y_kwargs: str
        Transformation keyword arguments for transform method applied to the y-axis
    """

    x = mongoengine.StringField()
    y = mongoengine.StringField()
    transform_x = mongoengine.StringField()
    transform_y = mongoengine.StringField()
    transform_x_kwargs = mongoengine.DictField()
    transform_y_kwargs = mongoengine.DictField()
    meta = {"allow_inheritance": True}


class ThresholdGeom(PopulationGeometry):
    """
    Threshold shape. Inherits from PopulationGeometry.
    NOTE: Thresholds should be stored as transformed values and converted to linear space
    using the 'transform_to_linear' method.

    Attributes
    -----------
    x_threshold: float
        Threshold applied to the X-axis
    y_threshold: float
        Threshold applied to the Y-axis
    """

    x_threshold = mongoengine.FloatField()
    y_threshold = mongoengine.FloatField()

    def transform_to_linear(self):
        """
        Thresholds are transformed to their equivalent value in linear space
        according to the transform defined. If transform is None, thresholds
        are returned as saved.

        Returns
        -------
        float, float
        """
        x, y = self.x_threshold, self.y_threshold
        if self.transform_x:
            kwargs = self.transform_x_kwargs or {}
            transformer = transform.TRANSFORMERS.get(self.transform_x)(**kwargs)
            x = transformer.inverse_scale(pd.DataFrame({"x": [self.x_threshold]}), features=["x"])["x"].values[0]
        if self.transform_y:
            kwargs = self.transform_y_kwargs or {}
            transformer = transform.TRANSFORMERS.get(self.transform_y)(**kwargs)
            y = transformer.inverse_scale(pd.DataFrame({"y": [self.y_threshold]}), features=["y"])["y"].values[0]
        return x, y


class PolygonGeom(PopulationGeometry):
    """
    Polygon shape. Inherits from PopulationGeometry.
    NOTE: X and Y values should be stored as transformed values and converted to linear space
    using the 'transform_to_linear' method.

    Attributes
    -----------
    x_values: list
        X-axis coordinates
    y_values: list
        Y-axis coordinates
    """

    x_values = mongoengine.ListField()
    y_values = mongoengine.ListField()

    @property
    def shape(self):
        assert self.x_values is not None and self.y_values is not None, "x and y values not defined for this Polygon"
        return create_polygon(self.x_values, self.y_values)

    def transform_to_linear(self):
        """
        x,y coordinates are transformed to their equivalent value in linear space
        according to the transform defined. If transform is None, coordinates
        are returned as saved.

        Returns
        -------
        numpy.ndarray, numpy.ndarray
        """
        x_values, y_values = self.x_values, self.y_values
        if self.transform_x:
            kwargs = self.transform_x_kwargs or {}
            transformer = transform.TRANSFORMERS.get(self.transform_x)(**kwargs)
            x_values = transformer.inverse_scale(pd.DataFrame({"x": self.x_values}), features=["x"])["x"].values
        if self.transform_y:
            kwargs = self.transform_y_kwargs or {}
            transformer = transform.TRANSFORMERS.get(self.transform_y)(**kwargs)
            y_values = transformer.inverse_scale(pd.DataFrame({"y": self.y_values}), features=["y"])["y"].values
        return x_values, y_values


def point_in_poly(coords: Tuple[float], poly: Polygon) -> pl.Series:
    return poly.contains(Point(coords))


def inside_polygon(df: Union[pl.DataFrame, pd.DataFrame], x: str, y: str, poly: Polygon) -> pl.DataFrame:
    """
    Return rows in dataframe who's values for x and y are contained in some polygon coordinate shape

    Parameters
    ----------
    df: polars.DataFrame or Pandas.DataFrame
        Data to query
    x: str
        name of x-axis plane
    y: str
        name of y-axis plane
    poly: shapely.geometry.Polygon
        Polygon object to search

    Returns
    --------
    polars.DataFrame
        Masked DataFrame containing only those rows that fall within the Polygon
    """
    df = df if isinstance(df, pl.DataFrame) else pandas_to_polars(data=df)
    point_inside_polygon = partial(point_in_poly, poly=poly)
    mask = df[[x, y]].apply(point_inside_polygon, return_dtype=pl.Boolean)
    return polars_to_pandas(data=df[mask, :])


def polygon_overlap(poly1: Polygon, poly2: Polygon, threshold: float = 0.0):
    """
    Compare the area of two polygons and give the fraction overlap.
    If fraction overlap does not exceed given threshold or the polygon's do not overlap,
    return 0.0

    Parameters
    ----------
    poly1: Polygon
    poly2: Polygon
    threshold: float (default = 0.0)

    Returns
    -------
    float
    """
    if poly1.intersects(poly2):
        overlap = float(poly1.intersection(poly2).area / poly1.area)
        if overlap >= threshold:
            return overlap
    return 0.0


def create_polygon(x: List[float], y: List[float]) -> Polygon:
    """
    Given a list of x coordinated and a list of y coordinates, generate a shapely Polygon

    Parameters
    ----------
    x: list
    y: list

    Returns
    -------
    shapely.geometry.Polygon
    """
    return Polygon([(x, y) for x, y in zip(x, y)])


@jit(nopython=True)
def inside_ellipse(
    data: np.array,
    center: tuple,
    width: int or float,
    height: int or float,
    angle: int or float,
) -> np.ndarray:
    """
    Return mask of two dimensional matrix specifying if a data point (row) falls
    within an ellipse

    Parameters
    -----------
    data: numpy.ndarray
        two dimensional matrix (x,y)
    center: tuple
        x,y coordinate corresponding to center of elipse
    width: int or float
        semi-major axis of eplipse
    height: int or float
        semi-minor axis of elipse
    angle: int or float
        angle of ellipse

    Returns
    --------
    Numpy.Array
        numpy array of indices for values inside specified ellipse
    """
    cos_angle = np.cos(np.radians(180.0 - angle))
    sin_angle = np.sin(np.radians(180.0 - angle))

    x = data[:, 0]
    y = data[:, 1]

    xc = x - center[0]
    yc = y - center[1]

    xct = xc * cos_angle - yc * sin_angle
    yct = xc * sin_angle + yc * cos_angle

    rad_cc = (xct ** 2 / (width / 2.0) ** 2) + (yct ** 2 / (height / 2.0) ** 2)

    in_ellipse = []

    for r in rad_cc:
        if r <= 1.0:
            # point in ellipse
            in_ellipse.append(True)
        else:
            # point not in ellipse
            in_ellipse.append(False)
    return in_ellipse


@jit(nopython=True)
def probabilistic_ellipse(covariances: np.ndarray, conf: float):
    """
    Given the covariance matrix of a mixture component, calculate a elliptical shape that
    represents a probabilistic confidence interval.

    Parameters
    ----------
    covariances: Numpy.Array
        Covariance matrix
    conf: float
        The confidence interval (e.g. 0.95 would give the region of 95% confidence)

    Returns
    -------
    float and float and float
        Width, Height and Angle of ellipse
    """
    eigen_val, eigen_vec = linalg.eigh(covariances)
    chi2 = stats.chi2.ppf(conf, 2)
    eigen_val = 2.0 * np.sqrt(eigen_val) * np.sqrt(chi2)
    u = eigen_vec[0] / linalg.norm(eigen_vec[0])
    angle = 180.0 * np.arctan(u[1] / u[0]) / np.pi
    return eigen_val[0], eigen_val[1], (180.0 + angle)


def create_envelope(x_values: np.array, y_values: np.array, alpha: float or None = 0.0) -> Polygon:
    """
    Given the x and y coordinates of a cloud of data points generate an envelope (alpha shape)
    that encapsulates these data points.

    Parameters
    ----------
    x_values: Numpy.Array
    y_values: Numpy.Array
    alpha: float or None (default = 0.0)
        By default alpha is 0, generating a convex hull (can be thought of as if wrapping an elastic band
        around the data points). Increase alpha to create a concave envelope. Warning, as alpha increases,
        more data points will fall outside the range of the envelope.


    Returns
    -------
    shapely.geometry.Polygon

    Raises
    ------
    GeometryError
        Failed to generate alpha shape; likely due to insufficient data or alpha being too large.
    """
    xy = np.array([[i[0], i[1]] for i in zip(x_values, y_values)])
    try:
        poly = alphashape.alphashape(points=xy, alpha=alpha)
        assert isinstance(poly, Polygon)
        return poly
    except AssertionError:
        raise GeometryError(
            "Failed to generate alpha shape. Check for insufficient data or whether alpha is too large"
        )


def ellipse_to_polygon(
    centroid: (float, float),
    width: float,
    height: float,
    angle: float,
    ellipse: Ellipse or None = None,
) -> Polygon:
    """
    Convert an ellipse to a shapely Polygon object.

    Parameters
    ----------
    centroid: (float, float)
    width: float
    height: float
    angle: float
    ellipse: Ellipse (optional)

    Returns
    -------
    shapely.geometry.Polygon
    """
    ellipse = ellipse or Ellipse(centroid, width, height, angle)
    vertices = ellipse.get_verts()
    return Polygon(vertices)
