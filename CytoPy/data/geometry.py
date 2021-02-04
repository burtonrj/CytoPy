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
from ..flow import transform
import numpy as np
import pandas as pd
from multiprocessing import Pool, cpu_count
from warnings import warn
from functools import partial
from matplotlib.patches import Ellipse
from scipy import linalg, stats
from scipy.spatial.qhull import ConvexHull, QhullError
from shapely.geometry import Polygon, Point
import mongoengine

__author__ = "Ross Burton"
__copyright__ = "Copyright 2020, CytoPy"
__credits__ = ["Ross Burton", "Simone Cuff", "Andreas Artemiou", "Matthias Eberl"]
__license__ = "MIT"
__version__ = "1.0.0"
__maintainer__ = "Ross Burton"
__email__ = "burtonrj@cardiff.ac.uk"
__status__ = "Production"


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
        Transformation method applied to the x-axis
    """
    x = mongoengine.StringField()
    y = mongoengine.StringField()
    transform_x = mongoengine.StringField()
    transform_y = mongoengine.StringField()
    transform_x_kwargs = mongoengine.DictField()
    transform_y_kwargs = mongoengine.DictField()
    meta = {'allow_inheritance': True}


class ThresholdGeom(PopulationGeometry):
    """
    Threshold shape. Inherits from PopulationGeometry.

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
        assert self.x_values is not None and self.y_values is not None, \
            "x and y values not defined for this Polygon"
        return create_polygon(self.x_values, self.y_values)

    def transform_to_linear(self):
        x_values, y_values = self.x_values, self.y_values
        if self.transform_x:
            kwargs = self.transform_x_kwargs or {}
            transformer = transform.TRANSFORMERS.get(self.transform_x)(**kwargs)
            x_values = transformer.inverse_scale(pd.DataFrame({"x": [self.x_values]}), features=["x"])["x"].values
        if self.transform_y:
            kwargs = self.transform_y_kwargs or {}
            transformer = transform.TRANSFORMERS.get(self.transform_y)(**kwargs)
            y_values = transformer.inverse_scale(pd.DataFrame({"y": [self.y_values]}), features=["y"])["y"].values
        return x_values, y_values


def point_in_poly(coords: np.array,
                  poly: Polygon):
    point = Point(coords)
    return poly.contains(point)


def inside_polygon(df: pd.DataFrame,
                   x: str,
                   y: str,
                   poly: Polygon,
                   njobs: int = -1):
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
    njobs: int
        Number of jobs to run in parallel, by default uses all available cores

    Returns
    --------
    Pandas.DataFrame
        Masked DataFrame containing only those rows that fall within the Polygon
    """
    if njobs < 0:
        njobs = cpu_count()
    xy = df[[x, y]].values
    f = partial(point_in_poly, poly=poly)
    with Pool(njobs) as pool:
        mask = list(pool.map(f, xy))
    return df.iloc[mask]


def polygon_overlap(poly1: Polygon,
                    poly2: Polygon,
                    threshold: float = 0.):
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
    return 0.


def create_polygon(x: list,
                   y: list):
    """
    Given a list of x coordinated and a list of y coordinates, generate a shapely Polygon

    Parameters
    ----------
    x: list
    y: list

    Returns
    -------
    Polygon
    """
    return Polygon([(x, y) for x, y in zip(x, y)])


def inside_ellipse(data: np.array,
                   center: tuple,
                   width: int or float,
                   height: int or float,
                   angle: int or float) -> object:
    """
    Return mask of two dimensional matrix specifying if a data point (row) falls
    within an ellipse

    Parameters
    -----------
    data: Numpy.array
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
    Numpy.array
        numpy array of indices for values inside specified ellipse
    """
    cos_angle = np.cos(np.radians(180. - angle))
    sin_angle = np.sin(np.radians(180. - angle))

    x = data[:, 0]
    y = data[:, 1]

    xc = x - center[0]
    yc = y - center[1]

    xct = xc * cos_angle - yc * sin_angle
    yct = xc * sin_angle + yc * cos_angle

    rad_cc = (xct ** 2 / (width / 2.) ** 2) + (yct ** 2 / (height / 2.) ** 2)

    in_ellipse = []

    for r in rad_cc:
        if r <= 1.:
            # point in ellipse
            in_ellipse.append(True)
        else:
            # point not in ellipse
            in_ellipse.append(False)
    return in_ellipse


def probablistic_ellipse(covariances: np.array,
                         conf: float):
    """
    Given the covariance matrix of a mixture component, calculate a elliptical shape that
    represents a probabilistic confidence interval.

    Parameters
    ----------
    covariances: np.array
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
    eigen_val = 2. * np.sqrt(eigen_val) * np.sqrt(chi2)
    u = eigen_vec[0] / linalg.norm(eigen_vec[0])
    angle = 180. * np.arctan(u[1] / u[0]) / np.pi
    return eigen_val[0], eigen_val[1], (180. + angle)


def create_convex_hull(x_values: np.array,
                       y_values: np.array):
    """
    Given the x and y coordinates of a cloud of data points, generate a convex hull,
    returning the x and y coordinates of its vertices.

    Parameters
    ----------
    x_values: Numpy.array
    y_values: Numpy.array

    Returns
    -------
    Numpy.array, Numpy.array
    """
    xy = np.array([[i[0], i[1]] for i in zip(x_values, y_values)])
    try:
        hull = ConvexHull(xy)
        x = [float(i) for i in xy[hull.vertices, 0]]
        y = [float(i) for i in xy[hull.vertices, 1]]
    except QhullError:
        warn("ConvexHull generated QhullError; cannot generate geometry")
        x, y = [], []
    return x, y


def ellipse_to_polygon(centroid: (float, float),
                       width: float,
                       height: float,
                       angle: float,
                       ellipse: Ellipse or None = None):
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
    Polygon
    """
    ellipse = ellipse or Ellipse(centroid, width, height, angle)
    vertices = ellipse.get_verts()
    return Polygon(vertices)
