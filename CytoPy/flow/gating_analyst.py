from ..utilities import inside_polygon
from ..data.populations import Population
from ..data.geometry import ThresholdGeom, PolygonGeom
from matplotlib.patches import Ellipse
from shapely.geometry import Point, Polygon as SPoly
from shapely.affinity import scale
from scipy.interpolate import splev, splrep
from scipy import linalg, stats
from scipy.signal import savgol_filter
from sklearn.cluster import *
from sklearn.mixture import *
from detecta import detect_peaks
from KDEpy import FFTKDE
from typing import List
from warnings import warn
import pandas as pd
import numpy as np
import string


def _polygon_overlap(polys: List[SPoly]):
    """
    Given a list of Polygon objects, iterate over them and generate a dictionary that
    provides the fraction of area overlap for each polygon compared to all other polygons
    in the list

    Parameters
    ----------
    polys: list

    Returns
    -------
    dict
    """
    overlaps = {i: [] for i, _ in enumerate(polys)}
    for i, c in enumerate(polys):
        for x in polys:
            if c.intersects(x):
                overlaps[i].append(c.intersection(x).area / c.area)
            else:
                overlaps[i].append(0)
    return overlaps


def _draw_circle(data: pd.DataFrame,
                 center: tuple):
    furthest = max(data.apply(lambda x: np.linalg.norm(x.values - center), axis=1))
    return scale(Point(center).buffer(1), xfact=furthest, yfact=furthest)


def _reduce_overlapping_circles(x: str,
                                y: str,
                                data: pd.DataFrame,
                                overlaps: dict,
                                circles: List[SPoly]):
    while all([sum(x) != 0 for x in overlaps.values()]):
        for i, overlap_frac in overlaps.items():
            if any([x > 0 for x in overlap_frac]):
                circles[i] = scale(circles[i],
                                   xfact=-(data[x].max() - data[x].min()) * 0.01,
                                   yfact=-(data[y].max() - data[y].min()) * 0.01)
        overlaps = _polygon_overlap(circles)
    return circles


class Analyst:
    """
    Base class for applying an autonomous gating method; an algorithm that generates a geometric
    shape. Generates a new Population object(s) from the resulting algorithm.

    Parameters
    -----------
    x: str
        Name of the x-axis variable
    y: str
        Name of the y-axis variable
    shape: str
        The type of shape this Analyst is expected to generate. Should be one of: 'threshold', 'polygon', 'ellipse'
    parent: str
        Parent population of the resulting population(s) of gating
    binary: bool
        Whether the gate is a binary gate
    model: str (optional)
        Name of the method used for gating (not required for DensityGate or ManualGate)
    conf: float (optional)
        Confidence interval to generate elliptical gate (only required for MixtureModel methods)
    """

    def __init__(self,
                 x: str or None,
                 y: str or None,
                 shape: str,
                 parent: str,
                 binary: bool,
                 model: str or None,
                 conf: float or None = None,
                 **kwargs):
        assert shape in ["threshold", "polygon", "ellipse"], """Invalid shape, must be one of: ["threshold", "polygon", "ellipse"]"""
        self.x = x
        self.y = y
        self.shape = shape
        self.parent = parent
        self.conf = conf
        self.binary = binary
        self.model = model
        if self.model is not None:
            if self.model == "HDBSCAN":
                from hdbscan import HDBSCAN
                self.model = HDBSCAN(**kwargs)
            else:
                assert model in globals().keys(), f"Module {model} not supported. See docs for supported methods. "
                self.model = globals()[model](**kwargs)

    def _threshold_2d(self,
                      data: pd.DataFrame,
                      x: float,
                      y: float):
        """
        Generate populations resulting for a two-dimensional threshold gate

        Parameters
        ----------
        data: Pandas.DataFrame
        x: str
        y: str

        Returns
        -------
        list
            List of Populations
        """
        bottom_left = data[(data[self.x] <= x) & (data[self.y] <= y)].index.values
        top_left = data[(data[self.x] <= x) & (data[self.y] > y)].index.values
        bottom_right = data[(data[self.x] > x) & (data[self.y] <= y)].index.values
        top_right = data[(data[self.x] > x) & (data[self.y] > y)].index.values
        populations = list()
        for definition, idx in zip(["--", "-+", "++", "+-"], [bottom_left, top_left, top_right, bottom_right]):
            geom = ThresholdGeom(x=self.x, y=self.y, x_threshold=x, y_threshold=y)
            populations.append(Population(population_name=definition,
                                          index=idx,
                                          parent=self.parent,
                                          geom=geom,
                                          definition=definition,
                                          n=len(idx)))
        return populations

    def _threshold_1d(self,
                      data: pd.DataFrame,
                      x: float):
        """
        Generate populations resulting for a one-dimensional threshold gate

        Parameters
        ----------
        data: Pandas.DataFrame
        x: str

        Returns
        -------
        list
            List of Populations
        """
        left = data[data[self.x] < x].index.values
        right = data[data[self.x] >= x].index.values
        populations = list()
        for definition, idx in zip(["-", "+"], [left, right]):
            geom = ThresholdGeom(x=self.x, x_threshold=x)
            populations.append(Population(population_name=definition,
                                          index=idx,
                                          parent=self.parent,
                                          geom=geom,
                                          definition=definition,
                                          n=len(idx)))
        return populations

    def _ellipse(self,
                 data: pd.DataFrame,
                 labels: list,
                 centers: list,
                 covar_matrix: np.array or None = None):
        """
        Generate populations as a result of circular gates (such as those generated from a centroid based clustering
        algorithm such as K-means).

        Parameters
        ----------
        data: Pandas.DataFrame
        labels: list
        centers: list
        covar_matrix: Numpy.array (optional)

        Returns
        -------
        list
            List of Populations
        """
        # if covar matrix is none, for each center, expand a circle to the most distant assigned point
        # if circles overlap, reduce circle until silhoutte is 0
        populations = list()
        names = list(string.ascii_uppercase)
        for i, label in enumerate(np.unique(labels)):
            if not self.conf:
                warn("No confidence interval set for mixture model, defaulting to 95%")
                self.conf = 0.95
            width, height, angle = probablistic_ellipse(covariances=covar_matrix[i],
                                                        conf=self.conf)
            vertices = Ellipse(centers[i], width, height, angle).get_verts()
            geom = PolygonGeom(x=self.x,
                               y=self.y,
                               x_values=vertices[:, 0],
                               y_values=vertices[:, 1])
            idx = data[inside_ellipse(data=data.values,
                                      center=centers[i],
                                      width=width,
                                      height=height,
                                      angle=angle)].index.values
            populations.append(Population(population_name=names[i],
                                          parent=self.parent,
                                          geom=geom,
                                          index=idx,
                                          n=len(idx)))
        return populations

    def _polygon(self,
                 data: pd.DataFrame,
                 labels: list):
        """
        Generate populations as a result of polygon gates

        Parameters
        ----------
        data: Pandas.DataFrame
        labels: list

        Returns
        -------
        list
            List of Populations
        """
        # Return N polygons, where N is the length of set of labels
        data = data.copy()
        data["labels"] = labels
        populations = list()
        names = list(string.ascii_uppercase)
        for i, label in enumerate(np.unique(labels)):
            label_df = data[data.labels == label]
            idx = label_df.index.values
            x_values, y_values = create_convex_hull(label_df[self.x].values, label_df[self.y].values)
            geom = PolygonGeom(x=self.x,
                               y=self.y,
                               x_values=x_values,
                               y_values=y_values)
            populations.append(Population(population_name=names[i],
                                          parent=self.parent,
                                          geom=geom,
                                          index=idx,
                                          n=len(idx)))
        return populations

    def fit_predict(self,
                    data: pd.DataFrame):
        """
        Wrapper for calling fit_predict from the chosen gating method. Inspects the method
        and generates the appropriate geometric shape that will create the resulting 'gate'

        Parameters
        ----------
        data: Pandas.DataFrame

        Returns
        -------
        list
            List of resulting Populations
        """
        labels = self.model.fit_predict(data[[self.x, self.y]])
        if self.shape == "polygon":
            return self._polygon(data=data, labels=labels)
        if self.shape == "ellipse":
            if "covariances_" in dir(self.model) and "means_" in dir(self.model):
                return self._ellipse(data=data,
                                     labels=labels,
                                     centers=self.model.means_,
                                     covar_matrix=self.model.covariances_)
            elif "cluster_centers_" in dir(self.model):
                return self._polygon(data=data,
                                     labels=labels)
            else:
                err = """Model does not contain attributes 'means_', 'covariances_', or 'cluster_centers_', 
                for an elliptical gate, valid automonous methods are: GaussianMixtureModel, BayesianGaussianMixtureModel
                or MiniBatchKMeans. Are you using one of these Scikit-Learn classes?"""
                raise ValueError(err)
        elif self.shape == "polygon":
            return self._polygon(data=data, labels=labels)
        else:
            raise ValueError("For threshold gates, please specify method as ManualGate or DensityGate")


class ManualGate(Analyst):
    """
    Class for manually generated (static) gates. Inherist from Analyst.
    """

    def __init__(self,
                 x: str or None,
                 y: str or None,
                 shape: str,
                 parent: str,
                 x_threshold: float or None = None,
                 y_threshold: float or None = None,
                 center: list or None = None,
                 width: float or None = None,
                 height: float or None = None,
                 angle: float or None = None,
                 x_values: list or None = None,
                 y_values: list or None = None,
                 rect: dict or None = None):
        super().__init__(x=x, y=y, shape=shape, binary=True, parent=parent, model=None)
        self.x_threshold = x_threshold
        self.y_threshold = y_threshold
        self.center = center
        self.width = width
        self.height = height
        self.angle = angle
        self.x_values = x_values
        self.y_values = y_values
        if rect is not None:
            assert all(i in rect.keys() for i in ["x_range", "y_range"]), "If specifying a manual rectangular gate, " \
                                                                          "then must provide x_range and y_range"
            min_x = int(rect.get("x_range")[0])
            max_x = int(rect.get("x_range")[1])
            min_y = int(rect.get("y_range")[0])
            max_y = int(rect.get("y_range")[1])
            self.x_values = [min_x, max_x, max_x, min_x, min_x]
            self.y_values = [min_y, min_y, max_y, max_y, min_y]

    def fit_predict(self,
                    data: pd.DataFrame):
        if self.shape == "threshold":
            if self.x_threshold is not None and self.y_threshold is not None:
                return self._threshold_2d(data=data,
                                          x=self.x_threshold,
                                          y=self.y_threshold)
            if self.x_threshold is not None and self.y_threshold is None:
                return self._threshold_1d(data=data, x=self.x_threshold)
            raise ValueError("For a threshold gate you must provide either x_threshold or both x_threshold "
                             "and y_threshold")
        if self.shape == "ellipse":
            err = "For an ellipse gate you must provide center, width, height, and angle"
            assert all([x is not None for x in [self.center,
                                                self.width,
                                                self.height,
                                                self.angle]]), err
            return self._manual_ellipse(data=data,
                                        center=self.center,
                                        width=self.width,
                                        height=self.height,
                                        angle=self.angle)
        assert all(
            [x is not None for x in [self.x_values, self.y_values]]), "Polygon gate requires x_values and y_values"
        return self._manual_polygon(data=data,
                                    x_values=self.x_values,
                                    y_values=self.y_values)

    def _manual_ellipse(self,
                        data: pd.DataFrame,
                        center: tuple,
                        width: float,
                        height: float,
                        angle: float):
        populations = list()
        vertices = Ellipse(center, width, height, angle).get_verts()
        geom = PolygonGeom(x=self.x,
                           y=self.y,
                           x_values=vertices[:, 0],
                           y_values=vertices[:, 1])
        idx = data[inside_ellipse(data=data.values,
                                  center=center,
                                  width=width,
                                  height=height,
                                  angle=angle)].index.values
        populations.append(Population(population_name="manual_ellipse",
                                      parent=self.parent,
                                      geom=geom,
                                      index=idx,
                                      n=len(idx)))
        return populations

    def _manual_polygon(self,
                        data: pd.DataFrame,
                        x_values: list,
                        y_values: list):
        populations = list()
        x_values, y_values = create_convex_hull(x_values, y_values)
        geom = PolygonGeom(x=self.x,
                           y=self.y,
                           x_values=x_values,
                           y_values=y_values)
        idx = inside_polygon(df=data, x=self.x, y=self.y, poly=geom.shape).index
        populations.append(Population(population_name="manual_polygon",
                                      parent=self.parent,
                                      geom=geom,
                                      index=idx,
                                      n=len(idx)))
        return populations


def _find_inflection_point(xx: np.array,
                           peaks: np.array,
                           probs: np.array,
                           incline: bool = False):
    # Find inflection point
    window_size = int(len(probs) * .1)
    if window_size % 2 == 0:
        # Window size must be odd
        window_size += 1
    # Fit a 3rd polynomial kernel
    smooth = savgol_filter(probs, window_size, 3)
    # Take the second derivative of this slope
    if incline:
        ddy = np.diff(np.diff(smooth[:peaks[0]]))
    else:
        ddy = np.diff(np.diff(smooth[peaks[0]:]))
    # Return the point where the second derivative peaks
    if incline:
        return float(xx[np.argmax(ddy)])
    return float(xx[peaks[0] + np.argmax(ddy)])


def silvermans(data: np.array):
    return float(0.9*min([np.std(data), stats.iqr(data)/1.34])*(len(data)**(-(1/5))))


class DensityGate(Analyst):

    def __init__(self, x: str or None, y: str or None, shape: str, parent: str, binary: bool, **kwargs):
        super().__init__(x=x, y=y, shape="threshold", parent=parent, binary=binary, model=None)
        self.min_peak_threshold = kwargs.get("min_peak_threshold", 0.01)
        self.peak_boundary = kwargs.get("peak_boundary", 0.1)
        self.threshold_method = kwargs.get("threshold_method", "density")
        self.q = kwargs.get("q", 0.95)
        self.low_memory = kwargs.get("low_memory", True)
        self.kde_bw = kwargs.get("kde_bw", "silverman")
        self.kde_kernel = kwargs.get("kde_kernel", "gaussian")
        self.cutoff_point = kwargs.get("cutoff_point", "inflection")
        self.biased_positive = kwargs.get("biased_positive", False)
        assert self.threshold_method in ["density", "quantile"]
        assert self.cutoff_point in ["inflection", "quantile"]

    def ctrl_gate(self,
                  data: pd.DataFrame,
                  ctrl: pd.DataFrame):
        thresholds = self.fit_predict(data=ctrl, return_thresholds=True)
        if len(thresholds) == 1:
            return self._threshold_1d(data=data, x=thresholds[0])
        else:
            return self._threshold_2d(data=data, x=thresholds[0], y=thresholds[1])

    def fit_predict(self,
                    data: pd.DataFrame,
                    return_thresholds: bool = False):
        thresholds = list()
        for d in [self.x, self.y]:
            if d is None:
                continue
            if self.threshold_method == "quantile":
                thresholds.append(data[d].quantile(self.q))
            else:
                xx, probs = FFTKDE(kernel=self.kde_kernel, bw=self.kde_bw).fit(data[d].values).evaluate()
                peaks = find_peaks(probs=probs, min_peak_threshold=self.min_peak_threshold, peak_boundary=self.peak_boundary)
                if len(peaks) == 0 and self.cutoff_point == "quantile":
                    thresholds.append(data[d].quantile(self.q))
                elif len(peaks) == 1:
                    thresholds.append(_find_inflection_point(xx=xx, probs=probs, peaks=peaks,
                                                             incline=self.biased_positive))
                elif len(peaks) == 2:
                    thresholds.append(find_local_minima(probs=probs, xx=xx, peaks=peaks))
                else:
                    # If peaks len > 2: incrementally increase the bw until the density estimate is smooth
                    # and the number of peaks is equal to two (in other words, increase the variance
                    # at expense of bias)
                    probs, peaks = multi_peak(probs, self.min_peak_threshold, self.peak_boundary)
                    # increment = bw * 0.05
                    if len(peaks) == 1:
                        if self.biased_positive:
                            thresholds.append(_find_inflection_point(xx=xx, probs=probs, peaks=peaks,
                                                                     incline=self.biased_positive))
                        elif self.cutoff_point == "quantile":
                            thresholds.append(data[d].quantile(self.q))
                        else:
                            thresholds.append(_find_inflection_point(xx=xx, probs=probs, peaks=peaks))
                    else:
                        thresholds.append(find_local_minima(probs=probs, xx=xx, peaks=peaks))
        if return_thresholds:
            return thresholds
        if len(thresholds) == 1:
            return self._threshold_1d(data=data, x=thresholds[0])
        else:
            return self._threshold_2d(data=data, x=thresholds[0], y=thresholds[1])


def multi_peak(probs: np.array,
               min_peak_threshold: float,
               peak_boundary: float,
               polyorder: int = 3):
    smoothed = probs.copy()
    window = 11
    while len(find_peaks(smoothed, min_peak_threshold, peak_boundary)) >= 3:
        if window >= len(smoothed)*.5:
            raise ValueError("Stable window size exceeded")
        smoothed = savgol_filter(smoothed, window, polyorder)
        window += 10
    return smoothed, find_peaks(smoothed, min_peak_threshold, peak_boundary)


def find_peaks(probs: np.array,
               min_peak_threshold: float,
               peak_boundary: float):
    """
    Internal function. Perform peak finding (see scipy.signal.find_peaks for details)

    Parameters
    -----------
    probs: Numpy.array
        array of probability estimates generated using flow.gating.utilities.kde

    Returns
    --------
    Numpy.array
        array of indices specifying location of peaks in `probs`
    """
    # Find peaks
    peaks = detect_peaks(probs,
                         mph=probs[np.argmax(probs)] * min_peak_threshold,
                         mpd=len(probs) * peak_boundary)
    return peaks


def _smooth_pdf(x: np.array,
                probs: np.array):
    spline = splrep(x, probs)
    x2 = np.linspace(x.min(), x.max(), 1000)
    return splev(x2, spline)