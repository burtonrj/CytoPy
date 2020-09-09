from .density_estimation import silvermans, kde
from ..utilities import inside_polygon
from ..data.populations import PopulationGeometry, Population
from shapely.geometry import Point, Polygon
from shapely.affinity import scale
from scipy.spatial import ConvexHull
from scipy import linalg, stats
from scipy.signal import savgol_filter
from sklearn.cluster import *
from sklearn.mixture import *
from detecta import detect_peaks
from typing import List
from warnings import warn
import pandas as pd
import numpy as np
import string


def create_convex_hull(x_values: np.array,
                       y_values: np.array):
    xy = np.array([[i[0], i[1]] for i in zip(x_values, y_values)])
    hull = ConvexHull(xy)
    return xy[hull.vertices, 0], xy[hull.vertices, 1]


def _probablistic_ellipse(covariances: np.array,
                          conf: float):
    eigen_val, eigen_vec = linalg.eigh(covariances)
    chi2 = stats.chi2.ppf(conf, 2)
    eigen_val = 2. * np.sqrt(eigen_val) * np.sqrt(chi2)
    u = eigen_vec[0] / linalg.norm(eigen_vec[0])
    angle = 180. * np.arctan(u[1] / u[0]) / np.pi
    return eigen_val[0], eigen_val[1], (180. + angle)


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


def _circle_overlap(circles: List[Polygon]):
    overlaps = {i: [] for i, _ in enumerate(circles)}
    for i, c in enumerate(circles):
        for x in circles:
            if c.intersects(x):
                overlaps[i].append(c.intersection(x).area / c.area)
            else:
                overlaps[i].append(0)
    return overlaps


def find_local_minima(probs: np.array,
                      xx: np.array,
                      peaks: np.array) -> float:
    """
    Find local minima between the two highest peaks in the density distribution provided

    Parameters
    -----------
    probs: Numpy.array
        probability for density estimate
    xx: Numpy.array
        x values for corresponding probabilities
    peaks: Numpy.array
        array of indices for identified peaks

    Returns
    --------
    float
        local minima between highest peaks
    """
    sorted_peaks = np.sort(probs[peaks])[::-1]
    if sorted_peaks[0] == sorted_peaks[1]:
        p1_idx, p2_idx = np.where(probs == sorted_peaks[0])[0]
    else:
        p1_idx = np.where(probs == sorted_peaks[0])[0][0]
        p2_idx = np.where(probs == sorted_peaks[1])[0][0]
    if p1_idx < p2_idx:
        between_peaks = probs[p1_idx:p2_idx]
    else:
        between_peaks = probs[p2_idx:p1_idx]
    local_min = min(between_peaks)
    return xx[np.where(probs == local_min)[0][0]]


class Analyst:
    def __init__(self,
                 x: str or None,
                 y: str or None,
                 shape: str,
                 parent: str,
                 binary: bool,
                 model: str or None,
                 conf: float or None = None,
                 **kwargs):
        assert shape in ["threshold", "polygon", "ellipse"], """Invalid shape, must be one of: 
        ["threshold", "polygon", "ellipse"]"""
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
        bottom_left = data[(data[self.x] <= x) & (data[self.y] <= y)].index.values
        top_left = data[(data[self.x] <= x) & (data[self.y] > y)].index.values
        bottom_right = data[(data[self.x] > x) & (data[self.y] <= y)].index.values
        top_right = data[(data[self.x] > x) & (data[self.y] > y)].index.values
        populations = list()
        for definition, idx in zip(["--", "-+", "++", "+-"], [bottom_left, top_left, top_right, bottom_right]):
            geom = PopulationGeometry(x=self.x, y=self.y, x_threshold=x, y_threshold=y)
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
        left = data[data[self.x] < x].index.values
        right = data[data[self.x] >= x].index.values
        populations = list()
        for definition, idx in zip(["-", "+"], [left, right]):
            geom = PopulationGeometry(x=self.x, x_threshold=x)
            populations.append(Population(population_name=definition,
                                          index=idx,
                                          parent=self.parent,
                                          geom=geom,
                                          definition=definition,
                                          n=len(idx)))
        return populations

    def _circle(self,
                data: pd.DataFrame,
                labels: list,
                centers: list):
        populations = list()
        circles = list()
        names = list(string.ascii_uppercase)
        data["label"] = labels
        # For each label
        for i, (label, center) in enumerate(zip(np.unique(labels), centers)):
            # Find the distance to the furthest data point from the center
            df = data[data.label == label]
            furthest = max(df.apply(lambda x: np.linalg.norm(x - center), axis=1))
            # draw a circle of width distance to furthest point
            circle = Point(center).buffer(1)
            circles.append(scale(circle, xfact=furthest, yfact=furthest))
        # measure overlap of circles
        overlaps = _circle_overlap(circles)
        # incrementally reduce circle size until no overlap occurs
        while all([sum(x) != 0 for x in overlaps.values()]):
            for i, overlap_frac in overlaps.items():
                if any([x > 0 for x in overlap_frac]):
                    circles[i] = scale(circles[i],
                                       xfact=-(data.x.max() - data.x.min()) * 0.01,
                                       yfact=-(data.y.max() - data.y.min()) * 0.01)
            overlaps = _circle_overlap(circles)
        # Now create populations
        for i, circle in enumerate(circles):
            box = circle.minimum_rotated_rectangle
            x, y = box.exterior.coords.xy
            width = max(Point(x[0], y[0]).distance(Point(x[1], y[1])))
            geom = PopulationGeometry(x=self.x,
                                      y=self.y,
                                      center=circle.centroid,
                                      width=width,
                                      height=width,
                                      angle=0)
            idx = data[inside_ellipse(data=data.values,
                                      center=circle.centroid,
                                      width=width,
                                      height=width,
                                      angle=0)].index.values
            populations.append(Population(population_name=names[i],
                                          parent=self.parent,
                                          geom=geom,
                                          index=idx,
                                          n=len(idx)))

        return populations

    def _ellipse(self,
                 data: pd.DataFrame,
                 labels: list,
                 centers: list,
                 covar_matrix: np.array or None = None):
        # if covar matrix is none, for each center, expand a circle to the most distant assigned point
        # if circles overlap, reduce circle until silhoutte is 0
        populations = list()
        names = list(string.ascii_uppercase)
        for i, label in enumerate(np.unique(labels)):
            if not self.conf:
                warn("No confidence interval set for mixture model, defaulting to 95%")
                self.conf = 0.95
            width, height, angle = _probablistic_ellipse(covariances=covar_matrix[i],
                                                         conf=self.conf)
            geom = PopulationGeometry(x=self.x,
                                      y=self.y,
                                      center=centers[i],
                                      width=width,
                                      height=height,
                                      angle=angle)
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
        # Return N polygons, where N is the length of set of labels
        data = data.copy()
        data["labels"] = labels
        populations = list()
        names = list(string.ascii_uppercase)
        for i, label in enumerate(np.unique(labels)):
            label_df = data[data.labels == label]
            idx = label_df.index.values
            x_values, y_values = create_convex_hull(label_df[self.x].values, label_df[self.y].values)
            geom = PopulationGeometry(x=self.x,
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
                return self._circle(data=data,
                                    labels=labels,
                                    centers=self.model.means_)
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
            min_x = rect.get("x_range")[0]
            max_x = rect.get("x_range")[1]
            min_y = rect.get("y_range")[0]
            max_y = rect.get("y_range")[1]
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
        geom = PopulationGeometry(x=self.x,
                                  y=self.y,
                                  center=center,
                                  width=width,
                                  height=height,
                                  angle=angle)
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
        geom = PopulationGeometry(x=self.x,
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
        return xx[np.argmax(ddy)]
    return xx[peaks[0] + np.argmax(ddy)]



class DensityGate(Analyst):

    def __init__(self, x: str or None, y: str or None, shape: str, parent: str, binary: bool, **kwargs):
        super().__init__(x=x, y=y, shape="threshold", parent=parent, binary=binary, model=None)
        self.min_peak_threshold = kwargs.get("min_peak_threshold", 0.05)
        self.peak_boundary = kwargs.get("peak_boundary", 0.25)
        self.threshold_method = kwargs.get("threshold_method", "density")
        self.q = kwargs.get("q", 0.95)
        self.low_memory = kwargs.get("low_memory", True)
        self.kde_bw = kwargs.get("kde_bw", None)
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
                probs, xx = kde(data, d, self.kde_bw)
                peaks = self._find_peaks(probs)
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
                    bw = silvermans(data[d].values)
                    df = data.copy()
                    if data.shape[0] > 5000 and self.low_memory:
                        df = df.sample(n=5000)
                    probs, xx = kde(df, d, kde_bw=bw)
                    peaks = self._find_peaks(probs)
                    increment = bw * 0.1
                    while len(peaks) > 2:
                        probs, xx = kde(df, d, kde_bw=bw)
                        peaks = self._find_peaks(probs)
                        bw = bw + increment
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

    def _find_peaks(self,
                    probs: np.array) -> np.array:
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
                             mph=probs[np.argmax(probs)] * self.min_peak_threshold,
                             mpd=len(probs) * self.peak_boundary)
        return peaks
