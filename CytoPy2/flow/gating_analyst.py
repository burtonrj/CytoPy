from ..data.fcs import Population, PopulationGeometry
from shapely.geometry import Point, Polygon
from shapely.affinity import scale
from sklearn.neighbors import KernelDensity
from scipy import linalg, stats
from scipy.signal import find_peaks, savgol_filter
from typing import List
from warnings import warn
import pandas as pd
import numpy as np
import importlib
import string

IMPORTS = {"GaussianMixture": "sklearn.mixture.GaussianMixture",
           "BayesianGaussianMixture": "sklearn.mixture.BayesianGaussianMixture",
           "Affinity": "sklearn.cluster.AffinityPropagation",
           "MiniBatchKMeans": "sklearn.cluster.MiniBatchKMeans",
           "Hierarchical": "sklearn.cluster.AgglomerativeClustering",
           "Birch": "sklearn.cluster.Birch",
           "DBSCAN": "sklearn.cluster.DBSCAN",
           "HDBSCAN": "hdbscan.HDBSCAN",
           "MeanShift": "sklearn.cluster.MeanShift",
           "Spectral": "sklearn.cluster.SpectralClustering"}


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


def _load_model(model: str,
                **kwargs):
    return importlib.import_module(IMPORTS.get(model))(**kwargs)


def _circle_overlap(circles: List[Polygon]):
    overlaps = {i: [] for i, _ in enumerate(circles)}
    for i, c in enumerate(circles):
        for x in circles:
            if c.intersects(x):
                overlaps[i].append(c.intersection(x).area / c.area)
            else:
                overlaps[i].append(0)
    return overlaps


def check_peak(peaks: np.array,
               probs: np.array,
               t: float = 0.05) -> np.array:
    """
    Check peaks against largest peak in list, if peak < t*largest peak, then peak is removed

    Parameters
    -----------
    peaks: Numpy.array
        array of indices for peaks
    probs: Numpy.array
        array of probability values of density estimate
    t: float, (default=0.05)
        height threshold as a percentage of highest peak

    Returns
    --------
    Numpy.array
        Sorted peaks
    """
    assert len(peaks) > 0, '"peak" array is empty'
    if peaks.geom[0] == 1:
        return peaks
    sorted_peaks = np.sort(probs[peaks])[::-1]
    real_peaks = list()
    real_peaks.append(np.where(probs == sorted_peaks[0])[0][0])
    for p in sorted_peaks[1:]:
        if p >= t*sorted_peaks[0]:
            real_peaks.append(np.where(probs == p)[0][0])
    return np.sort(np.array(real_peaks))


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


def silvermans(data: np.array):
    return 0.9*min([np.std(data), stats.iqr(data)/1.34])*(len(data)**(-(1/5)))


def kde(data: pd.DataFrame,
        x: str,
        kde_bw: float or None = None,
        kernel: str = 'gaussian') -> np.array:
    """
    Generate a 1D kernel density estimation using the scikit-learn implementation

    Parameters
    -----------
    data: Pandas.DataFrame
        Data for smoothing
    x: str
        column name for density estimation
    kde_bw: float
        bandwidth
    kernel: str, (default='gaussian')
        kernel to use for estimation (see scikit-learn documentation)

    Returns
    --------
    np.array
        Probability density function for array of 1000 x-axis values between min and max of data
    """
    if kde_bw is None:
        kde_bw = silvermans(data[x].values)
    density = KernelDensity(bandwidth=kde_bw, kernel=kernel)
    d = data[x].values
    density.fit(d[:, None])
    x_d = np.linspace(min(d), max(d), 1000)
    logprob = density.score_samples(x_d[:, None])
    return np.exp(logprob), x_d


class Analyst:
    def __init__(self,
                 x: str or None,
                 y: str or None,
                 shape: str,
                 parent: str,
                 model: str or None = None,
                 **kwargs):
        assert shape in ["threshold", "polygon", "ellipse"], """Invalid shape, must be one of: 
        ["threshold", "polygon", "ellipse"]"""
        self.x = x
        self.y = y
        self.shape = shape
        self.parent = parent
        self.model = None
        self._conf = None
        if 'conf' in kwargs:
            self._conf = kwargs.pop("conf")
        if model:
            self.model = _load_model(model=model, **kwargs)

    def _threshold_2d(self,
                      data: pd.DataFrame,
                      x: float,
                      y: float):
        bottom_left = data[(data[self.x] <= x) & (data[self.x] <= y)].index.values
        top_left = data[(data[self.x] <= x) & (data[self.x] > y)].index.values
        bottom_right = data[(data[self.x] > x) & (data[self.x] <= y)].index.values
        top_right = data[(data[self.x] > x) & (data[self.x] > y)].index.values
        populations = list()
        for definition, idx in zip(["--", "-+", "++", "+-"], [bottom_left, top_left, top_right, bottom_right]):
            geom = PopulationGeometry(x_threshold=x, y_threshold=y)
            populations.append(Population(index=idx,
                                          parent=self.parent,
                                          geom=geom,
                                          definition=definition))
        return populations

    def _threshold_1d(self,
                      data: pd.DataFrame,
                      x: float):
        left = data[data[self.x] < x].index.values
        right = data[data[self.x] >= x].index.values
        populations = list()
        for definition, idx in zip(["-", "+"], [left, right]):
            geom = PopulationGeometry(x_threshold=x)
            populations.append(Population(index=idx,
                                          parent=self.parent,
                                          geom=geom,
                                          definition=definition))
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
            geom = PopulationGeometry(center=circle.centroid,
                                      width=width,
                                      height=width,
                                      angle=0)
            populations.append(Population(population_name=names[i],
                                          parent=self.parent,
                                          geom=geom,
                                          index=data[inside_ellipse(data=data.values,
                                                                    center=circle.centroid,
                                                                    width=width,
                                                                    height=width,
                                                                    angle=0)].index.values))

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
            if not self._conf:
                warn("No confidence interval set for mixture model, defaulting to 95%")
                self._conf = 0.95
            width, height, angle = _probablistic_ellipse(covariances=covar_matrix[i],
                                                         conf=self._conf)
            geom = PopulationGeometry(center=centers[i],
                                      width=width,
                                      height=height,
                                      angle=angle)
            populations.append(Population(population_name=names[i],
                                          parent=self.parent,
                                          geom=geom,
                                          index=data[inside_ellipse(data=data.values,
                                                                    center=centers[i],
                                                                    width=width,
                                                                    height=height,
                                                                    angle=angle)].index.values))
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
            geom = PopulationGeometry(x_values=data[data.labels == label][self.x].values,
                                      y_values=data[data.labels == label][self.y].values)
            populations.append(Population(population_name=names[i],
                                          parent=self.parent,
                                          geom=geom))
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
                ir MiniBatchKMeans. Are you using one of these Scikit-Learn classes?"""
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
                 y_values: list or None = None):
        super().__init__(x=x, y=y, shape=shape, binary=True, parent=parent)
        self.x_threshold = x_threshold
        self.y_threshold = y_threshold
        self.center = center
        self.width = width
        self.height = height
        self.angle = angle
        self.x_values = x_values
        self.y_values = y_values

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
        geom = PopulationGeometry(center=center,
                                  width=width,
                                  height=height,
                                  angle=angle)
        populations.append(Population(population_name="manual_ellipse",
                                      parent=self.parent,
                                      geom=geom,
                                      index=data[inside_ellipse(data=data.values,
                                                                center=center,
                                                                width=width,
                                                                height=height,
                                                                angle=angle)].index.values))
        return populations

    def _manual_polygon(self,
                        data: pd.DataFrame,
                        x_values: list,
                        y_values: list):
        populations = list()
        geom = PopulationGeometry(x_values=x_values,
                                  y_values=y_values)
        populations.append(Population(population_name="manual_polygon",
                                      parent=self.parent,
                                      geom=geom))
        return populations


class DensityGate(Analyst):

    def __init__(self, x: str or None, y: str or None, shape: str, parent: str, **kwargs):
        super().__init__(x, y, shape, parent, **kwargs)
        self.peak_threshold = kwargs.get("peak_threshold", 0.05)
        self.threshold_method = kwargs.get("threshold_method", "density")
        self.q = kwargs.get("q", 0.95)
        self.kde_bw = kwargs.get("kde_bw", None)
        assert self.threshold_method in ["density", "quantile"]

    def fit_predict(self,
                    data: pd.DataFrame):
        if self.y is None:
            if self.threshold_method == "quantile":
                return self._threshold_1d(data=data, x=data[self.x].quantile(self.q))
            probs, xx = kde(data, self.x, self.kde_bw)
            peaks = self._find_peaks(probs)

            # 1D threshold
            pass
        # 2D threshold

    def _evaluate_peaks(self,
                        peaks: np.array,
                        probs: np.array):
        if len(peaks) == 1:
            # Find inflection point
            window_size = int(len(peaks)*.25)
            if window_size % 2 != 0:
                window_size += 1
            smooth = savgol_filter(probs, window_size, 3)
            dy = np.diff(np.diff(smooth))
            zero_crossings = np.where(np.diff(np.sign(dy)))[0]
            inflection_point_i = np.where(zero_crossings > peaks[0])[0]
            inflection_point = probs[inflection_point_i]
            return inflection_point
        # If peaks len == 2: find min between peaks
        # If p≥3⁠, for each peak, it calculates the height of the peak h
        # and its distance from the next adjacent peak d and calculates the score dh⁠.
        # It then finds the peak corresponding to the maximum of all computed scores and
        # chooses this peak and its next adjacent and goes to the case p = 2


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
        peaks = find_peaks(probs)[0]
        if self.peak_threshold:
            peaks = check_peak(peaks, probs, self.peak_threshold)
        return peaks