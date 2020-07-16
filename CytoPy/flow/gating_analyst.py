from ..data.fcs import Population, PopulationGeometry
from sklearn.cluster import SpectralClustering
from .utilities import inside_ellipse, inside_polygon
from scipy import linalg, stats
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
           "Dbscan": "sklearn.cluster.DBSCAN",
           "Hdbscan": "hdbscan.HDBSCAN",
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


def _load_model(model: str,
                **kwargs):
    return importlib.import_module(IMPORTS.get(model))(**kwargs)


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

        # For each label
        # ---- Find the furthest data point from the center
        # ---- draw a circle of width distance to furthest point
        # ---- measure overlap of circles
        #
        pass

    def _ellipse(self,
                 data: pd.DataFrame,
                 labels: list,
                 centers: list,
                 covar_matrix: np.array or None = None):
        # if cover matrix is none, for each center, expand a circle to the most distant assigned point
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
                                          geom=geom))
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
            return self._ellipse(data=data,
                                 center=self.center,
                                 width=self.width,
                                 height=self.height,
                                 angle=self.angle)
        assert all(
            [x is not None for x in [self.x_values, self.y_values]]), "Polygon gate requires x_values and y_values"
        return self._polygon(data=data,
                             x_values=self.x_values,
                             y_values=self.y_values)


class DensityGate(Analyst):
    pass
