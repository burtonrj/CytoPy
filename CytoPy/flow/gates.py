from ..data.fcs import Population, PopulationGeometry
from sklearn.cluster import SpectralClustering
from .utilities import inside_ellipse, inside_polygon
import pandas as pd
import numpy as np
import importlib

IMPORTS = {"GaussianMixture": "sklearn.mixture.GaussianMixture",
           "BayesianGaussianMixture": "sklearn.mixture.BayesianGaussianMixture",
           "Affinity": "sklearn.cluster.AffinityPropagation",
           "KMeans": "sklearn.cluster.KMeans",
           "Hierarchical": "sklearn.cluster.AgglomerativeClustering",
           "Birch": "sklearn.cluster.Birch",
           "Dbscan": "sklearn.cluster.DBSCAN",
           "Hdbscan": "hdbscan.HDBSCAN",
           "MeanShift": "sklearn.cluster.MeanShift",
           "Spectral": "sklearn.cluster.SpectralClustering"}


def _load_model(model: str,
                **kwargs):
    return importlib.import_module(IMPORTS.get(model))(**kwargs)


class Base:
    def __init__(self,
                 x: str or None,
                 y: str or None,
                 shape: str,
                 binary: bool,
                 parent: str,
                 model: str or None = None,
                 **kwargs):
        assert shape in ["threshold", "polygon", "ellipse"], """Invalid shape, must be one of: 
        ["threshold", "polygon", "ellipse"]"""
        self.x = x
        self.y = y
        self.shape = shape
        self.binary = binary
        self.parent = parent
        self.model = None
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
        for name, idx in zip(["--", "-+", "++", "+-"], [bottom_left, top_left, top_right, bottom_right]):
            geom = PopulationGeometry(x_threshold=x, y_threshold=y)
            populations.append(Population(population_name=name,
                                          index=idx,
                                          parent=self.parent,
                                          geom=geom,
                                          definition=name))
        return populations

    def _threshold_1d(self,
                      data: pd.DataFrame,
                      x: float):
        left = data[data[self.x] < x].index.values
        right = data[data[self.x] >= x].index.values
        populations = list()
        for name, idx in zip(["-", "+"], [left, right]):
            geom = PopulationGeometry(x_threshold=x)
            populations.append(Population(population_name=name,
                                          index=idx,
                                          parent=self.parent,
                                          geom=geom,
                                          definition=name))
        return populations

    def _ellipse(self,
                 data: pd.DataFrame,
                 center: list or tuple,
                 width: float,
                 height: float,
                 angle: float,
                 label: str or None = None):
        positive = data[inside_ellipse(data[[self.x, self.y]].values,
                                       center=center,
                                       width=width,
                                       height=height,
                                       angle=angle)].index.values
        negative = data[~data.index.isin(positive)].index.values
        geom = PopulationGeometry(center=center,
                                  width=width,
                                  height=height,
                                  angle=angle)
        if self.binary:
            return [Population(population_name="+",
                               index=positive,
                               parent=self.parent,
                               geom=geom,
                               definition="+"),
                    Population(population_name="-",
                               index=negative,
                               parent=self.parent,
                               geom=geom,
                               definition="-")]
        return Population(population_name=label,
                          index=positive,
                          parent=self.parent,
                          geom=geom,
                          definition="+")

    def _polygon(self,
                 data: pd.DataFrame,
                 x_values: list,
                 y_values: list,
                 label: str or None = None):
        geom = PopulationGeometry(x_values=x_values,
                                  y_values=y_values)
        positive = inside_polygon(df=data, x=self.x, y=self.y, poly=geom.shape).index.values
        negative = data[~data.index.isin(positive)].index.values
        if self.binary:
            return [Population(population_name="+",
                               index=positive,
                               parent=self.parent,
                               geom=geom,
                               definition="+"),
                    Population(population_name="-",
                               index=negative,
                               parent=self.parent,
                               geom=geom,
                               definition="-")]
        return Population(population_name=label,
                          index=positive,
                          parent=self.parent,
                          geom=geom,
                          definition="+")

    def fit_predict(self,
                    data: pd.DataFrame):
        pass

    def predict(self):
        pass


class ManualGate(Base):
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
        self.angle=angle
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
        assert all([x is not None for x in [self.x_values, self.y_values]]), "Polygon gate requires x_values and y_values"
        return self._polygon(data=data,
                             x_values=self.x_values,
                             y_values=self.y_values)


class DensityGate(Base):

