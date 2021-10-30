import logging
from functools import partial
from multiprocessing import cpu_count
from multiprocessing import Manager
from multiprocessing import Pool
from multiprocessing import Process
from string import ascii_uppercase
from typing import Dict
from typing import Iterable
from typing import List
from typing import Optional
from typing import Union

import mongoengine
import numpy as np
import pandas as pd
from hdbscan import HDBSCAN
from scipy import stats
from shapely.geometry import Polygon as ShapelyPolygon
from sklearn.cluster import *
from sklearn.linear_model import HuberRegressor
from sklearn.mixture import *
from sklearn.model_selection import ParameterGrid
from smm import SMM

from cytopy.data.errors import GateError
from cytopy.data.population import create_polygon
from cytopy.data.population import PolygonGeom
from cytopy.data.population import Population
from cytopy.gating.base import ChildPolygon
from cytopy.gating.base import Gate
from cytopy.utils.geometry import create_envelope
from cytopy.utils.geometry import ellipse_to_polygon
from cytopy.utils.geometry import GeometryError
from cytopy.utils.geometry import inside_polygon
from cytopy.utils.geometry import probabilistic_ellipse

logger = logging.getLogger(__name__)


class PolygonGate(Gate):
    children = mongoengine.EmbeddedDocumentListField(ChildPolygon)
    envelope_alpha = mongoengine.FloatField(default=0.0)
    x_values = mongoengine.ListField(required=False)
    y_values = mongoengine.ListField(required=False)

    def __init__(self, *args, **values):
        super().__init__(*args, **values)
        assert self.y is not None, "Polygon gate expects a y-axis variable"
        if self.method != "manual":
            err = "Invalid method, must be a valid Scikit-Learn class or supported model."
            assert self.method in list(globals().keys()), err
            params = self.method_kwargs or {}
            self.model = globals()[self.method](**params)

    def label_children(self, labels: Dict[str, str], drop: bool = True):
        if len(set(labels.values())) != len(labels.values()):
            raise GateError("Duplicate labels provided. Child merging not available for polygon gates")
        if drop:
            self.children = [c for c in self.children if c.name in labels.keys()]
        for c in self.children:
            c.name = labels.get(c.name)
        return self

    def _generate_populations(self, data: pd.DataFrame, polygons: List[ShapelyPolygon]) -> List[Population]:
        pops = list()
        for name, poly in zip(ascii_uppercase, polygons):
            pop_df = inside_polygon(data=data, x=self.x, y=self.y, poly=poly)
            geom = PolygonGeom(
                x=self.x,
                y=self.y,
                transform_x=self.transform_x,
                transform_y=self.transform_y,
                transform_x_kwargs=self.transform_x_kwargs,
                transform_y_kwargs=self.transform_y_kwargs,
                x_values=poly.exterior.xy[0],
                y_values=poly.exterior.xy[1],
            )
            pop = Population(population_name=name, source="gate", parent=self.parent, n=pop_df.shape[0], geom=geom)
            pop.index = pop_df.index.tolist()
            pops.append(pop)
        return pops

    def _match_to_children(self, new_populations: List[Population]) -> List[Population]:
        matched_populations = list()
        if len(new_populations) == 1 and len(self.children) == 1:
            new_populations[0].population_name = self.children[0].name
            return new_populations
        for child in self.children:
            hausdorff_distances = [child.geom.shape.hausdorff_distance(pop.geom.shape) for pop in new_populations]
            matching_population = new_populations[int(np.argmin(hausdorff_distances))]
            matching_population.population_name = child.name
            matched_populations.append(matching_population)
        return matched_populations

    def _fit(self, data: pd.DataFrame, **overwrite_kwargs):
        if overwrite_kwargs:
            self.model.set_params(**overwrite_kwargs)
        if self.method == "SMM":
            self.model.fit(data[[self.x, self.y]].to_numpy())
            labels = self.model.predict(data[[self.x, self.y]].to_numpy())
        else:
            labels = self.model.fit_predict(data[[self.x, self.y]].to_numpy())
        envelope_func = partial(create_envelope, alpha=self.envelope_alpha)
        xy = [data.iloc[np.where(labels == i)][[self.x, self.y]].values for i in np.unique(labels)]
        with Pool(cpu_count()) as pool:
            polygons = list(pool.map(envelope_func, xy))
        if len(polygons) == 0:
            raise GeometryError("Failed to generate Polygon geometries")
        return polygons

    def train(self, data: pd.DataFrame, transform: bool = True):
        data = self.preprocess(data=data, transform=transform)
        if self.downsample_method:
            data = self._downsample(data=data)
        if len(self.children) != 0:
            self.children.delete()
        polygons = self._fit(data=data)
        for name, poly in zip(ascii_uppercase, polygons):
            child = ChildPolygon(
                name=name,
                geom=PolygonGeom(
                    x=self.x,
                    y=self.y,
                    transform_x=self.transform_x,
                    transform_y=self.transform_y,
                    transform_x_kwargs=self.transform_x_kwargs,
                    transform_y_kwargs=self.transform_y_kwargs,
                    x_values=poly.exterior.xy[0].tolist(),
                    y_values=poly.exterior.xy[1].tolist(),
                ),
            )
            child.index = inside_polygon(data=data, x=self.x, y=self.y, poly=poly).index.tolist()
            self.children.append(child)
        self.reference = data
        return self

    def predict(self, data: pd.DataFrame, transform: bool = True, **overwrite_kwargs):
        assert len(self.children) > 0, "Call 'train' before predict."
        data = self.preprocess(data=data, transform=transform)
        if self.downsample_method:
            polygons = self._fit(data=self._downsample(data=data), **overwrite_kwargs)
        else:
            polygons = self._fit(data=data, **overwrite_kwargs)
        return self._match_to_children(self._generate_populations(data=data, polygons=polygons))

    def predict_with_hyperparameter_search(self, data: pd.DataFrame, parameter_grid: Dict, transform: bool = True):
        assert len(self.children) > 0, "Call 'train' before predict."
        data = self.preprocess(data=data, transform=transform)
        grid = ParameterGrid(parameter_grid)
        df = data if self.downsample_method is None else self._downsample(data=data)
        manager = Manager()
        polygons = manager.list()
        processes = [
            Process(target=self._fit_multiprocess_wrap, args=[df, polygons], kwargs=kwargs) for kwargs in grid
        ]
        for p in processes:
            p.start()
        for p in processes:
            p.join()
        populations = [
            self._match_to_children(self._generate_populations(data=data, polygons=polys)) for polys in polygons
        ]
        return polygon_gate_hyperparam_search(gate=self, populations=populations)


class EllipseGate(PolygonGate):
    conf = mongoengine.FloatField(default=0.95)
    prob_ellipse = mongoengine.BooleanField(default=True)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert (
            self.method_kwargs.get("covariance_type", "full") == "full"
        ), "EllipseGate only supports covariance_type of 'full'"
        valid = ["manual", "GaussianMixture", "BayesianGaussianMixture", "SMM"]
        assert self.method in valid, f"Elliptical gating method should be one of {valid}"

    def _fit(self, data: pd.DataFrame, **overwrite_kwargs):
        if not self.prob_ellipse:
            return super()._fit(data=data)
        if self.downsample_method:
            data = self._downsample(data=data)
        if overwrite_kwargs:
            self.model.set_params(**overwrite_kwargs)
        self.model.fit(data[[self.x, self.y]].to_numpy())
        if isinstance(self.model, SMM):
            ellipses = [probabilistic_ellipse(covar, conf=self.conf) for covar in self.model.covars_]
        else:
            ellipses = [probabilistic_ellipse(covar, conf=self.conf) for covar in self.model.covariances_]
        polygons = [ellipse_to_polygon(centroid, *ellipse) for centroid, ellipse in zip(self.model.means_, ellipses)]
        return polygons


class HuberGate(PolygonGate):
    conf = mongoengine.FloatField(default=0.95)

    def __init__(self, *args, **values):
        values["method"] = "HuberRegressor"
        super().__init__(*args, **values)

    def _predict_interval(self, data: pd.DataFrame, conf: Optional[float] = None):
        conf = conf or self.conf
        conf = stats.norm.ppf(1 - conf)
        x = np.array([data[self.x].min(), data[self.x].max()])
        y = np.array([data[self.y].min(), data[self.y].max()])
        y_pred = self.model.predict(x.reshape(-1, 1))
        stdev = np.sqrt(sum((y_pred - y) ** 2) / len(y) - 1)
        y_lower = y_pred - conf * stdev
        y_upper = y_pred + conf * stdev
        return y_lower, y_upper

    def _fit_model(self, data: pd.DataFrame):
        x = data[self.x].to_numpy().reshape(-1, 1)
        y = data[self.y].to_numpy()
        self.model.fit(x, y)

    def _fit(self, data: pd.DataFrame, **overwrite_kwargs) -> List[ShapelyPolygon]:
        overwrite_kwargs = overwrite_kwargs or {}
        conf = overwrite_kwargs.pop("conf", self.conf)
        if self.downsample_method:
            data = self._downsample(data=data)
        if overwrite_kwargs:
            self.model.set_params(**overwrite_kwargs)
        self._fit_model(data=data)
        y_lower, y_upper = self._predict_interval(data=data, conf=conf)
        return [
            create_polygon(
                [
                    data[self.x].min(),
                    data[self.x].max(),
                    data[self.x].max(),
                    data[self.x].min(),
                    data[self.x].min(),
                ],
                [y_lower[0], y_lower[1], y_upper[1], y_upper[0], y_lower[0]],
            )
        ]


def remove_null_populations(population_grid):
    """
    Remove populations with less than 4 events

    Parameters
    ----------
    population_grid: list

    Returns
    -------
    List
    """
    updated_grid = [[p for p in pops if p.n >= 3] for pops in population_grid]
    updated_grid = [x for x in updated_grid if len(x) > 0]
    return updated_grid


def polygon_gate_hyperparam_search(
    gate: Union[PolygonGate, EllipseGate, HuberGate], populations: List[List[Population]]
):
    populations = remove_null_populations(population_grid=populations)
    populations = [p for sl in populations for p in sl]
    optimal_populations = []
    for child in gate.children:
        search_space = [p for p in populations if p.population_name == child.name]
        idx = np.argmin([child.geom.shape.hausdorff_distance(p.geom.shape) for p in search_space])
        optimal_populations.append(search_space[int(idx)])
    return optimal_populations


def update_polygon(
    population: Population,
    parent_data: pd.DataFrame,
    x_values: Iterable[float],
    y_values: Iterable[float],
) -> Population:
    """
    Given an existing population and some new definition for it's polygon gate
    (different to what is already associated to the Population), update the Population
    index and geom accordingly. Any controls will have to be estimated again.

    Parameters
    ----------
    population: Population
    parent_data: Pandas.DataFrame
    x_values: list
    y_values: list

    Returns
    -------
    Population
    """
    if isinstance(x_values, np.ndarray):
        x_values = x_values.tolist()
    if isinstance(y_values, np.ndarray):
        y_values = y_values.tolist()
    poly = create_polygon(x=x_values, y=y_values)
    new_data = inside_polygon(data=parent_data, x=population.geom.x, y=population.geom.y, poly=poly)
    population.geom.x_values = x_values
    population.geom.y_values = y_values
    population.index = new_data.index.values
    population.n = len(population.index)
    return population
