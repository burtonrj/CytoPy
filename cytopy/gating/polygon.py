import logging
from functools import partial
from multiprocessing import cpu_count
from string import ascii_uppercase
from typing import Dict
from typing import Iterable
from typing import List
from typing import Union

import mongoengine
import numpy as np
import pandas as pd
from hdbscan import HDBSCAN
from joblib import delayed
from joblib import Parallel
from scipy import stats
from shapely.geometry import Polygon as ShapelyPolygon
from sklearn.cluster import *
from sklearn.linear_model import HuberRegressor
from sklearn.mixture import *
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


def _generic_polygon_model(
    model,
    data: np.ndarray,
):
    return model.fit_predict(data)


def _smm(model: SMM, data: np.ndarray):
    model.fit(data)
    return model.predict(data)


def _huber_regressor_gate(params: Dict, x: np.ndarray, y: np.ndarray, default_conf: float):
    conf = params.pop("conf", default_conf)
    model = HuberRegressor()
    model.set_params(**params)
    model.fit(x.reshape(-1, 1), y)
    conf = stats.norm.ppf(1 - conf)
    x = np.array([np.min(x), np.max(x)])
    y = np.array([np.min(y), np.max(y)])
    y_pred = model.predict(x.reshape(-1, 1))
    stdev = np.sqrt(sum((y_pred - y) ** 2) / len(y) - 1)
    y_lower = y_pred - conf * stdev
    y_upper = y_pred + conf * stdev
    return y_lower, y_upper


def _covariance(model, data: np.ndarray):
    model.fit(data)
    if isinstance(model, SMM):
        return model.covars_, model.means_
    return model.covariances_, model.means_


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

    def _generate_polygons(self, labels: np.ndarray, data: pd.DataFrame):
        with Parallel(cpu_count()) as parallel:
            polygons = parallel(
                delayed(create_envelope)(
                    xy,
                    alpha=self.envelope_alpha,
                )
                for xy in [data.iloc[np.where(labels == i)][[self.x, self.y]].values for i in np.unique(labels)]
            )
        if len(polygons) == 0:
            raise GeometryError("Failed to generate Polygon geometries")
        return polygons

    def _fit(self, data: pd.DataFrame, **overwrite_kwargs):
        overwrite_kwargs = overwrite_kwargs or {}
        params = self.method_kwargs or {}
        params.update(overwrite_kwargs)
        model = globals()[self.method](**params)
        if self.method == "SMM":
            labels = _smm(model=model, data=data[[self.x, self.y]].values)
        else:
            labels = _generic_polygon_model(model=model, data=data[[self.x, self.y]].values)
        return self._generate_polygons(labels=labels, data=data)

    def _fit_hyperparameter_search(self, data: pd.DataFrame, parameter_grid: Dict):
        grid = self._hyperparameter_grid(parameter_grid=parameter_grid)
        _worker = partial(_smm) if self.method == "SMM" else partial(_generic_polygon_model)
        models = [globals()[self.method](**params) for params in grid]
        with Parallel(n_jobs=cpu_count()) as parallel:
            labels = parallel(delayed(_worker)(m, data[[self.x, self.y]].values) for m in models)
        return [self._generate_polygons(labels=i, data=data) for i in labels]

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
        if self.reference_alignment:
            data = self._align_to_reference(data=data)
        df = data if self.downsample_method is None else self._downsample(data=data)
        polygons = self._fit(data=df, **overwrite_kwargs)
        return self._match_to_children(self._generate_populations(data=data, polygons=polygons))

    def predict_with_hyperparameter_search(self, data: pd.DataFrame, parameter_grid: Dict, transform: bool = True):
        assert len(self.children) > 0, "Call 'train' before predict."
        data = self.preprocess(data=data, transform=transform)
        if self.reference_alignment:
            data = self._align_to_reference(data=data)
        df = data if self.downsample_method is None else self._downsample(data=data)
        polygons = self._fit_hyperparameter_search(data=df, parameter_grid=parameter_grid)
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

    def _polygon_from_covar(self, covariance_matrix: np.ndarray, means: np.ndarray):
        ellipses = [probabilistic_ellipse(c, conf=self.conf) for c in covariance_matrix]
        polygons = [ellipse_to_polygon(centroid, *ellipse) for centroid, ellipse in zip(means, ellipses)]
        return polygons

    def _fit(self, data: pd.DataFrame, **overwrite_kwargs):
        if not self.prob_ellipse:
            return super()._fit(data=data)
        overwrite_kwargs = overwrite_kwargs or {}
        params = self.method_kwargs or {}
        params.update(overwrite_kwargs)
        model = globals()[self.method](**params)
        return self._polygon_from_covar(*_covariance(model=model, data=data[[self.x, self.y]].values))

    def _fit_hyperparameter_search(self, data: pd.DataFrame, parameter_grid: Dict):
        if not self.prob_ellipse:
            return super()._fit(data=data)
        grid = self._hyperparameter_grid(parameter_grid=parameter_grid)
        models = [globals()[self.method](**params) for params in grid]
        with Parallel(n_jobs=cpu_count()) as parallel:
            covar_means = parallel(delayed(_covariance)(m, data[[self.x, self.y]].values) for m in models)
        return [self._polygon_from_covar(cm[0], cm[1]) for cm in covar_means]


class HuberGate(PolygonGate):
    conf = mongoengine.FloatField(default=0.95)

    def __init__(self, *args, **values):
        values["method"] = "HuberRegressor"
        super().__init__(*args, **values)

    def _fit(self, data: pd.DataFrame, **overwrite_kwargs) -> List[ShapelyPolygon]:
        assert data is not None, "No data provided"
        overwrite_kwargs = overwrite_kwargs or {}
        kwargs = self.method_kwargs or {}
        kwargs.update(overwrite_kwargs)
        y_lower, y_upper = _huber_regressor_gate(
            params=self.method_kwargs or {}, x=data[self.x].values, y=data[self.y].values, default_conf=self.conf
        )
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

    def _fit_hyperparameter_search(self, data: pd.DataFrame, parameter_grid: Dict):
        grid = self._hyperparameter_grid(parameter_grid=parameter_grid)
        _worker = partial(_huber_regressor_gate, x=data[self.x].values, y=data[self.y].values, default_conf=self.conf)
        with Parallel(n_jobs=cpu_count()) as parallel:
            ybounds = parallel(delayed(_worker)(params) for params in grid)
        x = [
            data[self.x].min(),
            data[self.x].max(),
            data[self.x].max(),
            data[self.x].min(),
            data[self.x].min(),
        ]
        return [[create_polygon(x, [y[0][0], y[0][1], y[1][1], y[1][0], y[0][0]])] for y in ybounds]


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
