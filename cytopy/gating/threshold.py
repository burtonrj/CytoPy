import logging
from functools import partial
from multiprocessing import cpu_count
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union

import mongoengine
import numpy as np
import pandas as pd
from detecta import detect_peaks
from dtw import dtw
from joblib import delayed
from joblib import Parallel
from KDEpy import FFTKDE
from scipy.signal import savgol_filter

from cytopy.data.errors import GateError
from cytopy.data.population import Population
from cytopy.data.population import ThresholdGeom
from cytopy.gating.base import ChildThreshold
from cytopy.gating.base import Gate
from cytopy.utils.kde import kde_and_peak_finding

logger = logging.getLogger(__name__)


def find_inflection_point(
    x: np.array,
    p: np.array,
    peak_idx: int,
    incline: bool = False,
    window_size: Optional[int] = None,
    polyorder: int = 3,
    **kwargs,
) -> float:
    """
    Given some probability vector and grid space that represents a PDF as calculated by KDE,
    and assuming this vector has a single peak of highest density, calculate the inflection point
    at which the peak flattens. Probability vector is first smoothed using Savitzky-Golay filter.

    Parameters
    ----------
    x: np.array
        Grid space for the probability vector
    p: np.array
        Probability vector as calculated by KDE
    peak_idx: int
        Index of the peak
    incline: bool (default=False)
        If true, calculates the inflection point of the incline towards the peak
        as opposed to the decline away from the peak
    window_size: int, optional
        Window length of filter (must be an odd number). If not given then it is calculated as an
        odd integer nearest to a 10th of the grid length
    polyorder: int (default=3)
        Polynomial order for Savitzky-Golay filter
    kwargs: dict
        Additional keyword argument to pass to scipy.signal.savgol_filter

    Returns
    -------
    float
        Value of x at which the inflection point occurs
    """
    window_size = window_size or int(len(x) * 0.25)
    if window_size % 2 == 0:
        window_size += 1
    smooth = savgol_filter(p, window_size, polyorder, **kwargs)
    if incline:
        ddy = np.diff(np.diff(smooth[:peak_idx]))
    else:
        ddy = np.diff(np.diff(smooth[peak_idx:]))
    if incline:
        return float(x[np.argmax(ddy)])
    return float(x[peak_idx + np.argmax(ddy)])


def smoothed_peak_finding(
    p: np.array,
    starting_window_length: int = 11,
    polyorder: int = 3,
    min_peak_threshold: float = 0.05,
    peak_boundary: float = 0.1,
    **kwargs,
) -> (np.ndarray, np.ndarray):
    """
    Given the grid space and probability vector of some PDF calculated using KDE,
    first attempt to smooth the probability vector using a Savitzky-Golay filter
    (see scipy.signal.savgol_filter) and then perform peak finding until the
    number of peaks is less than 3. Window size will be incremented until the
    number of peaks is reduced. If window size exceeds half the length of the
    probability vector, will raise an AssertionError to avoid misrepresentation of
    the data.

    Parameters
    ----------
    p: np.array
        Probability vector resulting from KDE calculation
    starting_window_length: int (default=11)
        Window length of filter (must be > length of p, < length of p * 0.5, and an odd number)
    polyorder: int (default=3)
        Order of polynomial for filter
    min_peak_threshold: float (default=0.05)
        See cytopy.data.gate.find_peaks
    peak_boundary: float (default=0.1)
        See cytopy.data.gate.find_peaks
    kwargs: dict
        Additional keyword arguments to pass to scipy.signal.savgol_filter

    Returns
    -------
    np.array, np.array
        Smooth probability vector and index of peaks

    Raises
    ------
    ValueError
        Exceeded a safe number of iterations when expanding window of savgol filter. Likely
        means that there is a lack of data for correct estimation of peaks.
    """
    p = p.copy()
    window = starting_window_length

    while len(detect_peaks(p, mph=p[np.argmax(p)] * min_peak_threshold, mpd=len(p) * peak_boundary)) >= 3:
        if window >= len(p) * 0.5:
            raise ValueError("Stable window size exceeded")
        p = savgol_filter(p, window, polyorder, **kwargs)
        window += 10
    return p, detect_peaks(p, mph=p[np.argmax(p)] * min_peak_threshold, mpd=len(p) * peak_boundary)


def find_local_minima(p: np.array, x: np.ndarray, peaks: np.ndarray) -> float:
    """
    Find local minima between the two highest peaks in the density distribution provided

    Parameters
    -----------
    p: numpy array
        probability vector as generated from KDE
    x: numpy array
        Grid space for probability vector
    peaks: numpy array
        array of indices for identified peaks

    Returns
    --------
    float
        local minima between highest peaks
    """
    sorted_peaks = np.sort(p[peaks])[::-1]
    if sorted_peaks[0] == sorted_peaks[1]:
        p1_idx, p2_idx = np.where(p == sorted_peaks[0])[0]
    else:
        p1_idx = np.where(p == sorted_peaks[0])[0][0]
        p2_idx = np.where(p == sorted_peaks[1])[0][0]
    if p1_idx < p2_idx:
        between_peaks = p[p1_idx:p2_idx]
    else:
        between_peaks = p[p2_idx:p1_idx]
    local_min = min(between_peaks)
    return float(x[np.where(p == local_min)[0][0]])


def process_peaks(
    x: np.ndarray, p: np.ndarray, peaks: np.ndarray, x_grid: np.ndarray, incline: bool, q: Optional[float] = None
):
    if len(peaks) == 0:
        raise GateError("No peaks detected")
    if len(peaks) == 1:
        if q:
            return np.quantile(x, q=q)
        return find_inflection_point(x=x_grid, p=p, peak_idx=peaks[0], incline=incline)
    if len(peaks) == 2:
        return find_local_minima(p=p, x=x_grid, peaks=peaks)


def find_threshold(
    x: np.ndarray,
    kernel: str,
    bw: Union[str, float],
    min_peak_threshold: float,
    peak_boundary: float,
    incline: bool,
    q: Optional[float],
) -> float:
    peaks, x_grid, p = kde_and_peak_finding(
        x=x, kernel=kernel, bw=bw, min_peak_threshold=min_peak_threshold, peak_boundary=peak_boundary
    )
    if len(peaks) <= 2:
        return process_peaks(x=x, p=p, peaks=peaks, x_grid=x_grid, incline=incline, q=q)
    else:
        p, peaks = smoothed_peak_finding(p=p, min_peak_threshold=min_peak_threshold, peak_boundary=peak_boundary)
        return process_peaks(x=x, p=p, peaks=peaks, x_grid=x_grid, incline=incline, q=q)


def apply_threshold(
    data: pd.DataFrame,
    x: str,
    x_threshold: float,
    y: Optional[str] = None,
    y_threshold: Optional[float] = None,
) -> Dict[str, pd.DataFrame]:
    """
    Simpde wrapper for threshold_1d and threhsold_2d

    Parameters
    ----------
    data: Pandas.DataFrame
    x: str
    x_threshold: float
    y: str, optional
    y_threshold: float, optional

    Returns
    -------
    dict
    """
    if y is not None:
        return threshold_2d(data=data, x=x, y=y, x_threshold=x_threshold, y_threshold=y_threshold)
    else:
        return threshold_1d(data=data, x=x, x_threshold=x_threshold)


def threshold_1d(data: pd.DataFrame, x: str, x_threshold: float) -> Dict[str, pd.DataFrame]:
    """
    Appdy the given threshold (x_threshold) to the x-axis variable (x) and return the
    resulting dataframes corresponding to the positive and negative populations.
    Returns a dictionary of dataframes: {'-': Pandas.DataFrame, '+': Pandas.DataFrame}

    Parameters
    ----------
    data: Pandas.DataFrame
    x: str
    x_threshold: float

    Returns
    -------
    dict
        Negative population (less than threshold) and positive population (greater than or equal to threshold)
        in a dictionary as so: {'-': Pandas.DataFrame, '+': Pandas.DataFrame}
    """
    data = data.copy()
    return {"+": data[data[x] >= x_threshold], "-": data[data[x] < x_threshold]}


def threshold_2d(
    data: pd.DataFrame, x: str, y: str, x_threshold: float, y_threshold: float
) -> Dict[str, pd.DataFrame]:
    """
    Appdy the given threshold (x_threshold) to the x-axis variable (x) and the given threshold (y_threshold)
    to the y-axis variable (y), and return the  resulting dataframes as a dictionary:
        '++': Greater than or equal to threshold for both x and y
        '+-': Greater than or equal to threshold for x but less than threshold for y
        '-+': Greater than or equal to threshold for y but less than threshold for x
        '--': Less than threshold for both x and y

    Parameters
    ----------
    data: Pandas.DataFrame
    x: str
    y: str
    x_threshold: float
    y_threshold: float

    Returns
    -------
    dict
    """
    data = data.copy()
    return {
        "++": data[(data[x] >= x_threshold) & (data[y] >= y_threshold)],
        "--": data[(data[x] < x_threshold) & (data[y] < y_threshold)],
        "+-": data[(data[x] >= x_threshold) & (data[y] < y_threshold)],
        "-+": data[(data[x] < x_threshold) & (data[y] >= y_threshold)],
    }


class ThresholdBase(Gate):
    children = mongoengine.EmbeddedDocumentListField(ChildThreshold)

    def label_children(self, labels: Dict[str, str]):
        for c in self.children:
            c.name = labels.get(c.name)
        self._duplicate_children()
        return self

    def _generate_populations(
        self, data: Dict[str, pd.DataFrame], x_threshold: float, y_threshold: Optional[float]
    ) -> List[Population]:
        """
        Generate populations from a standard dictionary of dataframes tha t have had thresholds applied.

        Parameters
        ----------
        data: Pandas.DataFrame
        x_threshold: float
        y_threshold: float (optional)

        Returns
        -------
        List
            List of Population objects
        """
        pops = list()
        for definition, df in data.items():
            pop = Population(
                population_name=definition,
                definition=definition,
                parent=self.parent,
                n=df.shape[0],
                source="gate",
                geom=ThresholdGeom(
                    x=self.x,
                    y=self.y,
                    transform_x=self.transform_x,
                    transform_y=self.transform_y,
                    transform_x_kwargs=self.transform_x_kwargs,
                    transform_y_kwargs=self.transform_y_kwargs,
                    x_threshold=x_threshold,
                    y_threshold=y_threshold,
                ),
            )
            pop.index = df.index.to_list()
            pops.append(pop)
        return pops

    def _match_to_children(self, new_populations: List[Population]) -> List[Population]:
        """
        Given a list of newly create Populations, match the Populations to the gates children and
        return list of Populations with correct population names.

        Parameters
        ----------
        new_populations: list
            List of newly created Population objects

        Returns
        -------
        List
        """
        labeled = list()
        for c in self.children:
            matching_populations = [p for p in new_populations if c.match_definition(p.definition)]
            if len(matching_populations) == 0:
                continue
            elif len(matching_populations) > 1:
                idx = np.unique(np.concatenate([pop.index for pop in matching_populations], axis=0), axis=0)
                geom = matching_populations[0].geom
                pop = Population(
                    population_name=c.name,
                    definition=",".join([pop.definition for pop in matching_populations]),
                    parent=self.parent,
                    n=len(idx),
                    source="gate",
                    geom=geom,
                )
                pop.index = idx
            else:
                pop = matching_populations[0]
                pop.population_name = c.name
            labeled.append(pop)
        return labeled

    def _fit(self, data: pd.DataFrame, **overwrite_kwargs):
        return None, None

    def _fit_hyperparameter_search(self, data: pd.DataFrame, parameter_grid: Dict):
        return []

    def train(self, data: pd.DataFrame, transform: bool = True) -> (Dict[str, pd.DataFrame], List[Union[float, None]]):
        data = self.preprocess(data=data, transform=transform)
        if self.downsample_method:
            data = self._downsample(data=data)
        if len(self.children) > 0:
            self.children.delete()
        thresholds = self._fit(data=data)
        partitioned_data = apply_threshold(
            data=data,
            x=self.x,
            x_threshold=thresholds[0],
            y=self.y,
            y_threshold=thresholds[1],
        )
        for definition, df in partitioned_data.items():
            child = ChildThreshold(
                name=definition,
                definition=definition,
                geom=ThresholdGeom(
                    x=self.x,
                    y=self.y,
                    transform_x=self.transform_x,
                    transform_x_kwargs=self.transform_x_kwargs,
                    transform_y=self.transform_y,
                    transform_y_kwargs=self.transform_y_kwargs,
                    x_threshold=thresholds[0],
                    y_threshold=thresholds[1],
                ),
            )
            child.index = df.index.to_list()
            self.children.append(child)
        self.reference = data
        return self

    def predict(self, data: pd.DataFrame, transform: bool = True, **overwrite_kwargs):
        assert len(self.children) > 0, "Call 'train' before predict."
        data = self.preprocess(data=data, transform=transform)
        if data.shape[0] <= 3:
            raise GateError("Data provided contains 3 or less observations.")
        if self.reference_alignment:
            data = self._align_to_reference(data=data)
        df = data if self.downsample_method is None else self._downsample(data=data)
        thresholds = self._fit(data=df, **overwrite_kwargs)
        partitioned_data = apply_threshold(
            data=data,
            x=self.x,
            x_threshold=thresholds[0],
            y=self.y,
            y_threshold=thresholds[1],
        )
        pops = self._generate_populations(data=partitioned_data, x_threshold=thresholds[0], y_threshold=thresholds[1])
        return self._match_to_children(new_populations=pops)

    def _calc_optimal_populations(self, data: pd.DataFrame, populations: List[List[Population]]) -> List[Population]:
        features = [x for x in data.columns if "time" not in x.lower()]
        ref = self.reference[features]
        optimal_populations = None
        optimal_dist = np.inf
        for pops in populations:
            child_pop_data = [
                (
                    ref.loc[self.children.get(name=p.population_name).index]
                    .sample(n=len(p.index), replace=True)
                    .values,
                    data[features].loc[p.index].values,
                )
                for p in pops
            ]
            distance = np.sum([np.linalg.norm(x[0] - x[1]) for x in child_pop_data])
            if distance < optimal_dist:
                optimal_populations = pops
                optimal_dist = distance
        if optimal_populations is None:
            raise ValueError("Failed to compute minimal distance between reference and new populations.")
        return optimal_populations

    def predict_with_hyperparameter_search(self, data: pd.DataFrame, parameter_grid: Dict, transform: bool = True):
        try:
            assert len(self.children) > 0, "Call 'train' before predict."
            if data.shape[0] <= 3:
                raise GateError("Data provided contains 3 or less observations.")
            data = self.preprocess(data=data, transform=transform)
            if self.reference_alignment:
                data = self._align_to_reference(data=data)
            df = data if self.downsample_method is None else self._downsample(data=data)
            thresholds = self._fit_hyperparameter_search(data=df, parameter_grid=parameter_grid)
            with Parallel(n_jobs=cpu_count()) as parallel:
                partitioned_data = parallel(
                    delayed(apply_threshold)(data=data, x=self.x, x_threshold=t[0], y=self.y, y_threshold=t[1])
                    for t in thresholds
                )
            populations = [
                self._match_to_children(
                    new_populations=self._generate_populations(data=d, x_threshold=t[0], y_threshold=t[1])
                )
                for d, t in zip(partitioned_data, thresholds)
            ]
            return self._calc_optimal_populations(data=data, populations=populations)
        except Exception as e:
            logger.exception(e)
            logger.error(f"Failed to perform hyperparameter search, falling back to 'predict' method.")
            return self.predict(data=data, transform=transform)


class QuantileGate(ThresholdBase):
    children = mongoengine.EmbeddedDocumentListField(ChildThreshold)
    q = mongoengine.FloatField(required=True)

    def __init__(self, *args, **values):
        method = values.pop("method", "quantile")
        super().__init__(*args, **values, method=method)

    def _fit(self, data: pd.DataFrame, **overwrite_kwargs) -> Tuple[float, Union[float, None]]:
        overwrite_kwargs = overwrite_kwargs or {}
        q = overwrite_kwargs.get("q", self.q)
        if data.shape[0] <= 3:
            raise GateError("Data provided contains 3 or less observations.")
        thresholds = []
        for d in [self.x, self.y]:
            if d:
                thresholds.append(np.quantile(data[d].values, q=q))
            else:
                thresholds.append(None)
        return thresholds

    def _fit_hyperparameter_search(self, data: pd.DataFrame, parameter_grid: Dict):
        grid = self._hyperparameter_grid(parameter_grid=parameter_grid)
        q = [x["q"] for x in grid]
        with Parallel(cpu_count()) as parallel:
            x = parallel(delayed(np.quantile)(data[self.x].value, i) for i in q)
            if self.y:
                y = parallel(delayed(np.quantile)(data[self.y].value, i) for i in q)
                return [(i, j) for i, j in zip(x, y)]
            return [(i, None) for i in x]


class ThresholdGate(ThresholdBase):
    children = mongoengine.EmbeddedDocumentListField(ChildThreshold)
    kernel = mongoengine.StringField(required=True, default="gaussian")
    bw_method = mongoengine.StringField(required=True, default="silverman", choices=["ISJ", "silverman"])
    bw_x = mongoengine.FloatField(required=False)
    bw_y = mongoengine.FloatField(required=False)
    min_peak_threshold = mongoengine.FloatField(required=True, default=0.05)
    peak_boundary = mongoengine.FloatField(required=True, default=0.1)
    incline = mongoengine.BooleanField(required=True, default=False)
    single_peak_quantile = mongoengine.FloatField(required=False, default=0.99)
    x_threshold = mongoengine.FloatField(required=False)
    y_threshold = mongoengine.FloatField(required=False)

    def _fit(self, data: pd.DataFrame, **overwrite_kwargs) -> Tuple[float, Union[float, None]]:
        kwargs = overwrite_kwargs or {}
        kwargs["min_peak_threshold"] = kwargs.get("min_peak_threshold", self.min_peak_threshold)
        kwargs["q"] = kwargs.get("single_peak_quantile", self.single_peak_quantile)
        kwargs["peak_boundary"] = kwargs.get("peak_boundary", self.peak_boundary)
        kwargs["incline"] = kwargs.get("incline", self.incline)
        kwargs["kernel"] = kwargs.get("kernel", self.kernel)
        bw_x = kwargs.pop("bw_x", self.bw_method if not self.bw_x else self.bw_x)
        bw_y = kwargs.pop("bw_y", self.bw_method if not self.bw_y else self.bw_y)

        if self.method == "manual":
            if self.x:
                assert self.x_threshold, "For manual gating, provide an X threshold"
            if self.y:
                assert self.y_threshold, "For manual gating, provide an Y threshold"
            return self.x_threshold, self.y_threshold
        if data.shape[0] <= 3:
            raise GateError("Data provided contains 3 or less observations.")
        thresholds = []
        for d, bw in zip([self.x, self.y], [bw_x, bw_y]):
            if d:
                thresholds.append(find_threshold(x=data[d].values, bw=bw, **kwargs))
            else:
                thresholds.append(None)
        return thresholds

    def _fit_hyperparameter_search(self, data: pd.DataFrame, parameter_grid: Dict):
        grid = self._hyperparameter_grid(parameter_grid=parameter_grid)
        grid = [
            [
                (
                    g.get("kernel", self.kernel),
                    g.get(k, self.bw_method if not default else default),
                    g.get("min_peak_threshold", self.min_peak_threshold),
                    g.get("peak_boundary", self.peak_boundary),
                    g.get("incline", self.incline),
                    g.get("q", None),
                )
                for g in grid
            ]
            for k, default in [("bw_x", self.bw_x), ("bw_y", self.bw_y)]
        ]
        with Parallel(n_jobs=cpu_count()) as parallel:
            x_thresholds = parallel(delayed(find_threshold)(data[self.x].values, *args) for args in grid[0])
            if self.y:
                y_thresholds = parallel(delayed(find_threshold)(data[self.x].values, *args) for args in grid[1])
                return [(i, j) for i, j in zip(x_thresholds, y_thresholds)]
            return [(i, None) for i in x_thresholds]


def differential(x):
    idx = np.arange(1, len(x) - 1)
    return (x[idx] - x[idx - 1]) + ((x[idx + 1] - x[idx - 1]) / 2) / 2


class DDTW(ThresholdBase):
    kernel = mongoengine.StringField(required=True, default="gaussian")
    bw_method = mongoengine.StringField(required=True, default="silverman", choices=["ISJ", "silverman"])
    bw_x = mongoengine.FloatField(required=False)
    bw_y = mongoengine.FloatField(required=False)
    grid_n = mongoengine.IntField(required=True, default=1000)
    x_threshold = mongoengine.FloatField(required=True)
    y_threshold = mongoengine.FloatField(required=False)

    def __init__(self, *args, **kwargs):
        kwargs.pop("method", None)
        super().__init__(*args, **kwargs, method="DDTW")
        if self.y:
            if not self.y_threshold:
                raise AttributeError("No value provided for Y-axis threshold")

    def predict_with_hyperparameter_search(self, data: pd.DataFrame, parameter_grid: Dict, transform: bool = True):
        logger.warning(f"DDTW does not support hyperparameter search, defaulting to 'predict'")
        return self.predict(data=data, transform=transform)

    def _fit(self, data: pd.DataFrame, **overwrite_kwargs) -> Tuple[float, Union[float, None]]:
        thresholds = pd.DataFrame(
            {k: [t] for k, t in zip([self.x, self.y], [self.x_threshold, self.y_threshold]) if t is not None}
        )
        thresholds = self.transform(data=thresholds)
        x = thresholds[self.x].values[0]
        y = None
        if self.y:
            y = thresholds[self.y].values[0]
        if self._reference_cache is None:
            return [x, y]

        kwargs = overwrite_kwargs or {}
        kernel = kwargs.get("kernel", self.kernel)
        grid_n = kwargs.get("grid_n", self.grid_n)
        bw_x = kwargs.pop("bw_x", self.bw_method if not self.bw_x else self.bw_x)
        bw_y = kwargs.pop("bw_y", self.bw_method if not self.bw_y else self.bw_y)
        thresholds = []
        for d, t, bw in zip([self.x, self.y], [x, y], [bw_x, bw_y]):
            if not d:
                thresholds.append(None)
            else:
                ref = self.reference[d].values
                target = data[d].values
                xgrid = np.linspace(
                    np.min([np.min(ref), np.min(target)]) - 0.01, np.max([np.max(ref), np.max(target)]) + 0.01, grid_n
                )
                ref = FFTKDE(kernel=kernel, bw=bw).fit(ref).evaluate(xgrid)
                target = FFTKDE(kernel=kernel, bw=bw).fit(target).evaluate(xgrid)
                alignment = dtw(differential(ref), differential(target))
                nearest_threshold_idx = np.where(abs(xgrid - t) == np.min(abs(xgrid - t)))[0][0]
                thresholds.append(np.mean(xgrid[alignment.index2[alignment.index1 == nearest_threshold_idx]]))
        return thresholds


def update_threshold(
    population: Population,
    parent_data: pd.DataFrame,
    x_threshold: float,
    y_threshold: Optional[float] = None,
) -> Population:
    """
    Given an existing population and some new threshold(s) (different to what is already
    associated to the Population), update the Population index and geom accordingly.

    Parameters
    ----------
    population: Population
    parent_data: Pandas.DataFrame
    x_threshold: float
    y_threshold: float, optional
        Required if 2D threshold geometry

    Returns
    -------
    Population

    Raises
    ------
    ValueError
        If y_threshold is missing despite population y_threshold being defined
    """
    if population.geom.y_threshold is None:
        new_data = threshold_1d(data=parent_data, x=population.geom.x, x_threshold=x_threshold).get(
            population.definition
        )
        population.index = new_data.index.values
        population.n = len(population.index)
        population.geom.x_threshold = x_threshold
    else:
        if y_threshold is None:
            raise ValueError("2D threshold requires y_threshold")
        new_data = threshold_2d(
            data=parent_data,
            x=population.geom.x,
            x_threshold=x_threshold,
            y=population.geom.y,
            y_threshold=y_threshold,
        )
        definitions = population.definition.split(",")
        new_data = pd.concat([new_data.get(d) for d in definitions])
        population.index = new_data.index.values
        population.geom.x_threshold = x_threshold
        population.geom.y_threshold = y_threshold
    return population
