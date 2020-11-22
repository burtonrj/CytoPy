from ..data.gate import PolygonGate, ThresholdGate, EllipseGate, ChildPolygon, ChildThreshold
from ..feedback import vprint, progress_bar
from sklearn.model_selection import ParameterGrid
from scipy.spatial.distance import euclidean, cityblock
from functools import partial
from warnings import warn
import pandas as pd
import numpy as np


def signature_to_vector(signature: dict):
    """
    Convert a signature (dictionary of average parameters) to Numpy array

    Parameters
    ----------
    signature: dict

    Returns
    -------
    Numpy.Array
    """
    return np.array([v for v in signature.values()])


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


def cost_func(target: ChildPolygon or ChildThreshold,
              populations: list,
              method: str):
    """
    Given some Child of a Gate and the population grid of resulting populations after
    testing across the range of possible hyperparameters, filter the population grid
    for the population with the minimal 'cost' using the defined method.

    Parameters
    ----------
    target: ChildThreshold or ChildPolygon
        Child population definition
    populations: list
        List of nested lists, each containing the populations generated from the fitted gate
        under one set of hyperparameter conditions
    method: str
        Metric to minimise for choice of optimal population match

    Returns
    -------
    Population
    """
    search_space = [[x for x in pops if x.population_name == target.name]
                    for pops in populations]
    search_space = [x for sl in search_space for x in sl]
    if len(search_space) == 0:
        warn(f"No populations generated for target population {target.name}")
        return None
    if method in ["euclidean", "manhattan"]:
        assert hasattr(target, "signature"), "Invalid child populations for manhattan or euclidean dist; " \
                                             "requires 'signature' attribute"
        f = {"euclidean": euclidean, "manhattan": cityblock}.get(method, cityblock)
        idx = np.argmin([f(signature_to_vector(target.signature),
                           signature_to_vector(p.signature)) for p in search_space])
        return search_space[int(idx)]
    if method == "threshold_dist":
        if target.geom.y_threshold:
            idx = np.argmin([abs(target.geom.x_threshold - p.geom.x_threshold) +
                             abs(target.geom.y_threshold - p.geom.y_threshold)
                             for p in search_space])
            return search_space[int(idx)]
        idx = np.argmin([abs(target.geom.x_threshold - p.geom.x_threshold)
                         for p in search_space])
        return search_space[int(idx)]
    if method == "hausdorff":
        idx = np.argmin([target.geom.shape.hausdorff_distance(p.geom.shape)
                         for p in search_space])
        return search_space[int(idx)]
    raise ValueError("Unrecognised cost metrix; should be either euclidean, manhattan, threshold_dict "
                     "or hausdorff")


def fit_gate(updated_params: dict,
             gate: PolygonGate or ThresholdGate or EllipseGate,
             data: pd.DataFrame) -> list:
    """
    Update the Gate method parameters and fit to the given data, predicting matching
    Populations that are returned as a list
children
    Parameters
    ----------
    updated_params: dict
        Updated parameters to fit the gate with
    gate: PolygonGate or ThresholdGate or EllipseGate
        Gate object to fit to the data
    data: Pandas.DataFrame
        Parent data the gate acts upon

    Returns
    -------
    List
        List of new Population objects
    """
    gate.method_kwargs = updated_params
    return gate.fit_predict(data=data)


def optimal_populations(population_grid: list,
                        gate: PolygonGate or ThresholdGate or EllipseGate,
                        cost: str) -> list:
    """
    Given a grid Population results where each row represents the Populations generated
    under one set of hyperparameter conditions, find the optimal Population (i.e. the one
    that most resembles the original gate definition) by minimising the given cost e.g. the
    distance between the original and newly defined gate.

    Parameters
    ----------
    population_grid: list
        List of nested lists, where each row represents the Populations generated
        under one set of hyperparameter conditions
    gate: PolygonGate or ThresholdGate or EllipseGate
        Gate object to use to fit the given data, used here to access Children
    cost: str
        Name of method used to choose optimal population

    Returns
    -------
    List
        List of optimal Populations (in same order as gate Children)
    """
    f = partial(cost_func, populations=population_grid, method=cost)
    return list(map(f, gate.children))


def hyperparameter_gate(gate: ThresholdGate or PolygonGate or EllipseGate,
                        grid: dict,
                        cost: str,
                        parent: pd.DataFrame,
                        verbose: bool = True) -> list:
    """
    Fit a Gate to some parent data whilst searching the hyperparameter space (grid)
    for the optimal 'fit' as defined by minimising some cost (e.g. the distance between the
    originally defined gate and the newly generated gate). Populations from all possible
    combinations of hyperparameters will be generated and then populations matched
    according to the minimal cost.

    Parameters
    ----------
    gate: ThresholdGate or PolygonGate or EllipseGate
        Gate to be fitted
    grid: dict
        Hyperparameter space to search; must be a dictionary whom's values are lists
    cost: str
        Method to use for choosing optimal population matches; dependent on the type
        of Gate being fitted:
        * ThresholdGate:
            - "manhattan" (default): optimal parameters are those that result in the population whom's signature
              is of minimal distance to the original data used to define the gate. The manhattan distance is used
              as the distance metric.
            - "euclidean": optimal parameters are those that result in the population whom's signature
              is of minimal distance to the original data used to define the gate. The euclidean distance is used
              as the distance metric.
            - "threshold_dist": optimal parameters are those that result in the threshold
               whom's distance to the original threshold defined are smallest
        * PolygonGate & EllipseGate:
            - "hausdorff" (optional): parameters chosen that minimise the hausdorff distance
              between the polygon generated from new data and the original polgon gate created
              when the gate was defined
            - "manhattan" (default): optimal parameters are those that result in the population whom's signature
              is of minimal distance to the original data used to define the gate. The manhattan distance is used
              as the distance metric.
            - "euclidean": optimal parameters are those that result in the population whom's signature
              is of minimal distance to the original data used to define the gate. The euclidean distance is used
              as the distance metric.
    parent: Pandas.DataFrame
        Parent data that the gate is 'fitted' too
    verbose: bool (default=True)
        Whether to provide feedback to stdout

    Returns
    -------
    List
        List of optimal Populations
    """
    feedback = vprint(verbose)
    feedback(f"----- Hyperparameter optimisation: {gate.gate_name} -----")
    original_kwargs = gate.method_kwargs.copy()
    grid = grid.copy()

    grid = ParameterGrid(grid)
    feedback(f"Grid space: {len(grid)}")

    fitter = partial(fit_gate, gate=gate, data=parent)
    feedback("Fitting gates across parameter grid...")
    populations = list()
    for params in progress_bar(grid, verbose=verbose, total=len(grid)):
        populations.append(fitter(params))
    feedback("Matching optimal populations...")
    populations = remove_null_populations(population_grid=populations)
    pops = optimal_populations(population_grid=populations,
                               gate=gate,
                               cost=cost)
    gate.method_kwargs = original_kwargs
    feedback(f"------------------ complete ------------------")
    return [p for p in pops if p is not None]
