from ..data.gate import PolygonGate, ThresholdGate, EllipseGate, ChildPolygon, ChildThreshold
from ..feedback import vprint, progress_bar
from sklearn.model_selection import ParameterGrid
from scipy.spatial.distance import euclidean, cityblock
from multiprocessing import Pool, cpu_count
from functools import partial
import pandas as pd
import numpy as np


def cost_func(target: ChildPolygon,
              populations: list,
              method: str):
    assert all([hasattr(x, "signature") for x in target]), "Invalid child populations for manhattan dist; " \
                                                           "requires 'signature' attribute"
    search_space = np.array([[x for x in pops if x.population_name == target.name]
                            for pops in populations]).flatten()
    if method in ["euclidean", "manhattan"]:
        f = {"euclidean": euclidean, "manhattan": cityblock}.get(method, cityblock)
        idx = np.argmin([f(target.signature, p.signature) for p in search_space])
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
             data: pd.DataFrame):
    gate.method_kwargs = updated_params
    return gate.fit_predict(data=data)


def optimal_populations(population_grid: list,
                        gate: PolygonGate or ThresholdGate or EllipseGate,
                        cost: str):
    with Pool(cpu_count()) as pool:
        f = partial(cost_func, populations=population_grid, method=cost)
        return list(pool.map(f, gate.children))


def hyperparameter_gate(gate: ThresholdGate or PolygonGate or EllipseGate,
                        grid: dict,
                        cost: str,
                        parent: pd.DataFrame,
                        verbose: bool = True,
                        multiprocess: int or None = -1) -> list:
    feedback = vprint(verbose)
    feedback(f"----- Hyperparameter optimisation: {gate.gate_name} -----")
    original_kwargs = gate.method_kwargs.to_python()

    for k, v in original_kwargs.items():
        if k in grid.keys():
            grid[k].append(v)
        else:
            grid[k] = [v]

    grid = ParameterGrid(grid)
    feedback(f"Grid space: {len(grid)}")

    fitter = partial(fit_gate, gate=gate, data=parent)
    feedback("Fitting gates across parameter grid...")
    if multiprocess is not None:
        if multiprocess < 0:
            multiprocess = cpu_count()
        with Pool(multiprocess) as pool:
            populations = progress_bar(pool.imap(fitter, grid),
                                       verbose=verbose,
                                       total=len(grid))
    else:
        populations = list()
        for params in progress_bar(grid, verbose=verbose, total=len(grid)):
            populations.append(fitter(params))
    feedback("Matching optimal populations...")
    pops = optimal_populations(population_grid=populations,
                               gate=gate,
                               cost=cost)
    feedback(f"------------------ complete ------------------")
    return pops
