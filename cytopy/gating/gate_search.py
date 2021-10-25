#!/usr/bin.env/python
# -*- coding: utf-8 -*-
"""
This module provides functionality for hyperparameter search for autonomous gates


Copyright 2020 Ross Burton

Permission is hereby granted, free of charge, to any person
obtaining a copy of this software and associated documentation
files (the "Software"), to deal in the Software without restriction,
including without limitation the rights to use, copy, modify,
merge, publish, distribute, sublicense, and/or sell copies of the
Software, and to permit persons to whom the Software is furnished
to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included
in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""
from functools import partial
from multiprocessing import cpu_count
from multiprocessing import Pool
from typing import Dict
from typing import List
from typing import Union

import numpy as np
import pandas as pd
from numba import jit
from scipy.spatial.distance import cosine
from sklearn.model_selection import ParameterGrid

from cytopy.data import Population
from cytopy.feedback import progress_bar
from cytopy.gating.gate import EllipseGate
from cytopy.gating.gate import PolygonGate
from cytopy.gating.gate import ThresholdGate


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


def fit_gate(
    updated_params: dict,
    gate: PolygonGate or ThresholdGate or EllipseGate,
    data: pd.DataFrame,
    cached_data: bool,
    norm: bool,
) -> list:
    """
    Update the Gate method parameters and fit to the given data, predicting matching
    Populations that are returned as a list

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
    return gate.fit_predict(data=data, cached_data=cached_data, norm=norm)


def calculate_signatures(parent: pd.DataFrame, features: List[str], populations_idx: List[List[int]]) -> np.ndarray:
    sig = []
    features = [x for x in features if "time" not in x.lower()]
    for idx in populations_idx:
        sig.append(parent.loc[idx][features].mean(axis=0).values)
    return np.array(sig)


@jit(nopython=True)
def closest_signature(child_signature: np.ndarray, population_signatures: np.array) -> (int, float):
    return int(np.argmin([cosine(child_signature, signature) for signature in population_signatures]))


def threshold_gate_hyperparam_search(gate: ThresholdGate, parent: pd.DataFrame, populations: List[List[Population]]):
    populations = [p for sl in populations for p in sl]
    features = list(gate.children[0].geom.signature.keys())
    child_populations = {
        child.name: {
            "populations": [p.population_name == child.name for p in populations],
            "signature": np.array([child.signature[x] for x in features]),
        }
        for child in gate.children
    }
    optimal_populations = list()
    for child_name, data in child_populations.items():
        pops, child_signature = data["populations"], data["signature"]
        signatures = calculate_signatures(parent=parent, features=features, populations_idx=[p.index for p in pops])
        i = closest_signature(child_signature=child_signature, population_signatures=signatures)
        optimal_populations.append(pops[i])
    return optimal_populations


def hausdorff_distance(child, populations):
    search_space = [p for p in populations if p.population_name == child.name]
    idx = np.argmin([child.geom.shape.hausdorff_distance(p.geom.shape) for p in search_space])
    return search_space[int(idx)]


def polygon_gate_hyperparam_search(gate: Union[PolygonGate, EllipseGate], populations: List[List[Population]]):
    populations = remove_null_populations(population_grid=populations)
    populations = [p for sl in populations for p in sl]
    optimal_populations = []
    for child in gate.children:
        search_space = [p for p in populations if p.population_name == child.name]
        idx = np.argmin([child.geom.shape.hausdorff_distance(p.geom.shape) for p in search_space])
        optimal_populations.append(search_space[int(idx)])
    return optimal_populations


def hyperparameter_gate(
    gate: Union[ThresholdGate, PolygonGate, EllipseGate],
    grid: Dict,
    parent: pd.DataFrame,
    verbose: bool = True,
    norm: bool = False,
) -> list:
    original_kwargs = gate.method_kwargs.copy()
    grid = grid.copy()
    grid = ParameterGrid(grid)
    grid = list(grid.__iter__())
    first_populations, parent = fit_gate(updated_params=grid[0], data=parent, cached_data=False, gate=gate, norm=norm)
    populations = [first_populations]
    fitter = partial(fit_gate, gate=gate, data=parent, norm=norm, cached_data=True)
    for params in progress_bar(grid[1:], verbose=verbose, total=len(grid[1:])):
        populations.append(fitter(params)[0])
    gate.method_kwargs = original_kwargs

    if isinstance(gate, ThresholdGate):
        return threshold_gate_hyperparam_search(
            gate=gate,
            parent=parent,
            populations=populations,
        )
    return polygon_gate_hyperparam_search(gate=gate, populations=populations)
