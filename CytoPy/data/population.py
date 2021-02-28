#!/usr/bin.env/python
# -*- coding: utf-8 -*-
"""
When analysing single cell data we are ultimately interested in populations
of cells. This module contains the Population class, which controls the
data attaining to a single cell population. A FileGroup (see CytoPy.data.fcs)
can contain many Populations (which are embedded within the FileGroup).

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

from .geometry import PopulationGeometry, ThresholdGeom, PolygonGeom
from functools import reduce
from shapely.ops import unary_union
from typing import List
import numpy as np
import pandas as pd
import mongoengine

__author__ = "Ross Burton"
__copyright__ = "Copyright 2020, CytoPy"
__credits__ = ["Ross Burton", "Simone Cuff", "Andreas Artemiou", "Matthias Eberl"]
__license__ = "MIT"
__version__ = "2.0.0"
__maintainer__ = "Ross Burton"
__email__ = "burtonrj@cardiff.ac.uk"
__status__ = "Production"


class Population(mongoengine.EmbeddedDocument):
    """
    A population of cells identified by either a gate, clustering or supervised algorithm. Stores the
    index of events corresponding to a single population, where the index relates back
    to the primary data in the FileGroup in which a population is embedded.

    Attributes
    ----------
    population_name: str, required
        name of population
    n: int
        number of events associated to this population
    parent: str, required, (default: "root")
        name of parent population
    prop_of_parent: float, required
        proportion of events as a percentage of parent population
    prop_of_total: float, required
        proportion of events as a percentage of all events
    warnings: list, optional
        list of warnings associated to population
    geom: PopulationGeometry
        PopulationGeometry (see CytoPy.data.geometry) that defines the gate that
        captures this population.
    definition: str
        relevant for populations generated by a ThresholdGate; defines the source of this
        population e.g. "+" for a 1D threshold or "+-" for a 2D threshold
    index: numpy.ndarray
        numpy array storing index of events that belong to population
    signature: dict
        average of a population feature space (median of each channel); used to match
        children to newly identified populations for annotating
    source: str, required
        Source of the population i.e. what method was used to generate it. Valid choices are:
        "gate", "cluster", "root", or "classifier"
    """
    population_name = mongoengine.StringField()
    n = mongoengine.IntField()
    parent = mongoengine.StringField(required=True, default='root')
    prop_of_parent = mongoengine.FloatField()
    prop_of_total = mongoengine.FloatField()
    warnings = mongoengine.ListField()
    geom = mongoengine.EmbeddedDocumentField(PopulationGeometry)
    definition = mongoengine.StringField()
    source = mongoengine.StringField(required=True, choices=["gate", "cluster", "root", "classifier"])
    signature = mongoengine.DictField()

    def __init__(self, *args, **kwargs):
        # If the Population existed previously, fetch the index
        self._index = kwargs.pop("index", None)
        super().__init__(*args, **kwargs)

    @property
    def index(self):
        return self._index

    @index.setter
    def index(self, idx: np.array):
        assert isinstance(idx, np.ndarray), "idx should be type numpy.ndarray"
        self.n = len(idx)
        self._index = np.array(idx)


def _check_overlap(left: Population,
                   right: Population,
                   error: bool = True):
    """
    Given two Population objects assuming that they have Polygon geoms (raises assertion error otherwise),
    checks if the population geometries overlap.
    If error is True, raises assertion error if the geometries do not overlap.

    Parameters
    ----------
    left: Population
    right: Population
    error: bool (default = True)

    Returns
    -------
    bool or None

    Raises
    ------
    AssertionError
        If left or right population do not have a Polygon geometry or are not overlapping
    """
    assert all(
        [isinstance(x.geom, PolygonGeom) for x in [left, right]]), "Only Polygon geometries can be checked for overlap"
    overlap = left.geom.shape.intersects(right.geom.shape)
    if error:
        assert overlap, "Invalid: non-overlapping populations"
    return overlap


def _check_transforms_dimensions(left: Population,
                                 right: Population):
    """
    Given two Populations, checks if transformation methods and axis match. Raises assertion error if not.

    Parameters
    ----------
    left: Population
    right: Population

    Returns
    -------
    None

    Raises
    ------
    AssertionError
        If geometries are incompatible
    """
    assert left.geom.transform_x == right.geom.transform_x, \
        "X dimension transform differs between left and right populations"
    assert left.geom.transform_y == right.geom.transform_y, \
        "Y dimension transform differs between left and right populations"
    assert left.geom.x == right.geom.x, "X dimension differs between left and right populations"
    assert left.geom.y == right.geom.y, "Y dimension differs between left and right populations"


def _merge_index(left: Population,
                 right: Population) -> np.ndarray:
    """
    Merge the index of two populations.

    Parameters
    ----------
    left: Population
    right: Population

    Returns
    -------
    numpy.ndarray
    """
    return np.unique(np.concatenate([left.index, right.index], axis=0), axis=0)


def _merge_signatures(left: Population,
                      right: Population) -> dict:
    """
    Merge the signatures of two populations; taken as the mean of both signatures.

    Parameters
    ----------
    left: Population
    right: Population

    Returns
    -------
    dict
    """
    return pd.DataFrame([left.signature, right.signature]).mean().to_dict()


def _merge_thresholds(left: Population,
                      right: Population,
                      new_population_name: str):
    """
    Merge two Populations with ThresholdGeom geometries.

    Parameters
    ----------
    left: Population
    right: Population
    new_population_name: str

    Returns
    -------
    Population

    Raises
    ------
    AssertionError
        If geometries do not match
    """
    assert left.geom.x_threshold == right.geom.x_threshold, \
        "Threshold merge assumes that the populations are derived " \
        "from the same gate; X threshold should match between populations"
    assert left.geom.y_threshold == right.geom.y_threshold, \
        "Threshold merge assumes that the populations are derived " \
        "from the same gate; Y threshold should match between populations"

    new_geom = ThresholdGeom(x=left.geom.x,
                             y=left.geom.y,
                             transform_x=left.geom.transform_x,
                             transform_y=left.geom.transform_y,
                             x_threshold=left.geom.x_threshold,
                             y_threshold=left.geom.y_threshold)

    new_population = Population(population_name=new_population_name,
                                n=len(left.index) + len(right.index),
                                parent=left.parent,
                                warnings=left.warnings + right.warnings + ["MERGED POPULATION"],
                                index=_merge_index(left, right),
                                geom=new_geom,
                                source="gate",
                                definition=",".join([left.definition, right.definition]),
                                signature=_merge_signatures(left, right))
    return new_population


def _merge_polygons(left: Population,
                    right: Population,
                    new_population_name: str):
    """
    Merge two Populations with PolygonGeom geometries.

    Parameters
    ----------
    left: Population
    right: Population
    new_population_name: str

    Returns
    -------
    Population
    """
    _check_overlap(left, right)
    new_shape = unary_union([p.geom.shape for p in [left, right]])
    x, y = new_shape.exterior.coords.xy
    new_geom = PolygonGeom(x=left.geom.x,
                           y=left.geom.y,
                           transform_x=left.geom.transform_x,
                           transform_y=left.geom.transform_y,
                           x_values=x,
                           y_values=y)
    new_idx = _merge_index(left, right)
    new_population = Population(population_name=new_population_name,
                                n=len(new_idx),
                                parent=left.parent,
                                warnings=left.warnings + right.warnings + ["MERGED POPULATION"],
                                index=new_idx,
                                source="gate",
                                geom=new_geom,
                                signature=_merge_signatures(left, right))
    return new_population


def merge_non_geom_populations(populations: list,
                               new_population_name: str):
    """
    Merge populations arising from classification or clustering. Takes a list of Population objects
    and the name for the new population and merges their indexes, forming a new Population object.

    Parameters
    ----------
    populations: list
    new_population_name: str

    Returns
    -------
    Population

    Raises
    ------
    AssertionError
        Invalid populations provided
    """
    err = "merge_many_populations currently only supports 'cluster' or 'classifier' source " \
          "types. To merge populations from other sources, use merge_populations method"
    assert all([x.source == "cluster" or x.source == "classifier" for x in populations]), err
    assert len(set([x.parent for x in populations])) == 1, "Populations for merging should share the same parent"
    assert len(populations) > 1, "Provide two or more populations for merging"
    new_idx = np.unique(np.concatenate([x.index for x in populations], axis=0), axis=0)
    warnings = [i for sl in [x.warnings for x in populations] for i in sl] + ["MERGED POPULATIONS"]
    new_population = Population(population_name=new_population_name,
                                n=len(new_idx),
                                parent=populations[0].parent,
                                warnings=warnings,
                                index=new_idx,
                                source=populations[0].source,
                                signature=pd.DataFrame([x.signature for x in populations]).mean().to_dict())
    return new_population


def merge_gate_populations(left: Population,
                           right: Population,
                           new_population_name: str or None = None):
    """
    Merge two Population's. The indexes and signatures of these populations will be merged.
    The populations must have the same geometries.

    Parameters
    ----------
    left: Population
    right: Population
    new_population_name: str

    Returns
    -------
    Population

    Raises
    ------
    AssertionError
        Invalid populations provided
    """
    _check_transforms_dimensions(left, right)
    new_population_name = new_population_name or f"merge_{left.population_name}_{right.population_name}"
    assert left.parent == right.parent, "Parent populations do not match"
    assert left.source == right.source, "Populations must be from the same source"
    assert isinstance(left.geom, type(
        right.geom)), f"Geometries must be of the same type; left={type(left.geom)}, right={type(right.geom)}"
    if isinstance(left.geom, ThresholdGeom):
        return _merge_thresholds(left, right, new_population_name)
    return _merge_polygons(left, right, new_population_name)


def merge_multiple_gate_populations(populations: List[Population],
                                    new_population_name: str or None = None):
    """
    Merge multiple Population's. The indexes and signatures of these populations will be merged.
    The populations must have the same geometries.

    Parameters
    ----------
    populations: list
    new_population_name: str

    Returns
    -------
    Population

    Raises
    ------
    AssertionError
        Invalid populations provided
    """
    if new_population_name is None:
        assert len(set([p.population_name for p in populations])) == 1, \
            "If a new population name is not given the populations are expected to have the same population name"
    new_population_name = new_population_name or populations[0].population_name
    merged_pop = reduce(lambda p1, p2: merge_gate_populations(p1, p2), populations)
    merged_pop.population_name = new_population_name
    return merged_pop
