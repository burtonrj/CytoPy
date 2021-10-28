#!/usr/bin.env/python
# -*- coding: utf-8 -*-
"""
A 'gate' in traditional cytometry analysis is a manually 'drawn' shape in one or two dimensional
space that encapsulates or defines a population of events in that space. In CytoPy, a Gate is a procedure
applied to one or two dimensional space resulting in one or more geometries that either separate data in that
space into populations or encapsulate separate populations in that space. These geometries are stored within the
Gate as 'children', each defined by a Child class. The Gate is initially 'trained' on some example data,
generating one or more children within the Gate. When exposed to new data, using the fit_predict method,
the geometries are recreated in the new data and then matched to the children from the training data.

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
import logging
import pickle
from collections import Counter
from typing import List
from typing import Union

import mongoengine
import numpy as np
import pandas as pd
from bson import Binary
from shapely.ops import cascaded_union

from cytopy.data.errors import GateError
from cytopy.data.population import PolygonGeom
from cytopy.data.population import ThresholdGeom
from cytopy.data.read_write import BaseIndexDocument
from cytopy.gating.fda_norm import LandmarkRegistration
from cytopy.utils.sampling import density_dependent_downsampling
from cytopy.utils.sampling import faithful_downsampling
from cytopy.utils.sampling import uniform_downsampling
from cytopy.utils.transform import apply_transform

logger = logging.getLogger(__name__)


class Child(BaseIndexDocument):
    """
    Base class for a gate child population. This is representative of the 'population' of cells
    identified when a gate is first defined and will be used as a template to annotate
    the populations identified in new data.
    """

    name = mongoengine.StringField()
    meta = {"allow_inheritance": True}


class ChildThreshold(Child):
    """
    Child population of a Threshold gate. This is representative of the 'population' of cells
    identified when a gate is first defined and will be used as a tempdate to annotate
    the populations identified in new data.

    Attributes
    -----------
    name: str
        Name of the child
    definition: str
        Definition of population e.g "+" or "-" for 1 dimensional gate or "++" etc for 2 dimensional gate
    geom: ThresholdGeom
        Geometric definition for this child population
    """

    definition = mongoengine.StringField()
    geom = mongoengine.EmbeddedDocumentField(ThresholdGeom)

    def match_definition(self, definition: str) -> bool:
        """
        Given a definition, return True or False as to whether it matches this ChildThreshold's
        definition. If definition contains multiples separated by a comma, or the ChildThreshold's
        definition contains multiple, first split and then compare. Return True if matches any.

        Parameters
        ----------
        definition: str

        Returns
        -------
        bool
        """
        definition = definition.split(",")
        return any([x in self.definition.split(",") for x in definition])


class ChildPolygon(Child):
    """
    Child population of a Polgon or Ellipse gate. This is representative of the 'population' of cells
    identified when a gate is first defined and will be used as a tempdate to annotate
    the populations identified in new data.

    Attributes
    -----------
    name: str
        Name of the child
    geom: ChildPolygon
        Geometric definition for this child population
    """

    geom = mongoengine.EmbeddedDocumentField(PolygonGeom)


class Gate(mongoengine.Document):
    gate_name = mongoengine.StringField(required=True)
    parent = mongoengine.StringField(required=True)
    x = mongoengine.StringField(required=True)
    y = mongoengine.StringField(required=False)
    transform_x = mongoengine.StringField(required=False, default=None)
    transform_y = mongoengine.StringField(required=False, default=None)
    transform_x_kwargs = mongoengine.DictField()
    transform_y_kwargs = mongoengine.DictField()
    downsample_method = mongoengine.StringField(required=False)
    downsample_n = mongoengine.IntField(required=False, default=1000)
    downsample_kwargs = mongoengine.DictField(required=False)
    method = mongoengine.StringField(required=True)
    method_kwargs = mongoengine.DictField()
    children = mongoengine.EmbeddedDocumentListField(Child)
    reference_alignment = mongoengine.BooleanField(default=False)
    reference_kwargs = mongoengine.DictField()
    hyperparameter_search = mongoengine.DictField()
    _reference = mongoengine.FileField(db_alias="core", collection_name="gate_reference_data")
    meta = {"db_alias": "core", "collection": "gates", "allow_inheritance": True}

    def __init__(self, *args, **values):
        super().__init__(*args, **values)
        self.model = None
        self.x_transformer = None
        self.y_transformer = None
        self.validate()
        self._reference_cache = None

    @property
    def reference(self) -> pd.DataFrame:
        if self._reference_cache is not None:
            return self._reference_cache
        try:
            data = pickle.loads(self._reference.read())
            self._reference.seek(0)
            return data
        except TypeError:
            logger.error(f"Reference is empty!")
            return []

    @reference.setter
    def reference(self, data: pd.DataFrame):
        self._reference_cache = data

    def _xy_in_dataframe(self, data: pd.DataFrame):
        """
        Assert that the x and y variables defined for this gate are present in the given
        DataFrames columns

        Parameters
        ----------
        data: Pandas.DataFrame

        Returns
        -------
        None

        Raises
        -------
        GateError
            If required columns missing from provided data
        """
        try:
            assert self.x in data.columns, f"{self.x} missing from given dataframe"
            if self.y:
                assert self.y in data.columns, f"{self.y} missing from given dataframe"
        except AssertionError as e:
            raise GateError(e)

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Transform dataframe prior to gating

        Parameters
        ----------
        data: Pandas.DataFrame

        Returns
        -------
        Pandas.DataFrame
            Transformed dataframe
        """
        if self.transform_x is not None:
            logger.debug(f"Transforming x axis {self.x} with {self.transform_x} transform")
            kwargs = self.transform_x_kwargs or {}
            data, self.x_transformer = apply_transform(
                data=data,
                features=[self.x],
                method=self.transform_x,
                return_transformer=True,
                **kwargs,
            )
        if self.transform_y is not None and self.y is not None:
            logger.debug(f"Transforming x axis {self.y} with {self.transform_y} transform")
            kwargs = self.transform_y_kwargs or {}
            data, self.y_transformer = apply_transform(
                data=data,
                features=[self.y],
                method=self.transform_y,
                return_transformer=True,
                **kwargs,
            )
        return data

    def _downsample(self, data: pd.DataFrame) -> Union[pd.DataFrame, None]:
        """
        Perform down-sampling prior to gating. Returns down-sampled dataframe or
        None if sampling method is undefined.

        Parameters
        ----------
        data: Pandas.DataFrame

        Returns
        -------
        Pandas.DataFrame or None

        Raises
        ------
        GateError
            Invalid downsampling method provided or sampling kwargs are missing
        """
        data = data.copy()
        logger.debug(f"Downsampling data using {self.downsample_method} method")

        if self.downsample_method == "uniform":
            if self.downsample_n is None:
                raise GateError("Must provide 'n' for down-sampling")
            return uniform_downsampling(data=data, sample_size=self.downsample_n)

        if self.downsample_method == "density":
            kwargs = self.downsample_kwargs or {}
            features = [f for f in [self.x, self.y] if f is not None]
            return density_dependent_downsampling(data=data, features=features, **kwargs)

        if self.downsample_method == "faithful":
            kwargs = self.downsample_kwargs or {}
            kwargs["h"] = kwargs.get("h", 0.01)
            return faithful_downsampling(data=data.to_numpy(), **kwargs)

        raise GateError("Invalid downsample method, should be one of: 'uniform', 'density' or 'faithful'")

    def get_children(self, name: str):
        children = self.children.filter(name=name)
        if len(children) == 0:
            raise GateError(f"No such child ")
        return children

    def _duplicate_children(self, name: str):
        child_counts = Counter(self.get_children(name=name))
        if all([i == 1 for i in child_counts.values()]):
            return
        updated_children = []
        for name, count in child_counts.items():
            if count >= 2:
                updated_children.append(merge_children([c for c in self.children if c.name == name]))
            else:
                updated_children.append(self.children.get(name=name))
        self.children = updated_children

    def _align_to_reference(self, data: pd.DataFrame) -> pd.DataFrame:
        kwargs = self.reference_kwargs or {}
        lr = LandmarkRegistration(**kwargs)
        for d in [self.x, self.y]:
            if d:
                x = np.array([data[d].values, self.reference[d].values])
                data[d] = lr.fit(data=x).transform(data[d].values)
        return data

    def preprocess(self, data: pd.DataFrame, transform: bool):
        data = data.copy()
        if data.shape[0] <= 3:
            raise GateError("Data provided contains 3 or less observations.")
        self._xy_in_dataframe(data=data)
        if transform:
            data = self.transform(data=data)
        if self.reference_alignment:
            data = self._align_to_reference(data=data)
        return data

    def save(self, *args, **kwargs):
        for child in self.children:
            assert child.index is not None, f"Child {child.name} index is empty!"
        for child in self.children:
            child.write_index()
        if self._reference_cache is not None:
            if self._reference:
                self._reference.replace(Binary(pickle.dumps(self._reference_cache, protocol=pickle.HIGHEST_PROTOCOL)))
            else:
                self._reference.new_file()
                self._reference.write(Binary(pickle.dumps(self._reference_cache, protocol=pickle.HIGHEST_PROTOCOL)))
                self._reference.close()
        super(Gate, self).save()


def merge_children(children: List) -> Union[Child, ChildThreshold, ChildPolygon]:
    """
    Given a list of Child objects, merge and return single child

    Parameters
    ----------
    children: list

    Returns
    -------
    Child or ChildThreshold or ChildPolygon

    Raises
    ------
    AssertionError
        Invalid Children provided
    """
    assert (
        len(set([type(x) for x in children])) == 1
    ), f"Children must be of same type; not, {[type(x) for x in children]}"
    assert len(set([c.name for c in children])), "Children should all have the same name"
    if isinstance(children[0], ChildThreshold):
        definition = ",".join([c.definition for c in children])
        child = ChildThreshold(name=children[0].name, definition=definition, geom=children[0].geom)
        child.index = np.unique(np.concatenate([x.index for x in children], axis=0), axis=0)
        return child
    if isinstance(children[0], ChildPolygon):
        merged_poly = cascaded_union([c.geom.shape for c in children])
        x, y = merged_poly.exterior.xy[0], merged_poly.exterior.xy[1]
        child = ChildPolygon(
            name=children[0].name,
            geom=PolygonGeom(
                x=children[0].geom.x,
                y=children[0].geom.y,
                transform_x=children[0].geom.transform_x,
                transform_y=children[0].geom.transform_y,
                x_values=x,
                y_values=y,
            ),
        )
        child.index = np.unique(np.concatenate([x.index for x in children], axis=0), axis=0)
        return child
    raise GateError("Unrecognised Child type")
