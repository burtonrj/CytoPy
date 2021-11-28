#!/usr/bin.env/python
# -*- coding: utf-8 -*-
"""
Each subject in your analysis (human, mouse, cell line etc) can be
represented by a Subject document that can then be associated to
specimens in an Experiment. This Subject document is dynamic and can
house any relating meta-data.

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
import json
import logging
from functools import partial
from typing import Dict
from typing import List
from typing import Set
from typing import Union

import mongoengine
import numpy as np
import pandas as pd
import polars as pl
from mongoengine.base import BaseList
from scipy.stats import kurtosis
from scipy.stats import skew

from .read_write import polars_to_pandas

logger = logging.getLogger(__name__)


class Subject(mongoengine.DynamicDocument):
    """
    Document based representation of subject meta-data.
    Subjects are stored in a dynamic document, meaning
    new properties can be added ad-hoc.

    Attributes
    -----------
    subject_id: str, required
        Unique identifier for subject
    notes: str
        Additional notes
    """

    subject_id = mongoengine.StringField(required=True, unique=True)
    notes = mongoengine.StringField(required=False)

    meta = {"db_alias": "core", "collection": "subjects"}

    @property
    def fields(self) -> List[str]:
        return list(self.to_dict().keys())

    @fields.setter
    def fields(self, _):
        raise ValueError("Fields is read only, access individual fields to edit values.")

    def to_dict(self, *args, **kwargs) -> Dict:
        return json.loads(self.to_json(*args, **kwargs))

    def field_to_df(self, field: str, **kwargs) -> pd.DataFrame:
        if len(self.__getitem__(field)) == 0:
            logger.warning(f"filed '{field}' is empty")
            return None
        try:
            return pd.DataFrame(self.to_dict()[field], **kwargs)
        except KeyError:
            logger.error(f"{field} is not a recognised field")
        except ValueError:
            return (
                pd.DataFrame(self.to_dict()[field], index=["value"], **kwargs)
                .T.reset_index()
                .rename({"index": field}, axis=1)
            )

    @staticmethod
    def _list_node(node: BaseList, keys: List[str], summary: str):
        for k in keys:
            node = [n[k] for n in node]
        if any([isinstance(x, str) for x in node]):
            return ",".join([str(x) for x in node])
        if summary == "mean":
            return np.mean([float(x) for x in node])
        if summary == "median":
            return np.median([float(x) for x in node])
        if summary == "std":
            return np.std([float(x) for x in node])
        if summary == "kurtosis":
            return kurtosis([float(x) for x in node])
        if summary == "skew":
            return skew([float(x) for x in node])
        raise ValueError("Invalid value fo summary method, should be one of: mean, median, std, kurtosis, or skew")

    def lookup_var(self, key: Union[str, List[str]], summary: str = "mean"):
        try:
            if isinstance(key, list) and len(key) == 1:
                return self[key[0]]
            if isinstance(key, str):
                return self[key]
            node = self[key[0]]
            for i, k in enumerate(key[1:]):
                if isinstance(node, BaseList):
                    if len(node) == 0:
                        return None
                    if len(node) == 1:
                        node = node[0]
                        node = node[k]
                    else:
                        return self._list_node(node=node, keys=key[i:], summary=summary)
                else:
                    node = node[k]
            return node
        except KeyError:
            return np.nan


def common_fields(subjects: List[Subject]) -> Set:
    return set.intersection(*[set(s.fields) for s in subjects])


def safe_search(subject_id: str):
    try:
        return Subject.objects(subject_id=subject_id).get()
    except mongoengine.DoesNotExist:
        return None


def lookup_variable(subject_id: str, key: Union[str, List[str]]) -> Union[None, str]:
    try:
        subject = safe_search(subject_id=subject_id)
        if subject is None:
            return None
        return subject.lookup_var(key)
    except KeyError:
        return None


def add_meta_labels(data: Union[pd.DataFrame, pl.DataFrame], key: Union[str, List[str]], column_name: str):
    data = data if isinstance(data, pd.DataFrame) else polars_to_pandas(data=data)
    if column_name in data.columns:
        logger.warning(f"Data already contains column {column_name}. Will be overwritten.")
        data.drop(column_name, axis=1, inplace=True)
    if "subject_id" not in data.columns:
        raise ValueError(f"Data is missing 'subject_id' column")
    unique_subject_ids = data["subject_id"].unique()
    lookup_func = partial(lookup_variable, key=key)
    meta_var = list(map(lookup_func, unique_subject_ids))
    return data.merge(
        pd.DataFrame({"subject_id": unique_subject_ids, column_name: meta_var}), on="subject_id", how="left"
    )
