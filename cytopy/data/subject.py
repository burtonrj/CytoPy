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
        """
        List of available first level fields

        Returns
        -------
        List[str]
        """
        return list(self.to_dict().keys())

    @fields.setter
    def fields(self, _):
        raise ValueError("Fields is read only, access individual fields to edit values.")

    def __repr__(self):
        txt = "\n".join([f"Subject(subject_id={self.subject_id}) with the fields:"] + [f"- {x}" for x in self.fields])
        return txt

    def to_dict(self, *args, **kwargs) -> Dict:
        """
        Convert the document to a python dictionary, any additional arguments passed to mongoengine to_json function.

        Returns
        -------
        Dict
        """
        return json.loads(self.to_json(*args, **kwargs))

    def field_to_df(self, field: str, **kwargs) -> Union[pd.DataFrame, None]:
        """
        Convert a field to a Pandas DataFrame. This requires that the field be compatible
        with the creation of a Pandas DataFrame, this is valid if:
            * The field contains one or more keys with values as flat lists
            * The field is a list of flat dictionaries

        As an example, the following fields would create equivalent DataFrames:
         * [{"a": 1, "b": 2}, {"a": 5, "b": 5}]
         * {"a": [1, 5], "b": [2, 5]}

        Parameters
        ----------
        field: str
            Name of the field to convert
        kwargs:
            Additional keyword arguments passed to Pandas DataFrame constructor

        Returns
        -------
        Union[Panda.DataFrame, None]
            If the field is empty, will return None
        """
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
    def _list_node(node: BaseList, keys: List[str], summary: str) -> Union[str, np.ndarray]:
        """
        Summarise an iterable end-node

        Parameters
        ----------
        node: BaseList
        keys: List[str]
        summary: str
            Summary method, must be one of mean, median, std, kurtosis, or skew

        Returns
        -------
        Union[str, np.ndarray]
            If any of the values in this end-node are a string, will return a string of as comma-seperated values.
            Otherwise, returns a Numpy.Array
        """
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
            return skew(np.array([float(x) for x in node]))
        raise ValueError("Invalid value fo summary method, should be one of: mean, median, std, kurtosis, or skew")

    def lookup_var(self, key: Union[str, List[str]], summary: str = "mean") -> Union[str, float, None]:
        """
        Lookup a variable and return the value. Provide either the name of the field of interest, or
        provide a list of keys to navigate to the required field. For example, if the Subject had the following
        data structure and we wanted the monocyte count:

        - blood_results
        |
        --- full_blood_count
            |
            --- Lymphocyte count
            --- Neutrophil count
            --- Monocyte count

        We would provide as a key ["blood_results", "full_blood_count", "Monocyte count"].

        If the end node is a list of values (continuing with the example above, say the monocyte count contains multiple
        values) then you should provide a summary method. The summary method must be one of mean, median, std, kurtosis,
        or skew and specifies how the returned value is calculated. If the end node contains one or more string
        values the returned value will be a string of the values comma-seperated.

        Parameters
        ----------
        key: Union[str, List[str]]
        summary: str (default='mean')

        Returns
        -------
        Union[str, float, np.nan]
            If the key(s) is not recognised, will return None.
        """
        try:
            if isinstance(key, list) and len(key) == 1:
                return self[key[0]]
            if isinstance(key, str):
                return self[key]
            node = self[key[0]]
            for i, k in enumerate(key[1:]):
                if isinstance(node, BaseList):
                    if len(node) == 0:
                        return np.nan
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
    """
    Given a list of Subjects, return the common fields.

    Parameters
    ----------
    subjects: List[Subject]

    Returns
    -------
    Set
    """
    return set.intersection(*[set(s.fields) for s in subjects])


def safe_search(subject_id: str) -> Union[Subject, None]:
    """
    Search for a subject and return None if not found

    Parameters
    ----------
    subject_id: str

    Returns
    -------
    Union[Subject, None]
    """
    try:
        return Subject.objects(subject_id=subject_id).get()
    except mongoengine.DoesNotExist:
        return None


def lookup_variable(subject_id: str, key: Union[str, List[str]], **kwargs) -> Union[str, float, None]:
    """
    Lookup a variable in a Subject by the subject ID. See Subject.lookup_var for details.

    Parameters
    ----------
    subject_id: str
    key: Union[str, List[str]]
    kwargs:
        Keyword arguments passed to Subject.lookup_var

    Returns
    -------
    Union[str, float, np.nan]
    """
    try:
        subject = safe_search(subject_id=subject_id)
        if subject is None:
            return np.nan
        return subject.lookup_var(key, **kwargs)
    except KeyError:
        return np.nan


def add_meta_labels(
    data: Union[pd.DataFrame, pl.DataFrame], key: Union[str, List[str]], column_name: str, **kwargs
) -> pd.DataFrame:
    """
    Given a DataFrame with the column 'subject_id', iterate over the subject identifiers, load the
    Subject, search for a key, and then populate a new column with these values. If a subject is
    not identified or is missing the key, the value will be NaN. For details on how subject
    variables are obtained, see Subject.lookup_var.

    Parameters
    ----------
    data: Pandas.DataFrame
    key: Union[str, List[str]]
    column_name: str
        New column name
    kwargs:
        Keyword arguments passed to Subject.lookup_var

    Returns
    -------
    Pandas.DataFrame
    """
    data = data if isinstance(data, pd.DataFrame) else polars_to_pandas(data=data)
    if column_name in data.columns:
        logger.warning(f"Data already contains column {column_name}. Will be overwritten.")
        data.drop(column_name, axis=1, inplace=True)
    if "subject_id" not in data.columns:
        raise ValueError(f"Data is missing 'subject_id' column")
    unique_subject_ids = data["subject_id"].unique()
    lookup_func = partial(lookup_variable, key=key, **kwargs)
    meta_var = list(map(lookup_func, unique_subject_ids))
    return data.merge(
        pd.DataFrame({"subject_id": unique_subject_ids, column_name: meta_var}), on="subject_id", how="left"
    )
