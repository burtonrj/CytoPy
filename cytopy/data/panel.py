#!/usr/bin.env/python
# -*- coding: utf-8 -*-
"""
The panel module houses the Panel document, an embedded document of the Experiment class which handles
channel mappings and standardisation of columns names in single cell data.

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
from __future__ import annotations

import logging
import os
import re
from typing import Dict
from typing import List
from typing import Optional
from typing import Union

import mongoengine
import pandas as pd

from .errors import PanelError
from .read_write import match_file_ext
from .read_write import read_headers

logger = logging.getLogger(__name__)


def load_template(path: str) -> pd.DataFrame:
    """
    Given the filepath for a cytometry panel design, return the panel as a Pandas DataFrame.

    Parameters
    ----------
    path: str

    Returns
    -------
    Pandas.DataFrame

    Raises
    ------
    ValueError
        Panel template is not a CSV or Excel file

    KeyError
        Panel design missing required columns

    RuntimeError
        Duplicate channels or names present in the design
    """
    if match_file_ext(path=path, ext=".csv"):
        template = pd.read_csv(path)
    elif match_file_ext(path=path, ext=".xlsx") or match_file_ext(path=path, ext=".xls"):
        template = pd.read_excel(path)
    else:
        raise ValueError(f"Panel template should be a csv or excel file, not {path}.")
    required_columns = ["channel", "regex_pattern", "case_sensitive", "permutations", "name"]
    if not all([x in template.columns for x in required_columns]):
        raise KeyError(f"Template must contain columns {required_columns}.")
    unique_channels = len(template.channel.values) == template.channel.nunique()
    unique_names = len(template.name.values) == template.name.nunique()
    if not unique_channels and unique_names:
        raise RuntimeError("Duplicate channels and/or names detected in template.")
    return template


class Channel(mongoengine.EmbeddedDocument):
    channel = mongoengine.StringField(required=True)
    name = mongoengine.StringField(required=False)
    regex_pattern = mongoengine.StringField(required=False)
    case_sensitive = mongoengine.IntField(default=0)
    permutations = mongoengine.StringField(required=False)

    def query(self, x: str) -> Union[str, None]:
        """
        Given a some string, compare to the Channel definition and return the standardised channel name
        if it matches, otherwise return None.

        Parameters
        ----------
        x: str

        Returns
        -------
        Union[str, None]
        """
        if self.regex_pattern:
            if self.case_sensitive:
                if re.search(self.regex_pattern, x):
                    return self.name if self.name else self.channel
                return None
            if re.search(self.regex_pattern, x, re.IGNORECASE):
                return self.name if self.name else self.channel
        if self.permutations:
            for p in self.permutations.split(","):
                if x == p:
                    return self.name if self.name else self.channel
        if x == self.channel:
            return self.name
        return None


class Panel(mongoengine.EmbeddedDocument):
    """
    Document representation of channel/marker definition for an experiment. A panel, once associated to an experiment
    will standardise column names; when a FileGroup is retrieved from an Experiment, the channel/marker definitions
    in the data will be mapped to the associated panel.

    Attributes
    -----------
    channels: EmbeddedDocumentListField[Channel]
    """

    channels = mongoengine.EmbeddedDocumentListField(Channel)
    meta = {"db_alias": "core", "collection": "fcs_panels"}

    def list_channels(self) -> List[str]:
        return [x.name for x in self.channels]

    def query_channel(self, channel: str) -> str:
        """
        Given a string value of a channel from the raw data, query the Panel design and return the
        standardised channel name.

        Parameters
        ----------
        channel: str
            Raw channel name, potentially containing typos or permutations of the channel name

        Returns
        -------
        str

        Raises
        ------
        ValueError
            The given string either matched multiple Channel definitions or did not match any channel in this Panel.
        """
        matches = [channel_definition.query(channel) for channel_definition in self.channels]
        matches = [x for x in matches if x is not None]
        if len(matches) > 1:
            raise ValueError(f"Channel {channel} matched more than one definition in panel. Channels must be unique.")
        if len(matches) == 0:
            raise ValueError(f"No matching channel found in panel for {channel}")
        return matches[0]

    def build_mappings(self, path: Union[str, List[str]], s3_bucket: Optional[str] = None) -> Dict[str, str]:
        """
        Given one or more file paths for single cell data, generate a dictionary of column mappings to
        rename columns to standardised names defined by this Panel.

        Parameters
        ----------
        path: Union[str, List[str]]
            One or more list of file paths
        s3_bucket: str, optional

        Returns
        -------
        Dict[str, str]

        Raises
        ------
        ValueError
            When multiple file paths are provided column names must match between files.
        """
        try:
            if isinstance(path, str):
                path = [path]
            columns = None
            for path in path:
                if columns is None:
                    columns = read_headers(path=path, s3_bucket=s3_bucket)
                else:
                    assert set(columns) == set(read_headers(path=path, s3_bucket=s3_bucket))
            mappings = {}
            for col in [x for x in columns if x != "Index"]:
                mappings[col] = self.query_channel(channel=col)
            return mappings
        except AssertionError:
            err = "Columns must be identical for all files with related mappings"
            logger.error(err)
            raise ValueError(err)

    def create_from_tabular(self, path: str) -> Panel:
        """
        Populations the Panel definition using a tabular file (either CSV or Excel file)

        Parameters
        ----------
        path: str

        Returns
        -------
        Panel
        """
        logger.info(f"Generating new Panel definition from Excel file template {path}")
        if not os.path.isfile(path):
            raise PanelError(f"No such file {path}")
        template = load_template(path=path).fillna("").to_dict(orient="records")
        for row in template:
            self.channels.append(
                Channel(
                    channel=row["channel"],
                    regex_pattern=row["regex_pattern"],
                    case_sensitive=row["case_sensitive"],
                    permutations=row["permutations"],
                    name=row["name"],
                )
            )
        return self
