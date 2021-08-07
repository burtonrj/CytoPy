#!/usr/bin.env/python
# -*- coding: utf-8 -*-
"""
The experiment module houses the Experiment class, used to define
cytometry based experiments that can consist of one or more biological
specimens. An experiment should be defined for each cytometry staining
panel used in your analysis and the single cell data (contained in
*.fcs files) added to the experiment using the 'add_new_sample' method.
Experiments should be created using the Project class (see cytopy.data.projects).
All functionality for experiments and Panels are housed within this
module.

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
import gc
import logging
import os
import re
from collections import Counter
from datetime import datetime
from functools import wraps
from typing import Callable
from typing import Dict
from typing import List
from typing import Optional
from typing import Union

import matplotlib.pyplot as plt
import mongoengine
import numpy as np
import pandas as pd
import xlrd

from ..feedback import progress_bar
from ..flow.sampling import sample_dataframe
from .errors import DuplicateSampleError
from .errors import MissingSampleError
from .fcs import FileGroup
from .mapping import ChannelMap
from .read_write import FCSFile
from .subject import Subject

__author__ = "Ross Burton"
__copyright__ = "Copyright 2020, cytopy"
__credits__ = ["Ross Burton", "Simone Cuff", "Andreas Artemiou", "Matthias Eberl"]
__license__ = "MIT"
__version__ = "2.0.0"
__maintainer__ = "Ross Burton"
__email__ = "burtonrj@cardiff.ac.uk"
__status__ = "Production"
logger = logging.getLogger(__name__)


def _load_excel_to_dataframe(path: str) -> (pd.DataFrame, pd.DataFrame):
    """
    Check sheet names are as expected. That is: nomenclature and mappings.
    Then load Pandas DataFrame for each sheet.

    Parameters
    ----------
    path: str
        Path to excel file

    Returns
    -------
    Pandas.DataFrame, Pandas.DataFrame

    Raises
    ------
    ValueError
        If Excel sheet names are incorrect
    """
    xls = xlrd.open_workbook(path, on_demand=True)

    if not all([x in ["nomenclature", "mappings"] for x in xls.sheet_names()]):
        raise ValueError(f"Template must contain two sheets: nomenclature and mappings")

    nomenclature = pd.read_excel(path, sheet_name="nomenclature")
    mappings = pd.read_excel(path, sheet_name="mappings")
    return mappings, nomenclature


def _check_nomenclature_headings(nomenclature: pd.DataFrame) -> None:
    """
    Raise ValueError if columns in nomenclature DataFrame are invalid.

    Parameters
    ----------
    nomenclature: Pandas.DataFrame

    Returns
    -------
    None

    Raises
    -------
    ValueError
        If Nomenclature column names are incorrect
    """
    if not all([x in ["name", "regex", "permutations", "case"] for x in nomenclature.columns]):
        raise ValueError(
            "Nomenclature sheet of excel template must contain the following column headers: "
            "'name','regex','case','permutations'"
        )


def _check_mappings_headings(mappings: pd.DataFrame):
    """
    Raise ValueError if columns in mappings DataFrame are invalid.

    Parameters
    ----------
    mappings: Pandas.DataFrame

    Returns
    -------
    None

    Raises
    -------
    ValueError
        If Mappings column names are incorrect
    """
    err = "Mappings sheet of excel template must contain the following column headers: 'channel', 'marker'"
    if not all([x in ["channel", "marker"] for x in mappings.columns]):
        raise ValueError(err)


def check_excel_template(path: str) -> (pd.DataFrame, pd.DataFrame) or None:
    """
    Check excel template and if valid return pandas dataframes

    Parameters
    ----------
    path: str
        file path for excel template

    Returns
    --------
    (Pandas.DataFrame, Pandas.DataFrame) or None
        tuple of pandas dataframes (nomenclature, mappings) or None

    Raises
    ------
    ValueError
        If duplicate entries or missing entries in excel template
    """
    try:
        mappings, nomenclature = _load_excel_to_dataframe(path)
        _check_nomenclature_headings(nomenclature)
        _check_mappings_headings(mappings)
        # Check for duplicate entries
        err = "Duplicate entries in nomenclature, please remove duplicates before continuing"
        assert sum(nomenclature["name"].duplicated()) == 0, err
        # Check that all mappings have a corresponding entry in nomenclature
        for x in ["channel", "marker"]:
            for name in mappings[x]:
                if pd.isnull(name):
                    continue
                if name not in nomenclature.name.values:
                    logger.error(f"{name} missing from nomenclature, please review template")
                    raise ValueError(f"{name} missing from nomenclature, please review template")
        return nomenclature, mappings
    except AssertionError as e:
        raise ValueError(e)
    except ValueError as e:
        raise ValueError(e)


def check_duplication(x: List[str]) -> bool:
    """
    Internal method. Given a list check for duplicates. Warning generated for duplicates.

    Parameters
    ----------
    x: list

    Returns
    --------
    bool
        True if duplicates are found, else False
    """
    x = [i if i else None for i in x]
    duplicates = [item for item, count in Counter(x).items() if count > 1 and item is not None]
    if duplicates:
        logger.warning(f"Duplicate channel/markers identified: {duplicates}")
        return True
    return False


class NormalisedName(mongoengine.EmbeddedDocument):
    """
    Defines a standardised name for a channel or marker and provides method for testing if a channel/marker
    should be associated to standard

    Attributes
    ----------
    standard: str, required
        the "standard" name i.e. the nomenclature we used for a channel/marker in this panel
    regex_str: str
        regular expression used to test if a term corresponds to this standard
    permutations: str
        String values that have direct association to this standard (comma seperated values)
    case_sensitive: bool, (default=False)
        is the nomenclature case sensitive? This would be false for something like 'CD3' for example,
        where 'cd3' and 'CD3' are synonymous
    """

    standard = mongoengine.StringField(required=True)
    regex_str = mongoengine.StringField()
    permutations = mongoengine.StringField()
    case_sensitive = mongoengine.BooleanField(default=False)

    def query(self, x: str) -> Optional[str]:
        """
        Given a term 'x', determine if 'x' is synonymous to this standard. If so, return the standardised name.

        Parameters
        -----------
        x: str
            search term

        Returns
        --------
        str or None
            Standardised name if synonymous to standard, else None
        """
        logger.debug(f"Querying NormalisedName {self.standard} against {x}")
        if self.case_sensitive:
            if re.search(self.regex_str, x):
                return self.standard
            return None
        if re.search(self.regex_str, x, re.IGNORECASE):
            return self.standard
        if self.permutations:
            for p in self.permutations.split(","):
                if x == p:
                    return self.standard
        return None


def query_normalised_list(x: Optional[str], ref: List[NormalisedName]) -> str:
    """
    Query a channel/marker against a reference list of NormalisedName's

    Parameters
    ----------
    x: str or None
        channel/marker to query
    ref: list
        list of NormalisedName objects for reference search

    Returns
    --------
    str
        Standardised name

    Raises
    -------
    ValueError
        If no or multiple matches found in query
    """
    logger.debug(f"Normalising channels/markers {x}")
    corrected = list(filter(None.__ne__, [n.query(x) for n in ref]))

    if len(corrected) == 0:
        raise ValueError(f"Unable to normalise {x}; no match in linked panel")

    if len(corrected) > 1:
        raise ValueError(
            f"Unable to normalise {x}; matched multiple in linked panel, check "
            f"panel for incorrect definitions. Matches found: {corrected}"
        )
    return corrected[0]


def _is_empty(x: str):
    if x.isspace():
        return None
    if x == "":
        return None
    return x


def check_pairing(channel_marker: Dict[str, str], ref_mappings: List[ChannelMap]) -> bool:
    """
    Given a channel and marker check that a valid pairing exists in the list
    of given mappings.

    Parameters
    ----------
    channel_marker: dict
    ref_mappings: list
        List of ChannelMap objects

    Returns
    --------
    bool
        True if pairing exists, else False
    """
    logging.debug(f"Checking {channel_marker} has valid pairing in reference mappings")
    channel, marker = _is_empty(channel_marker.get("channel")), _is_empty(channel_marker.get("marker"))
    if not any([n.check_matched_pair(channel=channel, marker=marker) for n in ref_mappings]):
        return False
    return True


def _standardise(x: Optional[str], ref: List[NormalisedName], mappings: List[ChannelMap], alt: str) -> str:
    """
    Given a channel/marker, either return the corresponding standard name
    according to a list of standards (ref) or if the channel/marker is None,
    return the channel/marker in the ChannelMap object associated with it's
    matching channel/marker (alt)

    Parameters
    ----------
    x: str
    ref: list
    mappings: list
    alt: str

    Returns
    -------
    str
    """
    logging.debug(f"Normalising channel/marker name {x}")
    if x is not None:
        return query_normalised_list(x, ref)
    default = [m for m in mappings if m.channel == alt or m.marker == alt][0]
    if default.channel == alt:
        logging.debug(f"{x} = {default.marker}")
        return default.marker
    logging.debug(f"{x} = {default.channel}")
    return default.channel


def standardise_names(
    channel_marker: Dict,
    ref_channels: List[NormalisedName],
    ref_markers: List[NormalisedName],
    ref_mappings: List[ChannelMap],
) -> Dict[str, str]:
    """
    Given a dictionary detailing a channel/marker pair ({"channel": str, "marker": str})
    standardise its contents using the reference material provided.

    Parameters
    ----------
    channel_marker: dict
    ref_channels: list
    ref_markers: list
    ref_mappings: list

    Returns
    -------
    dict

    Raises
    ------
    ValueError
        If channel and marker are missing
    """
    logging.debug(f"Normalising channel/marker pairing {channel_marker}")
    channel, marker = _is_empty(channel_marker.get("channel")), _is_empty(channel_marker.get("marker"))
    if channel is None and marker is None:
        ValueError("Cannot standardise column names because both channel and marker missing from mappings")
    channel = _standardise(channel, ref_channels, ref_mappings, marker)
    marker = _standardise(marker, ref_markers, ref_mappings, channel)
    normalised = {"channel": channel, "marker": marker}
    logging.debug(f"Normalised: {normalised}")
    return normalised


def duplicate_mappings(mappings: List[dict]) -> None:
    """
    Check for duplicates in a list of dictionaries describing channel/marker mappings.
    Raise AssertionError if duplicates found.

    Parameters
    ----------
    mappings: list

    Returns
    -------
    None

    Raises
    ------
    ValueError
        If duplicate channel/marker found
    """
    logging.debug(f"Checking for duplicate mappings in {mappings}")
    channels = [x.get("channel") for x in mappings]
    if check_duplication(channels):
        raise ValueError("Duplicate channels provided")

    markers = [x.get("marker") for x in mappings]
    if check_duplication(markers):
        raise ValueError("Duplicate markers provided")


def missing_channels(mappings: List[dict], channels: List[NormalisedName], errors: str = "raise") -> None:
    """
    Check a list of channel/marker dictionaries for missing channels according to
    the reference channels given.

    Parameters
    ----------
    mappings: list
    channels: list
    errors: str

    Returns
    -------
    None

    Raises
    ------
    KeyError
        If channel is missing
    """
    logging.debug(f"Checking {mappings} for missing channels according to reference channels")
    existing_channels = [x.get("channel") for x in mappings]
    for x in channels:
        if x.standard not in existing_channels:
            if errors == "raise":
                raise KeyError(f"Missing channel {x.standard}")
            elif errors == "warn":
                logger.warning(f"Missing channel {x.standard}", stacklevel=2)


class Panel(mongoengine.EmbeddedDocument):
    """
    Document representation of channel/marker definition for an experiment. A panel, once associated to an experiment
    will standardise data upon input; when an fcs file is created in the database, it will be associated to
    an experiment and the channel/marker definitions in the fcs file will be mapped to the associated panel.

    Attributes
    -----------
    markers: EmbeddedDocListField
        list of marker names; see NormalisedName
    channels: EmbeddedDocListField
        list of channels; see NormalisedName
    mappings: EmbeddedDocListField
        list of channel/marker mappings; see ChannelMap
    initiation_date: DateTime
        date of creation

    """

    markers = mongoengine.EmbeddedDocumentListField(NormalisedName)
    channels = mongoengine.EmbeddedDocumentListField(NormalisedName)
    mappings = mongoengine.EmbeddedDocumentListField(ChannelMap)
    initiation_date = mongoengine.DateTimeField(default=datetime.now)
    meta = {"db_alias": "core", "collection": "fcs_panels"}

    def create_from_excel(self, path: str) -> None:
        """
        Populate panel attributes from an excel template

        Parameters
        ----------
        path: str
            path of file

        Returns
        --------
        None

        Raises
        ------
        ValueError
            If file path is invalid
        """
        logger.info(f"Generating new Panel definition from Excel file template {path}")

        if not os.path.isfile(path):
            raise ValueError("No such file {path}")

        nomenclature, mappings = check_excel_template(path)
        for col_name, attr in zip(["channel", "marker"], [self.channels, self.markers]):
            for name in mappings[col_name]:
                if not pd.isnull(name):
                    d = nomenclature[nomenclature["name"] == name].fillna("").to_dict(orient="list")
                    attr.append(
                        NormalisedName(
                            standard=d["name"][0],
                            regex_str=d["regex"][0],
                            case_sensitive=d["case"][0],
                            permutations=d["permutations"][0],
                        )
                    )
        mappings = mappings.fillna("").to_dict(orient="list")
        self.mappings = [ChannelMap(channel=c, marker=m) for c, m in zip(mappings["channel"], mappings["marker"])]

    def create_from_dict(self, x: dict):
        """
        Populate panel attributes from a python dictionary

        Parameters
        ----------
        x: dict
            dictionary object containing panel definition

        Returns
        --------
        None

        Raises
        ------
        ValueError
            If invalid dictionary template
        """

        # Check validity of input dictionary
        logger.info(f"Generating new Panel definition from dictionary template")

        err = (
            "Invalid template dictionary; must be a nested dictionary with parent keys: channels, markers, & mappings"
        )
        if not all([k in ["channels", "markers", "mappings"] for k in x.keys()]):
            raise ValueError(err)

        if not isinstance(x["mappings"], list):
            raise ValueError("Invalid template dictionary; mappings must be a list of tuples")

        if not all([len(i) == 2 for i in x["mappings"]]):
            raise ValueError(
                "Invalid template dictionary; mappings should be of shape (n,2) where n is the number of "
                "channel/marker pairs"
            )

        self.markers = [
            NormalisedName(
                standard=k["name"],
                regex_str=k["regex"],
                case_sensitive=k["case"],
                permutations=k["permutations"],
            )
            for k in x["markers"]
        ]
        self.channels = [
            NormalisedName(
                standard=k["name"],
                regex_str=k["regex"],
                case_sensitive=k["case"],
                permutations=k["permutations"],
            )
            for k in x["channels"]
        ]
        self.mappings = [ChannelMap(channel=c, marker=m) for c, m in x["mappings"]]

    def list_channels(self) -> list:
        """
        List of channels associated to panel

        Returns
        -------
        List
        """
        return [cm.channel for cm in self.mappings]

    def list_markers(self) -> list:
        """
        List of channels associated to panel

        Returns
        -------
        List
        """
        return [cm.marker for cm in self.mappings]


def compensate_data(x: np.ndarray, spill_matrix: np.ndarray) -> np.ndarray:
    """
    Compensate the given data, x, using the spillover matrix by solving for their linear
    combination.

    Parameters
    ----------
    x: numpy.ndarray
    spill_matrix: numpy.ndarray

    Returns
    -------
    numpy.ndarray
    """
    return np.linalg.solve(spill_matrix.T, x.T).T


def panel_defined(func: Callable) -> Callable:
    """
    Wrapper that will raise ValueError if Panel definition does not exist in provided Experiment
    """

    @wraps(func)
    def wrapper(*args, **kwargs):
        if args[0].panel is None:
            err = f"No panel defined for experiment {args[0].experiment_id}"
            raise ValueError(err)
        return func(*args, **kwargs)

    return wrapper


class Experiment(mongoengine.Document):
    """
    Container for Cytometry experiment. The correct way to generate and load these objects is using the
    Project.add_experiment method (see cytopy.data.project.Project). This object provides access
    to all experiment-wide functionality. New files can be added to an experiment using the
    add_new_sample method.

    Attributes
    -----------
    experiment_id: str, required
        Unique identifier for experiment
    panel: ReferenceField, required
        Panel object describing associated channel/marker pairs
    fcs_files: ListField
        Reference field for associated files
    flags: str, optional
        Warnings associated to experiment
    notes: str, optional
        Additional free text comments
    """

    experiment_id = mongoengine.StringField(required=True, unique=True)
    panel = mongoengine.EmbeddedDocumentField(Panel)
    fcs_files = mongoengine.ListField(mongoengine.ReferenceField(FileGroup, reverse_delete_rule=4))
    data_directory = mongoengine.StringField(required=True)
    flags = mongoengine.StringField(required=False)
    notes = mongoengine.StringField(required=False)

    meta = {"db_alias": "core", "collection": "experiments"}

    @staticmethod
    def _check_panel(panel_definition: str or None):
        """
        Check that parameters provided for defining a panel are valid.

        Parameters
        ----------
        panel_definition: str or None
            Path to a panel definition

        Returns
        -------
        None

        Raises
        ------
        ValueError
            Given parameters are invalid
        """
        if not os.path.isfile(panel_definition):
            raise ValueError(f"{panel_definition} does not exist")

        if not os.path.splitext(panel_definition)[1] in [".xls", ".xlsx"]:
            raise ValueError("Panel definition is not a valid Excel document")

    def generate_panel(self, panel_definition: Union[str, dict]) -> None:
        """
        Generate a new panel using the panel definition provided (path to a valid template).

        Parameters
        ----------
        panel_definition: Union[str, dict]
            Path to a panel definition

        Returns
        -------
        None

        Raises
        ------
        ValueError
            Panel definition is not a string or dict
        """
        new_panel = Panel()
        if isinstance(panel_definition, str):
            self._check_panel(panel_definition=panel_definition)
            new_panel.create_from_excel(path=panel_definition)
        elif isinstance(panel_definition, dict):
            new_panel.create_from_dict(panel_definition)
        else:
            raise ValueError("panel_definition should be type string or dict")
        self.panel = new_panel

    def delete_all_populations(self, sample_id: str) -> None:
        """
        Delete population data associated to experiment. Give a value of 'all' for sample_id to remove all population
        data for every sample.

        Parameters
        ----------
        sample_id: str
            Name of sample to remove populations from'; give a value of 'all'
            for sample_id to remove all population data for every sample.

        Returns
        -------
        None
        """
        for f in self.fcs_files:
            if sample_id == "all" or f.primary_id == sample_id:
                logger.info(f"Deleting all populations from FileGroup {sample_id}; {f.id}")
                f.populations = [p for p in f.populations if p.population_name == "root"]
                f.save()

    def sample_exists(self, sample_id: str) -> bool:
        """
        Returns True if the given sample_id exists in Experiment

        Parameters
        ----------
        sample_id: str
            Name of sample to search for

        Returns
        --------
        bool
            True if exists, else False
        """
        if sample_id not in list(self.list_samples()):
            return False
        return True

    def get_sample(self, sample_id: str) -> FileGroup:
        """
        Given a sample ID, return the corresponding FileGroup object

        Parameters
        ----------
        sample_id: str
            Sample ID for search

        Returns
        --------
        FileGroup

        Raises
        ------
        MissingSampleError
            If requested sample is not found in the experiment
        """
        logger.debug(f"Attempting to fetch FileGroup {sample_id}")
        if not self.sample_exists(sample_id):
            raise MissingSampleError(f"Invalid sample: {sample_id} not associated with this experiment")
        return [f for f in self.fcs_files if f.primary_id == sample_id][0]

    def filter_samples_by_subject(self, query: Union[str, mongoengine.queryset.visitor.Q]) -> List:
        """
        Filter FileGroups associated to this experiment based on some subject meta-data

        Parameters
        ----------
        query: str or mongoengine.queryset.visitor.Q
            Query to make on Subject

        Returns
        -------
        List
        """
        logger.debug(f"Fetching list of FileGroups associated to Subject on query {query}")
        matches = list()
        for f in self.fcs_files:
            try:
                Subject.objects(id=f.subject.id).filter(query).get()
                matches.append(f.primary_id)
            except mongoengine.DoesNotExist:
                logger.debug(f"No subject associated to {f.primary_id}; {f.id}")
                continue
            except mongoengine.MultipleObjectsReturned:
                logger.debug(f"Multiple matches to subject meta data for {f.primary_id}; {f.id}")
                matches.append(f.primary_id)
        return matches

    def list_samples(self, valid_only: bool = True) -> list:
        """
        Generate a list IDs of file groups associated to experiment

        Parameters
        -----------
        valid_only: bool
            If True, returns only valid samples (samples without 'invalid' flag)

        Returns
        --------
        List
            List of IDs of file groups associated to experiment
        """
        if valid_only:
            return [f.primary_id for f in self.fcs_files if f.valid]
        return [f.primary_id for f in self.fcs_files]

    def remove_sample(self, sample_id: str):
        """
        Remove sample (FileGroup) from experiment.

        Parameters
        -----------
        sample_id: str
            ID of sample to remove

        Returns
        --------
        None
        """
        logger.debug(f"Deleting {sample_id}")
        filegrp = self.get_sample(sample_id)
        self.fcs_files = [f for f in self.fcs_files if f.primary_id != sample_id]
        filegrp.delete()
        self.save()

    def _sample_exists(self, sample_id: str):
        if self.sample_exists(sample_id):
            raise DuplicateSampleError(f"A file group with id {sample_id} already exists")

    def _add_data(
        self,
        sample_id: str,
        compensated: bool,
        primary_data: Union[pd.DataFrame, FCSFile],
        mappings: Optional[List[Dict]] = None,
        controls: Optional[Dict] = None,
        subject_id: Optional[str] = None,
        processing_datetime: Optional[str] = None,
        collection_datetime: Optional[str] = None,
        missing_error: str = "raise",
    ) -> None:
        """
        Create a new FileGroup and associate with this experiment. This is an internal method
        called from add_dataframes or add_fcs_files. Expects data in the form of a DataFrame or
        FCSFile object.

        Parameters
        ----------
        sample_id: str
        compensated: bool
            Specifies if data has been compensated
        primary_data: Pandas.DataFrame or FCSFile
            The data attaining to the primary staining. Should either be a processed DataFrame (in which
            case 'mappings' should be provided) or an FCSFile object.
        mappings: List[Dict], optional
            List of mappings with keys 'channel' and 'marker'. Only required if passing DataFrame to primary
            data.
        controls: Dict, optional
            Dictionary of DataFrames or FCSFile objects - should correspond to control staining where the
            keys of the dictionaries are the control IDs
        subject_id: str, optional
            Subject to associate FileGroup to
        missing_error: str, (default="raise")
            How to handle missing channels (channels present in the experiment staining panel but
            absent from mappings). Should either be "raise" (raises an error) or "warn".
        processing_datetime: str, optional
        collection_datetime: str, optional

        Returns
        -------
        None
        """
        self._sample_exists(sample_id)
        processing_datetime = processing_datetime or datetime.now()
        collection_datetime = collection_datetime or datetime.now()
        controls = controls or {}

        try:
            mappings = mappings or primary_data.channel_mappings
            mappings = self._standardise_mappings(mappings, missing_error=missing_error)
        except ValueError as err:
            logger.exception(f"Failed to add {sample_id}: {err}")
            del primary_data
            del controls
            gc.collect()
            return

        logger.info("Adding primary data...")

        if isinstance(primary_data, pd.DataFrame):
            data = primary_data.values
        else:
            data = primary_data.event_data

        filegrp = FileGroup(
            data=data,
            channels=[x.get("channel") for x in mappings],
            markers=[x.get("marker") for x in mappings],
            primary_id=sample_id,
            compensated=compensated,
            collection_datetime=collection_datetime,
            processing_datetime=processing_datetime,
            data_directory=self.data_directory,
        )

        for ctrl_id, ctrl_data in controls.items():
            logger.info(f"Adding control file {ctrl_id}...")
            if isinstance(primary_data, pd.DataFrame):
                data = ctrl_data.values
            else:
                data = ctrl_data.event_data
            filegrp.add_ctrl_file(data=data, ctrl_id=ctrl_id)

        if subject_id is not None:
            try:
                filegrp.subject = Subject.objects(subject_id=subject_id).get()
            except mongoengine.errors.DoesNotExist:
                logger.warning(f"Error: no such patient {subject_id}, continuing without association.")
        filegrp.save()

        logger.info(f"Successfully created {sample_id} and associated to {self.experiment_id}")
        self.fcs_files.append(filegrp)
        self.save()
        del filegrp
        gc.collect()

    @panel_defined
    def add_dataframes(
        self,
        sample_id: str,
        primary_data: pd.DataFrame,
        mappings: List[Dict],
        controls: Optional[Dict] = None,
        comp_matrix: Optional[pd.DataFrame] = None,
        subject_id: Optional[str] = None,
        processing_datetime: Optional[str] = None,
        collection_datetime: Optional[str] = None,
        missing_error: str = "raise",
    ) -> None:
        """
        Add new single cell cytometry data to the experiment, under a new sample ID, using
        Pandas DataFrame(s) as the input; generates a new FileGroup associated to this experiment.
        The user must also provide the channel/marker mappings as a list of dictionary objects
        {"channel": <channel name>, "marker": <marker name>} which should match what is expected
        given the staining panel associated to this experiment.
        NOTE: the order in which this dictionaries are provided is assumed to match the order
        of the columns in the provided DataFrame(s).

        Parameters
        ----------
        sample_id: str
            Unique sample identifier (unique to this Experiment)
        primary_data: Pandas.DataFrame
            Single cell cytometry data for primary staining
        mappings: list
            List of dictionaries like so: {"channel": <channel name>, "marker": <marker name>}
        controls: dict, optional
            Dictionary of DataFrames(s) for single cell cytometry data for control staining e.g.
            FMOs or isotype controls
        comp_matrix: Pandas.DataFrame, optional
            Spill over matrix for compensation (if not provided, data is assumed to be compensated previously)
        subject_id: str, optional
            If a string value is provided, newly generated sample will be associated to this subject
        processing_datetime: str, optional
            Optional processing datetime string
        collection_datetime: str, optional
            Optional collection datetime string
        missing_error: str, (default="raise")
            How to handle missing channels (channels present in the experiment staining panel but
            absent from mappings). Should either be "raise" (raises an error) or "warn".

        Returns
        -------
        None

        Raises
        ------
        DuplicateSampleError
            If a FileGroup with the sample ID already exists
        """
        logger.info(
            f"Creating new FileGroup {sample_id} and adding to experiment {self.experiment_id} "
            f"using Pandas DataFrame(s)"
        )
        compensated = False
        if comp_matrix is not None:
            logger.info("Applying compensation...")
            primary_data = compensate_data(primary_data.values, comp_matrix.values)
            controls = {
                ctrl_id: compensate_data(ctrl_data, comp_matrix.values) for ctrl_id, ctrl_data in controls.items()
            }
            compensated = True
        self._add_data(
            sample_id=sample_id,
            compensated=compensated,
            primary_data=primary_data,
            mappings=mappings,
            controls=controls,
            subject_id=subject_id,
            processing_datetime=processing_datetime,
            collection_datetime=collection_datetime,
            missing_error=missing_error,
        )

    @panel_defined
    def add_fcs_files(
        self,
        sample_id: str,
        primary_data: Union[str, FCSFile],
        controls: Optional[Dict] = None,
        compensate: bool = True,
        comp_matrix: Optional[pd.DataFrame] = None,
        subject_id: Optional[str] = None,
        processing_datetime: Optional[str] = None,
        collection_datetime: Optional[str] = None,
        missing_error: str = "raise",
    ):
        """
        Add new single cell cytometry data to the experiment, under a new sample ID, using
        filepath to fcs file(s) as the input; generates a new FileGroup associated to this experiment.
        Alternatively, the user can also provide FCSFile object(s)

        Parameters
        ----------
        sample_id: str
            Unique sample identifier (unique to this Experiment)
        primary_data: str or FCSFile
            Single cell cytometry data for primary staining
        controls: dict, optional
            Dictionary of filepaths/FCSFiles for single cell cytometry data for control staining e.g.
            FMOs or isotype controls
        compensate: bool (default=True)
            If True, FCSFile will be searched for spillover matrix to apply to compensate data. If
            a spillover matrix has not been linked to the file, the filepath to a csv file containing
            the spillover matrix should be provided to 'comp_matrix'
        comp_matrix: str, optional
            Path to csv file containing spill over matrix for compensation
        subject_id: str, optional
            If a string value is provided, newly generated sample will be associated to this subject
        processing_datetime: str, optional
            Optional processing datetime string
        collection_datetime: str, optional
            Optional collection datetime string
        missing_error: str, (default="raise")
            How to handle missing channels (channels present in the experiment staining panel but
            absent from mappings). Should either be "raise" (raises an error) or "warn".

        Returns
        -------
        None

        Raises
        ------
        DuplicateSampleError
            If a FileGroup with the sample ID already exists

        AssertionError
            Raised on failure to standardise mappings using panel definition
        """
        logger.info(
            f"Creating new FileGroup {sample_id} and adding to experiment {self.experiment_id} using an FCS file"
        )

        if isinstance(primary_data, str):
            primary_data = FCSFile(filepath=primary_data, comp_matrix=comp_matrix)
        else:
            primary_data = primary_data

        compensated = False
        if compensate:
            compensated = True
            logger.info("Compensating primary file...")
            primary_data.compensate()

        control_data = {}
        for ctrl_id, ctrl in controls.items():
            if isinstance(ctrl, str):
                control_data[ctrl_id] = FCSFile(filepath=ctrl, comp_matrix=comp_matrix)
            else:
                control_data[ctrl_id] = ctrl

            if compensate:
                control_data[ctrl_id].compensate()

        self._add_data(
            sample_id=sample_id,
            compensated=compensated,
            primary_data=primary_data,
            mappings=None,
            controls=control_data,
            subject_id=subject_id,
            processing_datetime=processing_datetime,
            collection_datetime=collection_datetime,
            missing_error=missing_error,
        )

    def _standardise_mappings(self, mappings: List[Dict], missing_error: str) -> List[Dict]:
        """
        Given some mappings (list of dictionaries with keys: channel, marker) compare the
        mappings to the Experiment Panel. Returns the standardised mappings.

        Parameters
        ----------
        mappings: List[Dict]
        missing_error: str

        Returns
        -------
        List[Dict]

        Raises
        ------
        ValueError
            Channel/marker does not match panel mappings
        """
        logger.debug(f"Standardising channel/marker mappings {mappings}")
        mappings = list(
            map(
                lambda x: standardise_names(
                    channel_marker=x,
                    ref_channels=self.panel.channels,
                    ref_markers=self.panel.markers,
                    ref_mappings=self.panel.mappings,
                ),
                mappings,
            )
        )
        for cm in mappings:
            if not check_pairing(ref_mappings=self.panel.mappings, channel_marker=cm):
                raise ValueError(f"The channel/marker pairing {cm} does not correspond to any defined in panel")
        missing_channels(mappings=mappings, channels=self.panel.channels, errors=missing_error)
        duplicate_mappings(mappings)
        return mappings

    def control_counts(self, ax: Optional[plt.Axes] = None) -> plt.Axes:
        """
        Generates a barplot of total counts of each control in Experiment FileGroup's

        Parameters
        ----------
        ax: Matplotlib.Axes, optional

        Returns
        -------
        Matplotlib.Axes
        """
        ctrls = [f.controls for f in self.fcs_files]
        ctrl_counts = Counter([x for sl in ctrls for x in sl])
        ctrl_counts["Total"] = len(self.fcs_files)
        ax = ax or plt.subplots(figsize=(6, 6))[1]
        ax.bar(ctrl_counts.keys(), ctrl_counts.values())
        return ax

    def population_statistics(self, populations: Union[List, None] = None) -> pd.DataFrame:
        """
        Generates a Pandas DataFrame of population statistics for all FileGroups
        of an Experiment, for the given populations or all available populations
        if 'populations' is None.

        Parameters
        ----------
        populations: list, optional

        Returns
        -------
        Pandas.DataFrame
        """
        data = list()
        for f in self.fcs_files:
            for p in populations or f.list_populations():
                df = pd.DataFrame({k: [v] for k, v in f.population_stats(population=p).items()})
                df["sample_id"] = f.primary_id
                s = f.subject
                if s is not None:
                    df["subject_id"] = s.subject_id
                data.append(df)
        return pd.concat(data).reset_index(drop=True)

    def merge_populations(self, mergers: Dict):
        """
        For each FileGroup in sequence, merge populations. Given dictionary should contain
        a key corresponding to the new population name and value being a list of populations
        to merge. If one or more populations are missing, then available populations will be
        merged.

        Parameters
        ----------
        mergers: dict

        Returns
        -------
        None
        """
        logger.info(f"Merging populations: {mergers}")
        for new_population_name, targets in mergers.items():
            for f in self.fcs_files:
                pops = [p for p in targets if p in f.list_populations()]
                try:
                    f.merge_non_geom_populations(populations=pops, new_population_name=new_population_name)
                    f.save()
                except ValueError as e:
                    logger.warning(f"Failed to merge populations for {f.primary_id}: {str(e)}")

    def delete(self, signal_kwargs=None, **write_concern):
        """
        Delete Experiment; will delete all associated FileGroups.

        Returns
        -------
        None
        """
        logger.info(f"Attempting to delete experiment {self.experiment_id}")
        for f in self.fcs_files:
            logger.debug(f"deleting associated FileGroup {f.primary_id}")
            f.delete()
        self.save()
        super().delete(signal_kwargs=signal_kwargs, **write_concern)
        logger.info("Experiment successfully deleted.")


def single_cell_dataframe(
    experiment: Experiment,
    populations: Optional[Union[str, List[str]]] = None,
    regex: Optional[str] = None,
    transform: Union[str, Dict] = "logicle",
    transform_kwargs: Optional[Dict] = None,
    sample_ids: Optional[List[str]] = None,
    verbose: bool = True,
    ctrl: Optional[str] = None,
    label_parent: bool = False,
    frac_of: Optional[List[str]] = None,
    sample_size: Optional[Union[int, float]] = None,
    sampling_level: str = "file",
    sampling_method: str = "uniform",
    sampling_kwargs: Optional[Dict] = None,
):
    """
    Generate a single cell DataFrame that is a concatenation of population data from many
    samples from a single Experiment. Population level data is identifiable from the 'population_label'
    column, sample level data identifiable from the 'sample_id' column, and subject level information
    from the 'subject_id' column.

    Parameters
    ----------
    experiment: Experiment
    populations: list or str, optional
        * Single string value will load the matching population from samples in 'experiment'
        * List of strings will load the matching populations from samples in 'experiment'
        * None, to provide a regular expression (regex) for population matching
    regex: str, optional
        Match all populations matching the given pattern; if given, populations argument is ignored
    transform: str or dict (default='logicle')
        Transformation applied to the single cell data. If a string is provided, method is applied to
        all features. If a dictionary is provided, keys are interpreted as names of features and values
        the transform to be applied.
    transform_kwargs: dict, optional
        Additional keyword arguments passed to transform method
    sample_ids: list, optional
        List of samples to include. If None (default) then loads all available samples in experiment
    verbose: bool (default=True)
    ctrl: str, optional
        Loads data corresponding to the given control. NOTE: only supports loading of a single population
        from each sample in 'experiment'
    label_parent: bool (default=False)
        If True, additional column appended with parent name for each population
    frac_of: list, optional
        Provide a list of populations and additional columns will be appended to resulting
        DataFrame containing the fraction of the requested population compared to each population
        in this list
    sample_size: int or float, optional
        If given, the DataFrame will either be downsampled after aquiring data from each FileGroup
        or FileGroups are sampled individually - this behaviour is controlled by 'sampling_level'.
        If sampling_level = "file", then the sample_size is the number of events to obtain from each
        FileGroup. If sampling_level = "experiment", then the sampling size is the desired size of the
        resulting concatenated DataFrame.
    sampling_level: str, (default="file")
        If "file" (default) then each FileGroup is sampled before concatenating into a single DataFrame.
        If "experiment", then data is obtained from each FileGroup first, and then the concatenated
        data is sampled.
        If "population" then will attempt to sample the desired number of events from each population.
    sampling_method: str (default="uniform")
        The sampling method to use; see cytopy.flow.sampling
    sampling_kwargs: Dict, optional
        Additional keyword arguments passed to sampling method

    Returns
    -------
    Pandas.DataFrame
    """
    logger.debug(f"Loading data from {experiment.experiment_id}")
    sample_ids = sample_ids or list(experiment.list_samples())
    sampling_kwargs = sampling_kwargs or {}
    data = list()

    method = "load_population_df"
    kwargs = dict(
        population=populations,
        transform=transform,
        transform_kwargs=transform_kwargs,
        label_parent=label_parent,
        frac_of=frac_of,
    )

    if sample_size is not None and sampling_level == "file":
        kwargs = {**kwargs, **{"sample_size": sample_size, "sampling_method": sampling_method, **sampling_kwargs}}

    if isinstance(populations, list) or regex is not None:
        method = "load_multiple_populations"
        kwargs["sample_at_population_level"] = sampling_level == "population"
        kwargs["regex"] = regex
        kwargs["populations"] = populations
        kwargs.pop("population")
        if ctrl:
            raise ValueError(
                "load_multiple_populations does not support control data. Load ctrl populations "
                "individually and merge post-hoc."
            )
    elif ctrl:
        method = "load_ctrl_population_df"
        kwargs.pop("label_parent")
        kwargs.pop("frac_of")
        kwargs["ctrl"] = ctrl

    for _id in progress_bar(sample_ids, verbose=verbose):
        fg = experiment.get_sample(sample_id=_id)
        logger.debug(f"Loading FileGroup data from {_id}; {fg.id}")
        pop_data = getattr(fg, method)(**kwargs)
        pop_data["sample_id"] = _id
        pop_data["subject_id"] = None
        if fg.subject:
            pop_data["subject_id"] = fg.subject.subject_id
        data.append(pop_data)

    data = pd.concat([df.reset_index().rename({"index": "original_index"}, axis=1) for df in data]).reset_index(
        drop=True
    )
    data.index = list(data.index)

    if sample_size is not None and sampling_level == "experiment":
        data = sample_dataframe(
            data=data,
            sample_size=sample_size,
            method=sampling_method,
            **sampling_kwargs,
        ).reset_index(drop=True)
    return data
