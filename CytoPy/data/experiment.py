#!/usr/bin.env/python
# -*- coding: utf-8 -*-
"""
The experiment module houses the Experiment class, used to define
cytometry based experiments that can consist of one or more biological
specimens. An experiment should be defined for each cytometry staining
panel used in your analysis and the single cell data (contained in
*.fcs files) added to the experiment using the 'add_new_sample' method.
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

from ..feedback import vprint, progress_bar
from .fcs import FileGroup
from .subject import Subject
from .read_write import FCSFile
from .mapping import ChannelMap
from typing import List
from collections import Counter
from datetime import datetime
from warnings import warn
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import mongoengine
import shutil
import xlrd
import os
import re
import gc

__author__ = "Ross Burton"
__copyright__ = "Copyright 2020, CytoPy"
__credits__ = ["Ross Burton", "Simone Cuff", "Andreas Artemiou", "Matthias Eberl"]
__license__ = "MIT"
__version__ = "1.0.0"
__maintainer__ = "Ross Burton"
__email__ = "burtonrj@cardiff.ac.uk"
__status__ = "Production"


def _check_sheet_names(path: str) -> (pd.DataFrame, pd.DataFrame):
    """
    Check sheet names are as expected. That is: nomenclature and mappings.
    Return Pandas DataFrame for each sheet.

    Parameters
    ----------
    path: str
        Path to excel file

    Returns
    -------
    Pandas.DataFrame, Pandas.DataFrame
    """
    # Check sheet names
    xls = xlrd.open_workbook(path, on_demand=True)
    err = f"Template must contain two sheets: nomenclature and mappings"
    assert all([x in ['nomenclature', 'mappings'] for x in xls.sheet_names()]), err
    nomenclature = pd.read_excel(path, sheet_name='nomenclature')
    mappings = pd.read_excel(path, sheet_name='mappings')
    return mappings, nomenclature


def _check_nomenclature_headings(nomenclature: pd.DataFrame):
    """
    Raise AssertionError if columns in nomenclature DataFrame are invalid.

    Parameters
    ----------
    nomenclature: Pandas.DataFrame

    Returns
    -------
    None
    """
    err = "Nomenclature sheet of excel template must contain the following column headers: " \
          "'name','regex','case','permutations'"
    assert all([x in ['name', 'regex', 'permutations', 'case'] for x in nomenclature.columns]), err


def _check_mappings_headings(mappings: pd.DataFrame):
    """
    Raise AssertionError if columns in mappings DataFrame are invalid.

    Parameters
    ----------
    mappings: Pandas.DataFrame

    Returns
    -------
    None
    """
    err = "Mappings sheet of excel template must contain the following column headers: 'channel', 'marker'"
    assert all([x in ['channel', 'marker'] for x in mappings.columns]), err


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
    """
    mappings, nomenclature = _check_sheet_names(path)
    _check_nomenclature_headings(nomenclature)
    _check_mappings_headings(mappings)
    # Check for duplicate entries
    err = 'Duplicate entries in nomenclature, please remove duplicates before continuing'
    assert sum(nomenclature['name'].duplicated()) == 0, err
    # Check that all mappings have a corresponding entry in nomenclature
    for x in ['channel', 'marker']:
        for name in mappings[x]:
            if pd.isnull(name):
                continue
            assert name in nomenclature.name.values, f'{name} missing from nomenclature, please review template'
    return nomenclature, mappings


def check_duplication(x: list) -> bool:
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
        warn(f'Duplicate channel/markers identified: {duplicates}')
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

    def query(self, x: str) -> None or str:
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
        if self.case_sensitive:
            if re.search(self.regex_str, x):
                return self.standard
            return None
        if re.search(self.regex_str, x, re.IGNORECASE):
            return self.standard
        if self.permutations:
            for p in self.permutations.split(','):
                if x == p:
                    return self.standard
        return None


def query_normalised_list(x: str or None,
                          ref: List[NormalisedName]) -> str:
    """
    Internal method for querying a channel/marker against a reference list of
    NormalisedName's

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
    """
    corrected = list(filter(None.__ne__, [n.query(x) for n in ref]))
    assert len(corrected) != 0, f'Unable to normalise {x}; no match in linked panel'
    err = f'Unable to normalise {x}; matched multiple in linked panel, check ' \
          f'panel for incorrect definitions. Matches found: {corrected}'
    assert len(corrected) < 2, err
    return corrected[0]


def _is_empty(x: str):
    if x.isspace():
        return None
    if x == "":
        return None
    return x


def check_pairing(channel_marker: dict,
                  ref_mappings: List[ChannelMap]) -> bool:
    """
    Internal method. Given a channel and marker check that a valid pairing exists in the list
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
    channel, marker = _is_empty(channel_marker.get("channel")), _is_empty(channel_marker.get("marker"))
    if not any([n.check_matched_pair(channel=channel, marker=marker) for n in ref_mappings]):
        return False
    return True


def _standardise(x: str or None,
                 ref: List[NormalisedName],
                 mappings: List[ChannelMap],
                 alt: str):
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
    if x is not None:
        return query_normalised_list(x, ref)
    default = [m for m in mappings if m.channel == alt or m.marker == alt][0]
    if default.channel == alt:
        return default.marker
    return default.channel


def standardise_names(channel_marker: dict,
                      ref_channels: List[NormalisedName],
                      ref_markers: List[NormalisedName],
                      ref_mappings: List[ChannelMap]):
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
    """
    channel, marker = _is_empty(channel_marker.get("channel")), _is_empty(channel_marker.get("marker"))
    if channel is None and marker is None:
        raise ValueError("Cannot standardise column names because both channel and marker missing from mappings")
    channel = _standardise(channel, ref_channels, ref_mappings, marker)
    marker = _standardise(marker, ref_markers, ref_mappings, channel)
    return {"channel": channel, "marker": marker}


def duplicate_mappings(mappings: List[dict]):
    """
    Check for duplicates in a list of dictionaries describing channel/marker mappings.
    Raise AssertionError if duplicates found.

    Parameters
    ----------
    mappings: list

    Returns
    -------
    None
    """
    channels = [x.get("channel") for x in mappings]
    assert not check_duplication(channels), "Duplicate channels provided"
    markers = [x.get("marker") for x in mappings]
    assert not check_duplication(markers), "Duplicate markers provided"


def missing_channels(mappings: List[dict],
                     channels: List[NormalisedName],
                     errors: str = "raise"):
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
    """
    existing_channels = [x.get("channel") for x in mappings]
    for x in channels:
        if x.standard not in existing_channels:
            if errors == "raise":
                raise KeyError(f"Missing channel {x.standard}")
            elif errors == "warn":
                warn(f"Missing channel {x.standard}")


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
        date of creationfiles['controls']

    """
    markers = mongoengine.EmbeddedDocumentListField(NormalisedName)
    channels = mongoengine.EmbeddedDocumentListField(NormalisedName)
    mappings = mongoengine.EmbeddedDocumentListField(ChannelMap)
    initiation_date = mongoengine.DateTimeField(default=datetime.now)
    meta = {
        'db_alias': 'core',
        'collection': 'fcs_panels'
    }

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
        """
        assert os.path.isfile(path), f'Error: no such file {path}'
        nomenclature, mappings = check_excel_template(path)
        for col_name, attr in zip(['channel', 'marker'], [self.channels, self.markers]):
            for name in mappings[col_name]:
                if not pd.isnull(name):
                    d = nomenclature[nomenclature['name'] == name].fillna('').to_dict(orient='list')
                    attr.append(NormalisedName(standard=d['name'][0],
                                               regex_str=d['regex'][0],
                                               case_sensitive=d['case'][0],
                                               permutations=d['permutations'][0]))
        mappings = mappings.fillna('').to_dict(orient='list')
        self.mappings = [ChannelMap(channel=c, marker=m)
                         for c, m in zip(mappings['channel'], mappings['marker'])]

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
        """

        # Check validity of input dictionary
        err = 'Invalid template dictionary; must be a nested dictionary with parent keys: channels, markers, & mappings'
        assert all([k in ['channels', 'markers', 'mappings'] for k in x.keys()]), err

        assert isinstance(x['mappings'], list), 'Invalid template dictionary; mappings must be a list of tuples'
        err = 'Invalid template dictionary; mappings should be of shape (n,2) where n is the number of ' \
              'channel/marker pairs'
        assert all([len(i) == 2 for i in x['mappings']]), err
        self.markers = [NormalisedName(standard=k['name'],
                                       regex_str=k['regex'],
                                       case_sensitive=k['case'],
                                       permutations=k['permutations'])
                        for k in x['markers']]
        self.channels = [NormalisedName(standard=k['name'],
                                        regex_str=k['regex'],
                                        case_sensitive=k['case'],
                                        permutations=k['permutations'])
                         for k in x['channels']]
        self.mappings = [ChannelMap(channel=c, marker=m) for c, m in x['mappings']]

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


def data_dir_append_leading_char(path: str):
    """
    Format a file path to handle Win and Unix OS

    Parameters
    ----------
    path: str

    Returns
    -------
    str
    """
    leading_char = path[len(path) - 1]
    if leading_char not in ["\\", "/"]:
        if len(path.split("\\")) > 1:
            # Assuming a windows OS
            return path + "\\"
        else:
            # Assuming unix OS
            return path + "/"
    return path


def compenstate(x: np.ndarray,
                spill_matrix: np.ndarray):
    return np.linalg.solve(spill_matrix.T, x.T).T


class Experiment(mongoengine.Document):
    """
    Container for Cytometry experiment. The correct way to generate and load these objects is using the
    Project.add_experiment method (see CytoPy.data.project.Project). This object provides access
    to all experiment-wide functionality. New files can be added to an experiment using the
    add_new_sample method.

    Attributes
    -----------
    experiment_id: str, required
        Unique identifier for experiment
    panel: ReferenceField, required
        Panel object describing associated channel/marker pairs
    data_directory: str
        Address to drive for storage of single cell data
    fcs_files: ListField
        Reference field for associated files
    flags: str, optional
        Warnings associated to experiment
    notes: str, optional
        Additional free text comments
    gating_templates: ListField
        Reference to gating templates associated to this experiment
    """
    experiment_id = mongoengine.StringField(required=True, unique=True)
    data_directory = mongoengine.StringField(required=True)
    panel = mongoengine.EmbeddedDocumentField(Panel)
    fcs_files = mongoengine.ListField(mongoengine.ReferenceField(FileGroup, reverse_delete_rule=mongoengine.PULL))
    flags = mongoengine.StringField(required=False)
    notes = mongoengine.StringField(required=False)
    meta = {
        'db_alias': 'core',
        'collection': 'experiments'
    }

    def __init__(self, *args, **kwargs):
        panel_definition = kwargs.pop("panel_definition", None)
        super().__init__(*args, **kwargs)
        if self.data_directory:
            assert os.path.isdir(self.data_directory), f"data directory {self.data_directory} does not exist"
            self.data_directory = data_dir_append_leading_char(self.data_directory)
        else:
            raise ValueError("No data directory provided")
        if self.panel is None:
            assert panel_definition is not None, "No panel associated to this experiment, please provide a " \
                                                 "panel definition"
            self.panel = self.generate_panel(panel_definition=panel_definition)

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
            Raises AssertionError in the condition that the given parameters are invalid
        """
        assert os.path.isfile(panel_definition), f"{panel_definition} does not exist"
        err = "Panel definition is not a valid Excel document"
        assert os.path.splitext(panel_definition)[1] in [".xls", ".xlsx"], err

    def generate_panel(self,
                       panel_definition: str or dict):
        """
        Associate a panel to this Experiment, either by fetching an existing panel using the
        given panel name or by generating a new panel using the panel definition provided (path to a valid template).

        Parameters
        ----------
        panel_definition: str
            Path to a panel definition

        Returns
        -------
        Panel
        """
        new_panel = Panel()
        if isinstance(panel_definition, str):
            self._check_panel(panel_definition=panel_definition)
            new_panel.create_from_excel(path=panel_definition)
        elif isinstance(panel_definition, dict):
            new_panel.create_from_dict(panel_definition)
        else:
            raise ValueError("panel_definition should be type string or dict")
        return new_panel

    def update_data_directory(self,
                              new_path: str,
                              move: bool = True):
        """
        Update the data directory associated to this experiment. This will propagate to all
        associated FileGroup's. WARNING: this function will move the existing data directory
        and all of it's contents to the new given path.

        Parameters
        ----------
        new_path: str
        move: bool (default=True)
            If True, the data is assumed to be present at the old path and will be
            moved over to the new path by CytoPy

        Returns
        -------
        None
        """
        assert os.path.isdir(new_path), "Invalid directory given for new_path"
        for file in self.fcs_files:
            file.data_directory = new_path
            file.h5path = os.path.join(new_path, f"{file.id.__str__()}.hdf5")
            if move:
                shutil.move(f"{self.data_directory}{file.id}.hdf5", f"{new_path}/{file.id}.hdf5")
            file.save()
        self.data_directory = new_path
        shutil.move(self.data_directory, new_path)
        self.save()

    def delete_all_populations(self,
                               sample_id: str) -> None:
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
            if sample_id == 'all' or f.primary_id == sample_id:
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

    def get_sample(self,
                   sample_id: str) -> FileGroup:
        """
        Given a sample ID, return the corresponding FileGroup object

        Parameters
        ----------
        sample_id: str
            Sample ID for search

        Returns
        --------
        FileGroup
        """
        assert self.sample_exists(sample_id), f"Invalid sample: {sample_id} not associated with this experiment"
        return [f for f in self.fcs_files if f.primary_id == sample_id][0]

    def filter_subjects(self,
                        key: str or list,
                        value: str or int or float):
        matches = list()
        if isinstance(key, list) and len(key) == 1:
            key = key[0]
        if isinstance(key, str):
            for f in self.fcs_files:
                meta_var = fetch_subject_meta(sample_id=f.primary_id,
                                              experiment=self,
                                              meta_label=key)
                if meta_var == value:
                    matches.append(f.primary_id)
            return matches
        elif isinstance(key, list):
            for f in self.fcs_files:
                starting_node = fetch_subject_meta(sample_id=f.primary_id,
                                                   experiment=self,
                                                   meta_label=key[0])
                if key[1] not in starting_node.keys():
                    continue
                elif len(key) == 2:
                    if starting_node[key[1]] == value:
                        matches.append(f.primary_id)
                    continue
                else:
                    node = starting_node[key[1]]
                    for k in key[2:]:
                        if k in node.keys():
                            node = node[k]
                    if node == value:
                        matches.append(f.primary_id)
        return matches

    def list_samples(self,
                     valid_only: bool = True) -> list:
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
        filegrp = self.get_sample(sample_id)
        self.fcs_files = [f for f in self.fcs_files if f.primary_id != sample_id]
        filegrp.delete()
        self.save()

    def add_dataframes(self,
                       sample_id: str,
                       primary_data: pd.DataFrame,
                       mappings: list,
                       controls: dict or None = None,
                       comp_matrix: pd.DataFrame or None = None,
                       subject_id: str or None = None,
                       verbose: bool = True,
                       processing_datetime: str or None = None,
                       collection_datetime: str or None = None,
                       missing_error: str = "raise"):
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
        verbose: bool (default=True)
            If True, progress printed to stdout
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
        """
        processing_datetime = processing_datetime or datetime.now()
        collection_datetime = collection_datetime or datetime.now()
        controls = controls or {}
        feedback = vprint(verbose)
        assert not self.sample_exists(sample_id), f'A file group with id {sample_id} already exists'
        feedback("Loading data from csv files...")
        compensated = False
        if comp_matrix is not None:
            feedback("Applying compensation...")
            primary_data = compenstate(primary_data.values, comp_matrix.values)
            controls = {ctrl_id: compenstate(ctrl_data, comp_matrix) for ctrl_id, ctrl_data in controls.items()}
            compensated = True

        try:
            feedback("Checking channel/marker mappings...")
            mappings = self._standardise_mappings(mappings,
                                                  missing_error=missing_error)
        except AssertionError as err:
            warn(f"Failed to add {sample_id}: {str(err)}")
            del primary_data
            del controls
            gc.collect()
            return

        filegrp = FileGroup(primary_id=sample_id,
                            data_directory=self.data_directory,
                            compensated=compensated,
                            collection_datetime=collection_datetime,
                            processing_datetime=processing_datetime,
                            data=primary_data.values,
                            channels=[x.get("channel") for x in mappings],
                            markers=[x.get("marker") for x in mappings])
        for ctrl_id, ctrl_data in controls.items():
            feedback(f"Adding control file {ctrl_id}...")
            filegrp.add_ctrl_file(data=ctrl_data.values,
                                  ctrl_id=ctrl_id,
                                  channels=[x.get("channel") for x in mappings],
                                  markers=[x.get("marker") for x in mappings])
        if subject_id is not None:
            feedback(f"Associating to {subject_id} Subject...")
            try:
                p = Subject.objects(subject_id=subject_id).get()
                p.files.append(filegrp)
                p.save()
            except mongoengine.errors.DoesNotExist:
                warn(f'Error: no such patient {subject_id}, continuing without association.')
        feedback(f'Successfully created {sample_id} and associated to {self.experiment_id}')
        self.fcs_files.append(filegrp)
        self.save()
        del filegrp
        gc.collect()

    def add_fcs_files(self,
                      sample_id: str,
                      primary: str or FCSFile,
                      controls: dict or None = None,
                      subject_id: str or None = None,
                      comp_matrix: str or None = None,
                      compensate: bool = True,
                      verbose: bool = True,
                      processing_datetime: str or None = None,
                      collection_datetime: str or None = None,
                      missing_error: str = "raise"):
        """
        Add new single cell cytometry data to the experiment, under a new sample ID, using
        filepath to fcs file(s) as the input; generates a new FileGroup associated to this experiment.
        Alternatively, the user can also provide FCSFile object(s)

        Parameters
        ----------
        sample_id: str
            Unique sample identifier (unique to this Experiment)
        primary: str or FCSFile
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
        verbose: bool (default=True)
            If True, progress printed to stdout
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
        """
        processing_datetime = processing_datetime or datetime.now()
        collection_datetime = collection_datetime or datetime.now()
        controls = controls or {}
        feedback = vprint(verbose)
        assert not self.sample_exists(sample_id), f'A file group with id {sample_id} already exists'
        feedback("Creating new FileGroup...")
        if isinstance(primary, str):
            fcs_file = FCSFile(filepath=primary, comp_matrix=comp_matrix)
        else:
            fcs_file = primary

        if compensate:
            feedback("Compensating primary file...")
            fcs_file.compensate()
        feedback("Checking channel/marker mappings...")
        mappings = self._standardise_mappings(fcs_file.channel_mappings,
                                              missing_error=missing_error)

        filegrp = FileGroup(primary_id=sample_id,
                            data_directory=self.data_directory,
                            compensated=compensate,
                            collection_datetime=collection_datetime,
                            processing_datetime=processing_datetime,
                            data=fcs_file.event_data,
                            channels=[x.get("channel") for x in mappings],
                            markers=[x.get("marker") for x in mappings])
        for ctrl_id, path in controls.items():
            feedback(f"Adding control file {ctrl_id}...")
            if isinstance(path, str):
                fcs_file = FCSFile(filepath=path, comp_matrix=comp_matrix)
            else:
                fcs_file = path
            if compensate:
                feedback("Compensating...")
                fcs_file.compensate()
            mappings = self._standardise_mappings(fcs_file.channel_mappings,
                                                  missing_error=missing_error)
            filegrp.add_ctrl_file(data=fcs_file.event_data,
                                  ctrl_id=ctrl_id,
                                  channels=[x.get("channel") for x in mappings],
                                  markers=[x.get("marker") for x in mappings])
        if subject_id is not None:
            feedback(f"Associating too {subject_id} Subject...")
            try:
                p = Subject.objects(subject_id=subject_id).get()
                p.files.append(filegrp)
                p.save()
            except mongoengine.errors.DoesNotExist:
                warn(f'Error: no such patient {subject_id}, continuing without association.')
        feedback(f'Successfully created {sample_id} and associated to {self.experiment_id}')
        self.fcs_files.append(filegrp)
        self.save()
        del fcs_file
        del filegrp
        gc.collect()

    def _standardise_mappings(self,
                              mappings: list,
                              missing_error: str):
        """
        Given some mappings (list of dictionaries with keys: channel, marker) compare the
        mappings to the Experiment Panel. Returns the standardised mappings.

        Parameters
        ----------
        mappings: list
        missing_error: str

        Returns
        -------
        list
        """
        mappings = list(map(lambda x: standardise_names(channel_marker=x,
                                                        ref_channels=self.panel.channels,
                                                        ref_markers=self.panel.markers,
                                                        ref_mappings=self.panel.mappings),
                            mappings))
        for cm in mappings:
            err = f'The channel/marker pairing {cm} does not correspond to any defined in panel'
            assert check_pairing(ref_mappings=self.panel.mappings, channel_marker=cm), err
        missing_channels(mappings=mappings, channels=self.panel.channels, errors=missing_error)
        duplicate_mappings(mappings)
        return mappings

    def control_counts(self, ax: plt.Axes or None = None):
        ctrls = [f.controls for f in self.fcs_files]
        ctrl_counts = Counter([x for sl in ctrls for x in sl])
        ctrl_counts["Total"] = len(self.fcs_files)
        ax = ax or plt.subplots(figsize=(6, 6))[1]
        ax.bar(ctrl_counts.keys(), ctrl_counts.values())
        return ax

    def population_statistics(self,
                              populations: list or None = None):
        data = list()
        for f in self.fcs_files:
            for p in populations or f.list_populations():
                df = pd.DataFrame({k: [v] for k, v in f.population_stats(population=p).items()})
                df["sample_id"] = f.primary_id
                data.append(df)
        return pd.concat(data).reset_index(drop=True)

    def merge_populations(self,
                          mergers: dict):
        for new_population_name, targets in mergers.items():
            for f in self.fcs_files:
                pops = [p for p in targets if p in f.list_populations()]
                try:
                    f.merge_many_populations(populations=pops, new_population_name=new_population_name)
                    f.save()
                except AssertionError as e:
                    warn(f"Failed to merge populations for {f.primary_id}: {str(e)}", stacklevel=2)

    def delete(self,
               *args,
               **kwargs):
        """
        Delete Experiment.

        Parameters
        ----------
        args: list
        kwargs: dict

        Returns
        -------
        None
        """
        for f in self.fcs_files:
            f.delete()
        super().delete(*args, **kwargs)


def load_subject_id(pop_data: pd.DataFrame,
                    filegroup: FileGroup):
    """
    Given a FileGroup and a a DataFrame of population level data (where each row is a single cell),
    reverse search the Subjects to populate each row with the subject ID linked to this FileGroup.

    Parameters
    ----------
    pop_data: Pandas.DataFrame
    filegroup: FileGroup

    Returns
    -------
    Pandas.DataFrame
    """
    subject = fetch_subject(filegroup)
    if subject is not None:
        subject = subject.subject_id
    pop_data["subject_id"] = subject
    return pop_data


def load_population_data_from_experiment(experiment: Experiment,
                                         population: str,
                                         transform: str = "logicle",
                                         transform_kwargs: dict or None = None,
                                         sample_ids: list or None = None,
                                         verbose: bool = True,
                                         additional_columns: list or None = None):
    """
    Load Population from samples in the given Experiment and generate a
    standard exploration dataframe that contains the columns 'sample_id',
    'subject_id', 'meta_label' and initialises additional
    columns with null values if specified (additional_columns).


    Parameters
    ----------
    experiment: Experiment
    population: str
    transform: str
    sample_ids: list, optional
    verbose: bool (default=True)
    additional_columns: list, optional

    Returns
    -------
    Pandas.DataFrame
    """
    transform_kwargs = transform_kwargs or {}
    additional_columns = additional_columns or list()
    sample_ids = sample_ids or list(experiment.list_samples())
    population_data = list()
    for _id in progress_bar(sample_ids, verbose=verbose):
        fg = experiment.get_sample(sample_id=_id)
        pop_data = fg.load_population_df(population=population,
                                         transform=transform,
                                         transform_kwargs=transform_kwargs,
                                         label_downstream_affiliations=True)

        pop_data["sample_id"] = _id
        pop_data = load_subject_id(pop_data, fg)
        population_data.append(pop_data)
    data = pd.concat([df.reset_index().rename({"index": "original_index"}, axis=1)
                      for df in population_data]).reset_index(drop=True)
    data.index = list(data.index)
    for c in additional_columns:
        data[c] = None
    return data


def load_control_population_from_experiment(experiment: Experiment,
                                            population: str,
                                            ctrl: str,
                                            transform: str = "logicle",
                                            sample_ids: list or None = None,
                                            verbose: bool = True,
                                            additional_columns: list or None = None):
    """
    Load Population from a given control from samples in the given Experiment and generate a
    standard exploration dataframe that contains the columns 'sample_id',
    'subject_id', and initialises additional columns with null values if specified (additional_columns).


    Parameters
    ----------
    experiment: Experiment
    population: str
    ctrl: str,
    transform: str
    sample_ids: list, optional
    verbose: bool (default=True)
    additional_columns: list, optional

    Returns
    -------
    Pandas.DataFrame
    """
    additional_columns = additional_columns or list()
    sample_ids = sample_ids or list(experiment.list_samples())
    population_data = list()
    for _id in progress_bar(sample_ids, verbose=verbose):
        fg = experiment.get_sample(sample_id=_id)
        pop_data = fg.load_ctrl_population_df(population=population,
                                              transform=transform,
                                              ctrl=ctrl)
        pop_data["sample_id"] = _id
        pop_data["meta_label"] = None
        pop_data = load_subject_id(pop_data, fg)
        population_data.append(pop_data)
    data = pd.concat([df.reset_index().rename({"index": "original_index"}, axis=1)
                      for df in population_data]).reset_index(drop=True)
    data.index = list(data.index)
    for c in additional_columns:
        data[c] = None
    return data


def fetch_subject_meta(sample_id: str,
                       experiment: Experiment,
                       meta_label: str):
    """
    Fetch the Subject document through a reverse search of
    associated FileGroup and return the requested meta-label
    stored in the Subject. If no Subject is found or no
    meta-label matches the search, will return None

    Parameters
    ----------
    experiment: Experiment
        Experiment containing the FileGroup of interest
    sample_id: str
        FileGroup primary ID
    meta_label: str
        Meta variable to fetch

    Returns
    -------
    Subject or None
    """
    fg = experiment.get_sample(sample_id=sample_id)
    subject = fetch_subject(filegroup=fg)
    if subject is not None:
        return subject[meta_label]
    return None


def fetch_subject(filegroup: FileGroup):
    """
    Reverse search for Subject document using a FileGroup

    Parameters
    ----------
    filegroup: FileGroup

    Returns
    -------
    Subject or None
    """
    subject = Subject.objects(files=filegroup)
    if len(subject) != 1:
        warn(f"{filegroup.primary_id} is not associated to a Subject")
        return None
    return subject[0]


def experiment_subject_search(experiment: Experiment,
                              sample_id: str):
    f = experiment.get_sample(sample_id=sample_id)
    return fetch_subject(f)
