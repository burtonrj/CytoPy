from ..feedback import vprint
from .fcs import FileGroup
from .subject import Subject
from .read_write import FCSFile
from .mapping import ChannelMap
from typing import Generator, List, Dict
from collections import Counter
from datetime import datetime
from warnings import warn
import pandas as pd
import mongoengine
import shutil
import xlrd
import os
import re


def _check_sheet_names(path: str):
    # Check sheet names
    xls = xlrd.open_workbook(path, on_demand=True)
    err = f"Template must contain two sheets: nomenclature and mappings"
    assert all([x in ['nomenclature', 'mappings'] for x in xls.sheet_names()]), err
    nomenclature = pd.read_excel(path, sheet_name='nomenclature')
    mappings = pd.read_excel(path, sheet_name='mappings')
    return mappings, nomenclature


def _check_nomenclature_headings(nomenclature):
    err = "Nomenclature sheet of excel template must contain the following column headers: " \
          "'name','regex','case','permutations'"
    assert all([x in ['name', 'regex', 'permutations', 'case'] for x in nomenclature.columns]), err


def _check_mappings_headings(mappings):
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


def _check_duplication(x: list) -> bool:
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

    Parameters
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
        if re.search(self.regex_str, x, re.IGNORECASE):
            return self.standard
        if self.permutations:
            for p in self.permutations.split(','):
                if x == p:
                    return self.standard
        return None


def _query_normalised_list(x: str or None,
                           ref: List[NormalisedName]) -> str:
    """
    Internal static method for querying a channel/marker against a reference list of
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


def _check_pairing(channel_marker: dict,
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
    if x is not None:
        return _query_normalised_list(x, ref)
    default = [m for m in mappings if m.channel == alt or m.marker == alt][0]
    if default.channel == alt:
        return default.marker
    return default.channel


def standardise_names(channel_marker: dict,
                      ref_channels: List[NormalisedName],
                      ref_markers: List[NormalisedName],
                      ref_mappings: List[ChannelMap]):
    channel, marker = _is_empty(channel_marker.get("channel")), _is_empty(channel_marker.get("marker"))
    if channel is None and marker is None:
        raise ValueError("Cannot standardise column names because both channel and marker missing from mappings")
    channel = _standardise(channel, ref_channels, ref_mappings, marker)
    marker = _standardise(marker, ref_markers, ref_mappings, channel)
    return {"channel": channel, "marker": marker}


def _duplicate_mappings(mappings: List[dict]):
    channels = [x.get("channel") for x in mappings]
    assert not _check_duplication(channels), "Duplicate channels provided"
    markers = [x.get("marker") for x in mappings]
    assert not _check_duplication(markers), "Duplicate markers provided"


def _missing_channels(mappings: List[dict],
                      channels: List[NormalisedName],
                      errors: str = "raise"):
    existing_channels = [x.get("channel") for x in mappings]
    for x in channels:
        if x.standard not in existing_channels:
            if errors == "raise":
                raise KeyError(f"Missing channel {x.standard}")
            elif errors == "warn":
                warn(f"Missing channel {x.standard}")


class Panel(mongoengine.Document):
    """
    Document representation of channel/marker definition for an experiment. A panel, once associated to an experiment
    will standardise data upon input; when an fcs file is created in the database, it will be associated to
    an experiment and the channel/marker definitions in the fcs file will be mapped to the associated panel.

    Parameters
    -----------
    panel_name: str, required
        unique identifier for the panel
    markers: EmbeddedDocListField
        list of marker names; see NormalisedName
    channels: EmbeddedDocListField
        list of channels; see NormalisedName
    mappings: EmbeddedDocListField
        list of channel/marker mappings; see ChannelMap
    initiation_date: DateTime
        date of creationfiles['controls']

    """
    panel_name = mongoengine.StringField(required=True, unique=True)
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
        err = 'Invalid template dictionary; must be a nested dictionary with parent keys: channels, markers'
        assert all([k in ['channels', 'markers', 'mappings'] for k in x.keys()]), err
        err = f'Invalid template dictionary; nested dictionaries must contain keys: name, regex case, ' \
              f'and permutations'
        for k in ['channels', 'markers']:
            assert all([i.keys() == ['name', 'regex', 'case', 'permutations'] for i in x[k]]), err

        assert type(x['mappings']) == list, 'Invalid template dictionary; mappings must be a list of tuples'
        err = 'Invalid template dictionary; mappings must be a list of tuples'
        assert all([type(k) != tuple for k in x['mappings']]), err
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

    def get_channels(self) -> iter:
        """
        Yields list of channels associated to panel

        Returns
        -------
        Generator
        """
        for cm in self.mappings:
            yield cm.channel

    def get_markers(self) -> iter:
        """
        Yields list of channels associated to panel

        Returns
        -------
        Generator
        """
        for cm in self.mappings:
            yield cm.marker


def _data_dir_append_leading_char(path: str):
    leading_char = path[len(path) - 1]
    if leading_char not in ["\\", "/"]:
        if len(path.split("\\")) > 1:
            # Assuming a windows OS
            return path + "\\"
        else:
            # Assuming unix OS
            return path + "/"
    return path


def _add_file(path: str,
              filegrp: FileGroup,
              panel: Panel,
              compensate: bool = True,
              ctrl_id: str or None = None,
              comp_matrix: str or None = None,
              missing_error: str = "raise"):
    control = ctrl_id is not None
    fcs = FCSFile(filepath=path,
                  comp_matrix=comp_matrix)
    channel_mappings = list(map(lambda x: standardise_names(channel_marker=x,
                                                            ref_channels=panel.channels,
                                                            ref_markers=panel.markers,
                                                            ref_mappings=panel.mappings),
                                fcs.channel_mappings))
    for cm in channel_mappings:
        err = f'The channel/marker pairing {cm} does not correspond to any defined in panel'
        assert _check_pairing(ref_mappings=panel.mappings, channel_marker=cm), err
    _missing_channels(mappings=channel_mappings, channels=panel.channels, errors=missing_error)
    _duplicate_mappings(channel_mappings)
    if compensate:
        fcs.compensate()
    filegrp.add_file(data=fcs.event_data,
                     channel_mappings=channel_mappings,
                     control=control,
                     ctrl_id=ctrl_id)


class Experiment(mongoengine.Document):
    """
    Document representation of Flow Cytometry experiment

    Parameters
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
    gating_templates: ListField
        Reference to gating templates associated to this experiment
    """
    experiment_id = mongoengine.StringField(required=True, unique=True)
    data_directory = mongoengine.StringField(required=True)
    panel = mongoengine.ReferenceField(Panel, reverse_delete_rule=4)
    fcs_files = mongoengine.ListField(mongoengine.ReferenceField(FileGroup, reverse_delete_rule=mongoengine.PULL))
    flags = mongoengine.StringField(required=False)
    notes = mongoengine.StringField(required=False)
    meta = {
        'db_alias': 'core',
        'collection': 'experiments'
    }

    def __init__(self, *args, **kwargs):
        panel_definition = kwargs.pop("panel_definition", None)
        panel_name = kwargs.pop("panel_name", None)
        super().__init__(*args, **kwargs)
        if not self.panel:
            self.panel = self._generate_panel(panel_definition=panel_definition,
                                              panel_name=panel_name)
            self.panel.save()
        if self.data_directory:
            assert os.path.isdir(self.data_directory), f"data directory {self.data_directory} does not exist"
            self.data_directory = _data_dir_append_leading_char(self.data_directory)
        else:
            raise ValueError("No data directory provided")

    @staticmethod
    def _check_panel(panel_name: str or None,
                     panel_definition: str or None):
        """
        Check that parameters provided for defining a panel are valid.

        Parameters
        ----------
        panel_name: str or None
            Name of an existing panel
        panel_definition: str or None
            Path to a panel definition

        Returns
        -------
        None
            Raises AssertionError in the condition that the given parameters are invalid
        """
        if panel_definition is None and panel_name is None:
            raise ValueError("Must provide either path to panel definition or name of an existing panel")
        if panel_definition is not None:
            assert os.path.isfile(panel_definition), f"{panel_definition} does not exist"
            err = "Panel definition is not a valid Excel document"
            assert os.path.splitext(panel_definition)[1] in [".xls", ".xlsx"], err
        else:
            assert len(Panel.objects(panel_name=panel_name)) > 0, "Invalid panel name, panel does not exist"

    def _generate_panel(self,
                        panel_definition: str or None,
                        panel_name: str or None):
        """
        Associate a panel to this Experiment, either by fetching an existing panel using the
        given panel name or by generating a new panel using the panel definition provided (path to a valid template).

        Parameters
        ----------
        panel_name: str or None
            Name of an existing panel
        panel_definition: str or None
            Path to a panel definition

        Returns
        -------
        Panel
        """
        self._check_panel(panel_name=panel_name, panel_definition=panel_definition)
        if panel_definition is None:
            return Panel.objects(panel_name=panel_name).get()
        if panel_name is None:
            panel_name = f"{self.experiment_id}_panel"
        new_panel = Panel(panel_name=panel_name)
        new_panel.create_from_excel(path=panel_definition)
        return new_panel

    def update_data_directory(self, new_path: str):
        """
        Update the data directory associated to this experiment. This will propagate to all
        associated FileGroup's. WARNING: this function will move the existing data directory
        and all of it's contents to the new given path.

        Parameters
        ----------
        new_path: str

        Returns
        -------
        None
        """
        assert os.path.isdir(new_path), "Invalid directory given for new_path"
        shutil.move(self.data_directory, new_path)
        for file in self.fcs_files:
            file.data_directory = new_path
            file.save()
        self.data_directory = new_path
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
                f.populations = []
                f.save()

    def delete_gating_templates(self,
                                template_name: str) -> None:
        """
        Remove association and delete gating template. If template_name is 'all',
        then all associated gating templates will be deleted and removed

        Parameters
        ----------
        template_name: str
            Name of template to remove; if 'all', then all associated gating templates will be deleted and removed

        Returns
        --------
        None
        """
        for g in self.gating_templates:
            if template_name == 'all' or g.template_name == template_name:
                g.delete()
        if template_name == 'all':
            self.gating_templates = []
        else:
            self.gating_templates = [g for g in self.gating_templates if g.template_name != template_name]
        self.save()

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

    def list_samples(self,
                     valid_only: bool = True) -> Generator:
        """
        Generate a list IDs of file groups associated to experiment

        Parameters
        -----------
        valid_only: bool
            If True, returns only valid samples (samples without 'invalid' flag)

        Returns
        --------
        Generator
            List of IDs of file groups associated to experiment
        """
        for f in self.fcs_files:
            if valid_only:
                if f.valid:
                    yield f.primary_id
            else:
                yield f.primary_id

    def list_invalid(self) -> Generator:
        """
        Generate list of sample IDs for samples that have the 'invalid' flag in their flag attribute

        Returns
        --------
        Generator
            List of sample IDs for invalid samples
        """
        for f in self.fcs_files:
            if not f.valid():
                yield f.primary_id

    def get_sample_mid(self,
                       sample_id: str) -> str or None:
        """
        Given a sample ID (for a sample belonging to this experiment) return it's mongo ObjectID as a string

        Parameters
        -----------
        sample_id: str
            Sample ID for sample of interest

        Returns
        --------
        str or None
            string value for ObjectID
        """
        if not self.sample_exists(sample_id):
            return None
        return [f for f in self.fcs_files if f.primary_id == sample_id][0].id.__str__()

    def get_data(self,
                 sample_id: str,
                 sample_size: int or None = None,
                 include_controls: bool = True,
                 columns: str = "marker") -> dict:
        filegrp = self.get_sample(sample_id)
        return filegrp.load(sample_size=sample_size,
                            include_controls=include_controls,
                            columns=columns)

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

    def add_new_sample(self,
                       sample_id: str,
                       primary_path: str,
                       controls_path: dict or None = None,
                       subject_id: str or None = None,
                       comp_matrix: str or None = None,
                       compensate: bool = True,
                       verbose: bool = True,
                       processing_datetime: str or None = None,
                       collection_datetime: str or None = None,
                       missing_error: str = "raise",
                       require_equal_mappings: bool = True) -> FileGroup:
        """
        Add a new sample (FileGroup) to this experiment

        Parameters
        ----------
        sample_id: str
            Primary ID for identification of sample (FileGroup.primary_id)
        subject_id: str, optional
            ID for patient to associate sample too
        primary_path: dict
            Dictionary of file name and relative path to file for those files to be treated as "primary"
            files i.e. they are 'fully' stained and should not be treated as controls
        controls_path: dict
            Dictionary of control ID and relative path to file for those files to be treated as "control"
            files e.g. they are 'FMO' or 'isotype' controls
        comp_matrix: str, optional
            Path to csv file for spillover matrix for compensation calculation; if not supplied
            the matrix linked within the fcs file will be used, if not present will present an error
        compensate: bool, (default=True)
            Boolean value as to whether compensation should be applied before data entry (default=True)
        verbose: bool, (default=True)
            If True function will provide feedback in the form of print statements
            (default=True)
        missing_error: str (default="raise")
            How to handle missing channels (that is, channels that appear in the associated
            staining panel but cannot be found in the FCS data. Either "raise" to raise
            AssertionError or "warn" to flag a UserWarning but continue execution.
        require_equal_mappings: bool (default=True)
            Require that the channel/marker names are equivalent between primary data
            and control files.
        processing_datetime: str, optional
        collection_datetime: str, optional

        Returns
        --------
        FileGroup
            Newly created FileGroup object
        """

        controls_path = controls_path or {}
        feedback = vprint(verbose)
        assert not self.sample_exists(sample_id), f'A file group with id {sample_id} already exists'
        # Load files and store in dictionary as pandas dataframes
        feedback("Loading cytometry data...")
        files = _load_files(primary_path=primary_path,
                            controls_path=controls_path,
                            comp_matrix=comp_matrix)
        # Load mappings and check the mappings match between files
        feedback("Checking channel/marker mappings...")
        mappings = _load_and_check_mappings(fcs_files=files,
                                            panel=self.panel,
                                            missing_error=missing_error,
                                            require_equal_mappings=require_equal_mappings)
        if compensate:
            feedback("Compensating data...")
            for fcs_file in files.values():
                fcs_file.compensate()
        # Create FileGroup
        processing_datetime = processing_datetime or datetime.now()
        collection_datetime = collection_datetime or datetime.now()
        feedback("Generating FileGroup object...")
        filegrp = FileGroup(primary_id=sample_id,
                            data_directory=self.data_directory,
                            compensated=compensate,
                            collection_datetime=collection_datetime,
                            processing_datetime=processing_datetime,
                            mappings=mappings,
                            data={file_id: pd.DataFrame(fcs.event_data)
                                  for file_id, fcs in files.items()})
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
        return filegrp

    def delete(self, delete_panel: bool = True, *args, **kwargs):
        if delete_panel:
            self.panel.delete()
        super().delete(*args, **kwargs)


def _load_files(primary_path: str,
                controls_path: Dict[str, str],
                comp_matrix: str or None) -> Dict[str, FCSFile]:
    """
    Internal method for loading a dictionary of FCSFile objects

    Parameters
    ----------
    primary_path: str
    controls_path: dict
    comp_matrix: str (optional)

    Returns
    -------
    dict
    """
    files = {"primary": FCSFile(filepath=primary_path, comp_matrix=comp_matrix)}
    for ctr_id, path in controls_path.items():
        files[ctr_id] = FCSFile(filepath=path, comp_matrix=comp_matrix)
    return files


def _equal_lists(nested_lists: List[list]):
    """
    Given a list of nested lists, assert that all
    lists are equivalent.

    Parameters
    ----------
    nested_lists: list

    Returns
    -------
    list
    """
    for x in nested_lists:
        for y in nested_lists:
            assert x == y, f"Conflicting mappings: {x} != {y}"


def _load_and_check_mappings(fcs_files: Dict[str, FCSFile],
                             panel: Panel,
                             missing_error: str,
                             require_equal_mappings: bool) -> List[dict]:
    """
    Given a dictionary of FCSFile objects, check the channel mappings match
    what is expected from the Panel and (if require_equal_mappings is True)
    that the mappings match between the files.

    Parameters
    ----------
    fcs_files: dict
        Dictionary of FCSFile objects
    panel: Panel
        Panel object
    missing_error: str
        How to handle missing channels (that is, channels that appear in the associated
        staining panel but cannot be found in the FCS data. Either "raise" to raise
        AssertionError or "warn" to flag a UserWarning but continue execution.
    require_equal_mappings
        Require that the channel/marker names are equivalent between primary data
        and control files.
    Returns
    -------

    """
    mappings = {file_id: list(map(lambda x: standardise_names(channel_marker=x,
                                                              ref_channels=panel.channels,
                                                              ref_markers=panel.markers,
                                                              ref_mappings=panel.mappings),
                                  fcs_file.channel_mappings))
                for file_id, fcs_file in fcs_files.items()}
    for channel_mappings in mappings.values():
        for cm in channel_mappings:
            err = f'The channel/marker pairing {cm} does not correspond to any defined in panel'
            assert _check_pairing(ref_mappings=panel.mappings, channel_marker=cm), err
        _missing_channels(mappings=channel_mappings, channels=panel.channels, errors=missing_error)
        _duplicate_mappings(channel_mappings)
    if require_equal_mappings:
        _equal_lists([x.get("channel") for x in mappings.values()])
        _equal_lists([x.get("marker") for x in mappings.values()])
    return mappings.get("primary")
