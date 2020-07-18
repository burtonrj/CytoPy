from datetime import datetime
from warnings import warn
from collections import Counter
from functools import partial
import pandas as pd
import mongoengine
import xlrd
import os
import re


def create_regex(s: str, initials: bool = True) -> str:
    """
    Given a string representation of either a channel or marker, generate a standard
    regex string to be used in a panel template

    Parameters
    ----------
    s: str
        String value of channel or marker to generate regex term for
    initials: bool, (default=True)
        If True, account for use of initials to represent a marker/channel name

    Returns
    -------
    str
        Formatted regex string
    """

    def has_numbers(input_string):
        return any(char.isdigit() for char in input_string)

    s = [i for ls in [_.split('-') for _ in s.split(' ')] for i in ls]
    s = [i for ls in [_.split('.') for _ in s] for i in ls]
    s = [i for ls in [_.split('/') for _ in s] for i in ls]
    new_string = list()
    for i in s:
        if not has_numbers(i) and len(i) > 2 and initials:
            new_string.append(f'{i[0]}({i[1:]})*')
        else:
            new_string.append(i)
    new_string = '[\s.-]+'.join(new_string)
    new_string = '<*\s*' + new_string + '\s*>*'
    return new_string


def create_template(channel_mappings: list, file_name: str,
                    case_sensitive: bool = False, initials: bool = True):
    """
    Given a list of channel mappings from an fcs file, create an excel template for Panel creation.

    Parameters
    ----------
    channel_mappings: list
        List of channel mappings (list of dictionaries)
    file_name: str
        File name for saving excel template
    case_sensitive: bool, (default=False)
        If True, search terms for channels/markers will be case sensitive
    initials: bool, (default=True)
        If True, search terms for channels/markers will account for the use of initials of channels/markers

    Returns
    -------
    None
    """
    try:
        assert file_name.split('.')[1] == 'xlsx', 'Invalid file name, must be of format "NAME.xlsx"'
    except IndexError:
        raise Exception('Invalid file name, must be of format "NAME.xlsx"')

    mappings = pd.DataFrame()
    mappings['channel'] = [cm['channel'] for cm in channel_mappings]
    mappings['marker'] = [cm['marker'] for cm in channel_mappings]

    nomenclature = pd.DataFrame()
    names = mappings['channel'].tolist() + mappings['marker'].tolist()
    nomenclature['name'] = [n for n in names if n]
    f = partial(create_regex, initials=initials)
    nomenclature['regex'] = nomenclature['name'].apply(f)
    nomenclature['case'] = case_sensitive
    nomenclature['permutations'] = None
    writer = pd.ExcelWriter(file_name, engine='xlsxwriter')
    mappings.to_excel(writer, sheet_name='mappings')
    nomenclature.to_excel(writer, sheet_name='nomenclature')
    writer.save()


class ChannelMap(mongoengine.EmbeddedDocument):
    """
    Defines channel/marker mapping. Each document will contain a single value for channel and a single value for marker,
    these two values are treated as a pair within the panel.

    Parameters
    ----------
    channel: str
        name of channel (e.g. fluorochrome)
    marker: str
        name of marker (e.g. protein)
    """
    channel = mongoengine.StringField()
    marker = mongoengine.StringField()

    def check_matched_pair(self, channel: str, marker: str) -> bool:
        """
        Check a channel/marker pair for resemblance

        Parameters
        ----------
        channel: str
            channel to check
        marker: str
            marker to check

        Returns
        --------
        bool
            True if equal, else False
        """
        if self.channel == channel and self.marker == marker:
            return True
        return False

    def to_dict(self) -> dict:
        """
        Convert object to python dictionary

        Returns
        --------
        dict
        """
        return {'channel': self.channel, 'marker': self.marker}


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
    # Check sheet names
    xls = xlrd.open_workbook(path, on_demand=True)
    err = f"Template must contain two sheets: nomenclature and mappings"
    assert all([x in ['nomenclature', 'mappings'] for x in xls.sheet_names()]), err
    nomenclature = pd.read_excel(path, sheet_name='nomenclature')
    mappings = pd.read_excel(path, sheet_name='mappings')

    # Check nomenclature column headers
    err = "Nomenclature sheet of excel template must contain the following column headers: " \
          "'name','regex','case','permutations'"
    assert all([x in ['name', 'regex', 'permutations', 'case'] for x in nomenclature.columns]), err

    # Check mappings column headers
    err = "Mappings sheet of excel template must contain the following column headers: 'channel', 'marker'"
    assert all([x in ['channel', 'marker'] for x in mappings.columns]), err

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


def _query(x: str or None,
           ref: list) -> str:
    """
    Internal static method for querying a channel/marker against a reference list

    Parameters
    ----------
    x: str or None
        channel/marker to query
    ref: list
        list of ChannelMap objects for reference search

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


def _check_duplication(x: list) -> bool:
    """
    Internal method. Given a list check for duplicates. Duplicates are printed.

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
        print(f'Duplicate channel/markers identified: {duplicates}')
        return True
    return False


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
        date of creation

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

    def _check_pairing(self,
                       channel: str,
                       marker: str or None) -> bool:
        """
        Internal method. Given a channel and marker check that a valid pairing exists for this panel.

        Parameters
        ----------
        channel: str
            channel for checking
        marker: str
            marker for checking

        Returns
        --------
        bool
            True if pairing exists, else False
        """
        if marker is None:
            marker = ''
        if not any([n.check_matched_pair(channel=channel, marker=marker) for n in self.mappings]):
            return False
        return True

    def standardise_names(self, column_mappings: list) -> list:
        """
        Given a dictionary of column mappings, apply standardisation defined by this panel object and return
        standardised column mappings.

        Parameters
        ----------
        column_mappings: list
            List of dictionaries, where each dictionary corresponds to channel/marker mappings (channel, marker)

        Returns
        --------
        list or None and bool
            standardised column mappings
        """
        new_column_mappings = list()
        for channel, marker in column_mappings:
            # Normalise channel
            if channel:
                if channel.isspace():
                    channel = None
                else:
                    channel = _query(channel, self.channels)
            # Normalise marker
            if marker:
                marker, err = _query(marker, self.markers)
            else:
                # If marker is None, default to that assigned by panel
                default = [x for x in self.mappings if x.channel == channel]
                err = f'No marker name provided for channel {channel}. Was unable to establish default as ' \
                      f'{channel} is not recognised in this panel design.'
                assert len(default) > 0, err
                marker = default[0].marker
            # Check channel/marker pairing is correct
            err = f'The channel/marker pairing {channel}/{marker} does not correspond to any defined in panel'
            assert self._check_pairing(channel, marker), err
            new_column_mappings.append((channel, marker))

        # Check for duplicate channels/markers
        channels = [c for c, _ in new_column_mappings]
        assert not _check_duplication(channels), "Duplicate channels provided"
        markers = [m for _, m in new_column_mappings]
        assert not _check_duplication(markers), "Duplicate markers provided"
        # Check for missing channels
        for x in self.channels:
            assert x.standard in channels, f'Missing channel {x.standard}'
        return new_column_mappings

    def standardise(self,
                    data: pd.DataFrame) -> pd.DataFrame or None:
        """
        Given a dataframe of fcs events, standardise the columns according to the panel definition

        data: Pandas.DataFrame
            Pandas DataFrame of cell events

        Returns
        --------
        standardised Pandas DataFrame with columns ordered according to the panel definition
        """
        # Standardise the names
        # channel_marker -> channel_marker: [channel, marker]

        column_mappings = [(c, m) if not m.isspace() else (c, None)
                           for c, m in zip(data.channel.values, data.marker.values)]
        column_mappings = self.standardise_names(column_mappings)

        # Insert missing marker names using matched channel in panel
        updated_mappings = list()
        for channel, marker in column_mappings:
            if marker is None:
                marker = [p.marker for p in self.mappings if p.channel == channel][0]
            updated_mappings.append((channel, marker))
        return column_mappings

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
