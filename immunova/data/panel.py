from datetime import datetime
from collections import Counter
import pandas as pd
import mongoengine
import xlrd
import os
import re


class ChannelMap(mongoengine.EmbeddedDocument):
    """
    Embedded document -> Panel

    Defines channel/marker mapping. Each document will contain a single value for channel and a single value for marker,
    these two values are treated as a pair within the panel.
    """
    channel = mongoengine.StringField()
    marker = mongoengine.StringField()

    def check_matched_pair(self, channel: str, marker: str) -> bool:
        """
        Check a channel/marker pair for resemblance
        :param channel: channel to check
        :param marker: marker to check
        :return: True if equal, else False
        """
        if self.channel == channel and self.marker == marker:
            return True
        return False

    def to_python(self) -> dict:
        """
        Convert object to python dictionary
        :return: Dictionary object
        """
        return {'channel': self.channel, 'marker': self.marker}


class NormalisedName(mongoengine.EmbeddedDocument):
    """
    Embedded document -> Panel

    Defines a standardised name for a channel or marker and provides method for testing if a channel/marker
    should be associated to standard

    Attributes:
        standard - the "standard" name i.e. the nomenclature we used for a channel/marker in this panel
        regex_str - regular expression used to test if a term corresponds to this standard
        permutations - list of string values that have direct association to this standard
        case_sensitive - is the nomenclature case sensitive? This would be false for something like 'CD3' for example,
        where 'cd3' and 'CD3' are synonymous
    Methods:
         query - given a term 'x', determine if 'x' is synonymous to this standard. If so, return the standardised name.
    """
    standard = mongoengine.StringField(required=True)
    regex_str = mongoengine.StringField()
    permutations = mongoengine.StringField()
    case_sensitive = mongoengine.BooleanField(default=False)

    def query(self, x: str) -> None or str:
        """
        Given a term 'x', determine if 'x' is synonymous to this standard. If so, return the standardised name.
        :param x: search term
        :return: Standardised name if synonymous to standard, else None
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


class Panel(mongoengine.Document):
    """
    Embedded document -> FCSExperiment
    Document representation of channel/marker definition for an experiment. A panel, once associated to an experiment
    will standardise data upon input; when an fcs file is created in the database, it will be associated to
    an experiment and the channel/marker definitions in the fcs file will be mapped to the associated panel.

    Attributes:
        panel_name - unique identifier for the panel
        markers - list of marker names; see NormalisedName
        channels - list of channels; see NormalisedName
        mappings - list of channel/marker mappings; see ChannelMap
        initiation_date - date of creation
    Methods:
        check_excel_template - Given the file path of an excel template, check validity
        create_from_excel - Given the file path of an excel template, populate panel using template
        create_from_dict - Given a python dictionary object, populate panel using dictionary as template
        standardise_names - Given a dictionary of column mappings, apply standardisation defined by this panel object
        and return standardised column mappings.
        standardise - Given a dataframe of fcs events, standardise the columns according to the panel definition

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

    @staticmethod
    def check_excel_template(path: str) -> (pd.DataFrame, pd.DataFrame) or None:
        """
        Check excel template and if valid return pandas dataframes
        :param path: file path for excel template
        :return: tuple of pandas dataframes (nomenclature, mappings) or None
        """
        # Check sheet names
        xls = xlrd.open_workbook(path, on_demand=True)
        if not all([x in ['nomenclature', 'mappings'] for x in xls.sheet_names()]):
            return None
        else:
            nomenclature = pd.read_excel(path, sheet_name='nomenclature')
            mappings = pd.read_excel(path, sheet_name='mappings')
        # Check nomenclature column headers
        if not all([x in ['name', 'regex', 'permutations', 'case'] for x in nomenclature.columns]):
            print("Nomenclature sheet of excel template must contain the following column headers: "
                  "'name','regex','case','permutations'")
            return None
        # Check mappings column headers
        if not all([x in ['channel', 'marker'] for x in mappings.columns]):
            print("Mappings sheet of excel template must contain the following column headers:"
                  "'channel', 'marker'")
            return None
        # Check for duplicate entries
        if sum(nomenclature['name'].duplicated()) != 0:
            print('Duplicate entries in nomenclature, please remove duplicates before continuing')
            return None
        # Check that all mappings have a corresponding entry in nomenclature
        for x in ['channel', 'marker']:
            for name in mappings[x]:
                if pd.isnull(name):
                    continue
                if name not in nomenclature.name.values:
                    print(f'{name} missing from nomenclature, please review template')
                    return None
        return nomenclature, mappings

    def create_from_excel(self, path: str) -> bool:
        """
        Populate panel attributes from an excel template
        :param path: path of file
        :return: True is successful else False
        """
        if not os.path.isfile(path):
            print(f'Error: no such file {path}')
            return False
        templates = self.check_excel_template(path)
        if templates is None:
            print('Error: invalid excel template')
            return False
        nomenclature, mappings = templates

        def create_def(n):
            d = nomenclature[nomenclature['name'] == n].fillna('').to_dict(orient='list')
            return NormalisedName(standard=d['name'][0],
                                  regex_str=d['regex'][0],
                                  case_sensitive=d['case'][0],
                                  permutations=d['permutations'][0])
        for name in mappings['channel']:
            if not pd.isnull(name):
                definition = create_def(name)
                self.channels.append(definition)
        for name in mappings['marker']:
            if not pd.isnull(name):
                definition = create_def(name)
                self.markers.append(definition)
        mappings = mappings.fillna('').to_dict(orient='list')
        self.mappings = [ChannelMap(channel=c, marker=m)
                         for c, m in zip(mappings['channel'], mappings['marker'])]
        return True

    def create_from_dict(self, x: dict) -> bool:
        """
        Populate panel attributes from a python dictionary
        :param x: dictionary object containing panel definition
        :return: True if successful else False
        """

        # Check validity of input dictionary
        if not all([k in ['channels', 'markers', 'mappings'] for k in x.keys()]):
            print('Invalid template dictionary; must be a nested dictionary with parent keys: channels, markers')
            return False
        for k in ['channels', 'markers']:
            if not all([i.keys() == ['name', 'regex', 'case', 'permutations'] for i in x[k]]):
                print(f'Invalid template dictionary; nested dictionaries for {k} must contain keys: name, regex '
                      f'case, and permutations')
                return False
        if type(x['mappings']) != list:
            print('Invalid template dictionary; mappings must be a list of tuples')
            return False
        if not all([type(k) != tuple for k in x['mappings']]):
            print('Invalid template dictionary; mappings must be a list of tuples')
            return False
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
        return True

    @staticmethod
    def __query(x: str or None, ref: list, e: bool) -> str or None and bool:
        """
        Internal static method for querying a channel/marker against a reference list
        :param x: channel/marker to query
        :param ref: list of ChannelMap objects for reference search
        :param e: error state
        :return: Standardised name and error state
        """
        corrected = list(filter(None.__ne__, [n.query(x) for n in ref]))
        if len(corrected) == 0:
            print(f'Unable to normalise {x}; no match in linked panel')
            e = True
            return x, e
        if len(corrected) > 1:
            print(f'Unable to normalise {x}; matched multiple in linked panel, check'
                  f' panel for incorrect definitions. Matches found: {corrected}')
            e = True
            return x, e
        return corrected[0], e

    def __check_pairing(self, channel: str, marker: str or None) -> bool:
        """
        Internal method. Given a channel and marker check that a valid pairing exists for this panel.
        :param channel: channel for checking
        :param marker: marker for checking
        :return: True if pairing exists, else False
        """
        if marker is None:
            marker = ''
        if not any([n.check_matched_pair(channel=channel, marker=marker) for n in self.mappings]):
            return False
        return True

    @staticmethod
    def __check_duplication(x: list) -> bool:
        """
        Internal method. Given a list check for duplicates. Duplicates are printed.
        :param x:
        :return: True if duplicates are found, else False
        """
        duplicates = [item for item, count in Counter(x).items() if count > 1 and item is not None]
        if duplicates:
            print(f'Duplicate channel/markers identified: {duplicates}')
            return True
        return False

    def standardise_names(self, column_mappings: list) -> list or None and bool:
        """
        Given a dictionary of column mappings, apply standardisation defined by this panel object and return
        standardised column mappings.
        :param column_mappings: dictionary corresponding to channel/marker mappings (channel, marker)
        :return: standardised column mappings and error state
        """
        err = False
        new_column_mappings = list()
        for channel, marker in column_mappings:
            # Normalise channel
            if channel:
                channel, err = self.__query(channel, self.channels, err)
            # Normalise marker
            if marker:
                marker, err = self.__query(marker, self.markers, err)
            else:
                # If marker is None, default to that assigned by panel
                default = [x for x in self.mappings if x.channel == channel]
                if not default:
                    print(f'No marker name provided for channel {channel}. Was unable to establish default as'
                          f' {channel} is not recognised in this panel design.')
                    err = True
                else:
                    marker = default[0].marker
            # Check channel/marker pairing is correct
            if not self.__check_pairing(channel, marker):
                print(f'The channel/marker pairing {channel}/{marker} does not correspond to any defined in panel')
                err = True
            new_column_mappings.append((channel, marker))
        # Check for duplicate channels/markers
        channels = [c for c, _ in column_mappings]
        if self.__check_duplication(channels):
            err = True
        if self.__check_duplication([m for _, m in column_mappings]):
            err = True
        # Check for missing channels
        for x in self.channels:
            if x.standard not in channels:
                print(f'Missing channel {x.standard}')
                err = True
        return new_column_mappings, err

    def standardise(self, data: pd.DataFrame, catch_standardisation_errors: bool = False) -> pd.DataFrame or None:
        """
        Given a dataframe of fcs events, standardise the columns according to the panel definition
        :param catch_standardisation_errors: if True, any error in standardisation will cause function to return Null
        :param data: pandas dataframe of cell events
        :return: standardised pandas dataframe with columns ordered according to the panel definition
        """
        # Standardise the names
        # channel_marker -> channel_marker: [channel, marker]
        column_mappings = [[_ if _ != "" else None for _ in x.split('_')] for x in data.columns]
        column_mappings, err = self.standardise_names(column_mappings)
        if err and catch_standardisation_errors:
            return None

        # Insert missing marker names using matched channel in panel
        updated_mappings = list()
        for channel, marker in column_mappings:
            if marker is None:
                marker = [p.marker for p in self.mappings if p.channel == channel][0]
            updated_mappings.append((channel, marker))
        return column_mappings
