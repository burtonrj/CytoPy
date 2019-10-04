from datetime import datetime
from immunova.data.fcs import FileGroup, File
from immunova.data.gating import GatingStrategy
from immunova.data.utilities import data_from_file
from immunova.flow.readwrite.read_fcs import FCSFile
from collections import defaultdict
from multiprocessing import Pool, cpu_count
from functools import partial
import mongoengine
import pandas as pd
import numpy as np
import re
import os
import xlrd
import json


class ChannelMap(mongoengine.EmbeddedDocument):
    """
    Embedded document -> Panel

    Defines channel/marker mapping. Each document will contain a single value for channel and a single value for marker,
    these two values are treated as a pair within the panel.
    """
    channel = mongoengine.StringField()
    marker = mongoengine.StringField()


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
            if not all([k in ['name', 'regex', 'case', 'permutations']]):
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

    def standardise_names(self, column_mappings: dict) -> dict or None:
        """
        Given a dictionary of column mappings, apply standardisation defined by this panel object and return
        standardised column mappings.
        :param column_mappings: dictionary corresponding to channel/marker mappings (channel, marker)
        :return: standardised column mappings
        """

        def query(x, ref):
            corrected = list(filter(None.__ne__, [n.query(x) for n in ref]))
            if len(corrected) == 0:
                print(f'Unable to normalise {x}; no match in linked panel')
                return None
            if len(corrected) > 1:
                print(f'Unable to normalise {x}; matched multiple in linked panel, check'
                      f' panel for incorrect definitions. Matches found: {corrected}')
                return None
            return corrected[0]

        new_column_mappings = dict()
        for col_name, channel_marker in column_mappings.items():
            channel, marker = channel_marker
            # Normalise channel
            if channel:
                channel = query(channel, self.channels)
                if not channel:
                    return None
            # Normalise marker
            if marker:
                marker = query(marker, self.markers)
                if not marker:
                    return None
            new_column_mappings[col_name] = [channel, marker]
        return new_column_mappings

    def standardise(self, data: pd.DataFrame, skip_missing_channels: bool = True) -> pd.DataFrame or None:
        """
        Given a dataframe of fcs events, standardise the columns according to the panel definition
        :param data: pandas dataframe of cell events
        :return: standardised pandas dataframe with columns ordered according to the panel definition
        """
        # Standardise the names
        column_mappings = {x: [_ if _ != "" else None for _ in x.split('_')] for x in data.columns}
        column_mappings = self.standardise_names(column_mappings)
        if not column_mappings:
            return None
        # Insert missing marker names using matched channel in panel
        updated_mappings = defaultdict(list)
        for col_name, channel_marker in column_mappings.items():
            channel = channel_marker[0]
            marker = channel_marker[1]
            if marker is None:
                marker = [p.marker for p in self.mappings if p.channel == channel][0]
            updated_mappings[col_name] = [channel, marker]
        # Order the columns to match the panel definition
        ordered_columns = list()
        for channel_marker_pair in self.mappings:
            comparisons = {k: (v[0] == channel_marker_pair.channel, v[1] == channel_marker_pair.marker)
                           for k, v in updated_mappings.items()}
            comparisons = {k: sum(v) for k, v in comparisons.items() if sum(v) == 2}
            if len(comparisons) > 1:
                print(f'Multiple instances of {channel_marker_pair.channel}, {channel_marker_pair.marker} pair'
                      f'found!')
                return None
            if not comparisons:
                print(f'WARNING: {channel_marker_pair.channel}, {channel_marker_pair.marker} pair not found!')
                continue
            if not skip_missing_channels:
                print(f'Column mappings: {updated_mappings.items()}')
                return None
            ordered_columns.append(list(comparisons.keys())[0])
        return data[ordered_columns]


class FCSExperiment(mongoengine.Document):
    """
    Document representation of Flow Cytometry experiment

    Attributes:
        experiment_id - unique identifier for experiment
        panel - Panel object describing associated channel/marker pairs
        fcs_files - reference field for associated files
        flags - warnings associated to experiment
        notes - additional free text comments
        gating_templates - reference to gating templates associated to this experiment
    Methods:
        pull_sample_data - Given a sample ID, associated to this experiment, fetch the fcs data
        list_samples - Generate a list IDs of file groups associated to experiment
        remove_sample - Remove sample (FileGroup) from experiment.
        add_new_sample - Add a new sample (FileGroup) to this experiment

    """
    experiment_id = mongoengine.StringField(required=True, unique=True)
    panel = mongoengine.ReferenceField(Panel, reverse_delete_rule=4)
    fcs_files = mongoengine.ListField(mongoengine.ReferenceField(FileGroup, reverse_delete_rule=4))
    flags = mongoengine.StringField(required=False)
    notes = mongoengine.StringField(required=False)
    gating_templates = mongoengine.ListField(mongoengine.ReferenceField(GatingStrategy, reverse_delete_rule=4))

    meta = {
        'db_alias': 'core',
        'collection': 'fcs_experiments'
    }

    def pull_sample_data(self, sample_id: str, sample_size: int or None = None,
                         data_type: str = 'raw', include_controls: bool = True,
                         output_format: str = 'dataframe',
                         return_mappings: bool = True) -> (None, None) or (list, None) or (list, list):
        """
        Given a sample ID, associated to this experiment, fetch the fcs data
        :param sample_id: ID of sample to fetch data for
        :param sample_size: if provided with an integer value, a sample of data of given size will be returned
        (sample drawn from a uniform distribution)
        :param data_type: type of data to retrieve; either 'raw' or 'norm' normalised
        :param include_controls: if True (default) then control files associated to sample are included in the result
        :param output_format: preferred format of output; can either be 'dataframe' for a pandas dataframe, or 'matrix'
        for a numpy array
        :param return_mappings: if True (default) panel mappings (channel/marker pairs) returned as list
        :return: tuple; (list of dictionaries {id: file id, typ: data type, either raw or normalised,
        data: dataframe/matrix}, list of dictionaries defining channel/marker pairs
        {channel: channel name, marker: marker name})
        """
        if sample_id not in self.list_samples():
            print(f'Error: invalid sample_id, {sample_id} not associated to this experiment')
            return None, None
        file_grp = FileGroup.objects(primary_id=sample_id)
        if not file_grp:
            print(f'Error: invalid sample_id, no file entry for {sample_id}')
            return None, None
        file_grp = file_grp[0]
        files = file_grp.files
        mappings = [json.loads(x.to_json()) for x in self.panel.mappings]
        # Fetch data
        if not include_controls:  # Fetch data for primary file only
            f = [f for f in files if f.file_type == 'complete'][0]
            complete = data_from_file(f, data_type, sample_size, panel=self.panel)
            if return_mappings:
                return [complete], mappings
            return [complete], None
        # Fetch data for primary file & controls
        pool = Pool(cpu_count())
        f = partial(data_from_file, data_type=data_type, sample_size=sample_size,
                    output_format=output_format, panel=self.panel)
        data = pool.map(f, files)
        pool.close()
        pool.join()
        if return_mappings:
            return data, mappings
        return data, None

    def list_samples(self) -> list:
        """
        Generate a list IDs of file groups associated to experiment
        :return: List of IDs of file groups associated to experiment
        """
        return [f.primary_id for f in self.fcs_files]

    def remove_sample(self, sample_id: str, delete=False):
        """
        Remove sample (FileGroup) from experiment.
        :param sample_id: ID of sample to remove
        :param delete: if True, the FileGroup entry will be deleted from the database
        :return: True if successful
        """
        fg = FileGroup.objects(primary_id=sample_id)
        if not fg:
            print(f'Error: {sample_id} does not exist')
            return False
        self.fcs_files = [f for f in self.fcs_files if f.primary_id != sample_id]
        if delete:
            fg[0].delete()
        return True

    def add_new_sample(self, sample_id: str, file_path: str, controls: list,
                       comp_matrix: np.array or None = None, compensate: bool = True,
                       feedback: bool = True) -> None or str:
        """
        Add a new sample (FileGroup) to this experiment
        :param sample_id: primary ID for identification of sample (FileGroup.primary_id)
        :param file_path: file path of the primary fcs file (e.g. the fcs file that is of primary interest such as the
        file with complete staining)
        :param controls: list of file paths for control files e.g. a list of file paths for associated FMO controls
        :param comp_matrix: (optional) numpy array for spillover matrix for compensation calculation; if not supplied
        the matrix linked within the fcs file will be used, if not present will present an error
        :param compensate: boolean value as to whether compensation should be applied before data entry (default=True)
        :param feedback: boolean value, if True function will provide feedback in the form of print statements
        (default=True)
        :return: MongoDB ObjectID string for new FileGroup entry
        """
        def create_file_entry(path, file_id, control_=False):
            fcs = FCSFile(path, comp_matrix=comp_matrix)
            new_file = File()
            new_file.file_id = file_id
            if compensate:
                fcs.compensate()
                new_file.compensated = True
            if control_:
                new_file.file_type = 'control'
            data = fcs.dataframe
            data = self.panel.standardise(data)
            if data is None:
                print(f'Error: invalid channel/marker mappings for {file_id}, at path {file_path}, aborting.')
                return None
            new_file.put(data.values)
            return new_file

        if sample_id in self.list_samples():
            print(f'Error: a file group with id {sample_id} already exists')
            return None
        if feedback:
            print('Generating main file entry...')
        file_collection = FileGroup()
        file_collection.primary_id = sample_id
        primary_file = create_file_entry(file_path, sample_id)
        if not primary_file:
            return None
        file_collection.files.append(primary_file)
        if feedback:
            print('Generating file entries for controls...')
        for c in controls:
            control = create_file_entry(c['path'], f"{sample_id}_{c['control_id']}", control_=True)
            if not control:
                return None
            file_collection.files.append(control)
        file_collection.save()
        self.fcs_files.append(file_collection)
        if feedback:
            print(f'Successfully created {sample_id} and associated to {self.experiment_id}')
        self.save()
        return file_collection.id.__str__()
