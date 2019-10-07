import mongoengine
from bson.binary import Binary
from immunova.data.gating import Gate
from immunova.data.patient import Patient
from immunova.data.fcs_experiments import ChannelMap
from immunova.flow.gating.defaults import Geom
import numpy as np
import pandas as pd
import pickle


class Population(mongoengine.EmbeddedDocument):
    """
    Embedded document -> FileGroup
    Cached populations; stores the index of events associated to a population
    for quick loading.

    Attributes:
        population_name - name of population
        index - numpy array storing index of events that belong to population
        prop_of_parent - proportion of cells as a percentage of parent population
        prop_of_total - proportion of cells as a percentage of all events
        warnings - list of warnings associated to population
        parent - name of parent population
        children - list of child populations (list of strings)
        geom - list of key value pairs (tuples; (key, value)) for defining geom of population e.g.
        the defintion for an ellipse that 'captures' the population
    Methods:
        save_index - given a new numpy array of index values, serialise and commit data to database
        load_index - retrieve the index values for the given population
        to_python - generate a python dictionary object for this population
    """
    population_name = mongoengine.StringField()
    index = mongoengine.FileField(db_alias='core', collection_name='population_indexes')
    prop_of_parent = mongoengine.FloatField()
    prop_of_total = mongoengine.FloatField()
    warnings = mongoengine.ListField()
    parent = mongoengine.StringField()
    children = mongoengine.ListField()
    geom = mongoengine.ListField()

    def save_index(self, data: np.array):
        if self.index:
            self.index.replace(Binary(pickle.dumps(data, protocol=2)))
        else:
            self.index.new_file()
            self.index.write(Binary(pickle.dumps(data, protocol=2)))
            self.index.close()

    def load_index(self) -> np.array:
        return pickle.loads(bytes(self.index.read()))

    def to_python(self):
        population = dict(population_name=self.population_name, prop_of_parent=self.prop_of_parent,
                          prop_of_total=self.prop_of_total, warnings=self.warnings, parent=self.parent,
                          children=self.children)
        if self.population_name == 'root':
            population['geom'] = Geom(shape='NA', x='FSC-A', y='SSC-A')
        else:
            population['geom'] = Geom(**{k: v for k, v in self.geom})
        population['index'] = self.load_index()
        return population


class File(mongoengine.EmbeddedDocument):
    """
    Embedded document -> FileGroup
    Document representation of a single FCS file.

    Attributes:
        file_id: unique identifier for fcs file
        file_type: one of either 'complete' or 'control'; signifies the type of data stored
        data: numpy array of fcs events data
        norm: numpy array of normalised fcs events data
        compensated: boolean value, if True then data have been compensated
        channel_mappings: list of standarised channel/marker mappings (corresponds to column names of underlying data)

    Methods:
        raw_data - loads raw data returning a numpy array
        norm_data - loads normalised data returning a numpy array
        put - given a numpy array, data is serialised and stored

    """
    file_id = mongoengine.StringField(required=True, unique=True)
    file_type = mongoengine.StringField(default='complete')
    data = mongoengine.FileField(db_alias='core', collection_name='fcs_file_data')
    norm = mongoengine.FileField(db_alias='core', collection_name='fcs_file_norm')
    compensated = mongoengine.BooleanField(default=False)
    channel_mappings = mongoengine.ListField(ChannelMap)

    def raw_data(self, sample: int or None = None):
        """
        Load raw data
        :param sample: int value; produces a sample of given value
        :return:  Numpy array of events data (raw)
        """
        data = pickle.loads(self.data.read())
        if sample:
            return self.__sample(data, sample)
        return data

    def norm_data(self, sample: int or None = None):
        """
        Load normalised data
        :param sample: int value; produces a sample of given value
        :return:  Numpy array of events data (normalised)
        """
        data = pickle.loads(self.norm.read())
        if sample:
            return self.__sample(data, sample)
        return data

    @staticmethod
    def __sample(data, n):
        if n < data.shape[0]:
            idx = np.random.randint(data.shape[0], size=n)
            return data[idx, :]
        return data

    def put(self, data: np.array, typ: str = 'data'):
        """
        Save events data to database
        :param data: numpy array of events data
        :param typ: type of data; either `data` (raw) or `norm` (normalised)
        :return: None
        """
        if not any([typ == x for x in ['data', 'norm']]):
            print('Error: type must be one of [data, norm]')
            return None
        if typ == 'data':
            if self.data:
                self.data.replace(Binary(pickle.dumps(data, protocol=2)))
            else:
                self.data.new_file()
                self.data.write(Binary(pickle.dumps(data, protocol=2)))
                self.data.close()
        if typ == 'norm':
            if self.norm:
                self.norm.replace(Binary(pickle.dumps(data, protocol=2)))
            else:
                self.norm.new_file()
                self.norm.write(Binary(pickle.dumps(data, protocol=2)))
                self.norm.close()

    def data_from_file(self, data_type: str, sample_size: int or None, output_format: str = 'dataframe',
                       columns_default: str = 'marker') -> None or dict:
        """
        Pull data from a file document
        :param data_type: data type to retrieve; either 'raw' or 'norm' (normalised)
        :param sample_size: return a sample of given integer size
        :param output_format: preferred format of output; can either be 'dataframe' for a pandas dataframe, or 'matrix'
        for a numpy array
        :param columns_default: how to name columns if output_format='dataframe';
        either 'marker' or 'channel' (default = 'marker')
        :return: Dictionary output {id: file_id, typ: file_type, data: dataframe/matrix}
        """
        if data_type == 'raw':
            data = self.raw_data(sample=sample_size)

        elif data_type == 'norm':
            data = self.norm_data(sample=sample_size)
        else:
            print('Invalid data_type, must be raw or norm')
            return None
        if output_format == 'dataframe':
            data = self.__as_dataframe(data, columns_default=columns_default)
        return dict(id=self.file_id, typ=self.file_type, data=data)

    def __as_dataframe(self, matrix: np.array, columns_default: str = 'marker'):
        """
        Generate a pandas dataframe using a given numpy multi-dim array with specified column defaults
        :param matrix: numpy matrix to convert to dataframe
        :param columns_default: how to name columns; either 'marker' or 'channel' (default = 'marker')
        :return: Pandas dataframe
        """
        if columns_default == 'marker':
            mappings = [m.marker if not None else m.channel for m in self.channel_mappings]
        else:
            mappings = [m.channel for m in self.channel_mappings]
        return pd.DataFrame(matrix, columns=mappings, dtype='float32')


class FileGroup(mongoengine.Document):
    """
    Document representation of a file group; a selection of related fcs files (e.g. a sample and it's associated
    controls)

    Attributes:
        primary_id - unique ID to associate to group
        files - list of File objects
        flags - warnings associated to file group
        notes - additional free text
        populations - populations derived from this file group
        gates - gate objectes that have been applied to this file group
    """
    primary_id = mongoengine.StringField(required=True)
    files = mongoengine.EmbeddedDocumentListField(File)
    flags = mongoengine.StringField(required=False)
    notes = mongoengine.StringField(required=False)
    populations = mongoengine.EmbeddedDocumentListField(Population)
    gates = mongoengine.EmbeddedDocumentListField(Gate)
    patient = mongoengine.ReferenceField(Patient, reverse_delete_rule=mongoengine.PULL)
    meta = {
        'db_alias': 'core',
        'collection': 'fcs_files'
    }


