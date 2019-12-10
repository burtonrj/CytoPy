from bson.binary import Binary
import pandas as pd
import numpy as np
import mongoengine
import pickle


class DimReduction(mongoengine.EmbeddedDocument):
    """
    Embedded document -> Explorer. Used for storing cached results of dimensionality reduction.
    """
    method = mongoengine.StringField()
    features = mongoengine.ListField()
    data = mongoengine.FileField(db_alias='core', collection_name='explorer_embeddings')

    def pull(self) -> np.array:
        """
        Load embeddings
        :return:  Numpy array of embedded data
        """
        return pickle.loads(self.data.read())

    def put(self, data: np.array) -> None:
        """
        Save embeddings
        :param data: numpy array of events data
        :return: None
        """
        if self.data:
            self.data.replace(Binary(pickle.dumps(data, protocol=2)))
        else:
            self.data.new_file()
            self.data.write(Binary(pickle.dumps(data, protocol=2)))
            self.data.close()


class ExplorerData(mongoengine.Document):
    """
    Model for storing results of Explorer experiments.
    """
    name = mongoengine.StringField(required=True)
    transform = mongoengine.StringField(required=True)
    root_population = mongoengine.StringField(required=True)
    cache = mongoengine.EmbeddedDocumentField(DimReduction)
    column_names = mongoengine.ListField()
    data = mongoengine.FileField(db_alias='core', collection_name='concatenated_data')
    meta = {
        'db_alias': 'core',
        'collection': 'explorer'
    }

    def load(self) -> dict:
        """
        Load explorer data
        :return: Dictionary of explorer data
        """
        cache = None
        if self.cache:
            embeddings = self.cache.pull()
            cache = {'method': self.cache.method,
                     'features': self.cache.features,
                     'data': embeddings}
        return {'transform': self.transform,
                'root_population': self.root_population,
                'data': pd.DataFrame(pickle.loads(self.data.read()), columns=self.column_names),
                'cache': cache}

    def put(self, data: pd.DataFrame) -> None:
        """
        Save embeddings
        :param data: numpy array of events data
        :return: None
        """
        if self.data:
            self.column_names = data.columns.tolist()
            self.data.replace(Binary(pickle.dumps(data.values, protocol=2)))
        else:
            self.data.new_file()
            self.column_names = data.columns.tolist()
            self.data.write(Binary(pickle.dumps(data.values, protocol=2)))
            self.data.close()
