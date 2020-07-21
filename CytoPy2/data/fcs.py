from .mappings import ChannelMap
from shapely.geometry import Polygon, Point
from shapely.ops import unary_union
from shapely import affinity
from warnings import warn
from typing import List, Generator
from dask import dataframe as dd, array as da
import pandas as pd
import numpy as np
import mongoengine
import h5py
import os


class Population(mongoengine.EmbeddedDocument):
    """
    Cached populations; stores the index of events associated to a population for quick loading.

    Parameters
    ----------
    population_name: str, required
        name of population
    index: FileField
        numpy array storing index of events that belong to population
    prop_of_parent: float, required
        proportion of events as a percentage of parent population
    prop_of_total: float, required
        proportion of events as a percentage of all events
    warnings: list, optional
        list of warnings associated to population
    parent: str, required, (default: "root")
        name of parent population
    children: list, optional
        list of child populations (list of strings)
    geom: list, required
        list of key value pairs (tuples; (key, value)) for defining geom of population e.g.
        the definition for an ellipse that 'captures' the population
    clusters: EmbeddedDocListField
        list of associated Cluster documents
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        f = h5py.File(self._instance.primary_file.data)
        if
        self._index = None

    population_name = mongoengine.StringField()
    n = mongoengine.IntField()
    parent = mongoengine.StringField(required=True, default='root')
    prop_of_parent = mongoengine.FloatField()
    prop_of_total = mongoengine.FloatField()
    warnings = mongoengine.ListField()
    geom = mongoengine.EmbeddedDocument(PopulationGeometry)
    definition = mongoengine.StringField()
    clusters = mongoengine.EmbeddedDocumentListField(Cluster)
    control_idx = mongoengine.EmbeddedDocumentListField(ControlIndex)

    @property
    def index(self):
        return self._index

    @index.setter
    def index(self, arr: da):
        arr.to_hdf5(self._instance.primary_file.data, f'/index/{self.population_name}', arr)
        self._index = arr

    def get_ctrl(self,
                 ctrl_id: str) -> ControlIndex:
        """
        Returns
        -------
        ControlIndex
        """
        for c in self.control_idx:
            if c.control_id == ctrl_id:
                return c
        raise ValueError(f"No such control {ctrl_id}")


class File(mongoengine.EmbeddedDocument):
    """
    Document representation of a single FCS file.

    Parameters
    -----------
    file_id: str, required
        Unique identifier for fcs file
    file_type: str, required, (default='complete')
        One of either 'complete' or 'control'; signifies the type of data stored
    data: FileField
        Numpy array of fcs events data
    compensated: bool, required, (default=False)
        Boolean value, if True then data have been compensated
    channel_mappings: list
        List of standarised channel/marker mappings (corresponds to column names of underlying data)
    """
    file_id = mongoengine.StringField(required=True)
    file_type = mongoengine.StringField(default='complete')
    data = mongoengine.StringField(required=True)
    compensated = mongoengine.BooleanField(default=False)
    channel_mappings = mongoengine.EmbeddedDocumentListField(ChannelMap)


class FileGroup(mongoengine.Document):
    """
    Document representation of a file group; a selection of related fcs files (e.g. a sample and it's associated
    controls)

    Parameters
    ----------
    primary_id: str, required
        Unique ID to associate to group
    files: EmbeddedDocList
        List of File objects
    flags: str, optional
        Warnings associated to file group
    notes: str, optional
        Additional free text
    populations: EmbeddedDocList
        Populations derived from this file group
    gates: EmbeddedDocList
        Gate objects that have been applied to this file group
    collection_datetime: DateTime, optional
        Date and time of sample collection
    processing_datetime: DateTime, optional
        Date and time of sample processing
    """
    primary_id = mongoengine.StringField(required=True)
    primary_file = mongoengine.EmbeddedDocument(File)
    control_files = mongoengine.EmbeddedDocumentListField(File)
    flags = mongoengine.StringField(required=False)
    notes = mongoengine.StringField(required=False),
    collection_datetime = mongoengine.DateTimeField(required=False)
    processing_datetime = mongoengine.DateTimeField(required=False)
    populations = mongoengine.EmbeddedDocumentListField(Population)
    valid = mongoengine.BooleanField(default=True)
    meta = {
        'db_alias': 'core',
        'collection': 'fcs_files'
    }

    def load(self,
             sample_size: int or None = None,
             include_controls: bool = True,
             columns: str = "marker"):
        data = {"primary": dd.read_hdf(self.primary_file.data, self.primary_file.file_id,
                                       columns=[x[columns] for x in self.primary_file.channel_mappings])}
        if include_controls:
            data["controls"] = {f.file_id: dd.read_hdf(f.data, f.file_id,
                                                       columns=[x[columns] for x in f.channel_mappings])
                                for f in self.control_files if f.file_type == "control"}
        if sample_size is not None:
            data["primary"] = data["primary"].sample(n=sample_size)
            data["controls"] = {file_id: x.sample(n=sample_size) for file_id, x in data["controls"].keys()}
        return data

    def add_file(self,
                 file_id: str,
                 file_type: str,
                 data_directory: str,
                 data: np.array,
                 channel_mappings: List[dict],
                 chunks: bool = True,
                 compression: str = "lzf"):
        assert file_type in ["primary", "control"], "File type should be either 'primary' or 'control'"
        assert compression in ["lzf", "gzip"], "Valid compression values: 'lzf' or 'gzip'"
        assert file_id not in list(self.list_files()), f"File with id {file_id} already exists"
        assert os.path.isdir(data_directory), f"{data_directory} is not a valid directory"
        file = File(file_id=file_id,
                    file_type=file_type,
                    channel_mappings=[ChannelMap(channel=x["channel"],
                                                 marker=x["marker"]) for x in channel_mappings])
        data_path = os.path.join(data_directory, f"{file.id.__str__()}.hdf5")
        assert not os.path.isfile(data_path), f"hdf5 file already exists for {file_id}, check path: {data_path}"
        f = h5py.File(data_path, 'w')
        f.create_dataset(file_id, data=data, compression=compression, chunks=chunks, dtype='f8')
        file.data = data_path
        if file_type == "control":
            self.primary_file.append(file)
        elif file_type == "primary":
            self.control_files.append(file)

    def list_files(self) -> Generator:
        for f in [self.primary_file] + self.control_files:
            yield f.file_id

    def list_controls(self) -> Generator:
        """
        Return a list of file IDs for associated control files

        Returns
        -------
        list
        """
        for f in [self.primary_file] + self.control_files:
            if f.file_type == "control":
                yield f.file_id

    def list_gated_controls(self) -> Generator:
        """
        List ID of controls that have a cached index in each population of the saved population tree
        (i.e. they have been gated)

        Returns
        -------
        list
            List of control IDs for gated controls
        """
        for c in self.list_controls():
            if all([p.get_ctrl(c) is not None for p in self.populations]):
                yield c

    def list_populations(self) -> iter:
        """
        Yields list of population names
        Returns
        -------
        Generator
        """
        for p in self.populations:
            yield p.population_name

    def delete_clusters(self,
                        clustering_uid: str or None = None,
                        drop_all: bool = False):
        """
        Delete all cluster attaining to a given clustering UID

        Parameters
        ----------
        clustering_uid: str
            Unique identifier for clustering experiment that should have clusters deleted from file
        drop_all: bool
            If True, all clusters for every population are dropped from database regardless of the
            clustering experiment they are associated too
        Returns
        -------
        None
        """
        if not drop_all:
            assert clustering_uid, 'Must provide a valid clustering experiment UID'
        for p in self.populations:
            p.delete_clusters(clustering_uid, drop_all)
        self.save()

    def delete_populations(self, populations: list or str) -> None:
        """
        Delete given populations

        Parameters
        ----------
        populations: list or str
            Either a list of populations (list of strings) to remove or a single population as a string

        Returns
        -------
        None
        """
        if populations == all:
            self.populations = []
        else:
            self.populations = [p for p in self.populations if p.population_name not in populations]
        self.save()

    def update_population(self,
                          population_name: str,
                          new_population: Population):
        """
        Given an existing population name, replace that population with the new population document

        Parameters
        -----------
        population_name: str
            Name of population to be replaced
        new_population: Population
            Updated/new Population document

        Returns
        --------
        None
        """
        self.populations = [p for p in self.populations if p.population_name != population_name]
        self.populations.append(new_population)
        self.save()

    def get_population(self,
                       population_name: str) -> Population:
        """
        Given the name of a population associated to the FileGroup, returns the Population object, with
        index and control index ready loaded.

        Parameters
        ----------
        population_name: str
            Name of population to retrieve from database

        Returns
        -------
        Population
        """
        assert population_name in list(self.list_populations()), f'Population {population_name} does not exist'
        p = [p for p in self.populations if p.population_name == population_name][0]
        p.load_index()
        for ctrl in p.control_idx:
            ctrl.load_index()
        return p

    def get_population_by_parent(self,
                                 parent: str) -> Generator:
        """
        Given the name of some parent population, return a list of Population object whom's parent matches

        Parameters
        ----------
        parent: str
            Name of the parent population to search for

        Returns
        -------
        Generator
            List of Populations
        """
        for p in self.populations:
            if p.parent == parent:
                yield p

    def save(self, *args, **kwargs):
        root_n = [p for p in self.populations if p.population_name == "root"][0].n
        for p in self.populations:
            parent_n = [p for p in self.populations if p.population_name == p.parent][0].n
            p.prop_of_parent = p.n/parent_n
            p.prop_of_total = p.n/root_n
            for ctrl in p.control_idx:
                ctrl.save_index()
            p.save_index()
        super().save(*args, **kwargs)
