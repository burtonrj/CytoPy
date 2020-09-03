from .mappings import ChannelMap
from .populations import Population
from .gating_strategy import GatingStrategy
from typing import List, Generator
import pandas as pd
import numpy as np
import mongoengine
import h5py
import os


def _column_names(df: pd.DataFrame,
                  mappings: list,
                  preference: str = "marker"):
    assert preference in ["marker", "channel"], "preference should be either 'marker' or 'channel'"
    other = [x for x in ["marker", "channel"] if x != preference][0]
    col_names = list(map(lambda x: x[preference] if x[preference] else x[other], mappings))
    df.columns = col_names
    return df


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
    data_directory = mongoengine.StringField(required=True)
    controls = mongoengine.ListField()
    compensated = mongoengine.BooleanField(default=False)
    collection_datetime = mongoengine.DateTimeField(required=False)
    processing_datetime = mongoengine.DateTimeField(required=False)
    channel_mappings = mongoengine.EmbeddedDocumentListField(ChannelMap)
    populations = mongoengine.EmbeddedDocumentListField(Population)
    gating_strategy = mongoengine.ReferenceField(GatingStrategy, reverse_delete_rule=mongoengine.PULL)
    valid = mongoengine.BooleanField(default=True)
    notes = mongoengine.StringField(required=False)
    meta = {
        'db_alias': 'core',
        'collection': 'fcs_files'
    }

    def __init__(self, *args, **values):
        super().__init__(*args, **values)
        self.save()
        self.h5path = os.path.join(self.data_directory, f"{self.id.__str__()}.hdf5")

    def load(self,
             sample_size: int or float or None = None,
             include_controls: bool = True,
             columns: str = "marker"):
        data = {"primary": _column_names(df=pd.read_hdf(self.h5path, "primary"),
                                         mappings=self.channel_mappings,
                                         preference=columns)}
        if include_controls:
            data["controls"] = {ctrl_id: _column_names(df=pd.read_hdf(self.h5path, ctrl_id),
                                                       mappings=self.channel_mappings,
                                                       preference=columns)
                                for ctrl_id in self.controls}
        if sample_size is not None:
            if type(sample_size) == int:
                data["primary"] = data["primary"].sample(n=sample_size)
                data["controls"] = {file_id: x.sample(n=sample_size)
                                    for file_id, x in data["controls"].items()}
            else:
                data["primary"] = data["primary"].sample(frac=sample_size)
                data["controls"] = {file_id: x.sample(frac=sample_size)
                                    for file_id, x in data["controls"].items()}
        return data

    def add_file(self,
                 data: np.array,
                 channel_mappings: List[dict],
                 control: bool = False,
                 ctrl_id: str or None = None):

        if self.channel_mappings:
            self._valid_mappings(channel_mappings)
        else:
            self.channel_mappings = [ChannelMap(channel=x["channel"], marker=x["marker"])
                                     for x in channel_mappings]
        if control:
            assert ctrl_id, "No ctrl_id given"
            with h5py.File(self.h5path, "r") as f:
                assert ctrl_id not in f.keys(), f"Control file with ID {ctrl_id} already exists"
            pd.DataFrame(data).to_hdf(self.h5path, key=ctrl_id)
            self.controls.append(ctrl_id)
        else:
            with h5py.File(self.h5path, "r") as f:
                assert "primary" not in f.keys(), "There can only be one primary file associated to each file group"
            pd.DataFrame(data).to_hdf(self.h5path, key="primary")

    def _valid_mappings(self, channel_mappings: List[dict]):
        for cm in channel_mappings:
            err = f"{cm} doesn't match the expected channel mappings for this file group"
            assert sum([x.check_matched_pair(cm["channel"], cm["marker"])
                        for x in self.channel_mappings]) == 1, err

    def list_gated_controls(self) -> Generator:
        """
        List ID of controls that have a cached index in each population of the saved population tree
        (i.e. they have been gated)

        Returns
        -------
        list
            List of control IDs for gated controls
        """
        for c in self.controls():
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
        return [p for p in self.populations if p.population_name == population_name][0]

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
        # Calculate meta and save indexes to disk
        if self.populations:
            root_n = [p for p in self.populations if p.population_name == "root"][0].n
            with h5py.File(self.h5path, "r") as f:
                for p in self.populations:
                    parent_n = [p for p in self.populations if p.population_name == p.parent][0].n
                    p.prop_of_parent = p.n/parent_n
                    p.prop_of_total = p.n/root_n
                    f.create_dataset(f'/index/{p.population_name}', data=p.index)
                    for ctrl, idx in p.ctrl_index.items():
                        f.create_dataset(f'/index/{p.population_name}/{ctrl}', data=idx)
        super().save(*args, **kwargs)

