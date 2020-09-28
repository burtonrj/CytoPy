from .mappings import ChannelMap
from .populations import Population
from .gating_strategy import GatingStrategy
from warnings import warn
from typing import List, Generator
import pandas as pd
import numpy as np
import mongoengine
import h5py
import os


def _column_names(df: pd.DataFrame,
                  mappings: list,
                  preference: str = "marker"):
    """
    Given a dataframe of fcs events and a list of ChannelMapping objects, return the dataframe
    with column name updated according to the preference

    Parameters
    ----------
    df: pd.DataFrame
    mappings: list
    preference: str
        Valid values are: 'marker' or 'channel'

    Returns
    -------
    Pandas.DataFrame
    """
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
        if not self.id:
            self.populations = []
            self.save()
        self.h5path = os.path.join(self.data_directory, f"{self.id.__str__()}.hdf5")
        if self.populations:
            self._load_populations()

    def _load_populations(self):
        """
        Load indexes for existing populations from HDF5 file. This includes indexes for controls and clusters.

        Returns
        -------
        None
        """
        assert self._hdf5_exists(), f"Could not locate FileGroup HDF5 record {self.h5path}"
        with h5py.File(self.h5path, "r") as f:
            for pop in self.populations:
                k = f"/index/{pop.population_name}"
                if k + "/primary" not in f.keys():
                    warn(f"Population index missing for {pop.population_name}!")
                else:
                    pop.index = f[k + "/primary"][:]
                    ctrls = [x for x in f[k].keys() if x != "primary"]
                    for c in ctrls:
                        pop.set_ctrl_index(**{c: f[k + f"/{c}"][:]})
                k = f"/clusters/{pop.population_name}"
                for c in pop.clusters:
                    if c.cluster_id not in f[k].keys():
                        warn(f"Cluster index missing for {c.cluster_id} in population {pop.population_name}!")
                    else:
                        c.index = f[k + f"/{c.cluster_id}"][:]

    @staticmethod
    def _sample_data(data: dict,
                     sample_size: int or float):
        """
        Given a dictionary of dataframes like that produced from FileGroup.load() method, sample
        using given sample size.

        Parameters
        ----------
        data: dict
        sample_size: int or float

        Returns
        -------
        dict
        """
        data = data.copy()
        kwargs = dict(n=sample_size)
        if type(sample_size) == float:
            kwargs = dict(frac=sample_size)
        data["primary"] = data["primary"].sample(**kwargs)
        if "controls" in data.keys():
            data["controls"] = {file_id: x.sample(**kwargs)
                                for file_id, x in data["controls"].items()}
        return data

    def load(self,
             sample_size: int or float or None = None,
             include_controls: bool = True,
             columns: str = "marker"):
        """
        Load events data and return as a Pandas DataFrame.

        Parameters
        ----------
        sample_size: int or float or None (optional)
        include_controls: bool (default=True)
        columns: str (default="marker")

        Returns
        -------
        Pandas.DataFrame
        """
        data = {"primary": _column_names(df=pd.read_hdf(self.h5path, "primary"),
                                         mappings=self.channel_mappings,
                                         preference=columns)}
        if include_controls:
            data["controls"] = {ctrl_id: _column_names(df=pd.read_hdf(self.h5path, ctrl_id),
                                                       mappings=self.channel_mappings,
                                                       preference=columns)
                                for ctrl_id in self.controls}
        if sample_size is not None:
            data = self._sample_data(data=data, sample_size=sample_size)
        return data

    def _hdf5_exists(self):
        """
        Tests if associated HDF5 file exists.

        Returns
        -------
        bool
        """
        return os.path.isfile(self.h5path)

    def add_file(self,
                 data: np.array,
                 channel_mappings: List[dict],
                 control: bool = False,
                 ctrl_id: str or None = None):
        """
        Add new file to the FileGroup. Calls `save` upon completion.

        Parameters
        ----------
        data: Numpy.array
            Matrix of events data
        channel_mappings: List[dict]
            List of dictionary objects e.g {"channel": "PE-Cy7", "marker": "CD3"}
        control: bool (default=False)
            Indicates if file represents a control e.g. an FMO or isotype control
        ctrl_id: str or None
            Required if control=True

        Returns
        -------
        None
        """

        if self.channel_mappings:
            self._valid_mappings(channel_mappings)
        else:
            self.channel_mappings = [ChannelMap(channel=x["channel"], marker=x["marker"])
                                     for x in channel_mappings]
        if control:
            assert ctrl_id, "No ctrl_id given"
            if self._hdf5_exists():
                with h5py.File(self.h5path, "r") as f:
                    assert ctrl_id not in f.keys(), f"Control file with ID {ctrl_id} already exists"
            pd.DataFrame(data).to_hdf(self.h5path, key=ctrl_id)
            self.controls.append(ctrl_id)
        else:
            if self._hdf5_exists():
                with h5py.File(self.h5path, "r") as f:
                    assert "primary" not in f.keys(), "There can only be one primary file associated to each file group"
            pd.DataFrame(data).to_hdf(self.h5path, key="primary")
        self.save()

    def _valid_mappings(self, channel_mappings: List[dict]):
        """
        Given a list of dictionaries representing channel mappings, check that they match
        the channel mappings associated to this instance of FileGroup. Raises Assertion error in
        the event that the channel mappings do not match.

        Parameters
        ----------
        channel_mappings: List[dict]

        Returns
        -------
        None
        """
        for cm in channel_mappings:
            err = f"{cm} does not match the expected channel mappings for this file group"
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
                        drop_all: bool = False,
                        meta: bool = False):
        """
        Delete all cluster attaining to a given clustering UID

        Parameters
        ----------
        clustering_uid: str
            Unique identifier for clustering experiment that should have clusters deleted from file
        drop_all: bool
            If True, all clusters for every population are dropped from database regardless of the
            clustering experiment they are associated too
        meta: bool
            If True, delete is applied to meta clusters
        Returns
        -------
        None
        """
        if meta:
            for p in self.populations:
                p.delete_meta_clusters(clustering_uid, drop_all)
        else:
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
        if populations == "all":
            self.populations = [p for p in self.populations if p.population_name != "root"]
        else:
            assert isinstance(populations, list), "Provide a list of population names for removal"
            assert "root" not in populations, "Cannot delete root population"
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
        assert population_name in list(self.list_populations()), f"Invalid population {population_name} does not exist"
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

    def _write_populations(self):
        """
        Write population data to disk.

        Returns
        -------
        None
        """
        root_n = [x for x in self.populations if x.population_name == "root"][0].n
        with h5py.File(self.h5path, "a") as f:
            for p in self.populations:
                parent_n = [x for x in self.populations if x.population_name == p.parent][0].n
                p.prop_of_parent = p.n / parent_n
                p.prop_of_total = p.n / root_n
                f.create_dataset(f'/index/{p.population_name}/primary', data=p.index)
                for ctrl, idx in p.ctrl_index.items():
                    f.create_dataset(f'/index/{p.population_name}/{ctrl}', data=idx)
                for cluster in p.clusters:
                    cluster.prop_of_events = cluster.n/p.n
                    f.create_dataset(f'/clusters/{p.population_name}/{cluster.cluster_id}', data=cluster.index)

    def _hdf_create_population_grps(self):
        """
        Check if index group exists in HDF5 file, if not, create it. Then check that a group
        exists for each population and if absent create a group.

        Returns
        -------
        None
        """
        with h5py.File(self.h5path, "a") as f:
            if "index" not in f.keys():
                f.create_group("index")
            if "clusters" not in f.keys():
                f.create_group("clusters")
            for p in self.populations:
                if p.population_name not in f["index"].keys():
                    f.create_group(f"index/{p.population_name}")
                if p.population_name not in f["clusters"].keys():
                    f.create_group(f"clusters/{p.population_name}")

    def _hdf_reset_population_data(self):
        """
        For each population clear existing data ready for overwriting with
        current data.

        Returns
        -------
        None
        """
        with h5py.File(self.h5path, "a") as f:
            for p in self.populations:
                if "primary" in f[f"index/{p.population_name}"].keys():
                    del f[f"index/{p.population_name}/primary"]
                if p.population_name in f["clusters"].keys():
                    del f[f"clusters/{p.population_name}"]
                for ctrl_id in p.ctrl_index.keys():
                    if ctrl_id in f[f"index/{p.population_name}"].keys():
                        del f[f"index/{p.population_name}/{ctrl_id}"]

    def save(self, *args, **kwargs):
        # Calculate meta and save indexes to disk
        if self.populations:
            # self._hdf_create_population_grps()
            # Populate h5path for populations
            self._hdf_reset_population_data()
            self._write_populations()
        super().save(*args, **kwargs)

    def delete(self,
               delete_hdf5_file: bool = True,
               *args,
               **kwargs):
        if delete_hdf5_file:
            if os.path.isfile(self.h5path):
                os.remove(self.h5path)
            else:
                warn(f"Could not locate hdf5 file {self.h5path}")
        super().delete(*args, **kwargs)
