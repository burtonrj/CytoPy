from ..flow.tree import construct_tree
from .mapping import ChannelMap
from .population import Population
from warnings import warn
from typing import List, Generator, Dict
import pandas as pd
import numpy as np
import mongoengine
import anytree
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
    gating_strategy = mongoengine.ListField()
    valid = mongoengine.BooleanField(default=True)
    notes = mongoengine.StringField(required=False)
    meta = {
        'db_alias': 'core',
        'collection': 'fcs_files'
    }

    def __init__(self, *args, **values):
        data = values.pop("data", None)
        mappings = values.pop("mappings", None)
        super().__init__(*args, **values)
        if data is not None and mappings is not None:
            assert not self.id, "This FileGroup has already been defined"
            err = "Invalid data: should be a dictionary of dataframes"
            assert isinstance(data, dict), err
            assert all([isinstance(x, pd.DataFrame) for x in data.values()]), err
            self.channel_mappings = [ChannelMap(channel=x["channel"], marker=x["marker"])
                                     for x in mappings]
            self.save()
            self.h5path = os.path.join(self.data_directory, f"{self.id.__str__()}.hdf5")
            self._init_new_file(data=data)
            self.data = self.load(columns=values.get("columns", "marker"))
        else:
            self.h5path = os.path.join(self.data_directory, f"{self.id.__str__()}.hdf5")
            self.data = self.load(columns=values.get("columns", "marker"))
            self._load_populations()
            self.tree = construct_tree(populations=self.populations)

    def _init_new_file(self,
                       data: Dict[str, pd.DataFrame]):
        """
        Under the assumption that this FileGroup has not been previously defined,
        generate a HDF5 file and initialise the root Population

        Parameters
        ----------
        data: dict
            Dictionary of dataframes. Should contain a key for "primary" being
            the primary staining data. The remaining keys and values will be
            stored as control files.

        Returns
        -------
        None
        """
        err = "Data dictionary values should be pandas dataframes"
        assert all([isinstance(x, pd.DataFrame) for x in data.values()]), err
        ctrl_idx = {}
        for name, df in data.items():
            if name != "primary":
                df.to_hdf(self.h5path, key=name)
                self.controls.append(name)
                ctrl_idx[name] = df.index.values
            else:
                df.to_hdf(self.h5path, key="primary")
        self.populations = [Population(population_name="root",
                                       index=data.get("primary").index.values,
                                       parent="root",
                                       n=len(data.get("primary").index.values))]
        for ctrl_id, ctrl_data in ctrl_idx.items():
            self.get_population("root").set_ctrl_index(**{ctrl_id: ctrl_data})
        with h5py.File(self.h5path, "a") as f:
            f.create_group("index")
            f.create_group("index/root")
            f.create_group("clusters")
            f.create_group("cluster/root")
        self.tree = {"root": anytree.Node(name="root", parent=None)}
        self.save()

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

    def add_population(self,
                       population: Population):
        """
        Add a new Population to this FileGroup.

        Parameters
        ----------
        population: Population

        Returns
        -------
        None
        """
        err = f"Population with name '{population.population_name}' already exists"
        assert population.population_name not in self.tree.keys(), err
        self.populations.append(population)
        self.tree[population.population_name] = anytree.Node(name=population.population_name,
                                                             parent=self.tree.get(population.parent))

    def load(self,
             sample_size: int or float or None = None,
             include_controls: bool = True,
             columns: str = "marker") -> dict:
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
        assert os.path.isfile(self.h5path), "FileGroup is empty!"
        data = {"primary": _column_names(df=pd.read_hdf(self.h5path, "primary"),
                                         mappings=self.channel_mappings,
                                         preference=columns)}
        if include_controls:
            data["controls"] = {ctrl_id: _column_names(df=pd.read_hdf(self.h5path, ctrl_id),
                                                       mappings=self.channel_mappings,
                                                       preference=columns)
                                for ctrl_id in self.controls}
        if sample_size is not None:
            return self._sample_data(data=data, sample_size=sample_size)
        return data

    def _hdf5_exists(self):
        """
        Tests if associated HDF5 file exists.

        Returns
        -------
        bool
        """
        return os.path.isfile(self.h5path)

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

    def print_population_tree(self,
                              image: bool = False,
                              path: str or None = None):
        """
        Print population tree to stdout or save as an image if 'image' is True.

        Parameters
        ----------
        image: bool (default=False)
            Save tree as a png image
        path: str (optional)
            File path for image, ignored if 'image' is False.
            Defaults to working directory.

        Returns
        -------
        None
        """
        root = self.tree['root']
        if image:
            from anytree.exporter import DotExporter
            path = path or f'{os.getcwd()}/{self.id}_population_tree.png'
            DotExporter(root).to_picture(path)
        for pre, fill, node in anytree.RenderTree(root):
            print('%s%s' % (pre, node.name))

    def delete_clusters(self,
                        tag: str or None = None,
                        meta_label: str or None = None,
                        drop_all: bool = False):
        """

        Parameters
        ----------
        tag
        meta_label
        drop_all

        Returns
        -------

        """
        if drop_all:
            for p in self.populations:
                p.delete_all_clusters(clusters="all")
        elif tag:
            for p in self.populations:
                p.delete_cluster(tag=tag)
        elif meta_label:
            for p in self.populations:
                p.delete_cluster(meta_label=meta_label)
        else:
            raise ValueError("If drop_all is False, must provide tag or meta_label")

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
            self.populations = [p for p in self.populations if p.population_name == "root"]
            self.tree = {name: node for name, node in self.tree.items() if name == "root"}
        else:
            assert isinstance(populations, list), "Provide a list of population names for removal"
            assert "root" not in populations, "Cannot delete root population"
            downstream_effects = [self.list_downstream_populations(p) for p in populations]
            downstream_effects = set([x for sl in downstream_effects for x in sl])
            if len(downstream_effects) > 0:
                warn("The following populations are downstream of one or more of the "
                     "populations listed for deletion and will therefore be deleted: "
                     f"{downstream_effects}")
            populations = list(set(list(downstream_effects) + populations))
            self.populations = [p for p in self.populations if p.population_name not in populations]
            self.tree = {name: node for name, node in self.tree.items() if name not in populations}


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
            if p.parent == parent and p.population_name != "root":
                yield p

    def list_downstream_populations(self,
                                    population: str) -> list or None:
        """For a given population find all dependencies

        Parameters
        ----------
        population : str
            population name

        Returns
        -------
        list or None
            List of populations dependent on given population

        """
        assert population in self.tree.keys(), f'population {population} does not exist; ' \
                                               f'valid population names include: {self.tree.keys()}'
        root = self.tree['root']
        node = self.tree[population]
        dependencies = [x.name for x in anytree.findall(root, filter_=lambda n: node in n.path)]
        return [p for p in dependencies if p != population]

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
                    f.create_dataset(f'/clusters/{p.population_name}/{cluster.cluster_id}',
                                     data=cluster.index)

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
                if p.population_name in f["index"].keys():
                    if "primary" in f[f"index/{p.population_name}"].keys():
                        del f[f"index/{p.population_name}/primary"]
                    for ctrl_id in p.ctrl_index.keys():
                        if ctrl_id in f[f"index/{p.population_name}"].keys():
                            del f[f"index/{p.population_name}/{ctrl_id}"]
                if p.population_name in f["clusters"].keys():
                    del f[f"clusters/{p.population_name}"]

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
        super().delete(*args, **kwargs)
        if delete_hdf5_file:
            if os.path.isfile(self.h5path):
                os.remove(self.h5path)
            else:
                warn(f"Could not locate hdf5 file {self.h5path}")
