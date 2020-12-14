#!/usr/bin.env/python
# -*- coding: utf-8 -*-
"""
The fcs module houses all functionality for the management and manipulation
of data pertaining to a single biological specimen. This might include
multiple cytometry files (primary staining and controls) all of which
are housed within the FileGroup document. FileGroups should be generated
and access through the Experiment class.

Copyright 2020 Ross Burton

Permission is hereby granted, free of charge, to any person
obtaining a copy of this software and associated documentation
files (the "Software"), to deal in the Software without restriction,
including without limitation the rights to use, copy, modify,
merge, publish, distribute, sublicense, and/or sell copies of the
Software, and to permit persons to whom the Software is furnished
to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included
in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""

from ..feedback import vprint
from ..flow.tree import construct_tree
from ..flow.transforms import apply_transform
from ..flow.neighbours import knn, calculate_optimal_neighbours
from ..flow.sampling import uniform_downsampling
from .geometry import create_convex_hull
from .population import Population, merge_populations, PolygonGeom
from multiprocessing import Pool, cpu_count
from functools import partial
from warnings import warn
from typing import List, Generator
import pandas as pd
import numpy as np
import mongoengine
import anytree
import h5py
import os

__author__ = "Ross Burton"
__copyright__ = "Copyright 2020, CytoPy"
__credits__ = ["Ross Burton", "Simone Cuff", "Andreas Artemiou", "Matthias Eberl"]
__license__ = "MIT"
__version__ = "1.0.0"
__maintainer__ = "Ross Burton"
__email__ = "burtonrj@cardiff.ac.uk"
__status__ = "Production"


def h5_read_population(population_name: str,
                       filepath: str) -> dict:
    """
    Read the population data from a H5 file of a FileGroup. If the population
    is not found in the given H5 file, returns empty dictionary.

    Parameters
    ----------
    population_name: str
        Name of the population
    filepath: str
        Location of the H5 file

    Returns
    -------
    dict
        Nested dictionary of indexes for Population data:
        {"primary": the primary events index,
         "ctrl_index": dictionary of ctrl_id: ctrl file event indexes,
         "cluster_index": dictionary of cluster ID & tag: cluster event indexes}
    """
    with h5py.File(filepath, "r") as f:
        key = f"/index/{population_name}"
        if key not in f.keys():
            return {}
        data = {"primary": f[f"{key}/primary"][:],
                "ctrl_index": {},
                "cluster_index": {}}
        ctrls = [x for x in f[key].keys() if x != "primary"]
        for c in ctrls:
            data["ctrl_index"][c] = f[f"{key}/{c}"][:]
        for cluster in f[f"{key}/clusters"].keys():
            data["cluster_index"][cluster] = f[f"{key}/clusters/{cluster}"][:]
        return data


def h5file_exists(func: callable) -> callable:
    """
    Decorator that asserts the h5 file corresponding to the FileGroup exists.

    Parameters
    ----------
    func: callable
        Function to wrap

    Returns
    -------
    callable
        Wrapper function
    """
    def wrapper(*args, **kwargs):
        assert os.path.isfile(args[0].h5path), f"Could not locate FileGroup HDF5 record {args[0].h5path}"
        return func(*args, **kwargs)
    return wrapper


def _column_names(df: pd.DataFrame,
                  channels: list,
                  markers: list,
                  preference: str = "markers"):
    """
    Given a dataframe of fcs events and lists of channels and markers, set the
    column names according to the given preference.

    Parameters
    ----------
    df: pd.DataFrame
    channels: list
    markers: list
    preference: str
        Valid values are: 'markers' or 'channels'

    Returns
    -------
    Pandas.DataFrame
    """
    mappings = [{"channels": c, "markers": m} for c, m in zip(channels, markers)]
    assert preference in ["markers", "channels"], "preference should be either 'markers' or 'channels'"
    other = [x for x in ["markers", "channels"] if x != preference][0]
    col_names = list(map(lambda x: x[preference] if x[preference] else x[other], mappings))
    df.columns = col_names
    return df


class FileGroup(mongoengine.Document):
    """
    Document representation of a file group; a selection of related fcs files (e.g. a sample and it's associated
    controls).

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
        channels = values.pop("channels", None)
        markers = values.pop("markers", None)
        self.columns_default = values.pop("columns_default", "markers")
        assert self.columns_default in ["markers", "channels"], \
            "columns_default must be one of: 'markers', 'channels'"
        super().__init__(*args, **values)
        self.cell_meta_labels = {}
        if data is not None:
            assert not self.id, "This FileGroup has already been defined"
            assert channels is not None, "Must provide channels to create new FileGroup"
            assert markers is not None, "Must provide markers to create new FileGroup"
            self.save()
            self.h5path = os.path.join(self.data_directory, f"{self.id.__str__()}.hdf5")
            self._init_new_file(data=data, channels=channels, markers=markers)
        else:
            assert self.id is not None, "FileGroup has not been previously defined. Please provide primary data."
            self.h5path = os.path.join(self.data_directory, f"{self.id.__str__()}.hdf5")
            try:
                self._load_cell_meta_labels()
                self._load_population_indexes()
                self.tree = construct_tree(populations=self.populations)
            except AssertionError as err:
                warn(f"Failed to load data for {self.primary_id} ({self.id}); "
                     f"data may be corrupt or missing; {str(err)}")

    @h5file_exists
    def data(self,
             source: str,
             sample_size: int or float or None = None) -> pd.DataFrame:
        """
        Load the FileGroup dataframe for the desired source file.

        Parameters
        ----------
        source: str
            Name of the file to load from e.g. either "primary" or the name of a control
        sample_size: int or float (optional)
            Sample the DataFrame
        Returns
        -------
        Pandas.DataFrame
        """
        with h5py.File(self.h5path, "r") as f:
            assert source in f.keys(), f"Invalid source, expected one of: {f.keys()}"
            channels = [x.decode("utf-8") for x in f[f"mappings/{source}/channels"][:]]
            markers = [x.decode("utf-8") for x in f[f"mappings/{source}/markers"][:]]
            data = _column_names(df=pd.DataFrame(f[source][:]),
                                 channels=channels,
                                 markers=markers,
                                 preference=self.columns_default)
        if sample_size is not None:
            return uniform_downsampling(data=data,
                                        sample_size=sample_size)
        return data

    def _init_new_file(self,
                       data: np.array,
                       channels: List[str],
                       markers: List[str]):
        """
        Under the assumption that this FileGroup has not been previously defined,
        generate a HDF5 file and initialise the root Population

        Parameters
        ----------
        data: Numpy.Array
        channels: list
        markers: list

        Returns
        -------
        None
        """
        with h5py.File(self.h5path, "w") as f:
            f.create_dataset(name="primary", data=data)
            f.create_group("mappings")
            f.create_group("mappings/primary")
            f.create_dataset("mappings/primary/channels", data=np.array(channels, dtype='S'))
            f.create_dataset("mappings/primary/markers", data=np.array(markers, dtype='S'))
            f.create_group("index")
            f.create_group("index/root")
            f.create_group("clusters")
            f.create_group("clusters/root")
            f.create_group("cell_meta_labels")
        self.populations = [Population(population_name="root",
                                       index=np.arange(0, data.shape[0]),
                                       parent="root",
                                       n=data.shape[0])]
        self.tree = {"root": anytree.Node(name="root", parent=None)}
        self.save()

    def add_ctrl_file(self,
                      ctrl_id: str,
                      data: np.array,
                      channels: List[str],
                      markers: List[str]):
        """
        Add a new control file to this FileGroup.

        Parameters
        ----------
        ctrl_id: str
            Name of the control e.g ("CD45RA FMO" or "HLA-DR isotype control"
        data: Numpy.Array
            Single cell events data obtained for this control
        channels: list
            List of channel names
        markers: list
            List of marker names
        Returns
        -------
        None
        """
        with h5py.File(self.h5path, "a") as f:
            assert ctrl_id not in self.controls, f"Entry for {ctrl_id} already exists"
            f.create_dataset(name=ctrl_id, data=data)
            f.create_group(f"mappings/{ctrl_id}")
            f.create_dataset(f"mappings/{ctrl_id}/channels", data=np.array(channels, dtype='S'))
            f.create_dataset(f"mappings/{ctrl_id}/markers", data=np.array(markers, dtype='S'))
        root = self.get_population(population_name="root")
        root.set_ctrl_index(**{ctrl_id: np.arange(0, data.shape[0])})
        self.controls.append(ctrl_id)
        self.save()

    @h5file_exists
    def _load_cell_meta_labels(self):
        """
        Load single cell meta labels from disk

        Returns
        -------
        None
        """
        with h5py.File(self.h5path, "r") as f:
            if "cell_meta_labels" in f.keys():
                for meta in f["cell_meta_labels"].keys():
                    self.cell_meta_labels[meta] = np.array(f[f"cell_meta_labels/{meta}"][:],
                                                           dtype="U")

    @h5file_exists
    def _load_population_indexes(self):
        """
        Load population level event index data from disk

        Returns
        -------
        None
        """
        populations = [p.population_name for p in self.populations]
        read_func = partial(h5_read_population, filepath=self.h5path)
        with Pool(cpu_count()) as pool:
            population_indexes = list(pool.map(read_func, populations))
        for i, pop_idx in enumerate(population_indexes):
            if len(pop_idx) == 0:
                warn(f"Population {populations[i]} missing from H5 file")
            pop = self.get_population(populations[i])
            pop.index = pop_idx.get("primary", None)
            for ctrl_id, ctrl_idx in pop_idx.get("ctrl_index").items():
                pop.set_ctrl_index(**{ctrl_id: ctrl_idx})
            for cluster in pop.clusters:
                try:
                    cluster.index = pop_idx["cluster_index"][f"{cluster.cluster_id}_{cluster.tag}"]
                except KeyError:
                    warn(f"Cluster index missing for cluster ID {cluster.cluster_id}, tag {cluster.tag} "
                         f"in population {pop.population_name}")

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

    def load_ctrl_population_df(self,
                                ctrl: str,
                                population: str,
                                transform: str or dict or None = "logicle",
                                **kwargs):
        """
        Load the DataFrame for the events pertaining to a single population from a
        control. If the control is absent from this FileGroup it will raise an AssertionError.
        If the population has not been estimated for the given control, it will attempt to
        estimate the population using KNearestNeighbours classifier. See estimated_ctrl_population
        for details.

        Parameters
        ----------
        ctrl: str
            Name of the control sample to load
        population: str
            Name of the desired population
        transform: str or dict (optional)
            If given, transformation method applied to the columns of the DataFrame. If the
            value given is a string, it should be the name of the transform method applied
            to ALL columns. If it is a dictionary, keys should correspond to column names
            and values the transform to apply to said column.
        kwargs
            Additional keyword arguments passed to estimated_ctrl_population

        Returns
        -------

        """
        assert ctrl in self.controls, f"No such control {ctrl} associated to this FileGroup"
        if ctrl not in self.get_population(population_name=population).ctrl_index.keys():
            warn(f"Population {population} missing for control {ctrl}, will attempt to "
                 f"estimate population using KNN")
            self.estimate_ctrl_population(ctrl=ctrl, population=population, **kwargs)
        idx = self.get_population(population_name=population).ctrl_index.get(ctrl)
        data = self.data(source=ctrl).loc[idx]
        if isinstance(transform, dict):
            data = apply_transform(data=data, features_to_transform=transform)
        elif isinstance(transform, str):
            data = apply_transform(data, transform_method=transform)
        return data

    def estimate_ctrl_population(self,
                                 ctrl: str,
                                 population: str,
                                 verbose: bool = True,
                                 scoring: str = "balanced_accuracy",
                                 downsample: int or float or None = 0.1,
                                 sml_population_mappings: dict or None = None,
                                 **kwargs):
        """
        Estimate a population for a control sample by training a KNearestNeighbors classifier
        on the population in the primary data and using this model to predict membership
        in the control data. If n_neighbors parameter of Scikit-Learns KNearestNeighbors class
        is not given, it will be estimated using grid search cross-validation and optimisation
        of the given scoring parameter. See CytoPy.flow.neighbours for further details.

        If the
        Results of the population estimation will be saved to the populations ctrl_index property.

        Parameters
        ----------
        ctrl: str
            Control to estimate population for
        population: str
            Population to estimate
        verbose: bool (default=True)
        scoring: str (default="balanced_accuracy")
        downsample: int or float (optional)
        kwargs: dict
            Additional keyword arguments passed to initiate KNearestNeighbors object

        Returns
        -------
        None
        """
        feedback = vprint(verbose=verbose)
        feedback(f"====== Estimating {population} for {ctrl} control ======")
        population = self.get_population(population_name=population)
        if ctrl not in self.get_population(population_name=population.parent).ctrl_index.keys():
            feedback(f"Control missing parent {population.parent}, will attempt to estimate....")
            self.estimate_ctrl_population(ctrl=ctrl,
                                          population=population.parent,
                                          verbose=verbose,
                                          scoring=scoring,
                                          **kwargs)
            feedback(f"{population.parent} estimated, resuming estimation of {population.population_name}....")
        if sml_population_mappings is not None:
            features = sml_population_mappings.get(population.population_name).get("features")
            transformations = sml_population_mappings.get(population.population_name).get("transformations")
        else:
            features = [x for x in [population.geom.x, population.geom.y] if x is not None]
            transformations = {d: transform for d, transform in zip([population.geom.x, population.geom.y],
                                                                    [population.geom.transform_x,
                                                                     population.geom.transform_y])
                               if d is not None}
        training_data = self.load_population_df(population=population.parent,
                                                transform=transformations,
                                                label_downstream_affiliations=False).copy()
        assert training_data.shape[0] > 3, "Three or less events found in training data"
        training_data["labels"] = 0
        training_data.loc[population.index, "labels"] = 1
        if isinstance(downsample, int):
            training_data = pd.concat([training_data[training_data.labels == i].sample(n=downsample)
                                       for i in range(2)])
        if isinstance(downsample, float):
            training_data = pd.concat([training_data[training_data.labels == i].sample(frac=downsample)
                                       for i in range(2)])
        labels = training_data["labels"].values
        n = kwargs.get("n_neighbors", None)
        if n is None:
            feedback("Calculating optimal n_neighbours by grid search CV...")
            n, score = calculate_optimal_neighbours(x=training_data[features].values,
                                                    y=labels,
                                                    scoring=scoring,
                                                    **kwargs)
            feedback(f"Continuing with n={n}; chosen with balanced accuracy of {round(score, 3)}...")
        # Estimate control population using KNN
        feedback("Training KNN classifier....")
        train_acc, val_acc, model = knn(data=training_data,
                                        features=features,
                                        labels=labels,
                                        n_neighbours=n,
                                        holdout_size=0.2,
                                        random_state=42,
                                        return_model=True,
                                        **kwargs)
        feedback(f"...training balanced accuracy score: {train_acc}")
        feedback(f"...validation balanced accuracy score: {val_acc}")
        feedback(f"Predicting {population.population_name} for {ctrl} control...")
        ctrl_data = self.load_ctrl_population_df(ctrl=ctrl,
                                                 population=population.parent,
                                                 transform=transformations,
                                                 label_downstream_affiliations=False)
        assert ctrl_data.shape[0] > 3, "Three or less events found in parent data"
        ctrl_labels = model.predict(ctrl_data[features].values)
        ctrl_idx = ctrl_data.index.values[np.where(ctrl_labels == 1)[0]]
        population.set_ctrl_index(**{ctrl: ctrl_idx})
        feedback("===============================================")

    def load_population_df(self,
                           population: str,
                           transform: str or dict or None = "logicle",
                           label_downstream_affiliations: bool = False) -> pd.DataFrame:
        """
        Load the DataFrame for the events pertaining to a single population.

        Parameters
        ----------
        population: str
            Name of the desired population
        transform: str or dict (optional)
            If given, transformation method applied to the columns of the DataFrame. If the
            value given is a string, it should be the name of the transform method applied
            to ALL columns. If it is a dictionary, keys should correspond to column names
            and values the transform to apply to said column.
        label_downstream_affiliations: bool (default=False)
            If True, an additional column will be generated named "population_label" containing
            the end node membership of each event e.g. if you choose CD4+ population and
            there are subsequent populations belonging to this CD4+ population in a tree
            like: "CD4+ -> CD4+CD25+ -> CD4+CD25+CD45RA+" then the population label column
            will contain the name of the lowest possible "leaf" population that an event is
            assigned too.

        Returns
        -------
        Pandas.DataFrame
        """
        assert population in self.tree.keys(), f"Invalid population, {population} does not exist"
        idx = self.get_population(population_name=population).index
        data = self.data(source="primary").loc[idx]
        if isinstance(transform, dict):
            data = apply_transform(data=data, features_to_transform=transform)
        elif isinstance(transform, str):
            data = apply_transform(data, transform_method=transform)
        if label_downstream_affiliations:
            return self._label_downstream_affiliations(parent=population,
                                                       data=data)
        return data

    def _label_downstream_affiliations(self,
                                       parent: str,
                                       data: pd.DataFrame) -> pd.DataFrame:
        """
        An additional column will be generated named "population_label" containing
        the end node membership of each event e.g. if you choose CD4+ population and
        there are subsequent populations belonging to this CD4+ population in a tree
        like: "CD4+ -> CD4+CD25+ -> CD4+CD25+CD45RA+" then the population label column
        will contain the name of the lowest possible "leaf" population that an event is
        assigned too.

        Parameters
        ----------
        parent: str
        data: Pandas.DataFrame

        Returns
        -------
        Pandas.DataFrame
        """

        data["population_label"] = None
        dependencies = self.list_downstream_populations(parent)
        for pop in dependencies:
            idx = self.get_population(pop).index
            data.loc[idx, 'label'] = pop
        data["population_label"].fillna(parent, inplace=True)
        return data

    def _hdf5_exists(self):
        """
        Tests if associated HDF5 file exists.

        Returns
        -------
        bool
        """
        return os.path.isfile(self.h5path)

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
        Delete given populations. Populations downstream from delete population(s) will
        also be removed.

        Parameters
        ----------
        populations: list or str
            Either a list of populations (list of strings) to remove or a single population as a string.
            If a value of "all" is given, all populations are dropped.

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
            for name in populations:
                self.tree[name].parent = None
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

    def merge_populations(self,
                          left: Population,
                          right: Population,
                          new_population_name: str or None = None):
        """
        Merge two populations present in the current population tree.
        The merged population will have the combined index of both populations but
        will not inherit any clusters and will not be associated to any children
        downstream of either the left or right population. The population will be
        added to the tree as a descendant of the left populations parent. New
        population will be added to FileGroup.

        Parameters
        ----------
        left: Population
        right: Population
        new_population_name: str (optional)

        Returns
        -------
        None
        """
        self.add_population(merge_populations(left=left, right=right, new_population_name=new_population_name))

    def subtract_populations(self,
                             left: Population,
                             right: Population,
                             new_population_name: str or None = None):
        """
        Subtract the right population from the left population.
        The right population must either have the same parent as the left population
        or be downstream of the left population. The new population will descend from
        the same parent as the left population. The new population will have a
        PolygonGeom geom. New population will be added to FileGroup.

        Parameters
        ----------
        left: Population
        right: Population
        new_population_name: str (optional)

        Returns
        -------

        """
        same_parent = left.parent == right.parent
        downstream = right.population_name in list(self.list_downstream_populations(left.population_name))
        assert same_parent or downstream, "Right population should share the same parent as the " \
                                          "left population or be downstream of the left population"
        new_population_name = new_population_name or f"subtract_{left.population_name}_{right.population_name}"
        new_idx = np.array([x for x in left.index if x not in right.index])
        x, y = left.geom.x, left.geom.y
        transform_x, transform_y = left.geom.transform_x, left.geom.transform_y
        parent_data = self.load_population_df(population=left.parent,
                                              transform={x: transform_x,
                                                         y: transform_y})
        x_values, y_values = create_convex_hull(x_values=parent_data.loc[new_idx][x].values,
                                                y_values=parent_data.loc[new_idx][y].values)
        new_geom = PolygonGeom(x=x,
                               y=y,
                               transform_x=transform_x,
                               transform_y=transform_y,
                               x_values=x_values,
                               y_values=y_values)
        new_population = Population(population_name=new_population_name,
                                    parent=left.parent,
                                    n=len(new_idx),
                                    index=new_idx,
                                    geom=new_geom,
                                    warnings=left.warnings + right.warnings + ["SUBTRACTED POPULATION"])
        self.add_population(population=new_population)

    def _write_populations(self):
        """
        Write population data to disk.

        Returns
        -------
        None
        """
        root_n = self.get_population("root").n
        with h5py.File(self.h5path, "a") as f:
            if "cell_meta_labels" in f.keys():
                for meta, labels in self.cell_meta_labels.items():
                    ascii_labels = [x.encode("ascii", "ignore") for x in labels]
                    f.create_dataset(f'/cell_meta_labels/{meta}', data=ascii_labels)
            for p in self.populations:
                parent_n = self.get_population(p.parent).n
                p.prop_of_parent = p.n / parent_n
                p.prop_of_total = p.n / root_n
                f.create_dataset(f'/index/{p.population_name}/primary', data=p.index)
                for ctrl, idx in p.ctrl_index.items():
                    f.create_dataset(f'/index/{p.population_name}/{ctrl}', data=idx)
                for cluster in p.clusters:
                    cluster.prop_of_events = cluster.n / p.n
                    f.create_dataset(f'/clusters/{p.population_name}/{cluster.cluster_id}_{cluster.tag}',
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
            if "cell_meta_labels" in f.keys():
                for meta in self.cell_meta_labels.keys():
                    if meta in f["cell_meta_labels"]:
                        del f[f"cell_meta_labels/{meta}"]
            for p in self.populations:
                if p.population_name in f["index"].keys():
                    if "primary" in f[f"index/{p.population_name}"].keys():
                        del f[f"index/{p.population_name}/primary"]
                    for ctrl_id in p.ctrl_index.keys():
                        if ctrl_id in f[f"index/{p.population_name}"].keys():
                            del f[f"index/{p.population_name}/{ctrl_id}"]
                if p.population_name in f["clusters"].keys():
                    del f[f"clusters/{p.population_name}"]

    def population_stats(self,
                         population: str):
        """

        Parameters
        ----------
        population

        Returns
        -------

        """
        pop = self.get_population(population_name=population)
        parent = self.get_population(population_name=pop.parent)
        root = self.get_population(population_name="root")
        return {"population_name": population,
                "n": pop.n,
                "prop_of_parent": pop.n / parent.n,
                "prop_of_root": pop.n / root.n}

    def quantile_clean(self,
                       upper: float = 0.999,
                       lower: float = 0.001):
        df = self.data(source="primary")
        for x in df.columns:
            df = df[(df[x] >= df[x].quantile(lower)) & (df[x] <= df[x].quantile(upper))]
        clean_pop = Population(population_name="root_clean",
                               index=df.index.values,
                               parent="root",
                               n=df.shape[0])
        self.add_population(clean_pop)

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


def population_stats(filegroup: FileGroup) -> pd.DataFrame:
    """
    Given a FileGroup generate a DataFrame detailing the number of events, proportion
    of parent population, and proportion of total (root population) for each
    population in the FileGroup.

    Parameters
    ----------
    filegroup: FileGroup

    Returns
    -------
    Pandas.DataFrame
    """
    return pd.DataFrame([filegroup.population_stats(p)
                         for p in list(filegroup.list_populations())])