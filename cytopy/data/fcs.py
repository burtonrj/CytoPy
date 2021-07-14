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

from ..flow.tree import construct_tree
from ..flow.transform import apply_transform, apply_transform_map
from ..flow.sampling import uniform_downsampling
from ..flow.build_models import build_sklearn_model
from .geometry import create_envelope
from .population import Population, merge_gate_populations, merge_non_geom_populations, PolygonGeom
from .subject import Subject
from .errors import *
from sklearn.model_selection import StratifiedKFold, permutation_test_score
from imblearn.over_sampling import RandomOverSampler
from typing import *
import pandas as pd
import numpy as np
import mongoengine
import logging
import anytree
import h5py
import os
import re

__author__ = "Ross Burton"
__copyright__ = "Copyright 2020, cytopy"
__credits__ = ["Ross Burton", "Simone Cuff", "Andreas Artemiou", "Matthias Eberl"]
__license__ = "MIT"
__version__ = "2.0.0"
__maintainer__ = "Ross Burton"
__email__ = "burtonrj@cardiff.ac.uk"
__status__ = "Production"
logger = logging.getLogger("FileGroup")


def data_loaded(func: callable) -> callable:
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

    Raises
    ------
    ValueError
        HDF5 file does not exist
    """

    def wrapper(*args, **kwargs):
        try:
            assert args[0].h5path is not None, "Data directory and therefore HDF5 path has not been defined."
            assert os.path.isfile(args[0].h5path), f"Could not locate FileGroup HDF5 record {args[0].h5path}."
            return func(*args, **kwargs)
        except AssertionError as e:
            logger.exception(e)
            raise ValueError(str(e))

    return wrapper


def population_in_file(func: callable):
    """
    Wrapper to test if requested population passed to the given function
    exists in the given h5 file object

    Parameters
    ----------
    func: callable
        Function to wrap

    Returns
    -------
    callable
    """

    def wrapper(population_name: str,
                h5file: h5py.File):
        if population_name not in h5file["index"].keys():
            return None
        return func(population_name, h5file)

    return wrapper


@population_in_file
def h5_read_population_primary_index(population_name: str,
                                     h5file: h5py.File):
    """
    Given a population and an instance of a H5 file object, return the
    index of corresponding events

    Parameters
    ----------
    population_name: str
    h5file: h5py.File

    Returns
    -------
    numpy.ndarray
    """
    return h5file[f"/index/{population_name}/primary"][:]


def set_column_names(df: pd.DataFrame,
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

    Raises
    ------
    AssertionError
        Preference must be either 'markers' or 'channels'
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

    Attributes
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
    valid: BooleanField (default=True)
        True if FileGroup is valid
    subject: ReferenceField
        Reference to Subject. If Subject is deleted, this field is nullified but
        the FileGroup will persist
    """
    primary_id = mongoengine.StringField(required=True)
    controls = mongoengine.ListField()
    compensated = mongoengine.BooleanField(default=False)
    collection_datetime = mongoengine.DateTimeField(required=False)
    processing_datetime = mongoengine.DateTimeField(required=False)
    populations = mongoengine.EmbeddedDocumentListField(Population)
    gating_strategy = mongoengine.ListField()
    valid = mongoengine.BooleanField(default=True)
    notes = mongoengine.StringField(required=False)
    subject = mongoengine.ReferenceField(Subject, reverse_delete_rule=mongoengine.NULLIFY)
    data_directory = mongoengine.StringField()
    meta = {
        'db_alias': 'core',
        'collection': 'fcs_files'
    }

    def __init__(self, *args, **kwargs):
        data = kwargs.pop("data", None)
        channels = kwargs.pop("channels", None)
        markers = kwargs.pop("markers", None)
        super().__init__(*args, **kwargs)
        self._columns_default = "markers"
        self.cell_meta_labels = {}
        if self.id:
            self.h5path = os.path.join(self.data_directory, f"{self.id.__str__()}.hdf5")
            self.tree = construct_tree(populations=self.populations)
            self._load_cell_meta_labels()
            self._load_population_indexes()
        else:
            logger.info(f"Creating new FileGroup {self.primary_id}")
            if any([x is None for x in [data, channels, markers]]):
                raise ValueError("New instance of FileGroup requires that data, channels, and markers "
                                 "be provided to the constructor")
            self.save()
            self.h5path = os.path.join(self.data_directory, f"{self.id.__str__()}.hdf5")
            self.init_new_file(data=data, channels=channels, markers=markers)

    @property
    def columns_default(self):
        return self._columns_default

    @columns_default.setter
    def columns_default(self, value: str):
        assert value in ["markers", "channels"], "columns_default must be either 'markers' or 'channels'"
        self._columns_default = value

    @data_loaded
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

        Raises
        ------
        ValueError
            Invalid source
        """
        with h5py.File(self.h5path, "r") as f:
            if source not in f.keys():
                logging.error(f"Invalid source given on access to {self.primary_id} ({self.id}) HDF5, expected "
                              f"one of {f.keys()}")
                raise ValueError(f"Invalid source, expected one of: {f.keys()}")
            channels = [x.decode("utf-8") for x in f[f"mappings/{source}/channels"][:]]
            markers = [x.decode("utf-8") for x in f[f"mappings/{source}/markers"][:]]
            data = set_column_names(df=pd.DataFrame(f[source][:], dtype=np.float32),
                                    channels=channels,
                                    markers=markers,
                                    preference=self.columns_default)
        if sample_size is not None:
            return uniform_downsampling(data=data,
                                        sample_size=sample_size)
        return data

    def init_new_file(self,
                      data: np.array,
                      channels: List[str],
                      markers: List[str]):
        """
        Under the assumption that this FileGroup has not been previously defined,
        generate a HDF5 file and initialise the root Population

        Parameters
        ----------
        data: numpy.ndarray
        channels: list
        markers: list

        Returns
        -------
        None
        """
        logging.debug(f"Creating new HDF5 file for {self.primary_id}; {self.id} @ {self.h5path}")
        if os.path.isfile(self.h5path):
            logging.debug(f"{self.h5path} already exists, deleting.")
            os.remove(self.h5path)
        with h5py.File(self.h5path, "w") as f:
            f.create_dataset(name="primary", data=data)
            f.create_group("mappings")
            f.create_group("mappings/primary")
            f.create_dataset("mappings/primary/channels", data=np.array(channels, dtype='S'))
            f.create_dataset("mappings/primary/markers", data=np.array(markers, dtype='S'))
            f.create_group("index")
            f.create_group("index/root")
            f.create_group("cell_meta_labels")
        self.populations = [Population(population_name="root",
                                       index=np.arange(0, data.shape[0]),
                                       parent="root",
                                       n=data.shape[0],
                                       source="root")]
        self.tree = {"root": anytree.Node(name="root", parent=None)}
        self.save()
        logging.info(f"{self.h5path} created successfully.")

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
        data: numpy.ndarray
            Single cell events data obtained for this control
        channels: list
            List of channel names
        markers: list
            List of marker names

        Returns
        -------
        None

        Raises
        ------
        ValueError
            If control already exists
        """
        with h5py.File(self.h5path, "a") as f:
            if ctrl_id in self.controls:
                logger.error(f"{ctrl_id} already exists for {self.primary_id}; {self.id}")
                raise ValueError(f"Entry for {ctrl_id} already exists")
            f.create_dataset(name=ctrl_id, data=data)
            f.create_group(f"mappings/{ctrl_id}")
            f.create_dataset(f"mappings/{ctrl_id}/channels", data=np.array(channels, dtype='S'))
            f.create_dataset(f"mappings/{ctrl_id}/markers", data=np.array(markers, dtype='S'))
        self.controls.append(ctrl_id)
        self.save()
        logger.debug(f"Generated new control dataset in HDF5 file for {self.primary_id}")

    @data_loaded
    def _load_cell_meta_labels(self):
        """
        Load single cell meta labels from disk

        Returns
        -------
        None
        """
        with h5py.File(self.h5path, "r") as f:
            logger.debug(f"Loading cell meta labels from {self.primary_id}; {self.h5path}")
            if "cell_meta_labels" in f.keys():
                for meta in f["cell_meta_labels"].keys():
                    self.cell_meta_labels[meta] = np.array(f[f"cell_meta_labels/{meta}"][:],
                                                           dtype="U")

    @data_loaded
    def _load_population_indexes(self):
        """
        Load population level event index data from disk

        Returns
        -------
        None
        """
        with h5py.File(self.h5path, "r") as f:
            for p in self.populations:
                logger.debug(f"Loading {p} population idx from {self.primary_id}; {self.h5path}")
                primary_index = h5_read_population_primary_index(population_name=p.population_name,
                                                                 h5file=f)
                if primary_index is None:
                    continue
                p.index = primary_index

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

        Raises
        ------
        DuplicatePopulationError
            Population already exists

        AssertionError
            Population is missing index
        """
        logger.debug(f"Adding new population {population} to {self.primary_id}; {self.id}")
        if population.population_name in self.tree.keys():
            err = f"Population with name '{population.population_name}' already exists"
            raise DuplicatePopulationError(err)
        assert population.index is not None, "Population index is empty"
        if population.n is None:
            population.n = len(population.index)
        self.populations.append(population)
        self.tree[population.population_name] = anytree.Node(name=population.population_name,
                                                             parent=self.tree.get(population.parent))

    def update_population(self,
                          pop: Population):
        """
        Replace an existing population. Population to replace identified using 'population_name' field.
        Note: this method does not allow you to edit the

        Parameters
        ----------
        pop: Population
            New population object

        Returns
        -------
        None
        """
        logger.debug(f"Updating population {pop.population_name}")
        assert pop.population_name in self.list_populations(), 'Invalid population, does not exist'
        self.populations = [p for p in self.populations if p.population_name != pop.population_name]
        self.populations.append(pop)

    def load_ctrl_population_df(self,
                                ctrl: str,
                                population: str,
                                classifier: str = "XGBClassifier",
                                classifier_params: dict or None = None,
                                scoring: str = "balanced_accuracy",
                                transform: str = "logicle",
                                transform_kwargs: dict or None = None,
                                evaluate_classifier: bool = True,
                                kfolds: int = 5,
                                n_permutations: int = 25,
                                sample_size: int = 10000) -> pd.DataFrame:
        """
        Load a population from an associated control. The assumption here is that control files
        have been collected at the same time as primary staining and differ by the absence or
        permutation of a marker/channel/stain. Therefore the population of interest in the
        primary staining will be used as training data to identify the equivalent population in
        the control.

        The user should specify the control file, the population they want (which MUST already exist
        in the primary staining) and the type of classifier to use. Additional parameters can be
        passed to control the classifier and stratified cross validation with permutation testing
        will be performed if evalidate_classifier is set to True.

        Parameters
        ----------
        ctrl: str
            Control file to estimate population for
        population: str
            Population of interest. MUST already exist in the primary staining.
        classifier: str (default='XGBClassifier')
            Classifier to use. String value should correspond to a valid Scikit-Learn classifier class
            name or XGBClassifier for XGBoost.
        classifier_params: dict, optional
            Additional keyword arguments passed when initiating the classifier
        scoring: str (default='balanced_accuracy')
            Method used to evaluate the performance of the classifier if evaluate_classifier is True.
            String value should be one of the functions of Scikit-Learn's classification metrics:
            https://scikit-learn.org/stable/modules/model_evaluation.html.
        transform: str (default='logicle')
            Transformation to be applied to data prior to classification
        transform_kwargs: dict, optional
            Additional keyword arguments applied to Transformer
        evaluate_classifier: bool (default=True)
            If True, stratified cross validation with permutating testing is applied prior to
            predicting control population,  feeding back to stdout the performance of the classifier
            across k folds and n permutations
        kfolds: int (default=5)
            Number of cross validation rounds to perform if evaluate_classifier is True
        n_permutations: int (default=25)
            Number of rounds of permutation testing to perform if evaluate_classifier is True
        sample_size: int (default=10000)
            Number of events to sample from primary data for training

        Returns
        -------
        Pandas.DataFrame

        Raises
        ------
        AssertionError
            If desired population is not found in the primary staining

        MissingControlError
            If the chosen control does not exist
        """
        logger.info(
            f"Predicting {population} in control data in {self.primary_id} {ctrl} using {classifier} classifier")
        transform_kwargs = transform_kwargs or {}
        if ctrl not in self.controls:
            raise MissingControlError(f"No such control {ctrl} associated to this FileGroup")
        params = classifier_params or {}
        transform_kwargs = transform_kwargs or {}
        classifier = build_sklearn_model(klass=classifier, **params)
        assert population in self.list_populations(), f"Desired population {population} not found"

        logger.info("Loading primary and control data")
        training, ctrl, transformer = _load_data_for_ctrl_estimate(filegroup=self,
                                                                   target_population=population,
                                                                   ctrl=ctrl,
                                                                   transform=transform,
                                                                   sample_size=sample_size,
                                                                   **transform_kwargs)
        features = [x for x in training.columns if x != "label"]
        features = [x for x in features if x in ctrl.columns]
        x, y = training[features], training["label"].values

        if evaluate_classifier:
            logger.info("Evaluating classifier with permutation testing...")
            skf = StratifiedKFold(n_splits=kfolds, random_state=42, shuffle=True)
            score, permutation_scores, pvalue = permutation_test_score(classifier, x, y,
                                                                       cv=skf,
                                                                       n_permutations=n_permutations,
                                                                       scoring=scoring,
                                                                       n_jobs=-1,
                                                                       random_state=42)
            logger.info(f"...Performance (without permutations): {round(score, 4)}")
            logger.info(f"...Performance (average across permutations; standard dev): "
                        f"{round(np.mean(permutation_scores), 4)}; {round(np.std(permutation_scores), 4)}")
            logger.info(f"...p-value (comparison of original score to permuations): {round(pvalue, 4)}")

        logger.info("Predicting population for control data...")
        classifier.fit(x, y)
        ctrl_labels = classifier.predict(ctrl[features])
        training_prop_of_root = self.get_population(population).n / self.get_population("root").n
        ctrl_prop_of_root = np.sum(ctrl_labels) / ctrl.shape[0]
        logger.info(f"{population}: {round(training_prop_of_root, 3)}% of root in primary data")
        logger.info(f"Predicted in ctrl: {round(ctrl_prop_of_root, 3)}% of root in control data")
        ctrl = ctrl.iloc[np.where(ctrl_labels == 1)[0]]
        if transformer:
            return transformer.inverse_scale(data=ctrl, features=list(ctrl.columns))
        return ctrl

    def load_multiple_populations(self,
                                  populations: Optional[List[str]] = None,
                                  regex: Optional[str] = None,
                                  transform: Optional[Union[str, Dict]] = "logicle",
                                  features_to_transform: Optional[List] = None,
                                  transform_kwargs: Optional[Dict] = None,
                                  label_parent: bool = False,
                                  frac_of: Optional[List[str]] = None):
        """
        Load a DataFrame of single cell data obtained from multiple populations. Population data
        is merged and identifiable from the column 'population_label'

        Parameters
        ----------
        populations: List[str]
            Populations of interest (will log a warning and skip population if it doesn't exist)
        regex: str, optional
            Provide a regular expression pattern and will return matching populations (ignores populations
            if provided)
        transform: str or dict, optional (default="logicle")
            Transform to be applied; specify a value of None to not perform any transformation
        features_to_transform: list, optional
            Features (columns) to be transformed. If not provied, all columns transformed
        transform_kwargs: dict, optional
            Additional keyword arguments passed to Transformer
        label_parent: bool (default=False)
            If True, additional column appended with parent name for each population
        frac_of: list, optional
            Provide a list of populations and additional columns will be appended to resulting
            DataFrame containing the fraction of the requested population compared to each population
            in this list

        Returns
        -------
        Pandas.DataFrame

        Raises
        ------
        ValueError
            Must provide list of populations or a regex pattern
        """
        dataframe = list()
        if regex is None and populations is None:
            raise ValueError("Must provide list of populations or a regex pattern")

        if regex:
            populations = self.list_populations(regex=regex)
        for p in populations:
            try:
                pop_data = self.load_population_df(population=p,
                                                   transform=transform,
                                                   transform_kwargs=transform_kwargs,
                                                   features_to_transform=features_to_transform,
                                                   label_parent=label_parent,
                                                   frac_of=frac_of)
                pop_data["population_label"] = p
                dataframe.append(pop_data)
            except ValueError:
                logger.warning(f"{self.primary_id} does not contain population {p}")
        return pd.concat(dataframe)

    def load_population_df(self,
                           population: str,
                           transform: str or dict or None = "logicle",
                           features_to_transform: list or None = None,
                           transform_kwargs: dict or None = None,
                           label_parent: bool = False,
                           frac_of: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Load the DataFrame for the events pertaining to a single population.

        Parameters
        ----------
        population: str
            Name of the desired population
        transform: str or dict, optional (default="logicle")
            Transform to be applied; specify a value of None to not perform any transformation
        features_to_transform: list, optional
            Features (columns) to be transformed. If not provied, all columns transformed
        transform_kwargs: dict, optional
            Additional keyword arguments passed to Transformer
        label_parent: bool (default=False)
            If True, additional column appended with parent name for each population
        frac_of: list, optional
            Provide a list of populations and additional columns will be appended to resulting
            DataFrame containing the fraction of the requested population compared to each population
            in this list

        Returns
        -------
        Pandas.DataFrame

        Raises
        ------
        ValueError
            Invalid population, does not exist
        """
        if population not in self.tree.keys():
            logger.error(f"Invalid population, {population} does not exist for {self.primary_id}; {self.id}")
            raise ValueError(f"Invalid population, {population} does not exist")

        population = self.get_population(population_name=population)
        data = self.data(source="primary").loc[population.index]

        if transform is not None:
            features_to_transform = features_to_transform or list(data.columns)
            transform_kwargs = transform_kwargs or {}
            if isinstance(transform, dict):
                data = apply_transform_map(data=data, feature_method=transform, kwargs=transform_kwargs)
            else:
                data = apply_transform(data=data,
                                       method=transform,
                                       features=features_to_transform,
                                       return_transformer=False,
                                       **transform_kwargs)
        if label_parent:
            data["parent_label"] = population.parent

        if frac_of is not None:
            for comparison_pop in frac_of:
                if comparison_pop not in self.list_populations():
                    logger.warning(f"{comparison_pop} in 'frac_of' is not a recognised population")
                    continue
                comparison_pop = self.get_population(population_name=comparison_pop)
                data[f"frac of {comparison_pop.population_name}"] = population.n / comparison_pop.n

        return data

    def _hdf5_exists(self):
        """
        Tests if associated HDF5 file exists.

        Returns
        -------
        bool
        """
        return os.path.isfile(self.h5path)

    def list_populations(self,
                         regex: Optional[str] = None) -> list:
        """
        List population names

        Parameters
        ----------
        regex: str, optional
            Provide a regular expression pattern and only matching populations will be returned.

        Returns
        -------
        List
        """
        populations = [p.population_name for p in self.populations]
        if regex:
            regex = re.compile(regex)
            return list(filter(regex.match, populations))
        return populations

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

        Raises
        ------
        ValueError
            If invalid value given for populations
        """
        if populations == "all":
            logger.info(f"Deleting all populations in {self.primary_id}; {self.id}")
            for p in self.populations:
                self.tree[p.population_name].parent = None
            self.populations = [p for p in self.populations if p.population_name == "root"]
            self.tree = {name: node for name, node in self.tree.items() if name == "root"}
        else:
            try:
                logger.info(f"Deleting population(s) {populations} in {self.primary_id}; {self.id}")
                assert isinstance(populations, list), "Provide a list of population names for removal"
                assert "root" not in populations, "Cannot delete root population"
                downstream_effects = [self.list_downstream_populations(p) for p in populations]
                downstream_effects = set([x for sl in downstream_effects for x in sl])
                if len(downstream_effects) > 0:
                    logging.warning("The following populations are downstream of one or more of the "
                                    "populations listed for deletion and will therefore be deleted: "
                                    f"{downstream_effects}")
                populations = list(set(list(downstream_effects) + populations))
                self.populations = [p for p in self.populations if p.population_name not in populations]
                for name in populations:
                    self.tree[name].parent = None
                self.tree = {name: node for name, node in self.tree.items() if name not in populations}
            except AssertionError as e:
                logger.warning(e)
            except ValueError as e:
                logger.warning(e)

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

        Raises
        ------
        MissingPopulationError
            If population doesn't exist
        """
        if population_name not in list(self.list_populations()):
            raise MissingPopulationError(f'Population {population_name} does not exist')
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

        Raises
        ------
        AssertionError
            If Population does not exist
        """
        assert population in self.tree.keys(), f'population {population} does not exist; ' \
                                               f'valid population names include: {self.tree.keys()}'
        root = self.tree['root']
        node = self.tree[population]
        dependencies = [x.name for x in anytree.findall(root, filter_=lambda n: node in n.path)]
        return [p for p in dependencies if p != population]

    def merge_gate_populations(self,
                               left: Population or str,
                               right: Population or str,
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
        logger.info(f"Merging {left} and {right} populations for {self.primary_id}; {self.id}")
        if isinstance(left, str):
            left = self.get_population(left)
        if isinstance(right, str):
            right = self.get_population(right)
        self.add_population(merge_gate_populations(left=left, right=right, new_population_name=new_population_name))

    def merge_non_geom_populations(self,
                                   populations: list,
                                   new_population_name: str):
        """
        Merge multiple populations that are sourced either for classification or clustering methods.
        (Not supported for populations from autonomous gates)

        Parameters
        ----------
        populations: list
            List of populations to merge
        new_population_name: str
            Name of the new population

        Returns
        -------
        None

        Raises
        ------
        ValueError
            If populations is invalid
        """
        logger.info(f"Merging {populations} populations for {self.primary_id}; {self.id}")
        pops = list()
        for p in populations:
            if isinstance(p, str):
                pops.append(self.get_population(p))
            elif isinstance(p, Population):
                pops.append(p)
            else:
                raise ValueError("populations should be a list of strings or list of Population objects")
        self.add_population(merge_non_geom_populations(populations=pops, new_population_name=new_population_name))

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
        None

        Raises
        ------
        ValueError
            If left and right population are not sourced from root or Gate
        KeyError
            If left and right population do not share the same parent or the right population
            is not downstream of the left population
        """
        logger.info(f"Subtracting {right} population from {left} for {self.primary_id}; {self.id}")
        same_parent = left.parent == right.parent
        downstream = right.population_name in list(self.list_downstream_populations(left.population_name))

        if left.source not in ["root", "gate"] or right.source not in ["root", "gate"]:
            logger.error("Population source must be either 'root' or 'gate'")
            raise ValueError("Population source must be either 'root' or 'gate'")
        if not same_parent or not downstream:
            logger.error("Right population should share the same parent as the "
                         "left population or be downstream of the left population")
            raise KeyError("Right population should share the same parent as the "
                           "left population or be downstream of the left population")

        new_population_name = new_population_name or f"subtract_{left.population_name}_{right.population_name}"
        new_idx = np.setdiff1d(left.index, right.index)
        x, y = left.geom.x, left.geom.y
        transform_x, transform_y = left.geom.transform_x, left.geom.transform_y
        parent_data = self.load_population_df(population=left.parent,
                                              transform={x: transform_x,
                                                         y: transform_y})
        envelope = create_envelope(x_values=parent_data.loc[new_idx][x].values,
                                   y_values=parent_data.loc[new_idx][y].values)
        x_values, y_values = envelope.exterior.xy[0], envelope.exterior.xy[1]
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
        logger.debug(f"Writing populations in {self.primary_id}; {self.id} to disk")
        root_n = self.get_population("root").n
        with h5py.File(self.h5path, "r+") as f:
            for meta, labels in self.cell_meta_labels.items():
                logger.debug(f"Writing meta {meta}")
                ascii_labels = np.array([x.encode("ascii", "ignore") for x in labels])
                overwrite_or_create(file=f, data=ascii_labels, key=f"/cell_meta_labels/{meta}")
            for p in self.populations:
                logger.debug(f"Writing population {p}")
                parent_n = self.get_population(p.parent).n
                p._prop_of_parent = p.n / parent_n
                p.prop_of_total = p.n / root_n
                overwrite_or_create(file=f, data=p.index, key=f"/index/{p.population_name}/primary")

    def population_stats(self,
                         population: str,
                         warn_missing: bool = False):
        """
        Returns a dictionary of statistics (number of events, proportion of parent, and proportion of all events)
        for the requested population.

        Parameters
        ----------
        population: str
        warn_missing: bool (default=False)

        Returns
        -------
        Dict
        """
        try:
            pop = self.get_population(population_name=population)
            parent = self.get_population(population_name=pop.parent)
            root = self.get_population(population_name="root")
            return {"population_name": population,
                    "n": pop.n,
                    "frac_of_parent": pop.n / parent.n,
                    "frac_of_root": pop.n / root.n}
        except MissingPopulationError:
            if warn_missing:
                logger.warning(f"{population} not present in {self.primary_id} FileGroup")
            return {"population_name": population,
                    "n": 0,
                    "frac_of_parent": 0,
                    "frac_of_root": 0}

    def quantile_clean(self,
                       upper: float = 0.999,
                       lower: float = 0.001):
        """
        Iterate over every channel in the flow data and cut the upper and lower quartiles.

        Parameters
        ----------
        upper: float (default=0.999)
        lower: float (default=0.001)

        Returns
        -------
        None
        """
        df = self.load_population_df("root", transform="logicle")
        for x in df.columns:
            df = df[(df[x] >= df[x].quantile(lower)) & (df[x] <= df[x].quantile(upper))]
        clean_pop = Population(population_name="root_clean",
                               index=df.index.values,
                               parent="root",
                               source="root",
                               n=df.shape[0])
        self.add_population(clean_pop)

    def save(self, *args, **kwargs):
        """
        Save FileGroup and associated populations

        Returns
        -------
        None
        """
        logger.debug(f"Writing {self.primary_id}; {self.id} to disk")
        # Calculate meta and save indexes to disk
        if self.populations:
            # Populate h5path for populations
            self._write_populations()
        super().save(*args, **kwargs)

    def delete(self,
               delete_hdf5_file: bool = True,
               *args,
               **kwargs):
        """
        Delete FileGroup

        Parameters
        ----------
        delete_hdf5_file: bool (default=True)

        Returns
        -------
        None
        """
        logger.info(f"Deleting {self.primary_id}; {self.id}")
        super().delete(*args, **kwargs)
        if delete_hdf5_file:
            if os.path.isfile(self.h5path):
                os.remove(self.h5path)
            else:
                logger.warning(f"Could not locate hdf5 file {self.h5path}")


def overwrite_or_create(file: h5py.File,
                        data: np.ndarray,
                        key: str):
    """
    Check if node exists in hdf5 file. If it does exist, overwrite with the given
    array otherwise create a new dataset.

    Parameters
    ----------
    file: h5py File object
    data: Numpy Array
    key: str

    Returns
    -------
    None
    """
    if key in file:
        del file[key]
    file.create_dataset(key, data=data)


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


def _load_data_for_ctrl_estimate(filegroup: FileGroup,
                                 target_population: str,
                                 ctrl: str,
                                 transform: str,
                                 sample_size: int,
                                 **transform_kwargs):
    """
    Utility function for loading dataframes for estimating a control population. Given the FileGroup
    of interest, the target population, and the name of the control, the population from the primary
    staining will be loaded, class imbalance accounted for using random over sampling (resampling
    all classes except the majority) and down sampling performed if necessary (if sample_size <
    total population size). The root population of the control will also be loaded into a DataFrame.
    Both DataFrames are transformed and the training data (primary stain population) control root population,
    and the Transformer returned.

    Parameters
    ----------
    filegroup: FileGroup
    target_population: str
    ctrl: str
    transform: str
    sample_size: int
    transform_kwargs:
        Additional keyword arguments passed to apply_transform call

    Returns
    -------
    Pandas.DataFrame, Pandas.DataFrame, Transformer
    """
    training = filegroup.data(source="primary")
    population_idx = filegroup.get_population(target_population).index
    training["label"] = 0
    training.loc[population_idx, "label"] = 1
    ctrl = filegroup.data(source=ctrl)
    time_columns = training.columns[training.columns.str.contains("time", flags=re.IGNORECASE)].to_list()
    for t in time_columns:
        training.drop(t, axis=1, inplace=True)
        ctrl.drop(t, axis=1, inplace=True)
    features = [x for x in training.columns if x != "label"]
    sampler = RandomOverSampler(random_state=42)
    x_resampled, y_resampled = sampler.fit_resample(training[features].values, training["label"].values)
    training = pd.DataFrame(x_resampled, columns=features)
    training["label"] = y_resampled
    if training.shape[0] > sample_size:
        training = training.sample(n=sample_size)
    training = apply_transform(data=training, features=features, method=transform, **transform_kwargs)
    ctrl, transformer = apply_transform(data=ctrl,
                                        features=[x for x in features if x in ctrl.columns],
                                        method=transform,
                                        return_transformer=True,
                                        **transform_kwargs)
    return training, ctrl, transformer
