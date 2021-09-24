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
import logging
import os
import re
from typing import Dict
from typing import Generator
from typing import Iterable
from typing import List
from typing import Optional
from typing import Union

import anytree
import boto3
import flowio
import mongoengine
import numpy as np
import pandas as pd
import polars as pl
from botocore.errorfactory import ClientError
from imblearn.over_sampling import RandomOverSampler
from sklearn.model_selection import permutation_test_score
from sklearn.model_selection import StratifiedKFold

from ..flow.build_models import build_sklearn_model
from ..flow.sampling import sample_dataframe
from ..flow.transform import apply_transform
from ..flow.transform import apply_transform_map
from ..flow.transform import Transformer
from ..flow.tree import construct_tree
from .errors import DuplicatePopulationError
from .errors import MissingControlError
from .errors import MissingPopulationError
from .geometry import create_envelope
from .population import merge_gate_populations
from .population import merge_non_geom_populations
from .population import PolygonGeom
from .population import Population
from .read_write import load_compensation_matrix
from .read_write import read_from_disk
from .read_write import read_from_remote
from .setup import Config
from .subject import Subject

logger = logging.getLogger(__name__)
CONFIG = Config()


def feature_columns(data: pl.DataFrame):
    return [x for x in data.columns if x != "Index"]


def compensate(data: pl.DataFrame, spill_matrix: pl.DataFrame) -> pl.DataFrame:
    """
    Using the providing spillover matrix, compensate the given data by
    solving the linear matrix equation.

    Parameters
    ----------
    data: polars.DataFrame
    spill_matrix: polars.DataFrame

    Returns
    -------
    polars.DataFrame
    """
    features = [x for x in spill_matrix.columns if x != "Index"]
    other_columns = [x for x in data.columns if x not in features]
    compensated = pl.DataFrame(
        np.linalg.solve(spill_matrix[features].to_numpy().T, data[features].to_numpy().T).T, columns=features
    )
    return data[other_columns].hstack(compensated)


def valid_compensation_matrix_path(path: Union[str, None]):
    if path is not None and not path.lower().endswith(".csv"):
        raise mongoengine.errors.ValidationError("Compensation matrix should be a csv or parquet file")


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
    file_paths = mongoengine.DictField(required=True)
    compensate = mongoengine.BooleanField(default=True)
    compensation_matrix = mongoengine.StringField(required=False, validation=valid_compensation_matrix_path)
    collection_datetime = mongoengine.DateTimeField(required=False)
    processing_datetime = mongoengine.DateTimeField(required=False)
    s3_bucket = mongoengine.StringField(required=False)
    populations = mongoengine.EmbeddedDocumentListField(Population)
    gating_strategy = mongoengine.ListField()
    valid = mongoengine.BooleanField(default=True)
    notes = mongoengine.StringField(required=False)
    subject = mongoengine.ReferenceField(Subject, reverse_delete_rule=mongoengine.NULLIFY)
    channel_mappings = mongoengine.DictField(required=True)
    meta = {"db_alias": "core", "collection": "fcs_files"}

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.tree = {}
        if self.id:
            self.tree = construct_tree(populations=self.populations)
        else:
            data = self.data(source="primary")
            self.populations = [
                Population(
                    population_name="root",
                    n=data.shape[0],
                    parent="root",
                    source="root",
                    prop_of_parent=1.0,
                    prop_of_total=1.0,
                )
            ]
            self.populations[0].index = data.Index.to_list()
            self.tree = {"root": anytree.Node(name="root", parent=None)}
            self.save()

    def clean(self):
        if self.s3_bucket:
            s3 = boto3.resource("s3")
            for path in self.file_paths.values():
                try:
                    s3.head_object(self.s3_bucket, path)
                except ClientError as e:
                    logger.error(f"Could not locate {path} in bucket {self.s3_bucket}; {e}")
                    raise
            if self.compensate and self.compensation_matrix:
                try:
                    s3.head_object(self.s3_bucket, self.compensation_matrix)
                except ClientError as e:
                    logger.error(
                        f"Could not locate compensation matrix at {self.compensation_matrix} "
                        f"in bucket {self.s3_bucket}; {e}"
                    )
                    raise
        else:
            for path in self.file_paths.values():
                if not os.path.isfile(path):
                    err = f"Could not locate {path}. Has the file moved? If so make sure to update the database."
                    logger.error(err)
                    raise ValueError(err)
            if self.compensate and self.compensation_matrix:
                if not os.path.isfile(self.compensation_matrix):
                    err = (
                        f"Could not locate compensation matrix at {self.compensation_matrix}."
                        f"Has the file moved? If so make sure to update the database."
                    )
                    logger.error(err)
                    raise ValueError(err)

    def _load_data(self, source: str) -> pl.DataFrame:
        if self.s3_bucket:
            return read_from_remote(s3_bucket=self.s3_bucket, path=self.file_paths[source])
        else:
            return read_from_disk(path=self.file_paths[source])

    def _compensate_data(self, data: pl.DataFrame, source: str) -> pl.DataFrame:
        if self.compensation_matrix:
            if self.s3_bucket:
                spill_matrix = read_from_remote(s3_bucket=self.s3_bucket, path=self.compensation_matrix)
            else:
                spill_matrix = read_from_disk(path=self.compensation_matrix)
        else:
            if not self.file_paths[source].lower().endswith(".fcs"):
                raise TypeError(
                    "For file formats other than FCS a spillover matrix in the form "
                    "of a CSV or Parquet file must be provided"
                )
            spill_matrix = load_compensation_matrix(flowio.FlowData(filename=self.file_paths[source]))
        return compensate(data=data, spill_matrix=spill_matrix)

    def data(
        self,
        source: str = "primary",
        idx: Optional[Iterable[int]] = None,
        sample_size: Optional[Union[int, float]] = None,
        sampling_method: str = "uniform",
        **sampling_kwargs,
    ) -> pl.DataFrame:
        """
        Load the FileGroup dataframe for the desired source file e.g. "primary" for primary
        staining or name of a control for control staining.

        Parameters
        ----------
        source: str
            Name of the file to load from e.g. either "primary" or the name of a control
        idx: Iterable[int], optional
            Provide a list of indexes, will subset DataFrame on Index column. Used for subsetting for populations.
        sample_size: Union[int, float], optional
            If given an integer, will sample this number of events. If given a float, will downsample to given
            fraction of total events.
        sampling_method: str (default="uniform")
        sampling_kwargs:
            Additional keyword arguments passed to cytopy.flow.sampling.sample_dataframe

        Returns
        -------
        polars.DataFrame

        Raises
        ------
        ValueError
            Invalid source
        FileNotFoundError
            Could not locate cytometry data on disk or in remote storage. If the files have been moved, the
            database must be updated.
        TypeError
            Some file format other than FCS is being used but a path to a csv/parquet file for the spillover
            matric has not been provided.
        """
        try:
            data = self._load_data(source=source)
            if self.compensate:
                data = self._compensate_data(data=data, source=source)
            data = data.rename(mapping=self.channel_mappings)
            if idx is not None:
                if isinstance(idx, np.ndarray):
                    idx = idx.tolist()
                data = data.filter(pl.col("Index").is_in(idx))
            if sample_size is not None:
                data = pl.DataFrame(
                    sample_dataframe(data=data, sample_size=sample_size, method=sampling_method, **sampling_kwargs)
                )
            return data
        except KeyError as e:
            logger.error(
                f"Invalid source {source} for {self.primary_id}, expected one of {self.file_paths.keys()}; {e}"
            )
            raise
        except FileNotFoundError as e:
            logger.error(
                f"Could not locate file for {source} at {self.file_paths[source]}. Has the file moved? If so "
                f"make sure to update the database."
            )
            logger.exception(e)
            raise
        except ValueError as e:
            logger.exception(e)
            raise
        except TypeError as e:
            logger.exception(e)
            raise

    def add_population(self, population: Population) -> None:
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

        ValueError
            Population is missing index
        """
        logger.debug(f"Adding new population {population} to {self.primary_id}; {self.id}")
        if population.population_name in self.tree.keys():
            err = f"Population with name '{population.population_name}' already exists"
            raise DuplicatePopulationError(err)
        if population.index is None:
            raise ValueError("Population index is empty")
        if population.n is None:
            population.n = len(population.index)
        if population.prop_of_parent is None:
            population.prop_of_parent = population.n / self.get_population(population_name=population.parent).n
        if population.prop_of_total is None:
            population.prop_of_total = population.n / self.get_population(population_name="root").n
        self.populations.append(population)
        self.tree[population.population_name] = anytree.Node(
            name=population.population_name, parent=self.tree.get(population.parent)
        )

    def update_population(self, pop: Population) -> None:
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
        assert pop.population_name in self.list_populations(), "Invalid population, does not exist"
        self.populations = [p for p in self.populations if p.population_name != pop.population_name]
        self.populations.append(pop)

    def load_ctrl_population_df(
        self,
        ctrl: str,
        population: str,
        classifier: str = "XGBClassifier",
        classifier_params: Optional[Dict] = None,
        scoring: str = "balanced_accuracy",
        transform: str = "logicle",
        transform_kwargs: Optional[Dict] = None,
        evaluate_classifier: bool = True,
        kfolds: int = 5,
        n_permutations: int = 25,
        sample_size: int = 10000,
    ) -> pd.DataFrame:
        """
        Load a population from an associated control. The assumption here is that control files
        have been collected at the same time as primary staining and differ by the absence or
        permutation of a marker/channel/stain. Therefore the population of interest in the
        primary staining will be used as training data to identify the equivalent population in
        the control.

        The user should specify the control file, the population they want (which MUST already exist
        in the primary staining) and the type of classifier to use. Additional parameters can be
        passed to control the classifier and stratified cross validation with permutation testing
        will be performed if evaluate_classifier is set to True.

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
            Population data from control, as predicted using the primary staining

        Raises
        ------
        AssertionError
            If desired population is not found in the primary staining

        MissingControlError
            If the chosen control does not exist
        """
        logger.info(
            f"Predicting {population} in control data in {self.primary_id} {ctrl} using {classifier} classifier"
        )
        transform_kwargs = transform_kwargs or {}
        if ctrl not in self.file_paths.keys():
            raise MissingControlError(f"No such control {ctrl} associated to this FileGroup")
        params = classifier_params or {}
        transform_kwargs = transform_kwargs or {}
        classifier = build_sklearn_model(klass=classifier, **params)
        assert population in self.list_populations(), f"Desired population {population} not found"

        logger.info("Loading primary and control data")
        training, ctrl = _load_data_for_ctrl_estimate(
            filegroup=self,
            target_population=population,
            ctrl=ctrl,
            transform=transform,
            sample_size=sample_size,
            **transform_kwargs,
        )
        features = [x for x in training.columns if x not in ["label", "Index"]]
        features = [x for x in features if x in ctrl.columns]
        x, y = training[features].to_numpy(), training["label"].to_numpy()

        if evaluate_classifier:
            logger.info("Evaluating classifier with permutation testing...")
            skf = StratifiedKFold(n_splits=kfolds, random_state=42, shuffle=True)
            score, permutation_scores, pvalue = permutation_test_score(
                classifier,
                x,
                y,
                cv=skf,
                n_permutations=n_permutations,
                scoring=scoring,
                n_jobs=-1,
                random_state=42,
            )
            logger.info(f"...Performance (without permutations): {round(score, 4)}")
            logger.info(
                f"...Performance (average across permutations; standard dev): "
                f"{round(np.mean(permutation_scores), 4)}; {round(np.std(permutation_scores), 4)}"
            )
            logger.info(f"...p-value (comparison of original score to permuations): {round(pvalue, 4)}")

        logger.info("Predicting population for control data...")
        classifier.fit(x, y)
        ctrl_labels = classifier.predict(ctrl[features].to_numpy())
        training_prop_of_root = self.get_population(population).n / self.get_population("root").n
        ctrl_prop_of_root = np.sum(ctrl_labels) / ctrl.shape[0]
        logger.info(f"{population}: {round(training_prop_of_root, 3)}% of root in primary data")
        logger.info(f"Predicted in ctrl: {round(ctrl_prop_of_root, 3)}% of root in control data")
        ctrl = ctrl[np.where(ctrl_labels == 1)[0]]
        return ctrl[["Index"] + features].to_pandas().set_index("Index")

    def load_multiple_populations(
        self,
        populations: Optional[List[str]] = None,
        regex: Optional[str] = None,
        transform: Optional[Union[str, Dict]] = "logicle",
        features_to_transform: Optional[List] = None,
        transform_kwargs: Optional[Dict] = None,
        label_parent: bool = False,
        frac_of: Optional[List[str]] = None,
        sample_size: Optional[Union[int, float]] = None,
        sampling_method: str = "uniform",
        sample_at_population_level: bool = True,
        **sampling_kwargs,
    ) -> pd.DataFrame:
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
        sample_size: Union[int, float], optional
            If given an integer, will sample this number of events. If given a float, will downsample to given
            fraction of total events.
        sampling_method: str (default="uniform")
        sample_at_population_level: bool (default=True)
            Each population is downsampled independently, rather than downsampling the final DataFrame.
        sampling_kwargs:
            Additional keyword arguments passed to cytopy.flow.sampling.sample_dataframe

        Returns
        -------
        Pandas.DataFrame

        Raises
        ------
        ValueError
            Must provide list of populations or a regex pattern
        """
        dataframes = list()
        if regex is None and populations is None:
            raise ValueError("Must provide list of populations or a regex pattern")
        kwargs = dict(
            transform=transform,
            transform_kwargs=transform_kwargs,
            features_to_transform=features_to_transform,
            label_parent=label_parent,
            frac_of=frac_of,
        )
        if sample_size is not None and sample_at_population_level:
            kwargs["sample_size"] = sample_size
            kwargs["sampling_method"] = sampling_method
            kwargs["sampling_kwargs"] = sampling_kwargs
        if regex:
            populations = self.list_populations(regex=regex)
        for p in populations:
            try:
                pop_data = self.load_population_df(population=p, **kwargs)
                pop_data["population_label"] = [p for _ in range(pop_data.shape[0])]
                dataframes.append(pop_data)
            except ValueError:
                logger.warning(f"{self.primary_id} does not contain population {p}")
        if sample_size is not None and not sample_at_population_level:
            return sample_dataframe(
                data=pd.concat(dataframes), sample_size=sample_size, method=sampling_method, **sampling_kwargs
            )
        return pd.concat(dataframes)

    def load_population_df(
        self,
        population: str,
        transform: str or Optional[Dict] = "logicle",
        features_to_transform: list or None = None,
        transform_kwargs: Optional[Dict] = None,
        label_parent: bool = False,
        frac_of: Optional[List[str]] = None,
        sample_size: Optional[Union[int, float]] = None,
        sampling_method: str = "uniform",
        label_downstream_affiliations=None,
        **sampling_kwargs,
    ) -> pd.DataFrame:
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
        label_downstream_affiliations=None
            Depreciated in version >=3.0.
        sample_size: Union[int, float], optional
            If given an integer, will sample this number of events. If given a float, will downsample to given
            fraction of total events.
        sampling_method: str (default="uniform")
        sampling_kwargs:
            Additional keyword arguments passed to cytopy.flow.sampling.sample_dataframe

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
        transform_kwargs = transform_kwargs or {}
        data = self.data(
            source="primary",
            idx=population.index,
            sample_size=sample_size,
            sampling_method=sampling_method,
            **sampling_kwargs,
        )

        if isinstance(transform, str):
            features_to_transform = features_to_transform or feature_columns(data)
            data = apply_transform(data=data, method=transform, features=features_to_transform, **transform_kwargs)
        elif isinstance(transform, dict):
            data = apply_transform_map(data=data, feature_method=transform, kwargs=transform_kwargs)

        if label_parent:
            data["parent_label"] = [population.parent for _ in range(data.shape[0])]

        if frac_of is not None:
            for comparison_pop in frac_of:
                if comparison_pop not in self.list_populations():
                    logger.warning(f"{comparison_pop} in 'frac_of' is not a recognised population")
                    continue
                comparison_pop = self.get_population(population_name=comparison_pop)
                data[f"frac of {comparison_pop.population_name}"] = population.n / comparison_pop.n
        if isinstance(data, pl.DataFrame):
            return data.to_pandas()
        return data

    def list_populations(self, regex: Optional[str] = None) -> List[str]:
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

    def print_population_tree(self, image: bool = False, path: Optional[str] = None) -> None:
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
        root = self.tree["root"]
        if image:
            from anytree.exporter import DotExporter

            path = path or f"{os.getcwd()}/{self.id}_population_tree.png"
            DotExporter(root).to_picture(path)
        for pre, fill, node in anytree.RenderTree(root):
            print("%s%s" % (pre, node.name))

    def delete_populations(self, populations: Union[str, List[str]]) -> None:
        """
        Delete given populations. Populations downstream from delete population(s) will
        also be removed.

        Parameters
        ----------
        populations: str or list
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
            logger.debug(f"Deleting all populations in {self.primary_id}; {self.id}")
            for p in self.populations:
                self.tree[p.population_name].parent = None
            self.populations = [p for p in self.populations if p.population_name == "root"]
            self.tree = {name: node for name, node in self.tree.items() if name == "root"}
        else:
            try:
                logger.debug(f"Deleting population(s) {populations} in {self.primary_id}; {self.id}")
                assert isinstance(populations, list), "Provide a list of population names for removal"
                assert "root" not in populations, "Cannot delete root population"
                downstream_effects = [self.list_downstream_populations(p) for p in populations]
                downstream_effects = set([x for sl in downstream_effects for x in sl])
                if len(downstream_effects) > 0:
                    logger.warning(
                        "The following populations are downstream of one or more of the "
                        "populations listed for deletion and will therefore be deleted: "
                        f"{downstream_effects}"
                    )
                populations = list(set(list(downstream_effects) + populations))
                self.populations = [p for p in self.populations if p.population_name not in populations]
                for name in populations:
                    self.tree[name].parent = None
                self.tree = {name: node for name, node in self.tree.items() if name not in populations}
            except AssertionError as e:
                logger.warning(e)
            except ValueError as e:
                logger.warning(e)

    def get_population(self, population_name: str) -> Population:
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
            raise MissingPopulationError(f"Population {population_name} does not exist")
        return [p for p in self.populations if p.population_name == population_name][0]

    def get_population_by_parent(self, parent: str) -> Generator:
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

    def list_downstream_populations(self, population: str) -> Union[List[str], None]:
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
        MissingPopulationError
            If Population does not exist
        """
        if population not in self.tree.keys():
            raise MissingPopulationError(
                f"population {population} does not exist; valid population names include: {self.tree.keys()}"
            )
        root = self.tree["root"]
        node = self.tree[population]
        dependencies = [x.name for x in anytree.findall(root, filter_=lambda n: node in n.path)]
        return [p for p in dependencies if p != population]

    def merge_gate_populations(
        self,
        left: Union[Population, str],
        right: Union[Population, str],
        new_population_name: Optional[str] = None,
    ) -> None:
        """
        Merge two populations present in the current population tree.
        The merged population will have the combined index of both populations but
        will not inherit any clusters and will not be associated to any children
        downstream of either the left or right population. The population will be
        added to the tree as a descendant of the left populations parent. New
        population will be added to FileGroup.

        Parameters
        ----------
        left: Population or str
        right: Population or str
        new_population_name: str, optional

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

    def merge_non_geom_populations(self, populations: List[Union[str, Population]], new_population_name: str) -> None:
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

    def subtract_populations(
        self,
        left: Population,
        right: Population,
        new_population_name: Optional[str] = None,
    ) -> None:
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
            logger.error(
                "Right population should share the same parent as the "
                "left population or be downstream of the left population"
            )
            raise KeyError(
                "Right population should share the same parent as the "
                "left population or be downstream of the left population"
            )

        new_population_name = new_population_name or f"subtract_{left.population_name}_{right.population_name}"
        new_idx = np.setdiff1d(np.array(left.index), np.array(right.index))
        x, y = left.geom.x, left.geom.y
        transform_x, transform_y = left.geom.transform_x, left.geom.transform_y
        parent_data = self.load_population_df(population=left.parent, transform={x: transform_x, y: transform_y})
        envelope = create_envelope(
            x_values=parent_data.filter(pl.col("Index").is_in(new_idx))[x].to_numpy(),
            y_values=parent_data.filter(pl.col("Index").is_in(new_idx))[y].to_numpy(),
        )
        x_values, y_values = envelope.exterior.xy[0], envelope.exterior.xy[1]
        new_geom = PolygonGeom(
            x=x,
            y=y,
            transform_x=transform_x,
            transform_y=transform_y,
            x_values=x_values.tolist(),
            y_values=y_values.tolist(),
        )
        new_population = Population(
            population_name=new_population_name,
            parent=left.parent,
            n=len(new_idx),
            geom=new_geom,
            warnings=left.warnings + right.warnings + ["SUBTRACTED POPULATION"],
        )
        new_population.index = new_idx.tolist()
        self.add_population(population=new_population)

    def population_stats(self, population: str, warn_missing: bool = False) -> Dict:
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
            return {
                "population_name": population,
                "n": pop.n,
                "frac_of_parent": pop.n / parent.n,
                "frac_of_root": pop.n / root.n,
            }
        except MissingPopulationError:
            if warn_missing:
                logger.debug(f"{population} not present in {self.primary_id} FileGroup")
            return {
                "population_name": population,
                "n": 0,
                "frac_of_parent": 0,
                "frac_of_root": 0,
            }

    def quantile_clean(self, upper: float = 0.999, lower: float = 0.001) -> None:
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
        clean_pop = Population(
            population_name="root_clean",
            index=df.index.values,
            parent="root",
            source="root",
            n=df.shape[0],
        )
        self.add_population(clean_pop)

    def write_to_fcs(self, path: str, source: str = "primary"):
        with open(path, "wb") as f:
            channels, markers = zip(**self.channel_mappings)
            flowio.create_fcs(
                event_data=self.data(source=source).values.flatten(),
                file_handle=f,
                channel_names=channels,
                opt_channel_names=markers,
            )


def population_stats(filegroup: FileGroup) -> pl.DataFrame:
    """
    Given a FileGroup generate a DataFrame detailing the number of events, proportion
    of parent population, and proportion of total (root population) for each
    population in the FileGroup.

    Parameters
    ----------
    filegroup: FileGroup

    Returns
    -------
    polars.DataFrame
    """
    return pl.DataFrame([filegroup.population_stats(p) for p in list(filegroup.list_populations())])


def _load_data_for_ctrl_estimate(
    filegroup: FileGroup,
    target_population: str,
    ctrl: str,
    transform: str,
    sample_size: int,
    **transform_kwargs,
) -> (pl.DataFrame, pl.DataFrame, Transformer):
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
    transform_cache: bool
    transform_kwargs:
        Additional keyword arguments passed to apply_transform call

    Returns
    -------
    polars.DataFrame, polars.DataFrame, Transformer
    """
    training = filegroup.data(source="primary")
    population_idx = filegroup.get_population(target_population).index
    training["label"] = [0 for _ in range(training.shape[0])]
    training[training.Index.is_in(population_idx), "label"] = 1
    ctrl = filegroup.data(source=ctrl)

    features = [x for x in training.columns if x != "label"]
    if transform:
        training = apply_transform(data=training, features=features, method=transform, **transform_kwargs)
        ctrl = apply_transform(data=ctrl, features=features, method=transform, **transform_kwargs)

    time_columns = [col for col in training.columns if re.match("time", col, re.IGNORECASE)]
    for t in time_columns:
        training.drop(t, axis=1, inplace=True)
        ctrl.drop(t, axis=1, inplace=True)

    sampler = RandomOverSampler(random_state=42)
    x_resampled, y_resampled = sampler.fit_resample(training[features].values, training["label"].values)
    training = pl.DataFrame(x_resampled, columns=features)
    training["label"] = y_resampled
    if training.shape[0] > sample_size:
        training = training.sample(n=sample_size)
    return training, ctrl
