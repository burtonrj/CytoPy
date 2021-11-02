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
import gc
import logging
import os
import pickle
import re
from copy import deepcopy
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
from bson import Binary

from ..utils.geometry import inside_polygon
from ..utils.sampling import sample_dataframe
from ..utils.transform import apply_transform
from ..utils.transform import apply_transform_map
from .errors import DuplicatePopulationError
from .errors import MissingPopulationError
from .population import PolygonGeom
from .population import Population
from .population import ThresholdGeom
from .read_write import load_compensation_matrix
from .read_write import polars_to_pandas
from .read_write import read_from_disk
from .read_write import read_from_remote
from .setup import Config
from .subject import Subject
from cytopy.data.tree import construct_tree

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
        assert "primary" in self.file_paths.keys(), f"'primary' missing from file_paths"
        if self.id:
            for key in self.file_paths.keys():
                self.tree[key] = construct_tree(populations=[p for p in self.populations if p.data_source == key])
        else:
            for key in self.file_paths.keys():
                data = self.data(source=key)
                pop = Population(
                    population_name="root",
                    n=data.shape[0],
                    parent="root",
                    source="root",
                    data_source=key,
                    prop_of_parent=1.0,
                    prop_of_total=1.0,
                )
                pop.index = data.Index.to_list()
                self.populations.append(pop)
                self.tree[key] = {"root": anytree.Node(name="root", parent=None)}
            self.save()
            del data
            gc.collect()

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
            spill_matrix = load_compensation_matrix(flowio.FlowData(self.file_paths[source]))
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
            Additional keyword arguments passed to cytopy.utils.sampling.sample_dataframe

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
                    .reset_index()
                    .rename({"index": "Index"}, axis=1)
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

        if population.population_name in self.tree[population.data_source].keys():
            err = f"Population with name '{population.population_name}' already exists (for {population.data_source})"
            raise DuplicatePopulationError(err)
        if population.index is None:
            raise ValueError("Population index is empty")
        if population.n is None:
            population.n = len(population.index)
        if population.prop_of_parent is None:
            population.prop_of_parent = (
                population.n
                / self.get_population(population_name=population.parent, data_source=population.data_source).n
            )
        if population.prop_of_total is None:
            population.prop_of_total = (
                population.n / self.get_population(population_name="root", data_source=population.data_source).n
            )
        self.populations.append(population)
        self.tree[population.data_source][population.population_name] = anytree.Node(
            name=population.population_name, parent=self.tree[population.data_source].get(population.parent)
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
        old_pop = self.get_population(population_name=pop.population_name, data_source=pop.data_source)
        old_pop.n = pop.n
        old_pop.parent = pop.parent
        old_pop.prop_of_parent = pop.prop_of_parent
        old_pop.prop_of_total = pop.prop_of_total
        old_pop.normalised = pop.normalised
        old_pop.geom = pop.geom
        old_pop.definition = pop.definition
        old_pop.source = pop.source
        old_pop.index = pop.index

    def load_multiple_populations(
        self,
        populations: Optional[List[str]] = None,
        regex: Optional[str] = None,
        transform: Optional[Union[str, Dict]] = "asinh",
        features_to_transform: Optional[List] = None,
        transform_kwargs: Optional[Dict] = None,
        label_parent: bool = False,
        frac_of: Optional[List[str]] = None,
        sample_size: Optional[Union[int, float]] = None,
        sampling_method: str = "uniform",
        sample_at_population_level: bool = True,
        data_source: str = "primary",
        meta_vars: Optional[Dict] = None,
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
        transform: str or dict, optional (default="asinh")
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
            Additional keyword arguments passed to cytopy.utils.sampling.sample_dataframe

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
            data_source=data_source,
            meta_vars=meta_vars,
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
                logger.warning(f"{self.primary_id} ({data_source}) does not contain population {p}")
        if sample_size is not None and not sample_at_population_level:
            return sample_dataframe(
                data=pd.concat(dataframes), sample_size=sample_size, method=sampling_method, **sampling_kwargs
            )
        return pd.concat(dataframes)

    def load_population_df(
        self,
        population: str,
        transform: str or Optional[Dict] = "asinh",
        features_to_transform: list or None = None,
        transform_kwargs: Optional[Dict] = None,
        label_parent: bool = False,
        frac_of: Optional[List[str]] = None,
        sample_size: Optional[Union[int, float]] = None,
        sampling_method: str = "uniform",
        data_source: str = "primary",
        meta_vars: Optional[Dict] = None,
        label_downstream_affiliations=None,
        **sampling_kwargs,
    ) -> pd.DataFrame:
        """
        Load the DataFrame for the events pertaining to a single population.

        Parameters
        ----------
        population: str
            Name of the desired population
        transform: str or dict, optional (default="asinh")
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
            Additional keyword arguments passed to cytopy.utils.sampling.sample_dataframe

        Returns
        -------
        Pandas.DataFrame

        Raises
        ------
        ValueError
            Invalid population, does not exist
        """
        if population not in self.tree[data_source].keys():
            raise MissingPopulationError(population_id=population)

        population = self.get_population(population_name=population, data_source=data_source)
        transform_kwargs = transform_kwargs or {}
        data = self.data(
            source=data_source,
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

        if isinstance(data, pl.DataFrame):
            data = polars_to_pandas(data)

        if label_parent:
            data["parent_label"] = population.parent

        if meta_vars is not None and self.subject:
            for col_name, key in meta_vars.items():
                data[col_name] = self.subject.lookup_var(key=key)

        if frac_of is not None:
            for comparison_pop in frac_of:
                if comparison_pop not in self.list_populations():
                    logger.warning(f"{comparison_pop} in 'frac_of' is not a recognised population")
                    continue
                comparison_pop = self.get_population(population_name=comparison_pop)
                data[f"frac of {comparison_pop.population_name}"] = population.n / comparison_pop.n

        return data

    def list_populations(
        self, regex: Optional[str] = None, source: Optional[str] = None, data_source: str = "primary"
    ) -> List[str]:
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
        populations = [p for p in self.populations if p.data_source == data_source]
        if source:
            populations = [p for p in populations if p.source == source]
        populations = [p.population_name for p in populations]
        if regex:
            regex = re.compile(regex)
            return list(filter(regex.match, populations))
        return populations

    def print_population_tree(
        self, data_source: str = "primary", image: bool = False, path: Optional[str] = None
    ) -> None:
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
        root = self.tree[data_source]["root"]
        if image:
            from anytree.exporter import DotExporter

            path = path or f"{os.getcwd()}/{self.id}_population_tree.png"
            DotExporter(root).to_picture(path)
        for pre, fill, node in anytree.RenderTree(root):
            print("%s%s" % (pre, node.name))

    def rename_population(self, old_name: str, new_name: str, data_source: str = "primary"):
        assert new_name not in self.list_populations(data_source=data_source), f"{new_name} already exists!"
        pop = self.get_population(population_name=old_name, data_source=data_source)
        pop.population_name = new_name
        self.tree[data_source][old_name].name = new_name
        self.tree[data_source][new_name] = self.tree[data_source].pop(old_name)

    def delete_populations(self, populations: Union[str, List[str]], data_source: str = "primary") -> None:
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
            logger.debug(f"Deleting all populations in {self.primary_id}; {data_source}; {self.id}")
            for p in self.populations:
                self.tree[data_source][p.population_name].parent = None
            self.populations = [
                p for p in self.populations if p.population_name == "root" and p.data_source == data_source
            ]
            self.tree[data_source] = {name: node for name, node in self.tree[data_source].items() if name == "root"}
        else:
            try:
                logger.debug(f"Deleting population(s) {populations} in {self.primary_id}; {data_source}; {self.id}")
                assert isinstance(populations, list), "Provide a list of population names for removal"
                assert "root" not in populations, "Cannot delete root population"
                downstream_effects = [
                    self.list_downstream_populations(p, data_source=data_source) for p in populations
                ]
                downstream_effects = set([x for sl in downstream_effects for x in sl])
                if len(downstream_effects) > 0:
                    logger.warning(
                        "The following populations are downstream of one or more of the "
                        "populations listed for deletion and will therefore be deleted: "
                        f"{downstream_effects}"
                    )
                populations = list(set(list(downstream_effects) + populations))
                new_populations = []
                for p in self.populations:
                    if p.data_source != data_source:
                        new_populations.append(p)
                    else:
                        if p.population_name not in populations:
                            new_populations.append(p)
                self.populations = new_populations
                for name in populations:
                    self.tree[data_source][name].parent = None
                self.tree[data_source] = {
                    name: node for name, node in self.tree[data_source].items() if name not in populations
                }
            except AssertionError as e:
                logger.warning(e)
            except ValueError as e:
                logger.warning(e)

    def get_population(
        self,
        population_name: str,
        data_source: str = "primary",
    ) -> Population:
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
        if population_name not in list(self.list_populations(data_source=data_source)):
            raise MissingPopulationError(population_id=population_name)
        return [p for p in self.populations if p.population_name == population_name and p.data_source == data_source][
            0
        ]

    def get_population_by_parent(self, parent: str, data_source: str = "primary") -> Generator:
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
            if p.parent == parent and p.population_name != "root" and p.data_source == data_source:
                yield p

    def list_downstream_populations(self, population: str, data_source: str = "primary") -> Union[List[str], None]:
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
        if population not in self.tree[data_source].keys():
            raise MissingPopulationError(
                f"population {population} does not exist; valid population names include: {self.tree[data_source].keys()}"
            )
        root = self.tree[data_source]["root"]
        node = self.tree[data_source][population]
        dependencies = [x.name for x in anytree.findall(root, filter_=lambda n: node in n.path)]
        return [p for p in dependencies if p != population]

    def merge_populations(
        self, parent: str, populations: List[str], new_population_name: str, data_source: str = "primary"
    ) -> Population:
        """
        Merge two or more populations. Merged populations are tagged with the source "merger" and are treated
        like a "cluster" for the purposes of plotting - when plotted in a 1 dimensional space, they are shown as
        an overlaid KDE, when plotted in a 2 dimensional space, they are plotted as either an overlaid scatterplot
        or an estimated gate as an alpha-shape.

        Parameters
        ----------
        parent: str
        populations: List[Population]
        new_population_name: str

        Returns
        -------
        Population
        """
        populations = [self.get_population(population_name=p, data_source=data_source) for p in populations]

        for pop in populations:
            if parent in self.list_downstream_populations(population=pop.population_name):
                raise ValueError(
                    f"Cannot merge {pop.population_name} - parent {parent} is downstream from " f"this population"
                )

        new_idx = np.unique(np.concatenate([pop.index for pop in populations], axis=0), axis=0)
        new_population = Population(
            population_name=new_population_name,
            n=len(new_idx),
            parent=parent,
            source="merger",
            data_source=data_source,
        )
        new_population.index = new_idx
        self.add_population(population=new_population)

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
        KeyError
            If left and right population do not share the same parent or the right population
            is not downstream of the left population
        """
        logger.info(f"Subtracting {right} population from {left} for {self.primary_id}; {self.id}")
        same_parent = left.parent == right.parent
        downstream = right.population_name in list(self.list_downstream_populations(left.population_name))
        same_source = left.data_source == right.data_source
        if not same_parent or not downstream:
            err = (
                "Right population should share the same parent as the left population or be "
                "downstream of the left population"
            )
            logger.error(err)
            raise KeyError(err)
        if not same_source:
            err = "Right and left population must be from the same data source"
            logger.error(err)
            raise KeyError(err)

        new_population_name = new_population_name or f"subtract_{left.population_name}_{right.population_name}"
        new_idx = np.setdiff1d(np.array(left.index), np.array(right.index))
        new_population = Population(
            population_name=new_population_name, parent=left.parent, n=len(new_idx), source="subtraction"
        )
        new_population.index = new_idx
        self.add_population(population=new_population)

    def population_stats(self, population: str, warn_missing: bool = False, data_source: str = "primary") -> Dict:
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
            pop = self.get_population(population_name=population, data_source=data_source)
            parent = self.get_population(population_name=pop.parent, data_source=data_source)
            root = self.get_population(population_name="root", data_source=data_source)
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

    def write_to_fcs(self, path: str, source: str = "primary"):
        with open(path, "wb") as f:
            channels, markers = zip(**self.channel_mappings)
            flowio.create_fcs(
                event_data=self.data(source=source).values.flatten(),
                file_handle=f,
                channel_names=channels,
                opt_channel_names=markers,
            )

    def save(self, *args, **kwargs):
        for pop in self.populations:
            assert pop.index is not None, f"Population {pop.population_name} index is empty!"
        for pop in self.populations:
            pop.write_index()
        super(FileGroup, self).save(*args, **kwargs)


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


def copy_populations_to_controls_using_geoms(filegroup: FileGroup, ctrl: str, flag: float = 0.25):
    if ctrl not in filegroup.file_paths.keys():
        raise ValueError("Invalid ctrl, does not exist for given FileGroup")
    stats = {"Population": [], "% of parent (primary)": [], "% of parent (ctrl)": [], "Flag": []}
    queue = list(filegroup.tree["primary"]["root"].children)
    while len(queue) > 0:
        next_pop = queue.pop(0)
        pop = filegroup.get_population(next_pop.name, data_source="primary")
        queue = queue + list(filegroup.tree["primary"][pop.population_name].children)
        if not pop.geom or pop.parent not in filegroup.list_populations(data_source=ctrl):
            logger.warning(f"Skipping {pop.population_name}: missing parent or no geom defined")
            continue
        parent_df = filegroup.load_population_df(
            population=pop.parent,
            transform={pop.geom.x: pop.geom.transform_x, pop.geom.y: pop.geom.transform_y},
            transform_kwargs={
                pop.geom.x: pop.geom.transform_x_kwargs or {},
                pop.geom.y: pop.geom.transform_y_kwargs or {},
            },
            data_source=ctrl,
        )
        if isinstance(pop.geom, PolygonGeom):
            data = inside_polygon(data=parent_df, x=pop.geom.x, y=pop.geom.y, poly=pop.geom.shape)
        elif isinstance(pop.geom, ThresholdGeom):
            if pop.definition == "+":
                data = parent_df[parent_df[pop.geom.x] >= pop.geom.x_threshold]
            elif pop.definition == "-":
                data = parent_df[parent_df[pop.geom.x] < pop.geom.x_threshold]
            elif pop.definition == "--":
                data = parent_df[
                    (parent_df[pop.geom.x] < pop.geom.x_threshold) & (parent_df[pop.geom.y] < pop.geom.y_threshold)
                ]
            elif pop.definition == "-+":
                data = parent_df[
                    (parent_df[pop.geom.x] < pop.geom.x_threshold) & (parent_df[pop.geom.y] >= pop.geom.y_threshold)
                ]
            elif pop.definition == "+-":
                data = parent_df[
                    (parent_df[pop.geom.x] >= pop.geom.x_threshold) & (parent_df[pop.geom.y] < pop.geom.y_threshold)
                ]
            elif pop.definition == "++":
                data = parent_df[
                    (parent_df[pop.geom.x] >= pop.geom.x_threshold) & (parent_df[pop.geom.y] >= pop.geom.y_threshold)
                ]
            else:
                raise ValueError("Unrecognised definition for ThresholdGeom")
        else:
            logger.warning(f"Skipping {pop.population_name}: unrecognised geometry")
            continue
        ctrl_pop = Population(
            population_name=pop.population_name,
            parent=pop.parent,
            n=data.shape[0],
            geom=deepcopy(pop.geom),
            definition=pop.definition,
            source=pop.source,
            data_source=ctrl,
        )
        ctrl_pop.index = data.index.to_list()
        filegroup.add_population(population=ctrl_pop)
        ctrl_pop = filegroup.get_population(population_name=pop.population_name, data_source=ctrl)
        primary_prop = round(pop.prop_of_parent * 100, 5)
        ctrl_prop = round(ctrl_pop.prop_of_parent * 100, 5)
        stats["Population"].append(pop.population_name)
        stats["% of parent (primary)"].append(primary_prop)
        stats["% of parent (ctrl)"].append(ctrl_prop)
        floor = np.max([0, primary_prop - (primary_prop * flag)])
        ceil = np.min([100, primary_prop + (primary_prop * flag)])
        stats["Flag"].append(floor > ctrl_prop > ceil)
    return filegroup, pd.DataFrame(stats)
