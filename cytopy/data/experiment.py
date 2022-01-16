#!/usr/bin.env/python
# -*- coding: utf-8 -*-
"""
The experiment module houses the Experiment class, used to define
cytometry based experiments that can consist of one or more biological
specimens. An experiment should be defined for each cytometry staining
panel used in your analysis.

Experiments should be created using the Project class (see cytopy.data.projects).
All functionality for experiments and Panels are housed within this
module.

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
from __future__ import annotations

import gc
import logging
from collections import Counter
from typing import Dict
from typing import Iterable
from typing import List
from typing import Optional
from typing import Union

import matplotlib.pyplot as plt
import mongoengine
import numpy as np
import pandas as pd
from anndata import AnnData
from joblib import delayed
from joblib import Parallel

from ..feedback import progress_bar
from ..utils.sampling import sample_dataframe
from .errors import DuplicatePopulationError
from .errors import DuplicateSampleError
from .errors import EmptyPopulationError
from .errors import MissingPopulationError
from .errors import MissingSampleError
from .fcs import copy_populations_to_controls_using_geoms
from .fcs import effect_size
from .fcs import FileGroup
from .panel import Panel
from .subject import Subject

logger = logging.getLogger(__name__)


class Experiment(mongoengine.Document):
    """
    Container for Cytometry experiment. The correct way to generate and load these objects is using the
    Project.add_experiment method (see cytopy.data.project.Project). This object provides access
    to all experiment-wide functionality.

    Attributes
    -----------
    experiment_id: str, required
        Unique identifier for experiment
    panel: EmbeddedDocument[Panel], required
        Panel object describing associated channel/marker pairs
    fcs_files: List[ReferenceField]
        Reference field for associated files
    flags: str, optional
        Warnings associated to experiment
    notes: str, optional
        Additional free text comments
    """

    experiment_id = mongoengine.StringField(required=True, unique=True)
    panel = mongoengine.EmbeddedDocumentField(Panel)
    fcs_files = mongoengine.ListField(mongoengine.ReferenceField(FileGroup, reverse_delete_rule=4))
    flags = mongoengine.StringField(required=False)
    notes = mongoengine.StringField(required=False)

    meta = {"db_alias": "core", "collection": "experiments"}

    def __repr__(self):
        return f"Experiment(experiment_id={self.experiment_id})"

    def generate_panel(self, panel_definition: str) -> Experiment:
        """
        Generate a new panel using the panel definition provided (path to a valid template).

        Parameters
        ----------
        panel_definition: Union[str, dict]
            Path to a panel definition

        Returns
        -------
        Experiment

        Raises
        ------
        ValueError
            Panel definition is not a string or dict
        """
        new_panel = Panel()
        new_panel.create_from_tabular(path=panel_definition)
        self.panel = new_panel
        return self

    def delete_all_populations(self, sample_id: str) -> Experiment:
        """
        Delete population data associated to experiment. Give a value of 'all' for sample_id to remove all population
        data for every sample.

        Parameters
        ----------
        sample_id: str
            Name of sample to remove populations from'; give a value of 'all'
            for sample_id to remove all population data for every sample.

        Returns
        -------
        Experiment
        """
        logger.info(f"Deleting all populations in '{sample_id}'")
        for f in self.fcs_files:
            if sample_id == "all" or f.primary_id == sample_id:
                f.populations = [p for p in f.populations if p.population_name == "root"]
                f.save()
        logger.info("Populations deleted successfully!")
        return self

    def sample_exists(self, sample_id: str) -> bool:
        """
        Returns True if the given sample_id exists in Experiment

        Parameters
        ----------
        sample_id: str
            Name of sample to search for

        Returns
        --------
        bool
            True if exists, else False
        """
        if sample_id not in list(self.list_samples()):
            return False
        return True

    def get_sample(self, sample_id: str) -> FileGroup:
        """
        Given a sample ID, return the corresponding FileGroup object

        Parameters
        ----------
        sample_id: str
            Sample ID for search

        Returns
        --------
        FileGroup

        Raises
        ------
        MissingSampleError
            If requested sample is not found in the experiment
        """
        if not self.sample_exists(sample_id):
            raise MissingSampleError(f"Invalid sample: {sample_id} not associated with this experiment")
        return [f for f in self.fcs_files if f.primary_id == sample_id][0]

    def filter_samples_by_subject(self, query: Union[str, mongoengine.queryset.visitor.Q]) -> List[FileGroup]:
        """
        Filter FileGroups associated to this experiment based on some subject meta-data

        Parameters
        ----------
        query: str or mongoengine.queryset.visitor.Q
            Query to make on Subject

        Returns
        -------
        List[FileGroup]
        """
        logger.debug(f"Fetching list of FileGroups associated to Subject on query {query}")
        matches = []
        for f in self.fcs_files:
            try:
                Subject.objects(id=f.subject.id).filter(query).get()
                matches.append(f.primary_id)
            except mongoengine.DoesNotExist:
                logger.debug(f"No subject associated to {f.primary_id}; {f.id}")
                continue
            except mongoengine.MultipleObjectsReturned:
                logger.debug(f"Multiple matches to subject meta data for {f.primary_id}; {f.id}")
                matches.append(f.primary_id)
        return matches

    def list_samples(self, valid_only: bool = True) -> List[str]:
        """
        Generate a list of IDs for file groups associated to experiment

        Parameters
        -----------
        valid_only: bool
            If True, returns only valid samples (samples without 'invalid' flag)

        Returns
        --------
        List[str]
        """
        if valid_only:
            return [f.primary_id for f in self.fcs_files if f.valid]
        return [f.primary_id for f in self.fcs_files]

    def random_filegroup(self, seed: int = 42) -> str:
        """
        Return a random filegroup ID

        Parameters
        ----------
        seed: int = 42

        Returns
        -------
        str
        """
        np.random.seed(seed)
        return np.random.choice(self.list_samples())

    def remove_sample(self, sample_id: str) -> Experiment:
        """
        Remove sample (FileGroup) from experiment.

        Parameters
        -----------
        sample_id: str
            ID of sample to remove

        Returns
        --------
        Experiment
        """
        logger.debug(f"Deleting {sample_id}")
        filegrp = self.get_sample(sample_id)
        self.fcs_files = [f for f in self.fcs_files if f.primary_id != sample_id]
        filegrp.delete()
        self.save()
        return self

    def _sample_exists(self, sample_id: str):
        if self.sample_exists(sample_id):
            raise DuplicateSampleError(f"A file group with id {sample_id} already exists")

    def add_filegroup(
        self,
        sample_id: str,
        paths: Dict[str, str],
        compensate: bool = True,
        compensation_matrix: Optional[str] = None,
        s3_bucket: Optional[str] = None,
        subject_id: Optional[str] = None,
        processing_datetime: Optional[str] = None,
        collection_datetime: Optional[str] = None,
    ) -> Experiment:
        """
        Associate a new biological specimen to this Experiment and link to some single cell data source
        such as a CSV file, FCS file, HDF5 or Parquet file. This will generate a new FileGroup that will
        be linked to this Experiment and labelled with a unique ID (sample_id).

        A FileGroup can consist of multiple sources files, but expects a single source for the primary staining of
        cytometry data and then multiple files for subsequent controls. The data sources should be specified using a
        dictionary provided to 'path'. This must contain the key 'primary' whose value is the file path to the
        primary staining. All subsequent key-value pairs are treated as control ID's and corresponding file paths. If
        a value is provided for 's3_bucket', then the paths should be relative to this bucket.

        If the data must be compensated, make sure 'compensate' is True. If providing a path to an FCS file, then
        'compensation_matrix' can be left as None if a compensation matrix is embedded within this file. If a
        value is provided for 'compensation_matrix' it will overwrite the embedded matrix. If the source file is
        not an FCS file and 'compensate' is True, then a value MUST be provided for 'compensation_matrix' in the
        form of a file path to a valid CSV file containing the spillover matrix.

        NOTE: The file paths are stored within the database NOT the single cell data. Once an Experiment is
        setup you should try to keep filepaths static. If you must migrate date, see Project.migrate method.

        Parameters
        ----------
        sample_id: str
            Unique sample ID (unique to this Experiment)
        paths: Dict[str, str]
            File ID as key and filepath as value; must contain a 'primary' key.
        compensate: bool (default=True)
            Should the data be compensated upon reading?
        compensation_matrix: str, optional
            Filepath to spillover CSV file. If provided any embedded spillover matrix will be ignored.
        s3_bucket: str, optional
            Parent S3 bucket containing all filepaths
        subject_id: str, optional
            Subject to associate this FileGroup to
        processing_datetime: str, optional
        collection_datetime: str, optional

        Returns
        -------
        Experiment
        """
        if self.panel is None:
            raise AttributeError("No panel defined.")
        if "primary" not in paths.keys():
            err = "'primary' missing from paths"
            logger.error(err)
            raise ValueError(err)
        subject = None
        if subject_id is not None:
            try:
                subject = Subject.objects(subject_id=subject_id).get()
            except mongoengine.errors.DoesNotExist:
                logger.warning(f"Error: no such patient {subject_id}, continuing without association.")

        self.fcs_files.append(
            FileGroup(
                primary_id=sample_id,
                file_paths=paths,
                compensate=compensate,
                compensation_matrix=compensation_matrix,
                subject=subject,
                s3_bucket=s3_bucket,
                processing_datetime=processing_datetime,
                collection_datetime=collection_datetime,
                channel_mappings=self.panel.build_mappings(path=list(paths.values()), s3_bucket=s3_bucket),
            )
        )
        self.save()
        gc.collect()
        return self

    def control_counts(self, ax: Optional[plt.Axes] = None) -> plt.Axes:
        """
        Generates a barplot of total counts of each control in Experiment FileGroups

        Parameters
        ----------
        ax: Matplotlib.Axes, optional

        Returns
        -------
        Matplotlib.Axes
        """
        ctrls = [[x for x in f.file_paths.keys() if x != "primary"] for f in self.fcs_files]
        ctrl_counts = Counter([x for sl in ctrls for x in sl])
        ctrl_counts["Total"] = len(self.fcs_files)
        ax = ax or plt.subplots(figsize=(6, 6))[1]
        ax.bar(ctrl_counts.keys(), ctrl_counts.values())
        return ax

    @staticmethod
    def _prop_of_parent(df: pd.DataFrame, parent: str) -> pd.DataFrame:
        """
        Given a DataFrame of population statistics and the name of a parent
        population, calculate the proportion of events as a fraction of this
        parent and return the DataFrame with the new column 'frac_of_{parent}'.
        If the parent doesn't exist then the value in this column with be None.

        Parameters
        ----------
        df: Pandas.DataFrame
            Population statistics, including the number of events (n). Each row
            should correspond to a population.
        parent: str

        Returns
        -------
        Pandas.DataFrame
        """
        frac_of_parent = []
        parent_df = df[df.population_name == parent]
        if parent_df.shape[0] == 0:
            df[f"frac_of_{parent}"] = None
            return df
        parent_n = float(parent_df["n"].values[0])
        for _, row in df.iterrows():
            if row.n == 0:
                frac_of_parent.append(0)
            else:
                frac_of_parent.append(row.n / parent_n)
        df[f"frac_of_{parent}"] = frac_of_parent
        return df

    def population_statistics(
        self,
        populations: Optional[List[str]] = None,
        meta_vars: Optional[Dict] = None,
        additional_parent: Optional[str] = None,
        regex: Optional[str] = None,
        population_source: Optional[str] = None,
        data_source: str = "primary",
    ) -> pd.DataFrame:
        """
        Generates a Pandas DataFrame of population statistics for all FileGroups
        of an Experiment, for the given populations or all available populations
        if 'populations' is None.

        Parameters
        ----------
        populations: List[str], optional
            List of populations to generate statistics for. Leave as None to generate statistics
            for all populations
        meta_vars: Dict[str, Union[str, List[str]]], optional
            If a dictionary is provided, will generate additional columns of meta-variables sourced
            from the Subjects associated to the Experiment FileGroups. Let's take a look at an example.
            If our Experiments FileGroups had associated Subjects that contained the variable 'patient_type',
            we could add a column to our DataFrame called 'Patient Type' by passing the following:

                {'Patient Type': 'patient_type'}

            If this variable was embedded in our Subjects under the field 'labels', we would provide the following:

                {'Patient Type': ['labels', 'patient_type']}

            The DataFrame that this method returns will now contain an additional column called 'Patient Type'
            with values for each FileGroup derived from the Subject documents.
        additional_parent: str, optional
            If provided, will compute the proportion of events as a fraction of this population.
        regex: str, optional
            If provided and populations is None, will generate statistics for all populations matching this
            regular expression pattern.
        population_source: str, optional
            The type of populations to include e.g. gated, clusters etc
        data_source: str (default='primary')
            The data source of interest i.e. either primary or some control ID

        Returns
        -------
        Pandas.DataFrame
        """
        data = []
        for f in self.fcs_files:
            for p in populations or self.list_populations(
                regex=regex, population_source=population_source, data_source=data_source
            ):
                df = pd.DataFrame({k: [v] for k, v in f.population_stats(population=p).items()})
                df["sample_id"] = f.primary_id
                s = f.subject
                if s is not None:
                    df["subject_id"] = s.subject_id
                    if meta_vars is not None:
                        for col_name, key in meta_vars.items():
                            df[col_name] = s.lookup_var(key=key)
                data.append(df)
        data = pd.concat(data).reset_index(drop=True)
        if additional_parent:
            return data.groupby("sample_id").apply(lambda x: self._prop_of_parent(x, parent=additional_parent))
        return data

    def population_membership_boolean_matrix(
        self,
        regex: Optional[str] = None,
        population_source: Optional[str] = None,
        data_source: str = "primary",
        verbose: bool = True,
    ) -> pd.DataFrame:
        """
        For each FileGroup in this Experiment, generate a Pandas DataFrame where each row is an event and the columns
        are the populations contained within this FileGroup. The columns are boolean arrays that signify if an event
        is a member of that population.

        Parameters
        ----------
        regex: str, optional
            Only include populations that match this regular expression pattern
        population_source: str, optional
            Only include populations that match this population source e.g. gate or cluster
        data_source: str (default='primary')
            The data file of interest i.e. either primary or the name of a control file
        verbose: bool (default=True)

        Returns
        -------
        Pandas.DataFrame
        """
        data = []
        for fg in progress_bar(self.fcs_files, verbose=verbose):
            df = fg.population_membership_boolean_matrix(
                regex=regex, population_source=population_source, data_source=data_source
            )
            df["sample_id"] = fg.primary_id
            data.append(df)
        data = pd.concat(data).fillna(0)
        cluster_cols = [x for x in data.columns if x != "sample_id"]
        data[cluster_cols] = data[cluster_cols].astype(int)
        return data

    def population_membership_mapping(
        self,
        regex: Optional[str] = None,
        population_source: Optional[str] = None,
        data_source: str = "primary",
        verbose: bool = True,
    ) -> Dict[str, Dict[int, Iterable[str]]]:
        """
        For each FileGroup in this Experiment, search the populations and create a dictionary where each key is
        the index of an event and the values are the list of populations that the event is a member of. The resulting
        mappings are then embedded within a dictionary where each mapping is indexed by the FileGroup ID.

        Parameters
        ----------
        regex: str, optional
            Only include populations that match this regular expression pattern
        population_source: str, optional
            Only include populations that match this population source e.g. gate or cluster
        data_source: str (default='primary')
            The data file of interest i.e. either primary or the name of a control file
        verbose: bool (default=True)

        Returns
        -------
        Dict[int, Iterable[str]]
        """
        data = {}
        for fg in progress_bar(self.fcs_files, verbose=verbose):
            data[fg.primary_id] = fg.population_membership_mapping(
                regex=regex, population_source=population_source, data_source=data_source
            )
        return data

    def list_populations(
        self, regex: Optional[str] = None, population_source: Optional[str] = None, data_source: str = "primary"
    ) -> List[str]:
        """
        List all the populations contained within a data source

        Parameters
        ----------
        regex: str, optional
            Only include populations that match this regular expression pattern
        population_source: str, optional
            Only include populations that match this population source e.g. gate or cluster
        data_source: str (default='primary')
            The data file of interest i.e. either primary or the name of a control file

        Returns
        -------
        List[str]
        """
        populations = [
            fg.list_populations(regex=regex, population_source=population_source, data_source=data_source)
            for fg in self.fcs_files
        ]
        populations = [p for sl in populations for p in sl]
        return list(set(populations))

    def control_fold_change(
        self,
        population: List[str],
        ctrl: List[str],
        feature: List[str],
        transform: str = "asinh",
        transform_kwargs: Optional[Dict] = None,
        verbose: bool = True,
    ) -> pd.DataFrame:
        """
        Compute the fold change between the MFI of a channel in the primary staining
        compared to some control for a chosen population. This is repeated for every
        file in this experiment and returned as a Pandas DataFrame with the columns:
        population, feature, ctrl, fold_change and sample_id.
        Population, control, and feature to use should be provided as lists of equal length,
        these lists are then paired.

        Parameters
        ----------
        population: List[str]
            List of populations to compute fold change for, must be present in both
            primary and control, and contain more than 3 events for both. Will log error
            and exclude pairing if not.
        ctrl: List[str]
            List of control data to compute fold change for
        feature: List[str]
            List of channels to compute fold change for
        transform: str (default='asinh')
            How to transform data prior to computation. Values will be additionally scaled
            between 0 and 1 prior to computing fold change to handle negative values
        transform_kwargs: Optional[Dict]
            Additional keyword arguments passed to transform
        verbose: bool (default=True)
            Show progress bar

        Returns
        -------
        Pandas.DataFrame
        """
        fold_change = []
        for fg in progress_bar(self.fcs_files, verbose=verbose):
            fold_change.append(
                fg.control_fold_change(
                    population=population,
                    ctrl=ctrl,
                    feature=feature,
                    transform=transform,
                    transform_kwargs=transform_kwargs,
                )
            )
        return pd.concat(fold_change).reset_index(drop=True)

    def control_effect_size(
        self,
        population: str,
        ctrl: str,
        feature: str,
        eftype: str = "cohen",
        transform: str = "asinh",
        transform_kwargs: Optional[Dict] = None,
        verbose: bool = True,
        njobs: int = -1,
        **kwargs,
    ) -> pd.DataFrame:
        """
        For each FileGroup in this Experiment, compute the effect size for a population
        when comparing the primary staining to some control. By default, will compute
        Cohen's D, which is the standardised difference between the means. See
        https://pingouin-stats.org/generated/pingouin.compute_effsize.html for valid
        methods that can be used for effect size.

        Parameters
        ----------
        population: str
            The population of interest
        ctrl: str
            Name of the control for comparison
        feature: str
            The name of the channel to compare between the primary stain and control
        eftype: str (default='cohen')
            The effect size method to use. Can be any valid method according to
            https://pingouin-stats.org/generated/pingouin.compute_effsize.html
        transform: str (default='asinh')
        transform_kwargs: dict, optional
            Additional keyword arguments passed to transform method
        verbose: bool (default=True)
        njobs: int (default=-1)
        kwargs:
            Additional keyword arguments passed to pingouin.compute_effsize

        Returns
        -------
        Pandas.DataFrame
        """
        logger.info("Loading data...")
        data = []
        for fg in progress_bar(self.fcs_files, verbose=verbose):
            try:
                primary_data = fg.load_population_df(
                    population=population,
                    transform=transform,
                    transform_kwargs=transform_kwargs,
                    data_source="primary",
                )[feature].values
                ctrl_data = fg.load_population_df(
                    population=population, transform=transform, transform_kwargs=transform_kwargs, data_source=ctrl
                )[feature].values
                if ctrl_data.shape[0] < 3 or primary_data.shape[0] < 3:
                    logger.warning(f"Either primary or control data for {fg.primary_id} has < 3 events")
                    continue
                data.append({"id": fg.primary_id, "primary": primary_data, "ctrl": ctrl_data})
            except MissingPopulationError:
                logger.warning(
                    f"{fg.primary_id} missing requested population {population} either in "
                    f"the primary staining or control. Use the 'propagate_populations_to_control' "
                    f"method to ensure populations are present in controls."
                )
            except (KeyError, ValueError, TypeError, EmptyPopulationError) as e:
                logger.warning(f"Could not obtain control effect size for {fg.primary_id}: {e}")
        logger.info("Computing effect size...")
        with Parallel(n_jobs=njobs) as parallel:
            results = parallel(
                delayed(effect_size)(d.get("primary"), d.get("ctrl"), eftype=eftype, **kwargs)
                for d in progress_bar(data, verbose=verbose)
            )
        results = [
            {
                "sample_id": i.get("id"),
                "effsize": j[0],
                "lower_ci": j[1],
                "upper_ci": j[2],
            }
            for i, j in zip(data, results)
        ]
        del data
        return pd.DataFrame(results)

    def merge_populations(self, mergers: Dict[str : List[str]]) -> Experiment:
        """
        For each FileGroup in sequence, merge populations. Given dictionary should contain
        a key corresponding to the new population name and value being a list of populations
        to merge. If one or more populations are missing, then available populations will be
        merged.

        Parameters
        ----------
        mergers: dict

        Returns
        -------
        None
        """
        logger.info(f"Merging populations: {mergers}")
        for new_population_name, targets in mergers.items():
            for f in self.fcs_files:
                pops = [p for p in targets if p in f.list_populations()]
                try:
                    f.merge_populations(populations=pops, new_population_name=new_population_name)
                    f.save()
                except ValueError as e:
                    logger.warning(f"Failed to merge populations for {f.primary_id}: {str(e)}")
        return self

    def propagate_gates_to_control(self, ctrl: str, flag: float = 0.25) -> pd.DataFrame:
        """
        For each FileGroup, propagate gates in primary staining to control files, generating matching populations
        in each control. Returns a DataFrame of statistics for the new populations, including the number of events
        as a percentage of the parent population in the primary staining and the control. The fold difference
        in this percentage between the primary and control is also provided, along with a 'Flag' column with a value
        of 'True' where the absolute fold change exceeds the threshold given by 'flag'.

        Parameters
        ----------
        ctrl: str
        flag: float (default=0.25)

        Returns
        -------
        Pandas.DataFrame
        """
        stats = []
        for fg in progress_bar(self.fcs_files):
            try:
                fg, st = copy_populations_to_controls_using_geoms(filegroup=fg, ctrl=ctrl, flag=flag)
                stats.append(st)
                fg.save()
            except (ValueError, MissingPopulationError, DuplicatePopulationError, EmptyPopulationError) as e:
                logger.error(f"Unable to generate populations for {ctrl} in {fg.primary_id}: {e}")
        return pd.concat(stats).reset_index(drop=True)

    def delete(self, signal_kwargs=None, **write_concern):
        """
        Delete Experiment; will delete all associated FileGroups.

        Returns
        -------
        None
        """
        logger.info(f"Attempting to delete experiment {self.experiment_id}")
        for f in self.fcs_files:
            logger.debug(f"deleting associated FileGroup {f.primary_id}")
            f.delete()
        self.save()
        super().delete(signal_kwargs=signal_kwargs, **write_concern)
        logger.info("Experiment successfully deleted.")


def single_cell_dataframe(
    experiment: Experiment,
    populations: Optional[Union[str, List[str]]] = "root",
    regex: Optional[str] = None,
    transform: Optional[Union[str, Dict]] = "asinh",
    transform_kwargs: Optional[Dict] = None,
    sample_ids: Optional[List[str]] = None,
    verbose: bool = True,
    data_source: str = "primary",
    label_parent: bool = False,
    frac_of: Optional[List[str]] = None,
    sample_size: Optional[Union[int, float]] = None,
    sampling_level: str = "file",
    sampling_method: str = "uniform",
    sampling_kwargs: Optional[Dict] = None,
    meta_vars: Optional[Dict] = None,
    source_counts: bool = False,
    warn_missing: bool = True,
) -> pd.DataFrame:
    """
    Generate a single cell DataFrame (where each row is an event) that is a concatenation of population data from many
    samples from a single Experiment. Population level data is identifiable from the 'population_label'
    column, sample level data identifiable from the 'sample_id' column, and subject level information
    from the 'subject_id' column.

    Parameters
    ----------
    experiment: Experiment
    populations: Union[List[str], str], optional
        * Single string value will load the matching population from samples in 'experiment'
        * List of strings will load the matching populations from samples in 'experiment'
        * None, to provide a regular expression (regex) for population matching
    regex: str, optional
        Match all populations matching the given pattern; if given, populations argument is ignored
    transform: Union[str, Dict[str, str]] (default='asinh')
        Transformation applied to the single cell data. If a string is provided, method is applied to
        all features. If a dictionary is provided, keys are interpreted as names of features and values
        the transform to be applied.
    transform_kwargs: Dict, optional
        Additional keyword arguments passed to transform method
    sample_ids: List[str], optional
        List of samples to include. If None (default) then loads all available samples in experiment
    verbose: bool (default=True)
    data_source: str (default='primary')
        Specify the source file (i.e. primary or some control)
    label_parent: bool (default=False)
        If True, additional column appended with parent name for each population
    frac_of: List[str], optional
        Provide a list of populations and additional columns will be appended to resulting
        DataFrame containing the fraction of the requested population compared to each population
        in this list
    sample_size: Union[int, float], optional
        If given, the DataFrame will either be down-sampled after acquiring data from each FileGroup
        or FileGroups are sampled individually - this behaviour is controlled by 'sampling_level'.
        If sampling_level = "file", then the sample_size is the number of events to obtain from each
        FileGroup. If sampling_level = "experiment", then the sampling size is the desired size of the
        resulting concatenated DataFrame.
    sampling_level: str, (default="file")
        If "file" (default) then each FileGroup is sampled before concatenating into a single DataFrame.
        If "experiment", then data is obtained from each FileGroup first, and then the concatenated
        data is sampled.
        If "population" then will attempt to sample the desired number of events from each population.
    sampling_method: str (default="uniform")
        The sampling method to use; see cytopy.utils.sampling
    sampling_kwargs: Dict, optional
        Additional keyword arguments passed to sampling method
    meta_vars: Dict[str, Union[str, List[str]]], optional
        If provided, additional columns will be appended to the resulting DataFrame (with column names matching
        keys in provided dictionary)
    source_counts: bool (default=False)
        If True, an additional column is generated with an integer value of how many source methods a population
        was generated by (this is only relevant for consensus clustering where populations are the amalgamation
        of multiple clustering techniques)
    warn_missing: bool (default=True)
        Log a warning if a population is missing in a FileGroup.

    Returns
    -------
    Pandas.DataFrame
    """
    sample_ids = sample_ids or list(experiment.list_samples())
    sampling_kwargs = sampling_kwargs or {}
    data = []

    method = "load_population_df"
    kwargs = dict(
        population=populations,
        transform=transform,
        transform_kwargs=transform_kwargs,
        label_parent=label_parent,
        frac_of=frac_of,
        data_source=data_source,
        meta_vars=meta_vars,
        source_counts=source_counts,
    )

    if sample_size is not None and sampling_level == "file":
        kwargs = {**kwargs, **{"sample_size": sample_size, "sampling_method": sampling_method, **sampling_kwargs}}

    if isinstance(populations, list) or regex is not None:
        method = "load_multiple_populations"
        kwargs["sample_at_population_level"] = sampling_level == "population"
        kwargs["regex"] = regex
        kwargs["populations"] = populations
        kwargs["warn_missing"] = warn_missing
        kwargs.pop("population")

    for _id in progress_bar(sample_ids, verbose=verbose):
        try:
            fg = experiment.get_sample(sample_id=_id)
            if data_source not in fg.file_paths.keys():
                logger.warning(f"{_id} missing data source {data_source}")
                continue
            logger.debug(f"Loading FileGroup data from {_id}; {fg.id}")
            pop_data = getattr(fg, method)(**kwargs)
            pop_data["sample_id"] = _id
            pop_data["subject_id"] = None
            if fg.subject:
                pop_data["subject_id"] = fg.subject.subject_id
            data.append(pop_data)
        except MissingPopulationError as e:
            if warn_missing:
                logger.warning(f"{_id} missing population(s): {e}")
        except EmptyPopulationError:
            if warn_missing:
                logger.warning(f"No events found in {populations} within {_id}")

    data = pd.concat(data).reset_index().rename({"Index": "original_index"}, axis=1)

    if sample_size is not None and sampling_level == "experiment":
        data = sample_dataframe(
            data=data,
            sample_size=sample_size,
            method=sampling_method,
            **sampling_kwargs,
        ).reset_index(drop=True)
    return data


def single_cell_anndata(
    experiment: Experiment,
    populations: Optional[Union[str, List[str]]] = "root",
    regex: Optional[str] = None,
    transform: Optional[Union[str, Dict]] = "asinh",
    transform_kwargs: Optional[Dict] = None,
    sample_ids: Optional[List[str]] = None,
    verbose: bool = True,
    data_source: str = "primary",
    label_parent: bool = False,
    frac_of: Optional[List[str]] = None,
    sample_size: Optional[Union[int, float]] = None,
    sampling_level: str = "file",
    sampling_method: str = "uniform",
    sampling_kwargs: Optional[Dict] = None,
    meta_vars: Optional[Dict] = None,
    source_counts: bool = False,
    warn_missing: bool = True,
) -> AnnData:
    """
    This function is a wrapper to the single_cell_dataframe function, but returns an annotated DataFrame compatible
    with ScanPy.

    Generate a single cell DataFrame (where each row is an event) that is a concatenation of population data from many
    samples from a single Experiment. Population level data is identifiable from the 'population_label'
    column, sample level data identifiable from the 'sample_id' column, and subject level information
    from the 'subject_id' column.

    Parameters
    ----------
    experiment: Experiment
    populations: Union[List[str], str], optional
        * Single string value will load the matching population from samples in 'experiment'
        * List of strings will load the matching populations from samples in 'experiment'
        * None, to provide a regular expression (regex) for population matching
    regex: str, optional
        Match all populations matching the given pattern; if given, populations argument is ignored
    transform: Union[str, Dict[str, str]] (default='asinh')
        Transformation applied to the single cell data. If a string is provided, method is applied to
        all features. If a dictionary is provided, keys are interpreted as names of features and values
        the transform to be applied.
    transform_kwargs: Dict, optional
        Additional keyword arguments passed to transform method
    sample_ids: List[str], optional
        List of samples to include. If None (default) then loads all available samples in experiment
    verbose: bool (default=True)
    data_source: str (default='primary')
        Specify the source file (i.e. primary or some control)
    label_parent: bool (default=False)
        If True, additional column appended with parent name for each population
    frac_of: List[str], optional
        Provide a list of populations and additional columns will be appended to resulting
        DataFrame containing the fraction of the requested population compared to each population
        in this list
    sample_size: Union[int, float], optional
        If given, the DataFrame will either be down-sampled after acquiring data from each FileGroup
        or FileGroups are sampled individually - this behaviour is controlled by 'sampling_level'.
        If sampling_level = "file", then the sample_size is the number of events to obtain from each
        FileGroup. If sampling_level = "experiment", then the sampling size is the desired size of the
        resulting concatenated DataFrame.
    sampling_level: str, (default="file")
        If "file" (default) then each FileGroup is sampled before concatenating into a single DataFrame.
        If "experiment", then data is obtained from each FileGroup first, and then the concatenated
        data is sampled.
        If "population" then will attempt to sample the desired number of events from each population.
    sampling_method: str (default="uniform")
        The sampling method to use; see cytopy.utils.sampling
    sampling_kwargs: Dict, optional
        Additional keyword arguments passed to sampling method
    meta_vars: Dict[str, Union[str, List[str]]], optional
        If provided, additional columns will be appended to the resulting DataFrame (with column names matching
        keys in provided dictionary)
    source_counts: bool (default=False)
        If True, an additional column is generated with an integer value of how many source methods a population
        was generated by (this is only relevant for consensus clustering where populations are the amalgamation
        of multiple clustering techniques)
    warn_missing: bool (default=True)
        Log a warning if a population is missing in a FileGroup.

    Returns
    -------
    Pandas.DataFrame
    """
    data = single_cell_dataframe(
        experiment=experiment,
        populations=populations,
        regex=regex,
        transform=transform,
        transform_kwargs=transform_kwargs,
        sample_ids=sample_ids,
        verbose=verbose,
        data_source=data_source,
        label_parent=label_parent,
        frac_of=frac_of,
        sample_size=sample_size,
        sampling_level=sampling_level,
        sampling_method=sampling_method,
        sampling_kwargs=sampling_kwargs,
        meta_vars=meta_vars,
        source_counts=source_counts,
        warn_missing=warn_missing,
    )
    channels = experiment.panel.list_channels()
    x = data[channels].values
    meta_vars = meta_vars or {}
    frac_of = frac_of or []
    add_cols = list(meta_vars.keys()) + [f"frac_of_{x}" for x in frac_of]
    if label_parent:
        add_cols.append("parent_label")
    obs = data[["subject_id", "sample_id", "original_index"] + add_cols]
    var = pd.DataFrame(index=[x for x in data.columns if x not in obs.columns])
    return AnnData(X=x, obs=obs, var=var, obsm={"X": x})
