#!/usr/bin.env/python
# -*- coding: utf-8 -*-
"""
The experiment module houses the Experiment class, used to define
cytometry based experiments that can consist of one or more biological
specimens. An experiment should be defined for each cytometry staining
panel used in your analysis and the single cell data (contained in
*.fcs files) added to the experiment using the 'add_new_sample' method.
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
import gc
import logging
import os
from collections import Counter
from typing import Dict
from typing import List
from typing import Optional
from typing import Union

import matplotlib.pyplot as plt
import mongoengine
import numpy as np
import pandas as pd

from ..feedback import progress_bar
from ..utils.sampling import sample_dataframe
from .errors import DuplicatePopulationError
from .errors import DuplicateSampleError
from .errors import MissingPopulationError
from .errors import MissingSampleError
from .errors import PanelError
from .fcs import copy_populations_to_controls_using_geoms
from .fcs import FileGroup
from .panel import Panel
from .subject import Subject

__author__ = "Ross Burton"
__copyright__ = "Copyright 2020, cytopy"
__credits__ = ["Ross Burton", "Simone Cuff", "Andreas Artemiou", "Matthias Eberl"]
__license__ = "MIT"
__version__ = "2.0.0"
__maintainer__ = "Ross Burton"
__email__ = "burtonrj@cardiff.ac.uk"
__status__ = "Production"
logger = logging.getLogger(__name__)


class Experiment(mongoengine.Document):
    """
    Container for Cytometry experiment. The correct way to generate and load these objects is using the
    Project.add_experiment method (see cytopy.data.project.Project). This object provides access
    to all experiment-wide functionality. New files can be added to an experiment using the
    add_new_sample method.

    Attributes
    -----------
    experiment_id: str, required
        Unique identifier for experiment
    panel: ReferenceField, required
        Panel object describing associated channel/marker pairs
    fcs_files: ListField
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

    @staticmethod
    def _check_panel(panel_definition: str or None):
        """
        Check that parameters provided for defining a panel are valid.

        Parameters
        ----------
        panel_definition: str or None
            Path to a panel definition

        Returns
        -------
        None

        Raises
        ------
        ValueError
            Given parameters are invalid
        """
        if not os.path.isfile(panel_definition):
            raise PanelError(f"{panel_definition} does not exist")

        if not os.path.splitext(panel_definition)[1] in [".xls", ".xlsx"]:
            raise PanelError("Panel definition is not a valid Excel document")

    def generate_panel(self, panel_definition: str) -> None:
        """
        Generate a new panel using the panel definition provided (path to a valid template).

        Parameters
        ----------
        panel_definition: Union[str, dict]
            Path to a panel definition

        Returns
        -------
        None

        Raises
        ------
        ValueError
            Panel definition is not a string or dict
        """
        new_panel = Panel()
        new_panel.create_from_tabular(path=panel_definition)
        self.panel = new_panel

    def delete_all_populations(self, sample_id: str) -> None:
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
        None
        """
        for f in self.fcs_files:
            if sample_id == "all" or f.primary_id == sample_id:
                logger.info(f"Deleting all populations from FileGroup {sample_id}; {f.id}")
                f.populations = [p for p in f.populations if p.population_name == "root"]
                f.save()

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
        logger.debug(f"Attempting to fetch FileGroup {sample_id}")
        if not self.sample_exists(sample_id):
            raise MissingSampleError(f"Invalid sample: {sample_id} not associated with this experiment")
        return [f for f in self.fcs_files if f.primary_id == sample_id][0]

    def filter_samples_by_subject(self, query: Union[str, mongoengine.queryset.visitor.Q]) -> List:
        """
        Filter FileGroups associated to this experiment based on some subject meta-data

        Parameters
        ----------
        query: str or mongoengine.queryset.visitor.Q
            Query to make on Subject

        Returns
        -------
        List
        """
        logger.debug(f"Fetching list of FileGroups associated to Subject on query {query}")
        matches = list()
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
        Generate a list IDs of file groups associated to experiment

        Parameters
        -----------
        valid_only: bool
            If True, returns only valid samples (samples without 'invalid' flag)

        Returns
        --------
        List
            List of IDs of file groups associated to experiment
        """
        if valid_only:
            return [f.primary_id for f in self.fcs_files if f.valid]
        return [f.primary_id for f in self.fcs_files]

    def random_filegroup(self):
        return np.random.choice(self.list_samples())

    def remove_sample(self, sample_id: str):
        """
        Remove sample (FileGroup) from experiment.

        Parameters
        -----------
        sample_id: str
            ID of sample to remove

        Returns
        --------
        None
        """
        logger.debug(f"Deleting {sample_id}")
        filegrp = self.get_sample(sample_id)
        self.fcs_files = [f for f in self.fcs_files if f.primary_id != sample_id]
        filegrp.delete()
        self.save()

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
    ):
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
                channel_mappings=self.panel.build_mappings(path=paths.values(), s3_bucket=s3_bucket),
            )
        )
        self.save()
        gc.collect()

    def control_counts(self, ax: Optional[plt.Axes] = None) -> plt.Axes:
        """
        Generates a barplot of total counts of each control in Experiment FileGroup's

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
    def _prop_of_parent(df: pd.DataFrame, parent: str):
        frac_of_parent = []
        parent_n = float(df[df.population_name == parent]["n"].values[0])
        for _, row in df.iterrows():
            if row.n == 0:
                frac_of_parent.append(0)
            else:
                frac_of_parent.append(row.n / parent_n)
        df[f"frac_of_{parent}"] = frac_of_parent
        return df

    def population_statistics(
        self,
        populations: Union[List, None] = None,
        meta_vars: Optional[Dict] = None,
        additional_parent: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Generates a Pandas DataFrame of population statistics for all FileGroups
        of an Experiment, for the given populations or all available populations
        if 'populations' is None.

        Parameters
        ----------
        populations: list, optional

        Returns
        -------
        Pandas.DataFrame
        """
        data = list()
        for f in self.fcs_files:
            for p in populations or f.list_populations():
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

    def control_eff_size(
        self,
        population: str,
        ctrl: str,
        feature: str,
        method: str = "cohen",
        transform: str = "asinh",
        transform_kwargs: Optional[Dict] = None,
        verbose: bool = True,
        **kwargs,
    ):
        results = []
        for fg in progress_bar(self.fcs_files, verbose=verbose):
            try:
                effsize, lower, upper = fg.control_eff_size(
                    population=population,
                    ctrl=ctrl,
                    feature=feature,
                    method=method,
                    transform=transform,
                    transform_kwargs=transform_kwargs,
                    **kwargs,
                )
                if lower is None:
                    results.append(pd.DataFrame({"sample_id": [fg.primary_id], "effsize": [effsize]}))
                else:
                    results.append(
                        pd.DataFrame(
                            {
                                "sample_id": [fg.primary_id],
                                "effsize": [effsize],
                                "lower_ci": [lower],
                                "upper_ci": [upper],
                            }
                        )
                    )
            except MissingPopulationError:
                logger.error(
                    f"{fg.primary_id} missing requested population {population} either in "
                    f"the primary staining or control. Use the 'propagate_populations_to_control' "
                    f"method to ensure populations are present in controls."
                )
            except (KeyError, ValueError, TypeError) as e:
                logger.error(f"Could not obtain control effect size for {fg.primary_id}: {e}")
        return pd.concat(results).reset_index(drop=True)

    def merge_populations(self, mergers: Dict):
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
                    f.merge_non_geom_populations(populations=pops, new_population_name=new_population_name)
                    f.save()
                except ValueError as e:
                    logger.warning(f"Failed to merge populations for {f.primary_id}: {str(e)}")

    def propagate_populations_to_control(self, ctrl: str, flag: float = 0.25):
        stats = []
        for fg in progress_bar(self.fcs_files):
            try:
                fg, st = copy_populations_to_controls_using_geoms(filegroup=fg, ctrl=ctrl, flag=flag)
                stats.append(st)
                fg.save()
            except (ValueError, MissingPopulationError, DuplicatePopulationError) as e:
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
) -> pd.DataFrame:
    """
    Generate a single cell DataFrame that is a concatenation of population data from many
    samples from a single Experiment. Population level data is identifiable from the 'population_label'
    column, sample level data identifiable from the 'sample_id' column, and subject level information
    from the 'subject_id' column.

    Parameters
    ----------
    experiment: Experiment
    populations: list or str, optional
        * Single string value will load the matching population from samples in 'experiment'
        * List of strings will load the matching populations from samples in 'experiment'
        * None, to provide a regular expression (regex) for population matching
    regex: str, optional
        Match all populations matching the given pattern; if given, populations argument is ignored
    transform: str or dict (default='logicle')
        Transformation applied to the single cell data. If a string is provided, method is applied to
        all features. If a dictionary is provided, keys are interpreted as names of features and values
        the transform to be applied.
    transform_kwargs: dict, optional
        Additional keyword arguments passed to transform method
    sample_ids: list, optional
        List of samples to include. If None (default) then loads all available samples in experiment
    verbose: bool (default=True)
    ctrl: str, optional
        Loads data corresponding to the given control. NOTE: only supports loading of a single population
        from each sample in 'experiment'
    label_parent: bool (default=False)
        If True, additional column appended with parent name for each population
    frac_of: list, optional
        Provide a list of populations and additional columns will be appended to resulting
        DataFrame containing the fraction of the requested population compared to each population
        in this list
    sample_size: int or float, optional
        If given, the DataFrame will either be downsampled after aquiring data from each FileGroup
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

    Returns
    -------
    Pandas.DataFrame
    """
    logger.debug(f"Loading data from {experiment.experiment_id}")
    sample_ids = sample_ids or list(experiment.list_samples())
    sampling_kwargs = sampling_kwargs or {}
    data = list()

    method = "load_population_df"
    kwargs = dict(
        population=populations,
        transform=transform,
        transform_kwargs=transform_kwargs,
        label_parent=label_parent,
        frac_of=frac_of,
        data_source=data_source,
        meta_vars=meta_vars,
    )

    if sample_size is not None and sampling_level == "file":
        kwargs = {**kwargs, **{"sample_size": sample_size, "sampling_method": sampling_method, **sampling_kwargs}}

    if isinstance(populations, list) or regex is not None:
        method = "load_multiple_populations"
        kwargs["sample_at_population_level"] = sampling_level == "population"
        kwargs["regex"] = regex
        kwargs["populations"] = populations
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
            logger.error(f"{_id} missing population(s): {e}")

    data = pd.concat(data).reset_index().rename({"Index": "original_index"}, axis=1)

    if sample_size is not None and sampling_level == "experiment":
        data = sample_dataframe(
            data=data,
            sample_size=sample_size,
            method=sampling_method,
            **sampling_kwargs,
        ).reset_index(drop=True)
    return data
