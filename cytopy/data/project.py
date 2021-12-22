#!/usr/bin.env/python
# -*- coding: utf-8 -*-
"""
Every analysis is controlled using the Project class, the highest
structure in the hierarchy of documents in the central MongoDB database.
You can create multiple experiments for a Project, each attaining to
a different staining panel. Experiments are accessed and managed through
the Project class.

Projects also house the subjects (represented by the Subject class;
see cytopy.data.subject) of an analysis which can contain multiple
meta-data.

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

import os
import logging
import datetime
from copy import deepcopy
from typing import Dict, Mapping, Union
from typing import List
from typing import Optional

import mongoengine
import numpy as np
import pandas as pd

from ..feedback import progress_bar
from .experiment import Experiment
from .read_write import parse_directory_for_cytometry_files
from .subject import Subject
from .errors import DuplicateExperimentError
from .errors import DuplicateSubjectError
from .errors import MissingExperimentError
from .errors import MissingSubjectError

logger = logging.getLogger(__name__)


def _build_subject_records(data: Dict[str, pd.DataFrame]) -> Dict[str, Union[Mapping, List[Mapping]]]:
    """
    Convert subject data from one or more DataFrames into a JSON like data structure. Data should contain parent
    key name as keys and DataFrames as the value.

    Parameters
    ----------
    data: Dict[str, Pandas.DataFrame]

    Returns
    -------
    Dict[str, Union[Mapping, List[Mapping]]]
    """
    records = {}
    for parent_key, df in data.items():
        x = [row.to_dict() for _, row in df.iterrows()]
        if len(x) == 1:
            records[parent_key] = x[0]
        else:
            records[parent_key] = x
    return records


class Project(mongoengine.Document):
    """
    A project is the highest controlling structure of an analysis and houses
    all the experiments, their associated FileGroups and the populations
    contained in each FileGroup and the populations clusters.

    Project can be used to create new experiments and to load existing experiments
    to interact with.

    Single cell data is stored in HDF5 files and the meta-data stored in MongoDB.
    When creating a Project you should specify where to store these HDF5 files.
    This data is stored locally and the local path is stored in 'data_directory'.
    This will be checked each time the object is initiated but can be changed
    using the 'update_data_directory' method.


    Attributes
    ----------
    project_id: str, required
        unique identifier for project
    subjects: list
        List of references for associated subjects; see Subject
    start_date: DateTime
        date of creation
    owner: str, required
        user name of owner
    experiments: list
        List of references for associated fcs files
    """

    project_id = mongoengine.StringField(required=True, unique=True)
    subjects = mongoengine.ListField(mongoengine.ReferenceField(Subject, reverse_delete_rule=4))
    start_date = mongoengine.DateTimeField(default=datetime.datetime.now)
    owner = mongoengine.StringField(requred=True)
    experiments = mongoengine.ListField(mongoengine.ReferenceField(Experiment, reverse_delete_rule=4))

    meta = {"db_alias": "core", "collection": "projects"}

    def __repr__(self):
        return f"Project(project_id={self.project_id})"

    def get_experiment(self, experiment_id: str) -> Experiment:
        """
        Load the experiment object for a given experiment ID

        Parameters
        ----------
        experiment_id: str
            experiment to load

        Returns
        --------
        Experiment

        Raises
        -------
        MissingExperimentError
            If requested experiment does not exist in this project
        """
        try:
            return [e for e in self.experiments if e.experiment_id == experiment_id][0]
        except IndexError:
            logger.error(f"Invalid experiment; {experiment_id} does not exist")
            raise MissingExperimentError(f"Invalid experiment; {experiment_id} does not exist")

    def add_experiment(self, experiment_id: str, panel_definition: str) -> Experiment:
        """
        Add new experiment to project. Note you must provide either a path to a template for the panel
        definition (panel_definition) or the name of an existing panel (panel_name). If panel_definition is provided,
        then the panel_name will be used to name the new Panel document associated to this experiment.
        If no panel_name is provided, then the panel name will default to "{experiment_id}_panel".

        Parameters
        -----------
        experiment_id: str
            experiment name
        panel_definition: str
            Path to excel template for generating the panel

        Returns
        --------
        Experiment
            Newly created FCSExperiment

        Raises
        -------
        DuplicateExperimentError
            If given experiment ID already exists
        """
        if experiment_id in [x.experiment_id for x in self.experiments]:
            raise DuplicateExperimentError(f"Experiment with id {experiment_id} already exists!")
        exp = Experiment(experiment_id=experiment_id)
        exp.generate_panel(panel_definition=panel_definition)
        exp.save()
        self.experiments.append(exp)
        self.save()
        return exp

    def add_subject(self, subject_id: str, **kwargs) -> Subject:
        """
        Create a new subject and associated to project; a subject is an individual element of a study
        e.g. a patient or a mouse

        Parameters
        -----------
        subject_id: str
            subject ID for the new subject
        kwargs:
            Additional keyword arguments to pass to Subject initialisation (see cytopy.data.subject.Subject)

        Returns
        --------
        None

        Raises
        -------
        DuplicateSubjectError
            If subject already exists
        """
        if subject_id in [x.subject_id for x in self.subjects]:
            logger.error(f"Subject with ID {subject_id} already exists")
            raise DuplicateSubjectError(f"Subject with ID {subject_id} already exists")
        new_subject = Subject(subject_id=subject_id, **kwargs)
        new_subject.save()
        self.subjects.append(new_subject)
        self.save()
        return new_subject

    def add_subjects_with_metadata(
        self,
        target_dir: str,
        id_column: str,
        verbose: bool = True,
        exclude_columns: Optional[Dict[str, List[str]]] = None,
    ) -> Project:
        """
        Given a target directory containing CSV or Excel files, this function will parse these tabular files and
        search the 'id_column' within them; must not contain duplicate entries. New subjects will be generated with
        subject ID's according to the 'id_column'. All other columns within the tabular file(s) will populate
        meta variables in the Subject documents. Meta-data will be nested within keys using the base name of
        each file.

        Parameters
        ----------
        target_dir: str
        id_column: str
        verbose: bool (default=True)
        exclude_columns: Dict[str, List[str]], optional
            If provided, key should correspond to the name of a file within 'target_dir' and values should be
            columns to ignore inside the file.

        Returns
        -------
        Project
        """
        exclude_columns = exclude_columns or {}
        targets = {
            os.path.splitext(os.path.basename(x))[0]: os.path.join(target_dir, x)
            for x in os.listdir(target_dir)
            if os.path.splitext(os.path.basename(x))[1].lower() in [".csv", ".xlsx"]
        }
        targets = {
            x: pd.read_csv(path) if path.endswith(".csv") else pd.read_excel(path) for x, path in targets.items()
        }
        targets = {x: df[[i for i in df.columns if i not in exclude_columns.get(x, [])]] for x, df in targets.items()}
        unique_ids = set(
            list(np.concatenate([x[id_column].unique() for x in targets.values()], axis=0))
        )
        for _id in progress_bar(unique_ids, verbose=verbose):
            records = _build_subject_records(
                data={x: df[df[id_column] == _id].drop(id_column, axis=1) for x, df in targets.items()}
            )
            self.add_subject(subject_id=_id, **records)
        self.save()
        return self

    def add_cytometry_data_from_file_tree(
        self,
        target_directory: str,
        control_id: Optional[str] = None,
        controls: Optional[Dict[str, List[str]]] = None,
        exclude_files: Optional[str] = None,
        exclude_dir: Optional[str] = None,
        compensation_file: Optional[str] = None,
        compensate: bool = True,
        verbose: bool = True,
    ) -> Project:
        """
        Given some target directory, transverse the file tree and populate the project. The structure of the
        Project will then be modelled around the file tree. To use this method of project creation, the file
        tree should be structured like so:

        target_directory
        |
        --- Experiment 1
            |
            ---- Subject 1
                |
                ------ Primary.fcs
                ------ Control_1.fcs
                ------ Control_2.fcs
                ------ Control_n.fcs
                ------
            ---- Subject 2
            ---- Subject n
        --- Experiment 2
        --- Experiment n

        Each sub-folder should be an Experiment and the folder name must match an existing Experiment in this
        Project (if not, it will be skipped). Within the Experiment folder should be sub-folders named with existing
        Subject IDs (unrecognised subjects will be skipped). Within each subject folder there should be one or more
        FCS files; all other formats are ignored. Files that correspond to control data will be identified
        by the file naming containing the specified 'control_id' string. Control names can be specified with
        'controls' and should be a dictionary where the keys match experiment IDs.

        Parameters
        ----------
        target_directory: str
            File path to the parent directory
        control_id: str, optional
            If provided, files containing this string will be marked as controls
        controls: Dict[str, List[str]], optional
            Required if control_id provided. Keys must be the name of experiment IDs and values the identifiers
            for controls. Control files will be matched based on identifiers being present in the file name of
            fcs files.
        exclude_files: str, optional
            If provided, any files containing this string will be ignored
        exclude_dir: str, optional
            If provided, any directory containing this string will be ignored
        compensation_file: str, optional
            If provided, will search each subject directory for this file name (including file extension) and
            use this file as the compensation file for FCS files in this directory (Note: compensation_file can
            be a CSV file)
        compensate: bool (default=True)
            Specifies whether FileGroups compensate data
        verbose: bool (default=True)

        Returns
        -------
        Project
        """
        logger.info("Checking file tree and preparing for data entry.")
        cyto_files = {}
        experiment_dirs = os.listdir(target_directory)
        for exp_id in experiment_dirs:
            if exp_id not in self.list_experiments():
                logger.warning(f"{exp_id} is not a recognised experiment and will be ignored.")
                continue
            cyto_files[exp_id] = {}
            experiment_controls = controls.get(exp_id, [])
            if not experiment_controls:
                logger.warning(f"No control files provided for {exp_id}")
            for subject_id in os.listdir(os.path.join(target_directory, exp_id)):
                if subject_id not in self.list_subjects():
                    logger.warning(f"{subject_id} is not a recognised subject and will be ignored.")
                    continue
                cyto_files[exp_id][subject_id] = parse_directory_for_cytometry_files(
                    fcs_dir=os.path.join(target_directory, exp_id, subject_id),
                    control_id=control_id,
                    control_names=experiment_controls,
                    exclude_files=exclude_files,
                    exclude_dir=exclude_dir,
                    compensation_file=compensation_file,
                )
        for exp_id, subject_files in cyto_files.items():
            logger.info(f"Adding cytometry data for {exp_id}")
            experiment = self.get_experiment(experiment_id=exp_id)
            for subject_id, files in progress_bar(subject_files.items(), verbose=verbose):
                compensation_matrix_path = files.get("compensation_file", None)
                experiment.add_filegroup(
                    sample_id=f"{subject_id}_{exp_id}",
                    paths={x: path for x, path in files.items() if x != "compensation_file"},
                    compensate=compensate,
                    compensation_matrix=compensation_matrix_path,
                    subject_id=subject_id,
                )
        return self

    def list_subjects(self) -> List[str]:
        """
        Generate a list of subject ID for subjects associated to this project

        Returns
        --------
        List[str]
        """
        return [s.subject_id for s in self.subjects]

    def list_experiments(self) -> List[str]:
        """
        Lists experiments in project

        Returns
        -------
        List[str]
        """
        return [e.experiment_id for e in self.experiments]

    def get_subject(self, subject_id: str) -> Subject:
        """
        Given a subject ID associated to Project, return the Subject document

        Parameters
        -----------
        subject_id: str
            subject ID to pull

        Returns
        --------
        Subject

        Raises
        -------
        MissingSubjectError
            If desired subject does not exist
        """
        if subject_id not in self.list_subjects():
            raise MissingSubjectError(f"Invalid subject ID {subject_id}, does not exist")
        return Subject.objects(subject_id=subject_id).get()

    def delete_experiment(self, experiment_id: str) -> Project:
        """
        Delete experiment

        Parameters
        ----------
        experiment_id: str

        Returns
        -------
        Project
        """
        if experiment_id not in self.list_experiments():
            raise MissingExperimentError(f"No such experiment {experiment_id}")
        exp = self.get_experiment(experiment_id)
        exp.delete()
        return self

    def delete(self, delete_h5_data: bool = True, *args, **kwargs) -> None:
        """
        Delete project (wrapper function of mongoengine.Document.delete)

        Parameters
        -----------
        delete_h5_data: bool (default=True)
            Delete associated HDF5 data
        args:
            positional arguments to pass to parent call (see mongoengine.Document.delete)
        kwargs:
            keyword arguments to pass to parent call (see mongoengine.Document.delete)

        Returns
        --------
        None
        """
        logger.info(f"Deleting project {self.project_id}")
        logger.info("Deleting associated subjects...")
        for s in self.subjects:
            s.delete()
        logger.info("Deleting associated experiments...")
        for e in self.experiments:
            e.delete()
        super().delete(*args, **kwargs)
        logger.info("Project deleted.")


def merge_experiments(
    project: Project, experiment_left: str, experiment_right: str, new_experiment_id: str
) -> Experiment:
    """
    Merge two experiments in a Project (Must have equivalent panels!)

    Parameters
    ----------
    project: Project
    experiment_left: str
        Experiment ID of left experiment
    experiment_right: str
        Experiment ID of right experiment
    new_experiment_id: str
        New experiment ID

    Returns
    -------
    Experiment
        Newly created experiment
    """
    assert new_experiment_id not in project.list_experiments(), f"{new_experiment_id} already exists!"
    experiment_left = project.get_experiment(experiment_id=experiment_left)
    experiment_right = project.get_experiment(experiment_id=experiment_right)
    assert experiment_right.panel == experiment_left.panel, f"Experiments must have identical panels"
    new_experiment = Experiment(experiment_id=new_experiment_id)
    new_experiment.panel = deepcopy(experiment_left.panel)
    new_experiment.fcs_files = experiment_left.fcs_files + experiment_right.fcs_files
    new_experiment.save()
    project.experiments.append(new_experiment)
    project.save()
    return new_experiment
