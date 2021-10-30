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
import datetime
import os
from collections import defaultdict
from copy import deepcopy
from typing import Dict
from typing import List
from typing import Optional

import mongoengine
import numpy as np
import pandas as pd

from ..feedback import progress_bar
from .errors import *
from .experiment import Experiment
from .read_write import parse_directory_for_cytometry_files
from .subject import Subject

logger = logging.getLogger(__name__)


def _build_subject_records(data: Dict[str, pd.DataFrame]) -> Dict[str, List[Dict]]:
    records = defaultdict(list)
    for parent_key, df in data.items():
        records[parent_key] = [row.to_dict() for _, row in df.iterrows()]
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
    data_directory: str, required
        Path to the local directory for storing HDF5 files.
    """

    project_id = mongoengine.StringField(required=True, unique=True)
    subjects = mongoengine.ListField(mongoengine.ReferenceField(Subject, reverse_delete_rule=4))
    start_date = mongoengine.DateTimeField(default=datetime.datetime.now)
    owner = mongoengine.StringField(requred=True)
    experiments = mongoengine.ListField(mongoengine.ReferenceField(Experiment, reverse_delete_rule=4))

    meta = {"db_alias": "core", "collection": "projects"}

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
        Add new experiment to project. Note you must provide either a path to an excel template for the panel
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
    ):
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
        unique_ids = set(np.concatenate([x[id_column].unique() for x in targets.values()], axis=0).tolist())
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
    ):
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

    def list_subjects(self) -> list:
        """
        Generate a list of subject ID for subjects associated to this project

        Returns
        --------
        List
            List of subject IDs
        """
        return [s.subject_id for s in self.subjects]

    def list_experiments(self) -> list:
        """
        Lists experiments in project

        Returns
        -------
        List
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
            logger.error(f"Invalid subject ID {subject_id}, does not exist")
            raise MissingSubjectError(f"Invalid subject ID {subject_id}, does not exist")
        return Subject.objects(subject_id=subject_id).get()

    def delete_experiment(self, experiment_id: str):
        """
        Delete experiment

        Parameters
        ----------
        experiment_id: str

        Returns
        -------
        None
        """
        if experiment_id not in self.list_experiments():
            raise MissingExperimentError(f"No such experiment {experiment_id}")
        exp = self.get_experiment(experiment_id)
        exp.delete()

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
    project: Project, experiment_left: Experiment, experiment_right: Experiment, new_experiment_id: str
) -> Experiment:
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
