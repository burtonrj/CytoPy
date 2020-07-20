from .experiments import Experiment
from .subjects import Subject
from typing import Generator
import mongoengine
import datetime
import os


class Project(mongoengine.Document):
    """
    Document representation of Project

    Parameters
    ----------
    project_id: str, required
        unique identifier for project
    subjects: ListField
        List of references for associated subjects; see Subject
    start_date: DateTime
        date of creation
    owner: str, required
        user name of owner
    fcs_experiments: ListField
        List of references for associated fcs files
    """
    project_id = mongoengine.StringField(required=True, unique=True)
    _data_directory = mongoengine.StringField(db_field="data_directory")
    subjects = mongoengine.ListField(mongoengine.ReferenceField(Subject, reverse_delete_rule=4))
    start_date = mongoengine.DateTimeField(default=datetime.datetime.now)
    owner = mongoengine.StringField(requred=True)
    experiments = mongoengine.ListField(mongoengine.ReferenceField(Experiment, reverse_delete_rule=4))

    meta = {
        'db_alias': 'core',
        'collection': 'projects'
    }

    def __init__(self, *args, **kwargs):
        if "definition" in kwargs.keys():
            self.data_directory = kwargs.pop("data_directory")
        super().__init__(*args, **kwargs)

    @property
    def data_directory(self):
        return self._data_directory

    @data_directory.setter
    def data_directory(self, value):
        assert os.path.isdir(value)
        self._data_directory = value

    def list_experiments(self) -> Generator:
        """
        Generate a list of associated flow cytometry experiments

        Returns
        -------
        Generator
            list of experiment IDs
        """
        for e in self.experiments:
            yield e.experiment_id

    def load_experiment(self, experiment_id: str) -> Experiment:
        """
        For a given experiment in project, load the experiment object

        Parameters
        ----------
        experiment_id: str
            experiment to load

        Returns
        --------
        Experiment
        """
        assert experiment_id in list(self.list_experiments()), f'Error: no experiment {experiment_id} found'
        return Experiment.objects(experiment_id=experiment_id).get()

    def add_experiment(self,
                       experiment_id: str,
                       panel_name: str or None = None,
                       panel_definition: str or None = None) -> Experiment:
        """
        Add new experiment to project

        Parameters
        -----------
        experiment_id: str
            experiment name
        panel_name: str
            panel to associate to experiment

        Returns
        --------
        Experiment
            Newly created FCSExperiment
        """
        err = f'Error: Experiment with id {experiment_id} already exists!'
        assert experiment_id not in list(self.list_experiments()), err
        # CREATE EXPERIMENT
        exp = FCSExperiment(experiment_id=experiment_id,
                            panel_definition=panel_definition,
                            panel_name=panel_name)
        exp.save()
        self.experiments.append(exp)
        print(f'Experiment created successfully!')
        self.save()
        return exp