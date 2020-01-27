import mongoengine
import datetime
from .patient import Patient
from .fcs_experiments import FCSExperiment, Panel


class Project(mongoengine.Document):
    """
    Document representation of Project

    Attributes:
        project_id - unique identifier for project
        patients - reference field for associated patients; see Patient
        start_date - date of creation
        owner - user name of owner
        fcs_experiments - reference field for associated fcs files
    Methods:
        add_experiment - create new experiment and associate to project
        load_experiment - For a given experiment in project, return the experiment object
        list_fcs_experiments - generate a list of IDs for fcs experiments associated to this project
    """
    project_id = mongoengine.StringField(required=True, unique=True)
    patients = mongoengine.ListField(mongoengine.ReferenceField(Patient, reverse_delete_rule=4))
    start_date = mongoengine.DateTimeField(default=datetime.datetime.now)
    owner = mongoengine.StringField(requred=True)
    fcs_experiments = mongoengine.ListField(mongoengine.ReferenceField(FCSExperiment, reverse_delete_rule=4))

    meta = {
        'db_alias': 'core',
        'collection': 'projects'
    }

    def list_fcs_experiments(self) -> list:
        """
        Generate a list of associated flow cytometry experiments
        :return: list of experiment IDs
        """
        experiments = [e.experiment_id for e in self.fcs_experiments]
        return experiments

    def load_experiment(self, experiment_id: str) -> None or FCSExperiment:
        """
        For a given experiment in project, load the experiment object
        :param experiment_id: experiment to load
        :return: FCSExperiment object
        """
        if experiment_id not in self.list_fcs_experiments():
            print(f'Error: no experiment {experiment_id} found')
            return None
        e = [e for e in self.fcs_experiments if e.experiment_id == experiment_id][0]
        return e

    def add_experiment(self, experiment_id: str, panel_name: str) -> None or FCSExperiment:
        """
        Add new experiment to project
        :param experiment_id: experiment name
        :param panel_name: panel to associate to experiment
        :return: MongoDB document ID of newly created experiment
        """
        if FCSExperiment.objects(experiment_id=experiment_id):
            print(f'Error: Experiment with id {experiment_id} already exists!')
            return None
        panel = Panel.objects(panel_name=panel_name)
        if not panel:
            print(f'Error: Panel {panel_name} does not exist')
            return None
        exp = FCSExperiment()
        exp.experiment_id = experiment_id
        exp.panel = panel[0]
        exp.save()
        self.fcs_experiments.append(exp)
        print(f'Experiment created successfully!')
        self.save()
        return exp

