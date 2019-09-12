import mongoengine
import datetime
from data.patient import Patient
from data.fcs_experiments import FCSExperiment, Panel


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
        list_fcs_experiments - generate a list of IDs for fcs experiments associated to this project
    """
    project_id = mongoengine.StringField(required=True, unique=True)
    patients = mongoengine.ListField(mongoengine.ReferenceField(Patient))
    start_date = mongoengine.DateTimeField(default=datetime.datetime.now)
    owner = mongoengine.StringField(requred=True)
    fcs_experiments = mongoengine.ListField(mongoengine.ReferenceField(FCSExperiment))

    meta = {
        'db_alias': 'core',
        'collection': 'projects'
    }

    def list_fcs_experiments(self):
        """
        Generate a list of associated flow cytometry experiments
        :return: list of experiment IDs
        """
        experiments = [e.experiment_id for e in self.fcs_experiments]
        return experiments

    def add_experiment(self, experiment_id, panel_name):
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
        return exp.id.__str__()
