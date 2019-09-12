import mongoengine
import datetime
from data.patient import Patient
from data.fcs_experiments import FCSExperiment


class Project(mongoengine.Document):
    """
    Document representation of Project

    Attributes:
        project_id - unique identifier for project
        patients - reference field for associated patients; see Patient
        start_date - date of creation
        owner - user name of owner
        fcs_experiments - reference field for associated fcs files
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

    def add_experiment(self):