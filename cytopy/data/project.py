import mongoengine
import datetime
from .subject import Subject
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
        add_subject - create a new subject and associated to project
        list_subjects - generate a list of subject ID for all subjects associated to project
        pull_subject -given a subject ID, pull the subject document for corresponding subject
        delete - delete project (wraps parent call to delete, see mongoengine.Document.delete)
    """
    project_id = mongoengine.StringField(required=True, unique=True)
    subjects = mongoengine.ListField(mongoengine.ReferenceField(Subject, reverse_delete_rule=4))
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

    def add_subject(self, subject_id: str,
                    drug_data: list or None = None,
                    infection_data: list or None = None,
                    patient_biology: list or None = None,
                    **kwargs) -> None:
        """
        Create a new subject and associated to project; a subject is an individual element of a study
        e.g. a patient or a mouse
        :param subject_id: subject ID for the new subject
        :param drug_data: list of Drug documents to associated to subject (see cytopy.data.subject.Drug)
        :param infection_data: list of Bug documents to associated to subject (see cytopy.data.subject.Bug)
        :param patient_biology: list of Biology documents to associated to subject (see cytopy.data.subject.Biology)
        :param kwargs: addiitonal keyword arguments to pass to Subject initialisation (see cytopy.data.subject.Subject)
        :return: None
        """
        new_subject = Subject(subject_id=subject_id, **kwargs)
        if drug_data is not None:
            new_subject.drug_data = drug_data
        if infection_data is not None:
            new_subject.infection_data = infection_data
        if patient_biology is not None:
            new_subject.patient_biology = patient_biology
        new_subject.save()
        self.subjects.append(new_subject)
        self.save()

    def list_subjects(self) -> list:
        """
        Generate a list of subject ID for subjects associated to this project
        :return: List of subject IDs
        """
        return [p.subject_id for p in self.subjects]

    def pull_subject(self, subject_id: str) -> Subject:
        """
        Given a subject ID associated to Project, return the Subject document
        :param subject_id: subject ID to pull
        :return: Subject document
        """
        assert subject_id in self.list_subjects(), f'Invalid subject ID, valid subjects: {self.list_subjects()}'
        return [p for p in self.subjects if p.subject_id == subject_id][0]

    def delete(self, *args, **kwargs) -> None:
        """
        Delete project (wrapper function of mongoengine.Document.delete)
        :param args: positional arguments to pass to parent call (see mongoengine.Document.delete)
        :param kwargs: keyword arguments to pass to parent call (see mongoengine.Document.delete)
        :return: None
        """
        experiments = [self.load_experiment(e) for e in self.list_fcs_experiments()]
        for e in experiments:
            samples = e.list_samples()
            for s in samples:
                e.remove_sample(s)
            e.delete()
        for p in self.subjects:
            p.delete()
        super().delete(*args, **kwargs)
