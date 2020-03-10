import mongoengine
import datetime
from .subject import Subject
from .fcs_experiments import FCSExperiment, Panel


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

        Returns
        -------
        list
            list of experiment IDs
        """
        experiments = [e.experiment_id for e in self.fcs_experiments]
        return experiments

    def load_experiment(self, experiment_id: str) -> None or FCSExperiment:
        """
        For a given experiment in project, load the experiment object

        Parameters
        ----------
        experiment_id: str
            experiment to load

        Returns
        --------
        FCSExperiment or None
        """
        if experiment_id not in self.list_fcs_experiments():
            print(f'Error: no experiment {experiment_id} found')
            return None
        e = [e for e in self.fcs_experiments if e.experiment_id == experiment_id][0]
        return e

    def add_experiment(self, experiment_id: str, panel_name: str) -> None or FCSExperiment:
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
        FCSExperiment or None
            Newly created FCSExperiment
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

        Parameters
        -----------
        subject_id: str
            subject ID for the new subject
        drug_data: list, optional
            list of Drug documents to associated to subject (see cytopy.data.subject.Drug)
        infection_data: list, optional
            list of Bug documents to associated to subject (see cytopy.data.subject.Bug)
        patient_biology: list, optional
            list of Biology documents to associated to subject (see cytopy.data.subject.Biology)
        kwargs:
            Additional keyword arguments to pass to Subject initialisation (see cytopy.data.subject.Subject)

        Returns
        --------
        None
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

        Returns
        --------
        list
            List of subject IDs
        """
        return [p.subject_id for p in self.subjects]

    def pull_subject(self, subject_id: str) -> Subject:
        """
        Given a subject ID associated to Project, return the Subject document

        Parameters
        -----------
        subject_id: str
            subject ID to pull

        Returns
        --------
        Subject
        """
        assert subject_id in self.list_subjects(), f'Invalid subject ID, valid subjects: {self.list_subjects()}'
        return [p for p in self.subjects if p.subject_id == subject_id][0]

    def delete(self, *args, **kwargs) -> None:
        """
        Delete project (wrapper function of mongoengine.Document.delete)

        Parameters
        -----------
        args:
            positional arguments to pass to parent call (see mongoengine.Document.delete)
        kwargs:
            keyword arguments to pass to parent call (see mongoengine.Document.delete)

        Returns
        --------
        None
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
