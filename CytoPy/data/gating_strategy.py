from ..feedback import progress_bar, vprint
from .experiment import Experiment
from .fcs import FileGroup
from .gate import Gate
from datetime import datetime
import mongoengine


class Action(mongoengine.EmbeddedDocument):
    action_name = mongoengine.StringField()
    method = mongoengine.StringField(choices=["merge", "subtract"])
    left = mongoengine.StringField()
    right = mongoengine.StringField()
    new_population_name = mongoengine.StringField()


class GatingStrategy(mongoengine.Document):
    """
    Document representation of a gating template; a gating template is a collection of gating objects
    that can be applied to multiple fcs files or an entire experiment in bulk

    Parameters
    -----------
    template_name: str, required
        unique identifier for template
    gates: EmbeddedDocumentList
        list of Gate documents; see Gate
    creation_date: DateTime
        date of creation
    last_edit: DateTime
        date of last edit
    flags: str, optional
        warnings associated to this gating template
    notes: str, optional
        free text comments
    """
    name = mongoengine.StringField(required=True, unique=True)
    gates = mongoengine.ListField(mongoengine.ReferenceField(Gate, reverse_delete_rule=mongoengine.PULL))
    actions = mongoengine.EmbeddedDocumentListField(Action)
    creation_date = mongoengine.DateTimeField(default=datetime.now)
    last_edit = mongoengine.DateTimeField(default=datetime.now)
    flags = mongoengine.StringField(required=False)
    notes = mongoengine.StringField(required=False)
    meta = {
        'db_alias': 'core',
        'collection': 'gating_strategy'
    }

    def __init__(self, *args, **values):
        self.verbose = values.pop("verbose", True)
        self.print = vprint(verbose=self.verbose)
        super().__init__(*args, **values)
        self._filegroup = None
        self._tree = None

    def load_data(self):
        pass

    def preview_gate(self):
        pass

    def apply_gate(self):
        pass

    def apply_all(self):
        pass

    def delete_gate(self):
        pass

    def edit_gate(self):
        pass

    def delete_population(self):
        pass

    def delete_action(self):
        pass

    def plot_gate(self):
        pass

    def plot_backgate(self):
        pass

    def plot_population(self):
        pass

    def print_population_tree(self):
        pass

    def population_stats(self):
        pass

    def estimate_ctrl_population(self):
        pass

    def merge_populations(self):
        pass

    def subtract_populations(self):
        pass

    def save(self, *args, **kwargs):
        for g in self.gates:
            g.save()
        super().save(*args, **kwargs)

    def delete(self,
               delete_gates: bool = True,
               remove_associations: bool = True,
               *args, **kwargs):
        """
        Delete gating strategy. If delete_gates is True, then associated Gate objects will
        also be deleted. If remove_associations is True, then populations generated from
        this gating strategy will also be deleted.

        Parameters
        ----------
        delete_gates
        remove_associations
        args
        kwargs

        Returns
        -------

        """
        if delete_gates:
            self.print("Deleting gates...")
            for g in self.gates:
                g.delete()
        if remove_associations:
            self.print("Deleting associated populations...")
            for f in progress_bar(FileGroup.objects(), verbose=self.verbose):
                if self.name in f.gating_strategy:
                    f.gating_strategy = [gs for gs in f.gating_strategy if gs != self.name]
                    f.populations = []
                    f.save()
        super().delete(*args, **kwargs)
        self.print(f"{self.name} successfully deleted.")
