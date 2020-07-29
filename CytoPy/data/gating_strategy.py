from .gates import Gate
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
    template_name = mongoengine.StringField(required=True, unique=True)
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

    def save(self, *args, **kwargs):
        for g in self.gates:
            g.save()
        super().save(*args, **kwargs)
