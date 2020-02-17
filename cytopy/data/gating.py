import mongoengine
import datetime


class Gate(mongoengine.EmbeddedDocument):
    """
    Embedded document -> GatingStrategy
    Ref -> FCSExperiment

    Document representation of a Gate

    Attributes:
        gate_name - unique identifier for this gate (within the scope of a GatingStrategy)
        children - list of population names; populations derived from application of this gate
        parent - name of parent population; the population this gate acts upon
        class_ - name of gating class used to generate gate (see flow.gating.actions)
        method - name of class method used to generate gate (see flow.gating.actions)
        kwargs - list of keyword arguments (list of tuples; first element = key, second element = value) passed to
        class/method to generate gate
    Methods:
        to_python - convert gate document to a Python dictionary
    """
    gate_name = mongoengine.StringField(required=True)
    children = mongoengine.ListField()
    parent = mongoengine.StringField(required=True)
    class_ = mongoengine.StringField(required=True)
    method = mongoengine.StringField(required=True)
    kwargs = mongoengine.ListField(required=True)
    meta = {
        'abstract': True
    }

    def to_python(self) -> dict:
        """
        Convert document to Python dictionary object
        :return: Dictionary representation of document
        """
        return dict(gate_name=self.gate_name, children=self.children,
                    class_=self.class_, method=self.method, kwargs=self.kwargs)


class GatingStrategy(mongoengine.Document):
    """
    Document representation of a gating template; a gating template is a collection of gating objects
    that can be applied to multiple fcs files or an entire experiment in bulk

    Attributes:
        template_name - unique identifier for template
        gates - list of Gate documents; see Gate
        creation_date - date of creation
        last_edit - date of last edit
        flags - warnings associated to this gating template
        notes - free text comments
    """
    template_name = mongoengine.StringField(required=True)
    gates = mongoengine.EmbeddedDocumentListField(Gate)
    creation_date = mongoengine.DateTimeField(default=datetime.datetime.now)
    last_edit = mongoengine.DateTimeField(default=datetime.datetime.now)
    flags = mongoengine.StringField(required=False)
    notes = mongoengine.StringField(required=False)
    meta = {
        'db_alias': 'core',
        'collection': 'gating_strategy'
    }
