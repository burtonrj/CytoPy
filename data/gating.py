import mongoengine
import datetime


class Gate(mongoengine.EmbeddedDocument):
    """
    Document representation of a Gate

    Parameters
    -----------
    gate_name: str, required
        Unique identifier for this gate (within the scope of a GatingStrategy)
    children: list
        list of population names; populations derived from application of this gate
    parent: str, required
        name of parent population; the population this gate acts upon
    class_: str, required
        name of gating class used to generate gate (see flow.gating.actions)
    method: str, required
        name of class method used to generate gate (see flow.gating.actions)
    kwargs: list
        list of keyword arguments (list of tuples; first element = key, second element = value) passed to
        class/method to generate gate
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

        Returns
        --------
        dict
            Dictionary representation of document
        """
        return dict(gate_name=self.gate_name, children=self.children,
                    class_=self.class_, method=self.method, kwargs=self.kwargs)


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
