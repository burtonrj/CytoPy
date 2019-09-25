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
        x_feature - the name of the X dimension
        y_feature - the name of the Y dimension (optional)
        func - the function used to generate this gate
        func_args - list of key value pairs (tuple; (key, value)) forming the kwargs for func
        gate_type - either 'geom' or 'cluster'; does this gate produce geometric object that defines the gate, or
        does this gate 'cluster' the parent population into child populations
    """
    gate_name = mongoengine.StringField(required=True)
    children = mongoengine.ListField()
    parent = mongoengine.StringField(required=True)
    x = mongoengine.StringField(required=True)
    y = mongoengine.StringField(required=False)
    func = mongoengine.StringField(required=True)
    func_args = mongoengine.ListField(required=True)
    gate_type = mongoengine.StringField(required=True, choices=['geom', 'cluster', 'threshold'])
    boolean_gate = mongoengine.BooleanField(default=False)
    meta = {
        'abstract': True
    }


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
