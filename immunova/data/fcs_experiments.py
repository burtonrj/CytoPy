import mongoengine
from datetime import datetime
from data.fcs import FileGroup
from data.gating import GatingStrategy
import re


class ChannelMap(mongoengine.EmbeddedDocument):
    """
    Embedded document -> Panel

    Defines channel/marker mapping. Each document will contain a single value for channel and a single value for marker,
    these two values are treated as a pair within the panel.
    """
    channel = mongoengine.StringField()
    marker = mongoengine.StringField()


class NormalisedName(mongoengine.EmbeddedDocument):
    """
    Embedded document -> Panel

    Defines a standardised name for a channel or marker and provides method for testing if a channel/marker
    should be associated to standard

    Attributes:
        standard - the "standard" name i.e. the nomenclature we used for a channel/marker in this panel
        regex_str - regular expression used to test if a term corresponds to this standard
        permutations - list of string values that have direct association to this standard
        case_sensitive - is the nomenclature case sensitive? This would be false for something like 'CD3' for example,
        where 'cd3' and 'CD3' are synonymous
    Methods:
         query - given a term 'x', determine if 'x' is synonymous to this standard. If so, return the standardised name.
    """
    standard = mongoengine.StringField(required=True)
    regex_str = mongoengine.StringField()
    permutations = mongoengine.StringField()
    case_sensitive = mongoengine.BooleanField(default=False)

    def query(self, x: str) -> None or str:
        """
        Given a term 'x', determine if 'x' is synonymous to this standard. If so, return the standardised name.
        :param x: search term
        :return: Standardised name if synonymous to standard, else None
        """
        if self.case_sensitive:
            if re.search(self.regex_str, x):
                return self.standard
        if re.search(self.regex_str, x, re.IGNORECASE):
            return self.standard
        for p in self.permutations.split(','):
            if x == p:
                return self.standard
        return None


class Panel(mongoengine.Document):
    """
    Embedded document -> FCSExperiment
    Document representation of channel/marker definition for an experiment. A panel, once associated to an experiment
    will standardise data upon input; when an fcs file is created in the database, it will be associated to
    an experiment and the channel/marker definitions in the fcs file will be mapped to the associated panel.

    Attributes:
        panel_name - unique identifier for the panel
        markers - list of marker names; see NormalisedName
        channels - list of channels; see NormalisedName
        mappings - list of channel/marker mappings; see ChannelMap
        initiation_date - date of creation
    """
    panel_name = mongoengine.StringField(required=True, unique=True)
    markers = mongoengine.EmbeddedDocumentListField(NormalisedName)
    channels = mongoengine.EmbeddedDocumentListField(NormalisedName)
    mappings = mongoengine.EmbeddedDocumentListField(ChannelMap)
    initiation_date = mongoengine.DateTimeField(default=datetime.now)
    meta = {
        'db_alias': 'core',
        'collection': 'fcs_panels'
    }


class FCSExperiment(mongoengine.Document):
    """
    Document representation of Flow Cytometry experiment

    Attributes:
        experiment_id - unique identifier for experiment
        panel - Panel object describing associated channel/marker pairs
        fcs_files - reference field for associated files
        flags - warnings associated to experiment
        notes - additional free text comments
        gating_templates - reference to gating templates associated to this experiment
    """
    experiment_id = mongoengine.StringField(required=True, unique=True)
    panel = mongoengine.ReferenceField(Panel)
    fcs_files = mongoengine.ListField(mongoengine.ReferenceField(FileGroup))
    flags = mongoengine.StringField(required=False)
    notes = mongoengine.StringField(required=False)
    gating_templates = mongoengine.ListField(mongoengine.ReferenceField(GatingStrategy))

    meta = {
        'db_alias': 'core',
        'collection': 'fcs_experiments'
    }
