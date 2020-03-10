from .fcs import FileGroup
import mongoengine
import numpy as np


class MetaDataDictionary(mongoengine.Document):
    """
    Model for a custom dictionary that can be used for given descriptions to meta-data.
    Helpful when exploring single cell data that has been associated to meta-data in the Explorer object;
    see flow.clustering.main.Explorer)

    Parameters
    -----------
    key: str
        name of meta-data (column name)
    desc: str
        string value of writen description
    """
    key = mongoengine.StringField()
    desc = mongoengine.StringField()


class Drug(mongoengine.EmbeddedDocument):
    """
    Document representation of drug administration. Single document instance represents one event.

    Parameters
    -----------
    name: str
        name of therapy/drug
    init_date: DateTime
        date that therapy/drug started
    end_data: DateTime
        date that therapy/drug ended
    """
    name = mongoengine.StringField(required=True)
    init_date = mongoengine.DateTimeField(required=False)
    end_date = mongoengine.DateTimeField(required=False)


class Bug(mongoengine.EmbeddedDocument):
    """
    Document representation of isolated pathogen. Single document instance represents one pathogen.

    Parameters
    -----------
    gram_status: str, optional
        value of organisms gram status, valid choices are  ['P+ve', 'N-ve', 'Unknown']
    hmbpp_status: str, optional
        value of hmbpp status, valid choices are  ['P+ve', 'N-ve', 'Unknown']
    ribo_status: str, optional
        value of organisms ribo status, valid choices are  ['P+ve', 'N-ve', 'Unknown']
    org_name: str
        name of the organism
    id_method: str, optional
        method used to identify organism
    culture_source: str, optional
        site of isolated organism
    organism_type: str, optional
        type of organism isolated, valid choices are ['bacteria', 'fungi', 'virus']
    report_date: DateTime, optional
        date that organism was reported
    notes: str, optional
        string value for free text notes
    """
    gram_status = mongoengine.StringField(required=False, choices=['P+ve', 'N-ve', 'Unknown'])
    hmbpp_status = mongoengine.StringField(required=False, choices=['P+ve', 'N-ve', 'Unknown'])
    ribo_status = mongoengine.StringField(required=False, choices=['P+ve', 'N-ve', 'Unknown'])
    org_name = mongoengine.StringField(required=False)
    short_name = mongoengine.StringField(required=False)
    id_method = mongoengine.StringField(required=False)
    culture_source = mongoengine.StringField(required=False)
    organism_type = mongoengine.StringField(required=False, choices=['bacteria', 'fungi', 'virus'])
    report_date = mongoengine.DateTimeField(required=False)
    notes = mongoengine.StringField(required=False)


class Biology(mongoengine.EmbeddedDocument):
    """
    Document representation of biological test (blood pathology). Single document instance represents one test.

    Parameters
    -----------
    test_date: DateTime
        date that test was performed
    test: str
        name of pathology test
    result: float
        value of pathology test
    unit: str
        units reported
    ref_range: str
        reported reference range
    test_category: str
        category of test
    """
    test_date = mongoengine.DateTimeField()
    test = mongoengine.StringField()
    result = mongoengine.FloatField()
    unit = mongoengine.StringField()
    ref_range = mongoengine.StringField()
    test_category = mongoengine.StringField()


class Subject(mongoengine.DynamicDocument):
    """
    Document based representation of subject meta-data. Subjects are stored in a dynamic document, meaning
    new properties can be added ad-hoc.

    Parameters
    -----------
    subject_id: str, required
        Unique identifier for subject
    age: int
        Age of subject
    dob: Date
        Date of birth
    gender: int
        Gender of subject; 1 = Female, 0 = Male
    files: ListField
        List of references to files associated to subject
    drug_data: EmbeddedDocListField
        Associated drug data
    infection_data: EmbeddedDocListField
        Associated infection data
    patient_biology: EmbeddedDocListField
        Associated biological data
    notes: str
        Additional notes
    """
    subject_id = mongoengine.StringField(required=True, unique=True)
    age = mongoengine.IntField(required=False)
    dob = mongoengine.DateField(required=False)
    gender = mongoengine.IntField(required=False) # 1 = Female, 0 = Male

    # Associated FCS Files
    files = mongoengine.ListField(mongoengine.ReferenceField(FileGroup, reverse_delete_rule=mongoengine.PULL))

    # Embeddings
    drug_data = mongoengine.EmbeddedDocumentListField(Drug)
    infection_data = mongoengine.EmbeddedDocumentListField(Bug)
    patient_biology = mongoengine.EmbeddedDocumentListField(Biology)

    # Notes
    notes = mongoengine.StringField(required=False)

    meta = {
        'db_alias': 'core',
        'collection': 'patients'
    }


def gram_status(subject: Subject) -> str:
    """
    Given an instance of Subject, return the gram status of isolated organisms.
    Where multiple organisms are found, if gram status differs amongst orgs, returns 'mixed'

    Parameters
    ----------
    subject: Subject

    Returns
    --------
    str
        String value for gram status
    """
    if not subject.infection_data:
        return 'Unknown'
    orgs = [b.gram_status for b in subject.infection_data]
    if not orgs:
        return 'Unknown'
    if len(orgs) == 1:
        return orgs[0]
    return 'Mixed'


def bugs(subject: Subject, multi_org: str, short_name: bool = False) -> str:
    """
    Fetch the name of isolated organisms for each patient.

    Parameters
    -----------
    subject: Subject
    short_name: bool
        If True, the shortened name rather than whole latin name is returned
    multi_org: str
        If 'multi_org' equals 'list' then multiple organisms will be stored as a comma separated list
        without duplicates, whereas if the value is 'mixed' then multiple organisms will result in a value of 'mixed'.

    Returns
    --------
    str
        string of isolated organisms comma separated, or 'mixed' if multi_org == 'mixed' and multiple organisms
        listed for patient
    """
    if not subject.infection_data:
        return 'Unknown'
    if short_name:
        orgs = [b.short_name for b in subject.infection_data]
    else:
        orgs = [b.org_name for b in subject.infection_data]
    if not orgs:
        return 'Unknown'
    if len(orgs) == 1:
        return orgs[0]
    if multi_org == 'list':
        return ','.join(orgs)
    return 'mixed'


def org_type(subject: Subject) -> str:
    """
    Parse all infectious isolates for each patient and return the organism type isolated, one of either:
    'gram positive', 'gram negative', 'virus', 'mixed' or 'fungal'

    Parameters
    -----------
    subject: Subject

    Returns
    --------
    str
        common organism type isolated for patient
    """

    def bug_type(b: Bug):
        if not b.organism_type:
            return 'Unknown'
        if b.organism_type == 'bacteria':
            return b.gram_status
        return b.organism_type

    bugs = list(set(map(bug_type, subject.infection_data)))
    if len(bugs) == 0:
        return 'Unknown'
    if len(bugs) == 1:
        return bugs[0]
    return 'mixed'


def hmbpp_ribo(subject: Subject, field: str) -> str:
    """
    Given a value of either 'hmbpp' or 'ribo' for 'field' argument, return True if any Bug has a positive status
    for the given patient ID.

    Parameters
    -----------
    subject: Subject
    field: str
        field name to search for; expecting either 'hmbpp_status' or 'ribo_status'

    Returns
    --------
    str
        common value of hmbpp_status/ribo_status
    """
    if all([b[field] is None for b in subject.infection_data]):
        return 'Unknown'
    if all([b[field] == 'P+ve' for b in subject.infection_data]):
        return 'P+ve'
    if all([b[field] == 'N-ve' for b in subject.infection_data]):
        return 'N-ve'
    return 'mixed'


def biology(subject_id: str, test_name: str, method: str) -> np.float or None:
    """
    Given some test name, return a summary statistic of all results for a given patient ID

    Parameters
    -----------
    subject_id: str
        patient identifier
    test_name: str
        name of test to search for
    method: str
        summary statistic to use

    Returns
    --------
    Numpy.float or None
        Summary statistic (numpy float) or None if test does not exist
    """
    if subject_id is None:
        return None
    tests = Subject.objects(patient_id=subject_id).get().patient_biology
    tests = [t.result for t in tests if t.test == test_name]
    if not tests:
        return None
    if method == 'max':
        return np.max(tests)
    if method == 'min':
        return np.min(tests)
    if method == 'median':
        return np.median(tests)
    return np.average(tests)

