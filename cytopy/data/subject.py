from .fcs import FileGroup
import mongoengine
import numpy as np


class MetaDataDictionary(mongoengine.Document):
    """
    Model for a custom dictionary that can be used for given descriptions to meta-data. Helpful when exploring single cell data that has been
    associated to meta-data in the Explorer object; see flow.clustering.main.Explorer)

    Attributes:
        key - name of meta-data (column name)
        desc - string value of writen description
    """
    key = mongoengine.StringField()
    desc = mongoengine.StringField()


class Drug(mongoengine.EmbeddedDocument):
    """
    Embedded document -> Patient
    Document representation of drug administration. Single document instance represents one event.

    Attributes:
        name - name of therapy/drug
        init_date - date that therapy/drug started
        end_data - date that therapy/drug ended
    """
    name = mongoengine.StringField(required=True)
    init_date = mongoengine.DateTimeField(required=False)
    end_date = mongoengine.DateTimeField(required=False)


class Bug(mongoengine.EmbeddedDocument):
    """
    Embedded document -> Patient
    Document representation of isolated pathogen. Single document instance represents one pathogen.

    Attributes:
        gram_status - value of organisms gram status, valid choices are  ['P+ve', 'N-ve', 'Unknown']
        hmbpp_status - value of hmbpp status, valid choices are  ['P+ve', 'N-ve', 'Unknown']
        ribo_status - value of organisms ribo status, valid choices are  ['P+ve', 'N-ve', 'Unknown']
        org_name - name of the organism
        id_method - method used to identify organism
        culture_source - site of isolated organism
        organism_type - type of organism isolated, valid choices are ['bacteria', 'fungi', 'virus']
        report_date - date that organism was reported
        notes - string value for free text notes
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
    Embedded document -> Patient
    Document representation of biological test (blood pathology). Single document instance represents one test.

    Attributes:
        test_date - date that test was performed
        test - name of pathology test
        result - value of pathology test
        unit - units reported
        ref_range - reported reference range
        test_category - category of test
    """
    test_date = mongoengine.DateTimeField()
    test = mongoengine.StringField()
    result = mongoengine.FloatField()
    unit = mongoengine.StringField()
    ref_range = mongoengine.StringField()
    test_category = mongoengine.StringField()


class Subject(mongoengine.DynamicDocument):
    """
    Document based representation of patient meta-data
    """
    subject_id = mongoengine.StringField(required=True, unique=True)
    age = mongoengine.IntField(required=False)
    dob = mongoengine.DateField(required=False)
    gender = mongoengine.IntField(required=False) # 1 = Female, 0 = Male
    date_first_symptoms = mongoengine.DateTimeField(required=False)

    # Associated FCS Files
    files = mongoengine.ListField(mongoengine.ReferenceField(FileGroup, reverse_delete_rule=mongoengine.PULL))

    # Embeddings
    drug_data = mongoengine.EmbeddedDocumentListField(Drug)
    infection_data = mongoengine.EmbeddedDocumentListField(Bug)
    patient_biology = mongoengine.EmbeddedDocumentListField(Biology)

    # Admission
    admission_date_hosp = mongoengine.DateTimeField(required=False)
    admission_date_icu = mongoengine.DateTimeField(required=False)

    # Commodities
    comorbidity_notes = mongoengine.StringField(required=False)
    obesity = mongoengine.BooleanField(required=False)
    hypotension = mongoengine.BooleanField(required=False)
    myocardial_infarction = mongoengine.BooleanField(required=False)
    copd = mongoengine.BooleanField(required=False)
    diabetic = mongoengine.BooleanField(required=False)
    neurological_disorder = mongoengine.BooleanField(required=False)
    trauma = mongoengine.BooleanField(required=False)
    group_classification = mongoengine.StringField(required=False)
    chronic_kidney_disease = mongoengine.BooleanField(required=False)
    chronic_resp_disease = mongoengine.BooleanField(required=False)
    chronic_liver_disease = mongoengine.BooleanField(required=False)
    chronic_cardio_disease = mongoengine.BooleanField(required=False)
    malignancy = mongoengine.BooleanField(required=False)

    # Infection
    infection_source = mongoengine.StringField(require=False)
    infection_community_acquired = mongoengine.BooleanField(required=False)
    infection_notes = mongoengine.StringField(required=False)

    # Scores
    sofa = mongoengine.FloatField(required=False)
    apache_2 = mongoengine.FloatField(required=False)
    icnarc = mongoengine.FloatField(required=False)

    # Mechanical interventions
    mechanical_ventilation = mongoengine.BooleanField(required=False)
    mechanical_vent_days = mongoengine.FloatField(required=False)
    vaso_days = mongoengine.FloatField(required=False)
    dialysis_data = mongoengine.FloatField(required=False)

    # Notes
    notes = mongoengine.StringField(required=False)

    # Resp
    resp_inefficiency = mongoengine.BooleanField(required=False)
    oxygen_requirement = mongoengine.BooleanField(required=False)
    ards_criteria = mongoengine.StringField(required=False)

    # Other
    hemodynamic_impairment = mongoengine.BooleanField(required=False)
    vasopressor_drugs = mongoengine.BooleanField(required=False)
    renal_impairment = mongoengine.BooleanField(required=False)
    extra_renal_therapy = mongoengine.BooleanField(required=False)
    disseminated_iv_coag = mongoengine.BooleanField(required=False)

    # Outcome
    death_within_90days = mongoengine.BooleanField(required=False)
    death_date = mongoengine.DateTimeField(required=False)
    discharge_date_icu = mongoengine.DateTimeField(required=False)
    discharge_date_hosp = mongoengine.DateTimeField(required=False)
    outcome_category = mongoengine.StringField(required=False)

    meta = {
        'db_alias': 'core',
        'collection': 'patients'
    }


def gram_status(subject: Subject) -> str:
    """
    Given an instance of Subject, return the gram status of isolated organisms.
    Where multiple organisms are found, if gram status differs amongst orgs, returns 'mixed'
    :param subject: Subject document
    :return: String value for gram status
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
    :param short_name: If True, the shortened name rather than whole latin name is returned
    :param subject: Patient model object
    :param multi_org: If 'multi_org' equals 'list' then multiple organisms will be stored as a comma separated list
    without duplicates, whereas if the value is 'mixed' then multiple organisms will result in a value of 'mixed'.
    :return: string of isolated organisms comma separated, or 'mixed' if multi_org == 'mixed' and multiple organisms
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
    :param subject: Patient model object
    :return: common organism type isolated for patient
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
    :param subject: Patient model object
    :param field: field name to search for; expecting either 'hmbpp_status' or 'ribo_status'
    :return: common value of hmbpp_status/ribo_status
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
    :param subject_id: patient identifier
    :param test_name: name of test to search for
    :param method: summary statistic to use
    :return: Summary statistic (numpy float) or None if test does not exist
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

