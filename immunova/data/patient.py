from immunova.data.fcs import FileGroup
import mongoengine


class Drug(mongoengine.EmbeddedDocument):
    """
    Embedded document -> Patient
    Document representation of drug administration. Single document instance represents one event.
    """
    name = mongoengine.StringField(required=True)
    init_date = mongoengine.DateTimeField(required=False)
    end_date = mongoengine.DateTimeField(required=False)


class Bug(mongoengine.EmbeddedDocument):
    """
    Embedded document -> Patient
    Document representation of isolated pathogen. Single document instance represents one pathogen.
    """
    gram_status = mongoengine.StringField(required=False)
    hmbpp_status = mongoengine.BooleanField(required=False)
    ribo_status = mongoengine.BooleanField(required=False)
    org_name = mongoengine.StringField(required=False)
    id_method = mongoengine.StringField(required=False)
    culture_source = mongoengine.StringField(required=False)
    organism_type = mongoengine.StringField(required=False, choices=['bacteria', 'fungi', 'virus'])
    report_date = mongoengine.DateTimeField(required=False)
    notes = mongoengine.StringField(required=False)


class Biology(mongoengine.EmbeddedDocument):
    """
    Embedded document -> Patient
    Document representation of biological test (blood pathology). Single document instance represents one test.
    """
    test_date = mongoengine.DateTimeField()
    test = mongoengine.StringField()
    result = mongoengine.FloatField()
    unit = mongoengine.StringField()
    ref_range = mongoengine.StringField()
    test_category = mongoengine.StringField()


class Patient(mongoengine.DynamicDocument):
    """
    Document based representation of patient meta-data
    """
    patient_id = mongoengine.StringField(required=True, unique=True)
    age = mongoengine.IntField(required=False)
    dob = mongoengine.DateField(required=False)
    gender = mongoengine.IntField(required=False)
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


