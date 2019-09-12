import mongoengine


class Drug(mongoengine.EmbeddedDocument):
    name = mongoengine.StringField(required=True)
    init_date = mongoengine.DateTimeField(required=False)
    end_date = mongoengine.DateTimeField(required=False)


class Bug(mongoengine.EmbeddedDocument):
    gram_status = mongoengine.StringField(required=False)
    org_name = mongoengine.StringField(required=False)
    id_method = mongoengine.StringField(required=False)
    culture_source = mongoengine.StringField(required=False)
    organism_type = mongoengine.StringField(required=False)
    report_date = mongoengine.DateTimeField(required=False)
    notes = mongoengine.StringField(required=False)


class Biology(mongoengine.EmbeddedDocument):
    test_date = mongoengine.DateTimeField()
    test = mongoengine.StringField()
    result = mongoengine.FloatField()
    unit = mongoengine.StringField()
    ref_range = mongoengine.StringField()
    test_category = mongoengine.StringField()


class Sample(mongoengine.EmbeddedDocument):
    sample_id = mongoengine.StringField(required=True)
    collection_datetime = mongoengine.DateTimeField(required=True)
    processing_datetime = mongoengine.DateTimeField(required=True)
    flags = mongoengine.StringField(required=False)
    fcs_files = mongoengine.ListField()


class Patient(mongoengine.Document):
    patient_id = mongoengine.StringField(required=True, unique=True)
    age = mongoengine.IntField(required=False)
    gender = mongoengine.IntField(required=False)
    date_first_symptoms = mongoengine.DateTimeField(required=False)
    admission_date_hosp = mongoengine.DateTimeField(required=False)
    admission_date_icu = mongoengine.DateTimeField(required=False)
    drug_data = mongoengine.EmbeddedDocumentListField(Drug)
    comorbidity_notes = mongoengine.StringField(required=False)
    obesity = mongoengine.BooleanField(required=False)
    hypotension = mongoengine.BooleanField(required=False)
    myocardial_infarction = mongoengine.BooleanField(required=False)
    copd = mongoengine.BooleanField(required=False)
    diabetic = mongoengine.BooleanField(required=False)
    neurological_disorder = mongoengine.BooleanField(required=False)
    trauma = mongoengine.BooleanField(required=False)
    group_classification = mongoengine.StringField(required=False)
    infection_source = mongoengine.StringField(require=False)
    infection_community_acquired = mongoengine.BooleanField(required=False)
    infection_notes = mongoengine.StringField(required=False)
    infection_data = mongoengine.EmbeddedDocumentListField(Bug)
    sofa = mongoengine.FloatField(required=False)
    apache_2 = mongoengine.FloatField(required=False)
    icnarc = mongoengine.FloatField(required=False)
    mechanical_vent_days = mongoengine.FloatField(required=False)
    vaso_days = mongoengine.FloatField(required=False)
    dialysis_data = mongoengine.FloatField(required=False)
    patient_biology = mongoengine.EmbeddedDocumentListField(Biology)
    notes = mongoengine.StringField(required=False)
    chronic_kidney_disease = mongoengine.BooleanField(required=False)
    chronic_resp_disease = mongoengine.BooleanField(required=False)
    chronic_liver_disease = mongoengine.BooleanField(required=False)
    chronic_cardio_disease = mongoengine.BooleanField(required=False)
    malignancy = mongoengine.BooleanField(required=False)
    other_comorbidity = mongoengine.ListField(required=False)
    resp_inefficiency = mongoengine.BooleanField(required=False)
    oxygen_requirement = mongoengine.BooleanField(required=False)
    mechanical_ventilation = mongoengine.BooleanField(required=False)
    ards_criteria = mongoengine.StringField(required=False)
    hemodynamic_impairment = mongoengine.BooleanField(required=False)
    vasopressor_drugs = mongoengine.BooleanField(required=False)
    renal_impairment = mongoengine.BooleanField(required=False)
    extra_renal_therapy = mongoengine.BooleanField(required=False)
    disseminated_iv_coag = mongoengine.BooleanField(required=False)
    death_within_90days = mongoengine.BooleanField(required=False)
    death_date = mongoengine.DateTimeField(required=False)
    discharge_date_icu = mongoengine.DateTimeField(required=False)
    discharge_date_hosp = mongoengine.DateTimeField(required=False)
    outcome_category = mongoengine.StringField(required=False)
    samples = mongoengine.EmbeddedDocumentField(Sample)

    meta = {
        'db_alias': 'core',
        'collection': 'patients'
    }



