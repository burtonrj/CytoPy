#!/usr/bin.env/python
# -*- coding: utf-8 -*-
"""
Each subject in your analysis (human, mouse, cell line etc) can be
represented by a Subject document that can then be associated to
specimens in an Experiment. This Subject document is dynamic and can
house any relating meta-data.

Projects also house the subjects (represented by the Subject class;
see CytoPy.data.subject) of an analysis which can contain multiple
meta-data.

Copyright 2020 Ross Burton

Permission is hereby granted, free of charge, to any person
obtaining a copy of this software and associated documentation
files (the "Software"), to deal in the Software without restriction,
including without limitation the rights to use, copy, modify,
merge, publish, distribute, sublicense, and/or sell copies of the
Software, and to permit persons to whom the Software is furnished
to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included
in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""
from .fcs import FileGroup
import mongoengine
import numpy as np

__author__ = "Ross Burton"
__copyright__ = "Copyright 2020, CytoPy"
__credits__ = ["Ross Burton", "Simone Cuff", "Andreas Artemiou", "Matthias Eberl"]
__license__ = "MIT"
__version__ = "1.0.0"
__maintainer__ = "Ross Burton"
__email__ = "burtonrj@cardiff.ac.uk"
__status__ = "Production"


class MetaDataDictionary(mongoengine.Document):
    """
    Model for a custom dictionary that can be used for given descriptions to meta-data.
    Helpful when exploring single cell data that has been associated to meta-data in the Explorer object;
    see flow.clustering.main.Explorer)

    Attributes
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

    Attributes
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
    dose = mongoengine.StringField(required=False)
    notes = mongoengine.StringField(required=False)


class Bug(mongoengine.EmbeddedDocument):
    """
    Document representation of isolated pathogen. Single document instance represents one pathogen.

    Attributes
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
    growth_weight = mongoengine.StringField(required=False)
    notes = mongoengine.StringField(required=False)


class Biology(mongoengine.EmbeddedDocument):
    """
    Document representation of biological test (blood pathology). Single document instance represents one test.

    Attributes
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
    test_datetime = mongoengine.DateTimeField(required=False)
    test = mongoengine.StringField(required=False)
    result = mongoengine.FloatField(required=False)
    unit = mongoengine.StringField(required=False)
    ref_range = mongoengine.StringField(required=False)
    test_category = mongoengine.StringField(required=False)
    notes = mongoengine.StringField(required=False)


class Subject(mongoengine.DynamicDocument):
    """
    Document based representation of subject meta-data.
    Subjects are stored in a dynamic document, meaning
    new properties can be added ad-hoc.

    Attributes
    -----------
    subject_id: str, required
        Unique identifier for subject
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
        'collection': 'subjects'
    }

    def delete(self, *args, **kwargs):
        """
        Delete the Subject. The subject will automatically be pulled from associated Projects (reference field in
        Project model has reverse_delete_rile=4; see mongoengine API for info).

        WARNING: deletion of a subject will result in the automatic removal of all associated FCS data!

        Parameters
        ----------
        signal_kwargs: optional
            kwargs dictionary to be passed to the signal calls.
        write_concern
            Extra keyword arguments are passed down which will be used as options for the resultant getLastError command.
            For example, save(..., w: 2, fsync: True) will wait until at least two servers have recorded the write and
            will force an fsync on the primary server.

        Returns
        -------
        None
        """
        for f in self.files:
            f.delete()
        super().delete(*args, **kwargs)


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

