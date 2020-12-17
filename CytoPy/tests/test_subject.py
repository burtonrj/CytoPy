from CytoPy.data.fcs import FileGroup
from CytoPy.data import subject
import pytest


@pytest.fixture()
def create_subject(example_populated_experiment):
    exp = example_populated_experiment
    fg = exp.get_sample("test sample")
    s = subject.Subject(subject_id="test subject",
                        files=[fg])
    s.infection_data = [subject.Bug(org_name="staphylococcus aureus",
                                    short_name="STAAUR",
                                    gram_status="P+ve",
                                    organism_type="bacteria")]
    s.save()


@pytest.fixture()
def create_subject_bugs():
    s = subject.Subject(subject_id="test subject 0")
    s.save()

    s = subject.Subject(subject_id="test subject 1")
    s.infection_data = [subject.Bug(org_name="staphylococcus aureus",
                                    short_name="STAAUR",
                                    gram_status="P+ve",
                                    organism_type="bacteria")]
    s.save()

    s = subject.Subject(subject_id="test subject 2")
    s.infection_data = [subject.Bug(org_name="Staphylococcus aureus",
                                    short_name="STAAUR",
                                    gram_status="P+ve",
                                    organism_type="bacteria"),
                        subject.Bug(org_name="Escherichia coli",
                                    short_name="ESCCOL",
                                    gram_status="N-ve",
                                    organism_type="bacteria")]
    s.save()

    s = subject.Subject(subject_id="test subject 3")
    s.infection_data = [subject.Bug(org_name="Candida albicans",
                                    short_name="CANALB",
                                    organism_type="fungi")]
    s.save()


def test_delete_subject(create_subject):
    s = subject.Subject.objects(subject_id="test subject").get()
    s.delete()
    assert len(subject.Subject.objects(subject_id="test subject")) == 0
    assert len(FileGroup.objects(primary_id="test sample")) == 0


def test_gram_status(create_subject_bugs):
    assert subject.gram_status(subject.Subject.objects(subject_id="test subject 0").get()) == "Unknown"
    assert subject.gram_status(subject.Subject.objects(subject_id="test subject 1").get()) == "P+ve"
    assert subject.gram_status(subject.Subject.objects(subject_id="test subject 2").get()) == "Mixed"
    assert subject.gram_status(subject.Subject.objects(subject_id="test subject 3").get()) == "Unknown"


def test_get_bugs(create_subject_bugs):
    assert subject.get_bugs(subject.Subject.objects(subject_id="test subject 0").get(),
                            multi_org="mixed",
                            short_name=True) == "Unknown"
    assert subject.get_bugs(subject.Subject.objects(subject_id="test subject 1").get(),
                            multi_org="mixed",
                            short_name=True) == "STAAUR"
    assert subject.get_bugs(subject.Subject.objects(subject_id="test subject 2").get(),
                            multi_org="mixed",
                            short_name=True) == "Mixed"
    assert subject.get_bugs(subject.Subject.objects(subject_id="test subject 3").get(),
                            multi_org="mixed",
                            short_name=True) == "CANALB"