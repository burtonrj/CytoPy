from CytoPy2.data.subjects import Subject, Bug, Drug, Biology, gram_status, bugs, org_type, hmbpp_ribo, biology
from mongoengine import connect, disconnect
import unittest


class TextSubject(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        connect('testing', host='mongomock://localhost')

    @classmethod
    def tearDownClass(cls):
        disconnect()

    def test_gram_status(self):
        s = Subject(subject_id="test")
        self.assertEqual(gram_status(s), "Unknown")
        s.infection_date.append(Bug())
        self.assertEqual(gram_status(s), "Unknown")
        s.infection_date.append(Bug(gram_status='P+ve'))
        self.assertEqual(gram_status(s), 'P+ve')
        s.infection_date.append(Bug(gram_status='N-ve'))
        self.assertEqual(gram_status(s), 'Mixed')

    def test_bugs(self):
        s = Subject(subject_id="test")
        self.assertEqual(bugs(s), "Unknown")
        s.infection_date.append(Bug())
        self.assertEqual(bugs(s), "Unknown")
        s.infection_date.append(Bug(org_name="Staph Auerus", short_name="SAUR"))
        self.assertEqual(bugs(s), 'Staph Auerus')
        self.assertEqual(bugs(s, short_name=True), 'SAUR')
        s.infection_date.append(Bug(org_name="Staph Auerus", short_name="SAUR"))
        self.assertEqual(bugs(s), 'Staph Auerus')
        self.assertEqual(bugs(s, short_name=True), 'SAUR')
        s.infection_date.append(Bug(org_name="E.coli", short_name="ECOL"))
        self.assertEqual(bugs(s), 'Mixed')
        self.assertEqual(bugs(s, short_name=True), 'Mixed')
        self.assertEqual(bugs(s, short_name=True, multi_org=True), 'SAUR,ECOL')
        self.assertEqual(bugs(s, short_name=False, multi_org=True), 'Staph Auerus,E.coli')

    def test_org_type(self):
        s = Subject(subject_id="test")
        self.assertEqual(org_type(s), "Unknown")
        s.infection_date.append(Bug())
        self.assertEqual(org_type(s), "Unknown")
        s.infection_date.append(Bug(organism_type="Bacteria"))
        s.infection_date.append(Bug(organism_type="Bacteria"))
        self.assertEqual(org_type(s), "Bacteria")
        s.infection_date.append(Bug(organism_type="Virus"))
        self.assertEqual(org_type(s), "Mixed")
        s.infection_date.append(Bug(organism_type="Fungi"))
        self.assertEqual(org_type(s, multi_org=True), "Bacterua,Virus,Fungi")

    def biology(self):
        s = Subject(subject_id="test")
        self.assertIsNone(biology(s, "test"), "Unknown")
        s.append(Biology(name="test", result=5))
        s.append(Biology(name="test", result=10))
        s.append(Biology(name="test", result=5))
        self.assertEqual(Biology(s, "test", method="min"), 5)
        self.assertEqual(Biology(s, "test", method="max"), 10)
        self.assertEqual(Biology(s, "test", method="median"), 5)
        self.assertAlmostEqual(Biology(s, "test", method="average"), 6.666)

