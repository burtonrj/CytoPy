from CytoPy2.data.project import Project
from CytoPy2.data.experiments import Experiment
from CytoPy2.data.subjects import Subject
from mongoengine import connect, disconnect
import unittest


class TestProject(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        connect('testing', host='mongomock://localhost')

    @classmethod
    def tearDownClass(cls):
        disconnect()

    def test_init(self):
        p = Project(project_id="test",
                    data_directory="/media/ross/extdrive2/CytoPy_data/test")
        self.assertRaises(Project(project_id="test",
                                  data_directory="/media/ross/extdrive2/CytoPy_data/INVALID"),
                          AssertionError)

    def test_experiments(self):
        p = Project(project_id="test",
                    data_directory="/media/ross/extdrive2/CytoPy_data/test")
        p.add_experiment(experiment_id="test_exp1",
                         panel_definition="test_data/test_panel.xlsx")
        p.add_experiment(experiment_id="test_exp2",
                         panel_definition="test_data/test_panel.xlsx")
        self.assertEqual(p.list_experiments(), ["test_exp1", "text_exp2"])
        self.assertEqual(Experiment.objects(experiment_name="test_exp1").get(),
                         p.load_experiment("test_exp1"))
        p.delete_experiment("test_exp1")
        self.assertEqual(p.list_experiments(), ["test_exp2"])
        self.assertRaises(p.load_experiment("test_exp1"), AssertionError)

    def test_subjects(self):
        p = Project(project_id="test",
                    data_directory="/media/ross/extdrive2/CytoPy_data/test")
        p.add_subject(subject_id="test_pt1")
        p.add_subject(subject_id="test_pt2")
        self.assertRaises(p.add_subject(subject_id="test_pt2"), AssertionError)
        self.assertEqual(p.list_subjects(), ["test_pt1", "test_pt2"])
        self.assertEqual(Subject.objects(subject_id="test_pt1"),
                         p.load_subject("test_pt1"))
        p.delete_experiment("test_pt1")
        self.assertEqual(p.list_experiments(), ["test_pt2"])
        self.assertRaises(p.load_experiment("test_pt1"), AssertionError)


if __name__ == '__main__':
    unittest.main()
