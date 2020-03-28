import sys
sys.path.append('/home/ross/CytoPy')

from cytopy.data.project import Project
from cytopy.data.panel import Panel
from cytopy.data.mongo_setup import global_init
from mongoengine import connect
from mongoengine.errors import DoesNotExist
import unittest

db = connect('test')
db.drop_database('test')
global_init('test')


class TestProject(unittest.TestCase):

    def test_Project(self):
        test_panel = Panel(panel_name='test')
        test_panel.save()
        test_project = Project(project_id='test', owner='test')
        test_project.add_experiment('test_experiment_aml', panel_name='test')
        test_project.add_experiment('test_experiment_dummy', panel_name='test')
        test_project.add_subject('test_subject', testing=True)
        self.assertEqual(test_project.list_subjects(), ['test_subject'])
        s = test_project.pull_subject('test_subject')
        self.assertEqual(s.testing, True)
        self.assertEqual(s.subject_id, 'test_subject')
        self.assertEqual(test_project.list_fcs_experiments(), ['test_experiment_aml', 'test_experiment_dummy'])
        test_project.save()
        test_project.delete()
        err = False
        try:
            Project.objects(project_id='test').get()
        except DoesNotExist:
            err = True
        self.assertTrue(err)


if __name__ == '__main__':
    unittest.main()
