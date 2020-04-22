from CytoPy.data.panel import Panel
from cytopy.data.project import Project
from CytoPy.data.panel import ChannelMap
from cytopy.data.fcs import FileGroup, File
from sklearn.datasets import make_blobs
import numpy as np
import pandas as pd


def make_example_date(n_samples=100, centers=3, n_features=2):
    blobs = make_blobs(n_samples=n_samples,
                       centers=centers,
                       n_features=n_features,
                       random_state=42)
    columns = [f'feature{i}' for i in range(n_features)]
    columns = columns + ['blobID']
    example_data = pd.DataFrame(np.hstack((blobs[0], blobs[1].reshape(-1, 1))),
                                columns=columns)
    return example_data


def basic_setup():
    # Create panel
    test = Panel(panel_name='test')
    test.create_from_excel('../data/test_panel.xlsx')
    test.save()
    # Create Project
    test_project = Project(project_id='test', owner='test')
    test_project.add_experiment('test_experiment_aml', panel_name='test')
    test_project.add_experiment('test_experiment_dummy', panel_name='test')
    test_project.add_subject('test_subject', testing=True)
    test_project.save()


def setup_with_dummy_data():
    basic_setup()
    example_data = make_example_date(n_samples=100, centers=3, n_features=2)
    # Create dummy channel mappings
    mappings = [ChannelMap(channel='var0', marker='feature0'),
                ChannelMap(channel='var1', marker='feature1'),
                ChannelMap(channel='var2', marker='blobID')]
    # Populate data
    test_project = Project.objects(project_id='test').get()
    test_exp = test_project.load_experiment('test_experiment_dummy')
    test_grp = FileGroup(primary_id='dummy_test',
                         flags='dummy')
    test_file = File(file_id='dummy_test', channel_mappings=mappings)
    test_ctrl = File(file_id='dummy_ctrl', channel_mappings=mappings, file_type='control')
    test_file.put(example_data.values)
    test_ctrl.put(example_data.values)
    test_grp.files = [test_file, test_ctrl]
    test_grp.save()
    test_exp.fcs_files.append(test_grp)
    test_exp.save()
