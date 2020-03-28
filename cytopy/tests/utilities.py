from cytopy.data.panel import Panel
from cytopy.data.project import Project
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
