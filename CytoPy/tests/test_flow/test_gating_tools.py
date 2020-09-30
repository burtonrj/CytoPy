from ...data.experiments import Experiment
from ...data.populations import Population
from ...tests import assets
import pandas as pd
import pytest
import os


@pytest.fixture(autouse=True)
def create_example():
    test_exp = Experiment(experiment_id="test",
                          data_directory=f"{os.getcwd()}/test_data",
                          panel_definition=f"{assets.__path__._path[0]}/test_panel.xlsx")
    test_exp.add_new_sample(sample_id="test_sample",
                            primary_path=f"{assets.__path__._path[0]}/test.FCS",
                            compensate=False)
    fg = test_exp.get_sample("test_sample")
    data = pd.read_hdf(path_or_buf=fg.h5path, key="primary")
    fg.populations.append(Population(population_name="root",
                                     index=data.index.values,
                                     parent="root",
                                     n=data.shape[0]))
    fg.populations.append(Population(population_name="test_pop1",
                                     index=data.sample(100).index.values,
                                     parent="root",
                                     n=data.shape[0]))
    fg.populations.append(Population(population_name="test_pop2",
                                     index=data.sample(100).index.values,
                                     parent="root",
                                     n=data.shape[0]))
    fg.save()


def test_gating_init():
    pass

