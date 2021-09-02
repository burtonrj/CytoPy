"""
1. The generation of a feature matrix needs improving, I should be able to:
    1. Choose the populations of interest
    2. Choose how I wish to represent those populations - N events, frac of parent, ratio to some other pop
    3. I should be able to easily append meta data of varying complexity
2. Feature selection should be linear in the form of: filter → wrapper (feature importance) → embedded selection (Lasso) → partial dependence and feature interaction.
3. Dimension reduction should be focused on feature contribution in embedding
"""
import re
from collections import defaultdict
from typing import *

import pandas as pd

from ..data.experiment import Experiment
from ..data.experiment import FileGroup
from ..data.project import Project
from ..feedback import progress_bar


def list_all_populations(experiment: Experiment):
    pass


def get_population_as_perc_of(files: List[FileGroup], population: str, stat: str):
    pattern = re.compile("% of (.*)")
    parent_name = pattern.search(stat).group(1)
    data = defaultdict(list)
    for fg in files:
        if parent_name == "parent":
            parent_name = fg.get_population(population).parent
        elif parent_name == "grandparent":
            parent_name = fg.get_population(fg.get_population(population).parent).parent
        parent_n = fg.population_stats(population=parent_name)["n"]
        pop_n = fg.population_stats(population=population)["n"]
        data["sample_id"].append(fg.primary_id)
        data[stat].append(pop_n / parent_n * 100)
    return pd.DataFrame(data)


def get_population_n(files: List[FileGroup], population: str):
    data = defaultdict(list)
    for fg in files:
        data["sample_id"].append(fg.primary_id)
        data[f"{population}_n"].append(fg.population_stats(population=population)["n"])
    return pd.DataFrame(data)


class FeatureSpace:
    def __init__(self, project: Project, verbose: bool = True):
        self.project = project
        self.verbose = verbose
        self._data = pd.DataFrame(columns=["sample_id", "experiment"])

    @property
    def data(self):
        return self._data

    @data.setter
    def data(self, new_data: pd.DataFrame):
        self._data = self._data.merge(new_data, on=["sample_id", "experiment"], how="outer")

    def add_population_statistics(self, experiment: str, population: str, stat: str):
        exp = self.project.get_experiment(experiment_id=experiment)
        if "% of " in stat:
            data = get_population_as_perc_of(files=exp.fcs_files, population=population, stat=stat)
            data["experiment"] = exp.experiment_id
            self.data = data
        elif stat == "n":
            data = get_population_n(files=exp.fcs_files, population=population)
            data["experiment"] = exp.experiment_id
            self.data = data
        elif stat[0] == ":":
            pass
        else:
            raise ValueError(
                "stat must be either '% of parent', '% of grandparent', 'n' or '% of NAME' where "
                "'NAME' is the name of a valid population."
            )

    def add_mfi(self):
        pass

    def add_control_comparison(self):
        pass

    def add_meta(self):
        pass
