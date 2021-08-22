"""
1. The generation of a feature matrix needs improving, I should be able to:
    1. Choose the populations of interest
    2. Choose how I wish to represent those populations - N events, frac of parent, ratio to some other pop
    3. I should be able to easily append meta data of varying complexity
2. Feature selection should be linear in the form of: filter → wrapper (feature importance) → embedded selection (Lasso) → partial dependence and feature interaction.
3. Dimension reduction should be focused on feature contribution in embedding
"""
import re
from typing import *

import polars as pl

from ..data.experiment import Experiment
from ..data.experiment import FileGroup
from ..data.project import Project
from ..feedback import progress_bar


def list_all_populations(experiment: Experiment):
    pass


def get_population_as_perc_of(files: List[FileGroup], population: str, stat: str):
    pattern = re.compile("% of (.*)")
    parent_name = pattern.search(stat).group(1)
    data = {"sample_id": list(), stat: list()}
    for fg in files:
        parent_n = fg.population_stats(population=parent_name)["n"]
        pop_n = fg.population_stats(population=population)["n"]
        data["sample_id"].append(fg.primary_id)
        data[stat].append(pop_n / parent_n * 100)
    return pl.DataFrame(data)


class FeatureSpace:
    def __init__(self, project: Project, verbose: bool = True):
        self.project = project
        self.verbose = verbose
        self._data = pl.DataFrame()

    def add_population_statistics(self, experiment: str, population: str, stat: str):
        exp = self.project.get_experiment(experiment_id=experiment)
        if "% of " in stat:
            data = get_population_as_perc_of(files=exp.fcs_files, population=population, stat=stat)
            data["experiment"] = exp.experiment_id

    def add_population_ratios(self):
        pass

    def add_control_comparison(self):
        pass

    def add_meta(self):
        pass

    def dataframe(self):
        pass
