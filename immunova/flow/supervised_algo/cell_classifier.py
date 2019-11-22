from immunova.data.fcs_experiments import FCSExperiment
from immunova.flow.gating.actions import Gating
from immunova.flow.gating.defaults import ChildPopulationCollection
from immunova.flow.supervised_algo.utilities import standard_scale, norm_scale
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np


class CellClassifierError(Exception):
    pass


class CellClassifier:
    """
    DeepGating class for performing an adaption of DeepCyTOF in Immmunova.
    """
    def __init__(self, experiment: FCSExperiment, reference_sample: str, population_labels: list, features: list,
                 multi_label: bool = True, transform: str = 'log_transform', root_population: str = 'root',
                 threshold: float = 0.5, scale: str = 'Standardise'):

        self.experiment = experiment
        self.transform = transform
        self.multi_label = multi_label
        self.classifier = None
        self.preprocessor = None
        self.features = features
        self.root_population = root_population
        self.threshold = threshold

        if reference_sample == 'sample':
            # ToDo Create reference from sampling all available gated data
            # ref = self.sample_all()
            raise CellClassifierError(f'Error: training from concatenated sample currently not implemented')
        else:
            ref = Gating(self.experiment, reference_sample)
        self.population_labels = ref.valid_populations(population_labels)
        if len(ref.populations) < 2:
            raise CellClassifierError(f'Error: reference sample {reference_sample} does not contain any '
                                      f'gated populations, please ensure that the reference sample has '
                                      f'been gated prior to training.')

        if multi_label:
            self.ref_X, self.ref_y, self.mappings = self.dummy_data(ref, features)
        else:
            self.ref_X, self.ref_y, self.mappings = self.single_label_data(ref, features)

        if scale == 'Standardise':
            self.ref_X, self.preprocessor = standard_scale(self.ref_X)
        elif scale == 'Normalise':
            self.ref_X, self.preprocessor = norm_scale(self.ref_X)
        elif scale is None:
            print('Warning: it is recommended that data is scaled prior to training. Unscaled data can result '
                  'in some weights updating faster than others, having a negative effect on classifier performance')
        else:
            raise CellClassifierError('Error: scale method not recognised, must be either `Standardise` or `Normalise`')

    def dummy_data(self, ref: Gating, features) -> (pd.DataFrame, pd.DataFrame, list):
        root = ref.get_population_df(self.root_population, transform=True, transform_method=self.transform)[features]
        y = {pop: np.zeros((1, root.shape[0]))[0] for pop in self.population_labels}
        for pop in self.population_labels:
            pop_idx = ref.populations[pop].index
            np.put(y[pop], pop_idx, [1 for n in pop_idx])
        y = pd.DataFrame(y)
        return root, y, list(y.columns)

    def single_label_data(self, ref: Gating, features) -> (pd.DataFrame, np.array, list):
        if self.__check_downstream_overlaps(ref):
            raise CellClassifierError('Error: one or more population dependency errors')
        root = ref.get_population_df(self.root_population, transform=True, transform_method=self.transform)[features]
        y = np.zeros((0, root.shape[0]))[0]
        for i, pop in enumerate(self.population_labels):
            pop_idx = ref.populations[pop].index
            np.put(y, pop_idx, [i+1 for n in pop_idx])
        return root, y, self.population_labels

    def __check_downstream_overlaps(self, ref: Gating):
        downstream_overlaps = False
        for pop_i in self.population_labels:
            dependencies = ref.find_dependencies(pop_i)
            if self.root_population in dependencies:
                print(f'Error: population {pop_i} is upstream from the chosen root population {self.root_population}')
                downstream_overlaps = True
            for pop_j in self.population_labels:
                if pop_j == pop_i:
                    continue
                if pop_j in dependencies:
                    print(f'Error: population {pop_j} is a dependency of population {pop_i} (i.e. it is downstream '
                          f'from this population). This will result in invalid labelling. If you wish to continue '
                          f'with these population targets, please set multi_label parameter to True')
                    downstream_overlaps = True
        return downstream_overlaps

    def train_test_split(self, test_size):
        return train_test_split(self.ref_X, self.ref_y, test_size=test_size, random_state=42)

    def save_gating(self, target, y_hat):
        parent = target.get_population_df(population_name=self.root_population)
        new_populations = ChildPopulationCollection(gate_type='sml')
        if self.multi_label:
            y_hat = pd.DataFrame(y_hat, columns=self.mappings)
            for label in y_hat.columns:
                x = y_hat[label].values
                new_populations.add_population(name=label)
                new_populations.populations[label].update_index(x.nonzero())
                new_populations.populations[label].update_geom(shape='sml', x=None, y=None)
        target.update_populations(new_populations, parent, parent_name=self.root_population, warnings=[])
