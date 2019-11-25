from immunova.data.fcs_experiments import FCSExperiment
from immunova.data.fcs import FileGroup, File, ChannelMap
from immunova.data.panel import Panel
from immunova.flow.gating.actions import Gating
from immunova.flow.gating.defaults import ChildPopulationCollection
from immunova.flow.supervised_algo.utilities import standard_scale, norm_scale, find_common_features
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np


class CellClassifierError(Exception):
    pass


def __channel_mappings(features: list, panel: Panel):
    mappings = list()
    panel_mappings = panel.mappings
    for f in features:
        channel = list(filter(lambda x: x.channel == f, panel_mappings))
        marker = list(filter(lambda x: x.marker == f, panel_mappings))
        if (len(channel) > 1 or len(marker) > 1) or (len(channel) == 1 and len(marker) == 1):
            raise ValueError(f'Feature {f} found in multiple channel_marker mappings in panel {panel.panel_name}: '
                             f'{channel}; {marker}.')
        if len(channel) == 0:
            if len(marker) == 0:
                raise ValueError(f'Feature {f} not found in associated panel')
            mappings.append(ChannelMap(channel=marker[0].channel,
                                       marker=marker[0].marker))
            continue
        mappings.append(ChannelMap(channel=channel[0].channel,
                                   marker=channel[0].marker))
    return mappings


def create_reference_sample(experiment: FCSExperiment,
                            root_population='root',
                            exclude: list or None = None,
                            new_file_name: str or None = None,
                            sampling_method: str = 'uniform',
                            sample_n: int = 1000,
                            sample_frac: float or None = None):
    """
    Given some experiment and a root population that is common to all fcs file groups within this experiment, take
    a sample from each and create a new file group from the concatenation of these data
    :param experiment:
    :param root_population:
    :param exclude:
    :return:
    """
    def sample(d):
        if sampling_method == 'uniform':
            if sample_frac is None:
                if d.shape[0] > sample_n:
                    return d.sample(sample_n)
                return d
            return d.sample(frac=sample_frac)
        raise CellClassifierError('Error: currently only uniform sampling is implemented in this version of immunova')

    print('-------------------- Generating Reference Sample --------------------')
    if exclude is None:
        exclude = []
    if new_file_name is None:
        new_file_name = f'{experiment.experiment_id}_sampled_data'
    print('Finding features common to all fcs files...')
    features = find_common_features(experiment=experiment, exclude=exclude)
    channel_mappings = __channel_mappings(features,
                                          experiment.panel)
    files = [f for f in experiment.list_samples() if f not in exclude]
    data = pd.DataFrame()
    for f in files:
        print(f'Sampling {f}...')
        g = Gating(experiment, f, include_controls=False)
        if root_population not in g.populations.keys():
            print(f'Skipping {f} as {root_population} is absent from gated populations')
            continue
        df = g.get_population_df(root_population)[features]
        data = pd.concat([data, sample(df)])
    print('Sampling complete!')
    new_filegroup = FileGroup(primary_id=new_file_name)
    new_filegroup.flags = 'sampled data'
    new_file = File(file_id=new_file_name,
                    compensated=True,
                    channel_mappings=channel_mappings)
    print('Inserting sampled data to database...')
    new_file.put(data.values)
    new_filegroup.files.append(new_file)
    print('Saving changes...')
    mid = new_filegroup.save()
    experiment.fcs_files.append(new_filegroup)
    experiment.save()
    print(f'Complete! New file saved to database: {new_file_name}, {mid}')
    print('-----------------------------------------------------------------')


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
