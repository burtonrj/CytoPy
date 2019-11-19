from immunova.data.fcs_experiments import FCSExperiment
from immunova.flow.gating.transforms import apply_transform
from immunova.flow.gating.defaults import ChildPopulationCollection
from immunova.flow.gating.actions import Gating
from immunova.flow.normalisation import MMDResNet
from immunova.flow.deepcytof.classifier import cell_classifier, evaluate_model, predict_class
from keras.models import load_model
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from tqdm import tqdm
import pandas as pd
import numpy as np
import os


class DeepGateError(Exception):
    pass


def standard_scale(data: np.array):
    data = data.copy()
    preprocessor = StandardScaler().fit(data)
    data = preprocessor.transform(data)
    return data


def calculate_reference_sample(experiment: FCSExperiment) -> str:
    """
    Given an FCS Experiment with multiple FCS files, calculate the optimal reference file.

    This is performed as described in Li et al paper (https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5860171/) on
    DeepCyTOF: for every 2 samples i, j compute the Frobenius norm of the difference between their covariance matrics
    and then select the sample with the smallest average distance to all other samples.
    :param experiment: FCSExperiment with multiple FCS samples
    :return: sample ID for optimal reference sample
    """
    print('Warning: this process can take some time as comparisons are made between all samples in the experiment.')
    samples = experiment.list_samples()
    if len(samples) == 0:
        raise DeepGateError('Error: no samples associated to given FCSExperiment')
    n = len(samples)
    norms = np.zeros(shape=[n, n])
    ref_ind = None
    for i in tqdm(range(0, n)):
        data_i = experiment.pull_sample_data(sample_id=samples[i], data_type='raw',
                                             output_format='matrix')
        data_i = data_i[[x for x in data_i.columns if x != 'Time']]
        data_i = apply_transform(data_i, transform_method='log_transform')
        if data_i is None:
            print(f'Error: failed to fetch data for {samples[i]}. Skipping.')
            continue

        cov_i = np.cov(data_i, rowvar=False)
        for j in range(0, n):
            data_j = experiment.pull_sample_data(sample_id=samples[j], data_type='raw',
                                                 output_format='matrix')
            data_j = data_j[[x for x in data_j.columns if x != 'Time']]
            data_j = apply_transform(data_j, transform_method='log_transform')
            cov_j = np.cov(data_j, rowvar=False)
            cov_diff = cov_i - cov_j
            norms[i, j] = np.linalg.norm(cov_diff, ord='fro')
            norms[j, i] = norms[i, j]
            avg = np.mean(norms, axis=1)
            ref_ind = np.argmin(avg)[0]
    if ref_ind is not None:
        return samples[ref_ind]
    else:
        raise DeepGateError('Error: unable to calculate sample with minimum average distance. You must choose'
                            ' manually.')


class DeepGating:
    """
    DeepGating class for performing an adaption of DeepCyTOF in Immmunova.
    """
    def __init__(self, experiment: FCSExperiment, reference_sample: str, population_labels: list, features: list,
                 multi_label: bool = False, samples: str or list = 'all',
                 transform: str = 'log_transform', autoencoder: str or None = None,
                 calibrator: str or None = None, hidden_layer_sizes: list or None = None,
                 l2_penalty: float = 1e-4, root_population: str = 'root', threshold: float = 0.5):

        self.experiment = experiment
        self.transform = transform
        self.multi_label = multi_label
        self.classifier = None
        self.calibrator = None
        self.autoencoder = None
        self.l2_penalty = l2_penalty
        self.features = features
        self.root_population = root_population
        self.threshold = threshold

        ref = Gating(self.experiment, reference_sample)
        self.population_labels = ref.valid_populations(population_labels)
        if len(ref.populations) < 2:
            raise DeepGateError(f'Error: reference sample {reference_sample} does not contain any gated populations, '
                                f'please ensure that the reference sample has been gated prior to training.')

        if multi_label:
            self.ref_X, self.ref_y, self.mappings = self.dummy_data(ref, features)
        else:
            self.ref_X, self.ref_y, self.mappings = self.single_label_data(ref, features)

        if self.samples == 'all':
            self.samples = self.experiment.list_samples()
        else:
            self.samples = samples

        if autoencoder is not None:
            if not os.path.isfile(autoencoder):
                raise DeepGateError(f'Error: invalid file name passed for autoencoder model {autoencoder}')
            self.autoencoder = load_model(autoencoder)

        if calibrator is not None:
            if not os.path.isfile(calibrator):
                raise DeepGateError(f'Error: invalid file name passed for calibrator model {calibrator}')
            self.calibrator = MMDResNet.load_model(calibrator)

        if hidden_layer_sizes is None:
            self.hidden_layer_sizes = [12, 6, 3]
        else:
            self.hidden_layer_sizes = hidden_layer_sizes

    def dummy_data(self, ref: Gating, features, scale: bool = True) -> (pd.DataFrame, pd.DataFrame, list):
        root = ref.get_population_df(self.root_population, transform=True, transform_method=self.transform)[features]
        labels = {pop: np.zeros((1, root.shape[0]))[0] for pop in self.population_labels}
        for pop in self.population_labels:
            pop_idx = ref.populations[pop].index
            np.put(labels[pop], pop_idx, [1 for n in pop_idx])
        if scale:
            root = standard_scale(root)
        labels = pd.DataFrame(labels)
        return root, labels, list(labels.columns)

    def single_label_data(self, ref: Gating, features, scale: bool = True) -> \
            (pd.DataFrame, np.array, list):
        if self.__check_downstream_overlaps(ref):
            raise DeepGateError('Error: one or more population dependency errors')
        root = ref.get_population_df(self.root_population, transform=True, transform_method=self.transform)[features]
        y = np.zeros((0, root.shape[0]))[0]
        for i, pop in enumerate(self.population_labels):
            pop_idx = ref.populations[pop].index
            np.put(y, pop_idx, [i+1 for n in pop_idx])
        if scale:
            root = standard_scale(root)
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

    #def denoise(self, target):
    #    if self.autoencoder is None:
    #        print('No autoencoder found, training autoencoder...')
    #        self.train_autoencoder()
    #    return self.autoencoder.predict(target)

    def calibrate(self, sample, evaluate=False):
        source = sample[self.features].values
        target = self.ref_X.values
        mmdnet = MMDResNet.MMDNet(len(self.features), epochs=500, denoise=False, layer_sizes=[25, 25, 25],
                                  l2_penalty=self.l2_penalty)
        mmdnet.build_model()
        mmdnet.fit(source, target, initial_lr=1e-3, lr_decay=0.97)
        if evaluate:
            print('Evaluating calibration...')
            mmdnet.evaluate(source, target)
        calibrated_source = mmdnet.net.predict(source)
        return calibrated_source

    def load_model(self, path: str, model_type: str = 'classifier'):
        if not os.path.isfile(path):
            raise DeepGateError(f'Error: invalid file name passed to load_model {path}')
        if model_type == 'classifier':
            self.classifier = load_model(path)
            print('Classifier loaded successfully!')
        elif model_type == 'autoencoder':
            self.autoencoder = load_model(path)
            print('Autoencoder loaded successfully!')
        elif model_type == 'calibrator':
            self.calibrator = load_model(path)
            print('Calibrator loaded successfully!')
        print("Error: model_type not recognised, expecting one of: 'classifier', 'autoencoder', 'calibrator'")

    def save_autoencoder(self, path):
        self.autoencoder.save(path)
        print(f'Autoencoder saved to {path}')

    def save_calibrator(self, path):
        self.calibrator.save_weights(path)
        print(f'Calibrator saved to {path}')

    def save_classifier(self, path):
        self.classifier.save(path)
        print(f'Classifier saved to {path}')

    def __train(self, train_X, train_y):
        if self.multi_label:
            self.classifier = cell_classifier(train_X, train_y, self.hidden_layer_sizes, self.l2_penalty,
                                              activation='softmax', loss='binary_crossentropy',
                                              output_activation='sigmoid')
        else:
            self.classifier = cell_classifier(train_X, train_y, self.hidden_layer_sizes, self.l2_penalty)

    def train_cv(self, k=5):
        kf = KFold(n_splits=k)
        train_performance = list()
        test_performance = list()

        for i, (train_index, test_index) in enumerate(kf.split(self.ref_X)):
            train_X, test_X = self.ref_X[train_index], self.ref_X[test_index]
            train_y, test_y = self.ref_y[train_index], self.ref_y[test_index]
            self.__train(train_X, train_y)
            p = evaluate_model(self.classifier, train_X, train_y, self.multi_label, self.threshold)
            p['k'] = i
            train_performance.append(p)
            p = evaluate_model(self.classifier, test_X, test_y, self.multi_label, self.threshold)
            p['k'] = i
            test_performance.append(p)

        train_performance = pd.concat(train_performance)
        train_performance['average_performance'] = train_performance.mean(axis=1)
        train_performance['test_train'] = 'train'
        test_performance = pd.concat(test_performance)
        test_performance['average_performance'] = test_performance.mean(axis=1)
        test_performance['test_train'] = 'test'
        return pd.concat([train_performance, test_performance])

    def train_holdout(self, holdout_frac: float = 0.3):
        train_X, test_X, train_y, test_y = train_test_split(self.ref_X, self.ref_y,
                                                            test_size=holdout_frac,
                                                            random_state=42)
        self.__train(train_X, train_y)
        train_performance = evaluate_model(self.classifier, train_X, train_y, self.multi_label, self.threshold)
        train_performance['test_train'] = 'train'
        test_performance = evaluate_model(self.classifier, test_X, test_y, self.multi_label, self.threshold)
        test_performance['test_train'] = 'test'
        return pd.concat([test_performance, test_performance])

    def train_classifier(self, calc_performance: bool = True) -> pd.DataFrame or None:
        self.__train(self.ref_X, self.ref_y)
        if calc_performance:
            train_performance = evaluate_model(self.classifier, self.ref_X, self.ref_y,
                                               self.multi_label, self.threshold)
            train_performance['test_train'] = 'train'
            return train_performance

    def predict(self, target_sample, denoise=False, calibrate=False, threshold=None):
        sample_gates = Gating(self.experiment, target_sample)
        sample = standard_scale(sample_gates.get_population_df(self.root_population,
                                                               transform=True,
                                                               transform_method=self.transform)[self.features])
        if self.classifier is None:
            raise DeepGateError('Error: cell classifier has not been trained, either load an existing model using '
                                'the `load_model` method or train the classifier using the `train_classifier` method')
        #if denoise:
            #print('Removing noise with Autoencoder')
            #sample = self.denoise(sample)

        if calibrate:
            sample = self.calibrate(sample)

        y_probs = self.classifier.predict(sample)
        y_hat = predict_class(y_probs, threshold)
        new_populations = ChildPopulationCollection(gate_type='sml')
        if self.multi_label:
            y_hat = pd.DataFrame(y_hat, columns=self.mappings)
            for label in y_hat.columns:
                x = y_hat[label].values
                new_populations.add_population(name=label)
                new_populations.populations[label].update_index(x.nonzero())
                new_populations.populations[label].update_geom(shape='sml', x=None)
        sample_gates.update_populations(new_populations, sample, parent_name=self.root_population, warnings=[])
        print('Deep gating complete!')