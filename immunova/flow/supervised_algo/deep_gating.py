from immunova.data.fcs_experiments import FCSExperiment
from immunova.flow.gating.transforms import apply_transform
from immunova.flow.gating.defaults import ChildPopulationCollection
from immunova.flow.gating.actions import Gating
from keras.models import Model
from keras.layers import Input, Dense
from keras.models import load_model
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from keras.regularizers import l2
from keras import callbacks as cb
from keras.callbacks import LearningRateScheduler
import keras.optimizers
import pandas as pd
import numpy as np
import math
import os


class DeepGateError(Exception):
    pass


def step_decay(epoch):
    '''
    Learning rate schedule.
    '''
    initial_lrate = 1e-3
    drop = 0.5
    epochs_drop = 50.0
    lrate = initial_lrate * math.pow(drop,math.floor((1+epoch)/epochs_drop))
    return lrate


def standard_scale(data: np.array):
    data = data.copy()
    preprocessor = StandardScaler().fit(data)
    data = preprocessor.transform(data)
    return data


def cell_classifier(train_x, train_y, hidden_layer_sizes, l2_penalty=1e-4,
                    activation='softplus', loss='sparse_categorical_crossentropy',
                    output_activation='softmax'):
    # Expand labels, to work with sparse categorical cross entropy.
    if loss == 'sparse_categorical_crossentropy':
        train_y = np.expand_dims(train_y, -1)

    # Construct a feed-forward neural network.
    input_layer = Input(shape=(train_x.shape[1],))
    hidden1 = Dense(hidden_layer_sizes[0], activation=activation,
                    W_regularizer=l2(l2_penalty))(input_layer)
    hidden2 = Dense(hidden_layer_sizes[1], activation=activation,
                    W_regularizer=l2(l2_penalty))(hidden1)
    hidden3 = Dense(hidden_layer_sizes[2], activation=activation,
                    W_regularizer=l2(l2_penalty))(hidden2)
    num_classes = len(np.unique(train_y)) - 1
    output_layer = Dense(num_classes, activation=output_activation)(hidden3)

    net = Model(input=input_layer, output=output_layer)
    lrate = LearningRateScheduler(step_decay)
    optimizer = keras.optimizers.rmsprop(lr=0.0)

    net.compile(optimizer=optimizer, loss=loss)
    net.fit(train_x, train_y, nb_epoch=80, batch_size=128, shuffle=True,
            validation_split=0.1,
            callbacks=[lrate, cb.EarlyStopping(monitor='val_loss',
                                               patience=25, mode='auto')])
    return net


class DeepGating:
    """
    DeepGating class for performing an adaption of DeepCyTOF in Immmunova.
    """
    def __init__(self, experiment: FCSExperiment, reference_sample: str, population_labels: list, features: list,
                 multi_label: bool = True, samples: str or list = 'all',
                 transform: str = 'log_transform', hidden_layer_sizes: list or None = None,
                 l2_penalty: float = 1e-4, root_population: str = 'root', threshold: float = 0.5,
                 scale: str = 'Standardise'):

        self.experiment = experiment
        self.transform = transform
        self.multi_label = multi_label
        self.classifier = None
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

        if scale == 'Standardise':
            self.ref_X = standard_scale(self.ref_X)
        elif scale == 'Normalise':
            self.ref_X = norm_scale(self.ref_X)
        elif scale is None:
            print('Warning: it is recommended that data is scaled prior to training. Unscaled data can result '
                  'in some weights updating faster than others, having a negative effect on classifier performance')
        else:
            raise DeepGateError('Error: scale method not recognised, must be either `Standardise` or `Normalise`')

        if self.samples == 'all':
            self.samples = self.experiment.list_samples()
        else:
            self.samples = samples

        if hidden_layer_sizes is None:
            self.hidden_layer_sizes = [12, 6, 3]
        else:
            self.hidden_layer_sizes = hidden_layer_sizes

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
            raise DeepGateError('Error: one or more population dependency errors')
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

    def load_model(self, path: str):
        if not os.path.isfile(path):
            raise DeepGateError(f'Error: invalid file name passed to load_model {path}')
        self.classifier = load_model(path)
        print('Classifier loaded successfully!')

    def save_classifier(self, path):
        self.classifier.save(path)
        print(f'Classifier saved to {path}')

    def __train(self, train_x, train_y):
        if self.multi_label:
            self.classifier = cell_classifier(train_x, train_y, self.hidden_layer_sizes, self.l2_penalty,
                                              activation='softmax', loss='binary_crossentropy',
                                              output_activation='sigmoid')
        else:
            self.classifier = cell_classifier(train_x, train_y, self.hidden_layer_sizes, self.l2_penalty)

    def train_cv(self, k=5):
        kf = KFold(n_splits=k)
        train_performance = list()
        test_performance = list()

        for i, (train_index, test_index) in enumerate(kf.split(self.ref_X)):
            train_x, test_x = self.ref_X[train_index], self.ref_X[test_index]
            train_y, test_y = self.ref_y[train_index], self.ref_y[test_index]
            self.__train(train_x, train_y)
            p = evaluate_model(self.classifier, train_x, train_y, self.multi_label, self.threshold)
            p['k'] = i
            train_performance.append(p)
            p = evaluate_model(self.classifier, test_x, test_y, self.multi_label, self.threshold)
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
        train_x, test_x, train_y, test_y = train_test_split(self.ref_X, self.ref_y,
                                                            test_size=holdout_frac,
                                                            random_state=42)
        self.__train(train_x, train_y)
        train_performance = evaluate_model(self.classifier, train_x, train_y, self.multi_label, self.threshold)
        train_performance['test_train'] = 'train'
        test_performance = evaluate_model(self.classifier, test_x, test_y, self.multi_label, self.threshold)
        test_performance['test_train'] = 'test'
        return pd.concat([test_performance, test_performance])

    def train_classifier(self, calc_performance: bool = True) -> pd.DataFrame or None:
        self.__train(self.ref_X, self.ref_y)
        if calc_performance:
            train_performance = evaluate_model(self.classifier, self.ref_X, self.ref_y,
                                               self.multi_label, self.threshold)
            train_performance['test_train'] = 'train'
            return train_performance

    def predict(self, target_sample, threshold=None):
        sample_gates = Gating(self.experiment, target_sample)
        sample = standard_scale(sample_gates.get_population_df(self.root_population,
                                                               transform=True,
                                                               transform_method=self.transform)[self.features])
        if self.classifier is None:
            raise DeepGateError('Error: cell classifier has not been trained, either load an existing model using '
                                'the `load_model` method or train the classifier using the `train_classifier` method')

        y_probs = self.classifier.predict(sample)
        y_hat = predict_class(y_probs, threshold)
        new_populations = ChildPopulationCollection(gate_type='sml')
        if self.multi_label:
            y_hat = pd.DataFrame(y_hat, columns=self.mappings)
            for label in y_hat.columns:
                x = y_hat[label].values
                new_populations.add_population(name=label)
                new_populations.populations[label].update_index(x.nonzero())
                new_populations.populations[label].update_geom(shape='sml', x=None, y=None)
        sample_gates.update_populations(new_populations, sample, parent_name=self.root_population, warnings=[])
        print('Deep gating complete!')