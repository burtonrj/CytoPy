from immunova.flow.gating.defaults import ChildPopulationCollection
from immunova.flow.gating.actions import Gating
from immunova.flow.supervised_algo.evaluate import evaluate_model
from immunova.flow.supervised_algo.cell_classifier import CellClassifier, CellClassifierError
from immunova.flow.supervised_algo.utilities import predict_class
from keras.models import Model
from keras.layers import Input, Dense
from keras.models import load_model
from sklearn.model_selection import KFold
from keras.regularizers import l2
from keras import callbacks as cb
from keras.callbacks import LearningRateScheduler
import keras.optimizers
import pandas as pd
import numpy as np
import math
import os


def step_decay(epoch):
    '''
    Learning rate schedule.
    '''
    initial_lrate = 1e-3
    drop = 0.5
    epochs_drop = 50.0
    lrate = initial_lrate * math.pow(drop,math.floor((1+epoch)/epochs_drop))
    return lrate


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


class DeepGating(CellClassifier):
    """
    DeepGating class for performing an adaption of DeepCyTOF in Immmunova.
    """
    def __init__(self, hidden_layer_sizes: list or None = None, l2_penalty: float = 1e-4, **kwargs):
        super().__init__(**kwargs)
        self.l2_penalty = l2_penalty

        if hidden_layer_sizes is None:
            self.hidden_layer_sizes = [12, 6, 3]
        else:
            self.hidden_layer_sizes = hidden_layer_sizes

    def load_model(self, path: str):
        if not os.path.isfile(path):
            raise CellClassifierError(f'Error: invalid file name passed to load_model {path}')
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
        train_x, test_x, train_y, test_y = self.train_test_split(test_size=holdout_frac)
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
        if self.preprocessor is not None:
            sample = self.preprocessor.fit_transform(sample_gates.get_population_df(self.root_population,
                                                                                    transform=True,
                                                                                    transform_method=self.transform)[self.features])
        else:
            sample = sample_gates.get_population_df(self.root_population,
                                                    transform=True,
                                                    transform_method=self.transform)[self.features]
        if self.classifier is None:
            raise CellClassifierError('Error: cell classifier has not been trained, either '
                                      'load an existing model using the `load_model` method or train '
                                      'the classifier using the `train_classifier` method')

        y_probs = self.classifier.predict(sample)
        y_hat = predict_class(y_probs, threshold)
        self.save_gating(sample_gates, y_hat)
        print('Deep gating complete!')