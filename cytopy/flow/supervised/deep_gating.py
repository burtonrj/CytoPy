from cytopy.flow.supervised.cell_classifier import CellClassifier
from keras.models import Sequential
from keras.layers import Input, Dense
from keras.models import load_model
from keras.regularizers import l2
from keras import callbacks as cb
from keras.callbacks import LearningRateScheduler
import keras.optimizers
import numpy as np
import math
import os


def step_decay(epoch):
    """
    Learning rate scheduler.
    """
    initial_lrate = 1e-3
    drop = 0.5
    epochs_drop = 50.0
    lrate = initial_lrate * math.pow(drop, math.floor((1+epoch)/epochs_drop))
    return lrate


class DeepGating(CellClassifier):
    """
    DeepGating class for performing an adaption of DeepCyTOF in Immmunova.
    """
    def __init__(self, hidden_layer_sizes: list or None = None, l2_penalty: float = 1e-4,
                 activation_func: str = 'softplus', loss_func: str = 'sparse_categorical_crossentropy',
                 output_activation_func: str = 'softmax', **kwargs):
        super().__init__(**kwargs)
        self.l2_penalty = l2_penalty
        self.activation_func = activation_func
        self.loss_func = loss_func
        self.output_activation_func = output_activation_func
        self.prefix = 'DeepNeuralNet'

        if hidden_layer_sizes is None:
            self.hidden_layer_sizes = [12, 6, 3]
        else:
            self.hidden_layer_sizes = hidden_layer_sizes

    def load_model(self, path: str):
        assert os.path.isfile(path), f'Invalid file name passed to load_model {path}'
        self.classifier = load_model(path)
        print('Classifier loaded successfully!')

    def build_model(self):
        # Expand labels, to work with sparse categorical cross entropy.
        if self.loss_func == 'sparse_categorical_crossentropy':
            self.train_y = np.expand_dims(self.train_y, -1)

        model = Sequential()
        # Construct a feed-forward neural network.
        model.add(Input(shape=(self.train_X.shape[1],)))
        model.add(Dense(self.hidden_layer_sizes[0], activation=self.activation_func,
                        W_regularizer=l2(self.l2_penalty)))
        model.add(Dense(self.hidden_layer_sizes[1], activation=self.activation_func,
                        W_regularizer=l2(self.l2_penalty)))
        model.add(Dense(self.hidden_layer_sizes[2], activation=self.activation_func,
                        W_regularizer=l2(self.l2_penalty)))
        num_classes = len(np.unique(self.train_y)) - 1
        model.add(Dense(num_classes, activation=self.output_activation_func))
        optimizer = keras.optimizers.rmsprop(lr=0.0)

        model.compile(optimizer=optimizer, loss=self.loss_func)
        self.classifier = model

    def _fit(self, x, y, **kwargs):
        lrate = LearningRateScheduler(step_decay)
        self.classifier.fit(x, y, nb_epoch=80, batch_size=128, shuffle=True,
                            validation_split=0.1, callbacks=[lrate, cb.EarlyStopping(monitor='val_loss',
                                                                                     patience=25, mode='auto')])

    def save_classifier(self, path):
        self.classifier.save(path)
        print(f'Classifier saved to {path}')


