from .cell_classifier import CellClassifier
from keras.models import Sequential
from keras.layers import Dense
from keras.models import load_model
from keras.regularizers import l2
from keras import callbacks as cb
from keras.callbacks import LearningRateScheduler
import keras.optimizers
import numpy as np
import math
import os


def step_decay(epoch: int):
    """
    Learning rate scheduler.

    Parameters
    -----------
    epoch: int
        Number of epochs to perform

    Returns
    --------
    float
        Learning rate
    """
    initial_lrate = 1e-3
    drop = 0.5
    epochs_drop = 50.0
    lrate = initial_lrate * math.pow(drop, math.floor((1+epoch)/epochs_drop))
    return lrate


class DeepGating(CellClassifier):
    """
    Identify populations using a supervised multi-layer feed-forward neural network, inspired by the work
    of Li, H et al (https://doi.org/10.1093/bioinformatics/btx448). Neural network is implemented using Keras
    and Tensorflow

    Parameters
    -----------
    hidden_layer_sizes: List
        Size of hidden layers (length must equal number of layers)
    n_layer: int, (default=3)
        Number of layers in neural network
    l2_penalty: float, (default=1e-4)
        Parameter for L2 regularisation
    activation_func: str, (default='softplus')
        Name of activation function to use of hidden layers
    loss_func: str, (default='sparse_categorical_crossentropy')
        Name of loss function to use for updating weights
    output_activation_func: str, (default='softmax')
        Name of activation function to use on outer layer
    kwargs:
        Keyword arguments for CellClassifier
    """
    def __init__(self,
                 hidden_layer_sizes: list or None = None,
                 n_layers: int = 3,
                 l2_penalty: float = 1e-4,
                 activation_func: str = 'softplus',
                 loss_func: str = 'sparse_categorical_crossentropy',
                 output_activation_func: str = 'softmax',
                 **kwargs):
        super().__init__(**kwargs)
        self.l2_penalty = l2_penalty
        self.activation_func = activation_func
        self.loss_func = loss_func
        self.output_activation_func = output_activation_func
        self.prefix = 'DeepNeuralNet'

        if hidden_layer_sizes is None:
            self.hidden_layer_sizes = list()
            self.hidden_layer_sizes.append(12)
            for i in range(n_layers - 2):
                self.hidden_layer_sizes.append(6)
            self.hidden_layer_sizes.append(3)
        else:
            assert len(hidden_layer_sizes) == n_layers, 'Hidden layer sizes does not match number of specified layers'
            if n_layers < 3:
                print('n_layers cannot be less than 3')
                self.n_layers = 3
            else:
                self.n_layers = n_layers
            self.hidden_layer_sizes = hidden_layer_sizes

    def load_model(self,
                   path: str):
        """
        Load existing model

        Parameters
        ----------
        path: str
            Path to existing Kera model

        Returns
        -------
        None
        """
        assert os.path.isfile(path), f'Invalid file name passed to load_model {path}'
        self.classifier = load_model(path)
        print('Classifier loaded successfully!')

    def build_model(self):
        """
        Build model (must be called prior to fit)

        Returns
        -------
        None
        """
        # Expand labels, to work with sparse categorical cross entropy.
        if self.loss_func == 'sparse_categorical_crossentropy':
            self.train_y = np.expand_dims(self.train_y, -1)
            num_classes = len(np.unique(self.train_y))
        elif self.loss_func == 'categorical_crossentropy':
            num_classes = len(np.unique(self.train_y))
            self.train_y = keras.utils.to_categorical(self.train_y,
                                                      num_classes=num_classes)
        else:
            num_classes = len(np.unique(self.train_y)) - 1

        model = Sequential()
        # Construct a feed-forward neural network.
        # Input layer
        model.add(Dense(self.hidden_layer_sizes[0], activation=self.activation_func,
                        W_regularizer=l2(self.l2_penalty), input_shape=(self.train_X.shape[1],)))

        for i in range(self.n_layers - 2):
            model.add(Dense(self.hidden_layer_sizes[i+1], activation=self.activation_func,
                            W_regularizer=l2(self.l2_penalty)))

        # Output layer
        model.add(Dense(num_classes, activation=self.output_activation_func))
        optimizer = keras.optimizers.rmsprop(lr=0.0)

        model.compile(optimizer=optimizer, loss=self.loss_func)
        self.classifier = model

    def _fit(self,
             x: np.array,
             y: np.array,
             **kwargs):
        """
        Should be called internally. Overwrites base class fit method. Fit's data to model.

        Parameters
        ----------
        x: Numpy.array
            feature space
        y: Numpy.array
            labels
        kwargs:
            Additional keyword arguments to pass to fit call

        Returns
        -------
        None
        """
        lrate = LearningRateScheduler(step_decay)
        epochs = kwargs.get('epochs', 80)
        batch_size = kwargs.get('batch_size', 128)
        validation_split = kwargs.get('validation_split', 0.1)
        callbacks = [lrate, cb.EarlyStopping(monitor='val_loss', patience=25, mode='auto')]
        if self.class_weights is not None:
            self.classifier.fit(x, y, nb_epoch=epochs, batch_size=batch_size, shuffle=True,
                                validation_split=validation_split, callbacks=callbacks,
                                class_weight=self.class_weights, **kwargs)
        else:
            self.classifier.fit(x, y, nb_epoch=epochs, batch_size=batch_size, shuffle=True,
                                validation_split=validation_split, callbacks=callbacks, **kwargs)

    def save_classifier(self, path):
        """
        Save classifier to disk

        Parameters
        ----------
        path: str
            File path

        Returns
        -------
        None
        """
        self.classifier.save(path)
        print(f'Classifier saved to {path}')


