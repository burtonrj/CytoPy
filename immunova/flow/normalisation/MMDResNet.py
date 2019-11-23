'''
@author: urishaham
'''

# Keras
import keras.optimizers
from keras.layers import Input, Dense, merge, Activation, add
from keras.layers.normalization import BatchNormalization
from keras.callbacks import LearningRateScheduler
from keras.models import Model, load_model
from keras import callbacks as cb
from keras import initializers
from keras.regularizers import l2
import tensorflow as tf
import keras.backend as K
# Immunova imports
from immunova.flow.normalisation import CostFunctions as cf
from immunova.flow.normalisation import Monitoring as mn
# Scikit-Learn
from sklearn import decomposition
import sklearn.preprocessing as prep
# Scipy and other imports
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib
import math
np.random.seed(42)
# detect display
import os
havedisplay = "DISPLAY" in os.environ
# if we have a display use a plotting backend
if havedisplay:
    matplotlib.use('TkAgg')
else:
    matplotlib.use('Agg')


def train_preprocessor(data, method):
    if method == 'Standardise':
        return prep.StandardScaler().fit(data)
    if method == 'MixMax':
        return prep.MinMaxScaler().fit(data)
    raise ValueError('Currently only Z-score standardisation and MinMax normalisation are valid '
                     'scaling methods')


# learning rate schedule
def step_decay(epoch):
    initial_lrate = 0.001
    drop = 0.1
    epochs_drop = 150.0
    lrate = initial_lrate * math.pow(drop, math.floor((1+epoch)/epochs_drop))
    return lrate


class MMDNet:
    def __init__(self, data_dim, epochs=500, layer_sizes=None, l2_penalty=1e-2,
                 batch_size=1000, verbose=1):
        self.data_dim = data_dim
        self.epochs = epochs
        self.batch_size = batch_size
        self.verbose = verbose
        self.l2_penalty = l2_penalty
        self.layers = None
        self.model = None
        if layer_sizes is None:
            self.layer_sizes = [25, 25]
        else:
            self.layer_sizes = layer_sizes

    def fit(self, source, target, initial_lr=1e-3, lr_decay=0.97, scale_method='Standardise',
            evaluate=True):
        # rescale source
        preprocessor = train_preprocessor(source, scale_method)
        source = preprocessor.transform(source)
        target = preprocessor.transform(target)

        inputDim = target.shape[1]
        calibInput = Input(shape=(inputDim,))
        block1_bn1 = BatchNormalization()(calibInput)
        block1_a1 = Activation('relu')(block1_bn1)
        block1_w1 = Dense(self.layer_sizes[0], activation='linear', kernel_regularizer=l2(self.l2_penalty),
                          kernel_initializer=initializers.RandomNormal(stddev=1e-4))(block1_a1)
        block1_bn2 = BatchNormalization()(block1_w1)
        block1_a2 = Activation('relu')(block1_bn2)
        block1_w2 = Dense(inputDim, activation='linear', kernel_regularizer=l2(self.l2_penalty),
                          kernel_initializer=initializers.RandomNormal(stddev=1e-4))(block1_a2)
        block1_output = add([block1_w2, calibInput])
        block2_bn1 = BatchNormalization()(block1_output)
        block2_a1 = Activation('relu')(block2_bn1)
        block2_w1 = Dense(self.layer_sizes[1], activation='linear', kernel_regularizer=l2(self.l2_penalty),
                          kernel_initializer=initializers.RandomNormal(stddev=1e-4))(block2_a1)
        block2_bn2 = BatchNormalization()(block2_w1)
        block2_a2 = Activation('relu')(block2_bn2)
        block2_w2 = Dense(inputDim, activation='linear', kernel_regularizer=l2(self.l2_penalty),
                          kernel_initializer=initializers.RandomNormal(stddev=1e-4))(block2_a2)
        block2_output = add([block2_w2, block1_output])
        block3_bn1 = BatchNormalization()(block2_output)
        block3_a1 = Activation('relu')(block3_bn1)
        block3_w1 = Dense(self.layer_sizes[1], activation='linear', kernel_regularizer=l2(self.l2_penalty),
                          kernel_initializer=initializers.RandomNormal(stddev=1e-4))(block3_a1)
        block3_bn2 = BatchNormalization()(block3_w1)
        block3_a2 = Activation('relu')(block3_bn2)
        block3_w2 = Dense(inputDim, activation='linear', kernel_regularizer=l2(self.l2_penalty),
                          kernel_initializer=initializers.RandomNormal(stddev=1e-4))(block3_a2)
        block3_output = add([block3_w2, block2_output])

        self.model = Model(inputs=calibInput, outputs=block3_output)
        lrate = LearningRateScheduler(step_decay)
        optimizer = keras.optimizers.rmsprop(lr=0.0)

        self.model.compile(optimizer=optimizer,
                           loss=lambda y_true,
                                       y_pred:
                           cf.MMD(block3_output, target, MMDTargetValidation_split=0.1).KerasCost(y_true, y_pred))
        K.get_session().run(tf.global_variables_initializer())
        source_labels = np.zeros(source.shape[0])
        self.model.fit(source, source_labels, nb_epoch=self.epochs, batch_size=self.batch_size,
                       validation_split=0.1, verbose=self.verbose,
                       callbacks=[lrate, mn.monitorMMD(source, target, self.model.predict),
                                  cb.EarlyStopping(monitor='val_loss', patience=50, mode='auto')])

        if evaluate:
            self.evaluate(source, target)

    def save_model(self, model_path, weights_path=None):
        self.model.save(model_path)
        if weights_path is not None:
            self.model.save_weights(weights_path)

    def load_model(self, path):
        self.model = load_model(path)

    def evaluate(self, source, target):
        source = source.copy()
        target = target.copy()
        calibrated_source = self.model.predict(source)

        # ----- PCA ----- #
        pca = decomposition.PCA()
        pca.fit(target)

        # project data onto PCs
        target_sample_pca = pd.DataFrame(pca.transform(target)[:, 0:2], columns=['PCA1', 'PCA2'])
        target_sample_pca['Label'] = 'Target'
        projection_before = pd.DataFrame(pca.transform(source)[:, 0:2], columns=['PCA1', 'PCA2'])
        projection_before['Label'] = 'Source before calibration'
        projection_after = pd.DataFrame(pca.transform(calibrated_source)[:, 0:2], columns=['PCA1', 'PCA2'])
        projection_after['Label'] = 'Source after calibration'

        fig, ax = plt.subplots(figsize=(8, 8))
        ax.scatter(target_sample_pca['PCA1'], target_sample_pca['PCA2'],
                   c='blue', s=4, alpha=0.4, label=f'Target')
        ax.scatter(projection_before['PCA1'], projection_before['PCA2'],
                   c='red', s=4, alpha=0.4, label=f'Before')
        ax.scatter(projection_after['PCA1'], projection_after['PCA2'],
                   c='green', s=4, alpha=0.4, label=f'After')
        ax.set_xlabel('PCA1')
        ax.set_ylabel('PCA2')
        ax.set_title('Performance of normalisation')
        ax.legend()
        plt.show()

        # ----- Histograms ----- #
        target['Label'] = 'Target'
        source['Label'] = 'Source '
        calibrated_source['Label'] = 'After'

        data = pd.concat([target, source, calibrated_source])
        data = pd.melt(data, id_vars=['Label'], var_name='Marker', value_name='MFI')
        g = sns.FacetGrid(data, col="Marker", col_wrap=2, height=3, aspect=2, hue='Label', sharey=False)
        g.map(sns.distplot, "MFI", hist=False, kde=True).add_legend()
        return g






