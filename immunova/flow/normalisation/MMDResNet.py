'''
Created on Dec 5, 2016

@author: urishaham
'''

import keras.optimizers
from keras.layers import Input, Dense, Activation, add
from keras.models import Model, load_model
from keras import callbacks as cb
import numpy as np
from keras.layers.normalization import BatchNormalization

import immunova.flow.normalisation.CostFunctions as cf
import immunova.flow.normalisation.Monitoring as mn
from keras.regularizers import l2
from sklearn import decomposition
from keras.callbacks import LearningRateScheduler
import math
import immunova.flow.normalisation.ScatterHist as sh
from keras import initializers
import sklearn.preprocessing as prep
import tensorflow as tf
import keras.backend as K


def create_block(x_input, layer_size, l2_penalty):
    input_dim = int(x_input.get_shape()[-1])
    block_bn1 = BatchNormalization()(x_input)
    block_a1 = Activation('relu')(block_bn1)
    block_w1 = Dense(layer_size, activation='linear',
                     kernel_regularizer=l2(l2_penalty),
                     kernel_initializer=initializers.RandomNormal(stddev=1e-4))(block_a1)
    block_bn2 = BatchNormalization()(block_w1)
    block_a2 = Activation('relu')(block_bn2)
    block_w2 = Dense(input_dim, activation='linear',
                     kernel_regularizer=l2(l2_penalty),
                     kernel_initializer=initializers.RandomNormal(stddev=1e-4))(block_a2)
    block_output = add([block_w2, x_input])
    return block_output


class MMDNet:
    def __init__(self, data_dim, epochs=500, denoise=False,
                 ae_keep_prob=.8, ae_latent_dim=25, ae_l2_penalty=1e-2,
                 layer_sizes=None, l2_penalty=1e-2):
        self.data_dim = data_dim
        self.epochs = epochs
        self.denoise = denoise
        self.ae_keep_prob = ae_keep_prob
        self.ae_latent_dim = ae_latent_dim
        self.ae_l2_penalty = ae_l2_penalty
        self.l2_penalty = l2_penalty
        self.layers = None
        self.net = None
        if layer_sizes is None:
            self.layer_sizes = [25, 25, 25]
        else:
            self.layer_sizes = layer_sizes

    def build_model(self):
        calib_input = Input(shape=(self.data_dim,))

        # create all layers of MMDNet
        layers = [calib_input]
        for layer_size in self.layer_sizes:
            layers.append(create_block(layers[-1], layer_size, self.l2_penalty))

        calibMMDNet = Model(inputs=calib_input, outputs=layers[-1])

        if self.denoise:
            input_cell = Input(shape=(self.data_dim,))
            encoded = Dense(self.ae_latent_dim, activation='relu',W_regularizer=l2(self.ae_l2_penalty))(input_cell)
            encoded1 = Dense(self.ae_latent_dim, activation='relu',W_regularizer=l2(self.ae_l2_penalty))(encoded)
            decoded = Dense(self.data_dim, activation='linear',W_regularizer=l2(self.ae_l2_penalty))(encoded1)
            autoencoder = Model(input=input_cell, output=decoded)
            autoencoder.compile(optimizer='rmsprop', loss='mse')
            self.ae = autoencoder

        self.layers = layers
        self.net = calibMMDNet

    def fit(self, source, target, initial_lr=1e-3, lr_decay=0.97):
        # preprocess data

        if self.denoise:
            # denoise with autoencoder
            numZerosOK = 1
            s_to_keep = np.sum(source == 0 , axis=1) <= numZerosOK
            t_to_keep = np.sum(target == 0, axis=1) <= numZerosOK

            ae_y = np.concatenate([source[s_to_keep], target[t_to_keep]], axis=0)
            np.random.shuffle(ae_y)
            ae_x = ae_y * np.random.binomial(n=1, p=self.ae_keep_prob, size=ae_y.shape)
            self.ae.fit(ae_x, ae_y, epochs=self.epochs, batch_size=128, shuffle=True,  validation_split=0.1,
                            callbacks=[mn.monitor(), cb.EarlyStopping(monitor='val_loss', patience=25,  mode='auto')])
            source = self.ae.predict(source)
            target = self.ae.predict(target)

        # rescale source to have zero mean and unit variance
        # apply same transformation to the target
        preprocessor = prep.StandardScaler().fit(source)
        source = preprocessor.transform(source)
        target = preprocessor.transform(target)

        # compile net
        step_decay = lambda epoch: initial_lr * math.pow(lr_decay, epoch)
        lrate = LearningRateScheduler(step_decay)

        optimizer = keras.optimizers.rmsprop(lr=0.0)

        self.net.compile(optimizer=optimizer, loss=lambda y_true,y_pred:
                       cf.MMD(self.layers[-1], target, MMDTargetValidation_split=0.1).KerasCost(y_true, y_pred))

        # initialize all variables
        K.get_session().run(tf.global_variables_initializer())

        # train model
        sourceLabels = np.zeros(source.shape[0])
        self.net.fit(source, sourceLabels, epochs=self.epochs, batch_size=1000,validation_split=0.1, verbose=1,
                   callbacks=[lrate, mn.monitorMMD(source, target, self.net.predict),
                              cb.EarlyStopping(monitor='val_loss',patience=50,mode='auto')])

    def save_model(self, model_path, weights_path=None):
        self.net.save(model_path)
        if weights_path is not None:
            self.net.save_weights(weights_path)

    def load_model(self, path):
        self.net = load_model(path)

    def evaluate(self, source, target):
        calibratedSource = self.net.predict(source)

        #
        # qualitative evaluation: PCA
        #
        pca = decomposition.PCA()
        pca.fit(target)

        # project data onto PCs
        target_sample_pca = pca.transform(target)
        projection_before = pca.transform(source)
        projection_after = pca.transform(calibratedSource)

        # choose PCs to plot
        pc1 = 0
        pc2 = 1
        axis1 = 'PC'+str(pc1)
        axis2 = 'PC'+str(pc2)
        sh.scatterHist(target_sample_pca[:,pc1], target_sample_pca[:,pc2], projection_before[:,pc1], projection_before[:,pc2], axis1, axis2)
        sh.scatterHist(target_sample_pca[:,pc1], target_sample_pca[:,pc2], projection_after[:,pc1], projection_after[:,pc2], axis1, axis2)

