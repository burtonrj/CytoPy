from keras.layers import Input, Dense
from keras.models import Model
import keras.optimizers
from keras.regularizers import l2
from keras import callbacks as cb
from keras.callbacks import LearningRateScheduler
from sklearn.metrics import confusion_matrix, accuracy_score, roc_auc_score, precision_score, recall_score
import numpy as np
import math


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

    encoder = Model(input=input_layer, output=output_layer)
    net = Model(input=input_layer, output=output_layer)
    lrate = LearningRateScheduler(step_decay)
    optimizer = keras.optimizers.rmsprop(lr=0.0)

    net.compile(optimizer=optimizer, loss=loss)
    net.fit(train_x, train_y, nb_epoch=80, batch_size=128, shuffle=True,
            validation_split=0.1,
            callbacks=[lrate, cb.EarlyStopping(monitor='val_loss',
                                               patience=25, mode='auto')])
    return net


def f1score(confusionMatrix):
    '''
    Calculate the F1 score of a given confusion matrix.
    '''
    col1 = confusionMatrix[1:, :1]
    confusionMatrix = confusionMatrix[1:, 1:]
    temp = np.zeros(confusionMatrix.shape)

    for i in range(0, confusionMatrix.shape[0]):
        for j in range(0, confusionMatrix.shape[1]):
            if col1[i, 0] > 0:
                temp[i, j] = np.random.randint(0, col1[i, 0])
                col1[i, 0] = col1[i, 0] - temp[i, j]

    confusionMatrix = confusionMatrix + temp
    confusionMatrix = confusionMatrix.astype(int)

    sum_C = np.sum(confusionMatrix, axis=1)  # sum of each row
    sum_K = np.sum(confusionMatrix, axis=0)  # sum of each column

    Pr = np.divide(confusionMatrix, np.matlib.repmat(np.array([sum_C]).T, 1,
                                                     confusionMatrix.shape[0]))
    Re = np.divide(confusionMatrix,
                   np.matlib.repmat(sum_K, confusionMatrix.shape[1], 1))

    F = np.divide(2 * np.multiply(Pr, Re), Pr + Re)

    for i in range(0, F.shape[0]):
        for j in range(0, F.shape[1]):
            if np.isnan(F[i, j]):
                F[i, j] = 0

    F = np.max(F, axis=1)
    return np.dot(sum_C, F) / np.sum(sum_C)


def evaluate_model(classifier, x, y):
    y_hat = classifier.predict(x)
    conf_matrix = confusion_matrix(y_true=y, y_pred=y_hat)

