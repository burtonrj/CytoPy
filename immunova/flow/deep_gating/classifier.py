from keras.layers import Input, Dense
from keras.models import Model
import keras.optimizers
from keras.regularizers import l2
from keras import callbacks as cb
from keras.callbacks import LearningRateScheduler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, multilabel_confusion_matrix, \
    roc_auc_score, label_ranking_average_precision_score, label_ranking_loss
import pandas as pd
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


def predict_class(y_probs, threshold):
    """
    Returns the predicted class given the probabilities of each class. If threshold = None, the class with
    the highest probability is returned for each value in y, otherwise assumed to be multi-class prediction
    and converts output to one-hot-encoded multi-label output using the given threshold.
    :param y_probs:
    :param threshold:
    :return:
    """
    def convert_ml(y):
        if y > threshold:
            return 1
        return 0
    if threshold is not None:
        y_hat = list()
        for x in y_probs:
            y_hat.append(list(map(lambda i: convert_ml(i), x)))
        return y_hat
    return list(map(lambda j: np.argmax(j), y_probs))


def multi_label_performance(y, y_probs):
    return pd.DataFrame(dict(ranking_avg_precision=label_ranking_average_precision_score(y_true=y,
                                                                                         y_score=y_probs),
                             label_ranking_loss=label_ranking_loss(y_true=y, y_score=y_probs)))


def evaluate_model(classifier, x, y, multi_label, threshold=None):
    y_probs = classifier.predict(x)
    y_hat = predict_class(y_probs, threshold)
    if multi_label:
        return multi_label_performance(y, y_hat)
    return pd.DataFrame(dict(f1_score=f1_score(y_true=y, y_pred=y_hat, average='weighted'),
                             accuracy=accuracy_score(y_true=y, y_pred=y_hat),
                             precision=precision_score(y_true=y, y_pred=y_hat, average='weighted'),
                             recall=recall_score(y_true=y, y_pred=y_hat, average='weighted'),
                             auc_score=roc_auc_score(y_true=y, y_score=[x[i] for i, x in zip(y_hat, y_probs)])))

