from keras import callbacks as cb
from keras.callbacks import LearningRateScheduler
from keras import initializers
from keras.layers import Input, Dense, merge, Dropout, Activation
from keras.layers.normalization import BatchNormalization
from keras.models import Model
import keras.optimizers as opt
from keras.regularizers import l2
import math
import matplotlib.pyplot as plt
import numpy as np
import os.path

