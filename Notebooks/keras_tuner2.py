# -*- coding: utf-8 -*-
"""
Created on Fri Jun  3 21:14:43 2022

@author: user
"""

import tensorflow as tf
import keras
from  keras.layers import Dense
from keras.models import Sequential
from keras.layers import Dropout
from keras.callbacks import LearningRateScheduler
import numpy as np
from matplotlib import pyplot as plt
from sklearn.model_selection import GridSearchCV
from keras.constraints import maxnorm
from numpy.random import seed
print(tf.__version__)
print(tf.keras.__version__)
np.random.seed(42)
# define the model
model = Sequential()
model.add(Dense(10, activation="relu", input_shape = (n_features,)))
model.add(Dense(10,activation="relu"))


