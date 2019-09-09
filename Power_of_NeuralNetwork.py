""" Created on Mon Sep  9 14:36:35 2019 """

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math
import scipy
import random
import keras

from keras.models import Sequential
from keras.layers import Dense
from keras import optimizers

x = np.array([[0 , 0] , [0 , 1] , [1 , 0] , [1 , 1]])
y = np.array([0 , 1 , 1 , 0])

model = Sequential()

model.add(Dense(units = 2 , input_shape = (2 , ) , activation = "relu"))
model.add(Dense(units = 2 , activation = "relu" ))
model.add(Dense(units = 1 , activation = "sigmoid"))

# Compiling the model
sgd = optimizers.SGD(lr = .01 , momentum = .1)
model.compile(optimizer = sgd , loss = "binary_crossentropy" , metrics = ["accuracy"])

# Fitting the model
model.fit(x , y , epochs = 2000 , batch_size = 2 )

y_pred = model.predict(x)
