# -*- coding: utf-8 -*-
"""
Created on Fri Jun  3 20:30:10 2022

@author: user
"""


#import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from tensorflow import keras
from tensorflow.keras import layers
import keras_tuner
from kerastuner.tuners import RandomSearch
df = pd.read_csv("Notebooks/fina_data.csv")

# Split data into training and test

train_data=df.iloc[:7188,:]
test_data=df.iloc[7188:,:]
len(test_data)


#split data
from sklearn.model_selection import train_test_split
from sklearn.model_selection import train_test_split
X=train_data.drop(columns="Amount",axis=1)
Y=train_data["Amount"]
x_train,x_test,y_train,y_test = train_test_split(X,Y,test_size = 0.2,random_state =1)


from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import GridSearchCV

from keras.models import Sequential
from keras.layers import Dense,Activation,Embedding,Flatten,LeakyReLU,BatchNormalization,Dropout
from keras.activations import relu,sigmoid


def create_model(layers,activation):
    model=Sequential()
    for i,nodes in enumerate(layers):
        if i==0:
            model.add(Dense(nodes,input_dim=x_train.shape[1]))
            model.add(Activation(activation))
            model.add(Dropout(0.3))
        else:
            model.add(Dense(nodes))
            model.add(Activation(activation))
            model.add(Dropout(0.3))
        model.add(Dense(1,kernel_initializer="glorot_uniform",activation="linear"))
    
        model.compile(optimizer="Adam",loss="mean_squared_error",metrics=["mean_squared_error"])
    return model
    
    
    
        
model=KerasRegressor(build_fn=create_model,verbose=0)

layers=[[20],[40,20],[60,45,30,15]]
activations=["relu"]
param_grid=dict(layers=layers,activation=activations,batch_size=[32,64],epochs=[50])
grid=GridSearchCV(estimator=model,param_grid=param_grid,cv=10,n_jobs = -1,verbose=0)


grid_result=grid.fit(x_train,y_train)


pred_y=grid.predict(x_test)
preds = np.exp(pred_y)
preds

y = np.exp(y_test)
y
from sklearn.metrics import mean_squared_error
mean_squared_error(y,preds,squared = False)

