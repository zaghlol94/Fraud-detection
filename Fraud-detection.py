#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 23 15:50:22 2017

@author: zaghlol
"""

#import data and lib 
import numpy as np

import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Credit_Card_Applications.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# Feature Scaling
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (0, 1))
X = sc.fit_transform(X)

# Training the SOM
from minisom import MiniSom
som = MiniSom(x = 10, y = 10, input_len = 15, sigma = 1.0, learning_rate = 0.5)
som.random_weights_init(X)
som.train_random(data = X, num_iteration = 100)

# Visualizing the results
from pylab import bone, pcolor, colorbar, plot, show
bone()
pcolor(som.distance_map().T)
colorbar()
markers = ['o', 's']
colors = ['r', 'g']
for i, x in enumerate(X):
    w = som.winner(x)
    plot(w[0] + 0.5,
         w[1] + 0.5,
         markers[y[i]],
         markeredgecolor = colors[y[i]],
         markerfacecolor = 'None',
         markersize = 10,
         markeredgewidth = 2)
show()

# Finding the frauds
mappings = som.win_map(X)
frauds = np.concatenate((mappings[(4,3)], mappings[(3,3)]), axis = 0)
frauds = sc.inverse_transform(frauds)

#from unsupervised to supervised
customers = dataset.iloc[:, 1:].values

is_fraud=np.zeros(len(dataset))

for i in range(len(dataset)):
    if dataset.iloc[i,0] in frauds:
        is_fraud[i]=1


#feat scaling 
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
customers = sc.fit_transform(customers)


#ANN
import keras 
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout

classifier=Sequential()

classifier.add(Dense(output_dim=2,init='uniform',activation='relu',input_dim=15))

classifier.add(Dense(output_dim=1,init='uniform',activation='sigmoid'))

classifier.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])

classifier.fit(customers,is_fraud,batch_size=1,nb_epoch=2)

Ypred=classifier.predict(customers)
Ypred=np.concatenate((dataset.iloc[:,0:1],Ypred),axis=1)
Ypred=Ypred[Ypred[:,1].argsort()]