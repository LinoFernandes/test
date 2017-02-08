#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 30 00:06:42 2017

@author: Lino
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

np.set_printoptions(threshold=np.nan) # Show complete array

dataset = pd.read_csv('~/Documents/MLUdemy/Data.csv')
X = dataset.iloc[:, :-1].values
Y = dataset.iloc[:, 3].values

# Missing Data
""""from sklearn.preprocessing import Imputer

imputer = Imputer(missing_values='NaN',strategy='mean',axis=0)
imputer = imputer.fit(X[:, 1:3])
X[:,1:3] = imputer.transform(X[:, 1:3])"""

#Categorical DAta

"""from sklearn.preprocessing import LabelEncoder
labelencoder_X = LabelEncoder()
X[:, 0] = labelencoder_X.fit_transform(X[:, 0])
labelencoder_Y = LabelEncoder()
Y = labelencoder_Y.fit_transform(Y)"""
# Problem with this: Creates relational order in the variables. I.E the ML mode
# will think France is greater than Germany (or wtv number it gave the variables).
# Solution, create dummy variables (Note: For variable Y that is not a problem,
# because Y will be given to the model as the dependent variable):

"""from sklearn.preprocessing import OneHotEncoder
onehotencoder = OneHotEncoder(categorical_features = [0])
X = onehotencoder.fit_transform(X).toarray()"""

# Training and Test Set

from sklearn.cross_validation import train_test_split
X_Train, X_Test, Y_Train, Y_Test = train_test_split(X, Y, test_size=0.2, random_state=0) 
#random_state->seed

# Feature Scaling
"""
from sklearn.preprocessing import StandardScaler

sc_X = StandardScaler()
X_Train = sc_X.fit_transform(X_Train)
X_Test = sc_X.transform(X_Test)"""

# Note: In train we use .fit_transform and in test we only use .transform. Why?
# IMPORTANT: Because it was already fitted before (sc_X is the same). ALWAYS FIT TRAIN
# SET FIRST.
# Note: Here we are also scaling the dummy variables. Usually for the sake of interpretability
# the dummy variables are not scaled (but the model has less accuracy that way). In the 
# tutorial they were not interested in interpretability so the scaled them for simplicity.
# For classification (unlike regression) we don't need to scale Y.

                     
