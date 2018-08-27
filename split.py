#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 22 20:14:17 2018

@author: mirandayuan
"""
from sklearn.model_selection import train_test_split
import pickle

pkl_data = 'data_pre.pkl'
data_do = 'data_gender.pkl'


###load dataset
with open(pkl_data, 'rb') as file:  
    features, df = pickle.load(file)


### divide to X and y
X = df[features]
###gender  
y = df["sex_target"]
###race
#y = df["race_target"]
###age
#y = df["age_target"]

### divide to training and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0, stratify=y)
X_train.reset_index(drop=True, inplace=True)
y_train.reset_index(drop=True, inplace=True)
X_test.reset_index(drop=True, inplace=True)
y_test.reset_index(drop=True, inplace=True)

### size of different group
print('training dataset: ', y_train.value_counts())
print('test dataset: ', y_test.value_counts())


### binarize target
y_train_ = y_train[:]
for i in range(len(y_train_)):
    if y_train_[i] % 2 == 0:
        y_train.loc[i] = 1
    else:
        y_train.loc[i] = 0

y_test_ = y_test[:]
for i in range(len(y_test_)):
    if y_test_[i] % 2 == 0:
        y_test.loc[i] = 1
    else:
        y_test.loc[i] = 0


### save -> pickle file
data_objects = (X_train, X_test, y_train, y_test)

with open(data_do, 'wb') as file:
    pickle.dump(data_objects, file)
