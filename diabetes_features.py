#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 21 21:29:48 2018

@author: mirandayuan
"""

import pandas as pd
from sklearn.utils import resample
from sklearn.model_selection import train_test_split
import pickle

pkl_data = 'data.pkl'



### Read the dataset into a dataframe and map the labels to numbers
df = pd.read_csv('diabetes.csv')
### Separate the input features from the label
attributes = df.columns.tolist()
attributes.remove("Outcome")
X = df[attributes]
y = df["Outcome"]
### divide to training and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)


### bootstrap
### ---divide to different groups
age2y = []
age2n = []
age4y = []
age4n = []
age6y = []
age6n = []
for i in range(0,len(X_train)):
    if X_train.iloc[i]['Age'] < 36:
        if df.iloc[i]['Outcome'] == 1:
            age2y.append(i)
        else:
            age2n.append(i)
    elif X_train.iloc[i]['Age'] < 51:
        if df.iloc[i]['Outcome'] == 1:
            age4y.append(i)
        else:
            age4n.append(i)
    else:
        if df.iloc[i]['Outcome'] == 1:
            age6y.append(i)
        else:
            age6n.append(i)
"""         
print(len(age2y))
print(len(age2n))
print(len(age4y))
print(len(age4n))
print(len(age6y))
print(len(age6n))

"""
### ---resample
index = []
sample_num = [91, 159, 71, 179, 98, 152]
sample_list = [age2y, age2n, age4y, age4n, age6y, age6n]
for i in range(len(sample_num)):
    index.extend(resample(sample_list[i], n_samples = sample_num[i]))

X_train_s = []
y_train_s = []
for train in index:
    X_train_s.append(X_train.iloc[train].tolist())
    y_train_s.append(y_train.iloc[train].tolist())

### save -> pickle file
data_objects = (X_train, X_test.values.tolist(), y_train, y_test.tolist())
with open(pkl_data, 'wb') as file:  
    pickle.dump(data_objects, file)

