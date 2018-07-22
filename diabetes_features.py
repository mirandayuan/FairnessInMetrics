#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 21 21:29:48 2018

@author: mirandayuan
"""

import pandas as pd
from sklearn.utils import resample
import json
import random

Data = 'diabetes.json'

###read and save json file
def read(name):
    with open(name, 'r') as file:
        return file.read()

def save(data, name, typ):
	with open(name, typ) as file:
		file.write(data)

### Read the dataset into a dataframe and map the labels to numbers
df = pd.read_csv('diabetes.csv')


### bootstrap
### ---divide to different groups
age2y = []
age2n = []
age4y = []
age4n = []
age6y = []
age6n = []
for i in range(0,len(df)):
    if df.iloc[i]['Age'] < 36:
        if df.iloc[i]['Outcome'] == 1:
            age2y.append(i)
        else:
            age2n.append(i)
    elif df.iloc[i]['Age'] < 51:
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
sample_num = [66, 184, 131, 119, 117, 133]
sample_list = [age2y, age2n, age4y, age4n, age6y, age6n]
for i in range(len(sample_num)):
    index.append(resample(sample_list[i], n_samples = sample_num[i]))

### Separate the input features from the label
attributes = df.columns.tolist()
attributes.remove("Outcome")
X = df[attributes]
y = df["Outcome"]

###Separate to training and test
index_train = []
index_test = []
sample_train = [int(i * 0.8) for i in sample_num]
for i in range(len(index)):
    random.shuffle(index[i])
    index_train.extend(index[i][:sample_train[i]])
    index_test.extend(index[i][sample_train[i]:])

diabetes = {}
diabetes['X_train'] = []
diabetes['X_test'] = []
diabetes['y_train'] = []
diabetes['y_test'] = []
for train in index_train:
    diabetes['X_train'].append(X.iloc[train].tolist())
    diabetes['y_train'].append(int(y.iloc[train]))
for test in index_test:
    diabetes['X_test'].append(X.iloc[test].tolist())
    diabetes['y_test'].append(int(y.iloc[test]))



### save -> json file
save(json.dumps(diabetes), Data, 'w')
