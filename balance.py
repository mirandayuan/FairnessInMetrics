#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  2 22:47:56 2018

@author: mirandayuan
"""
import pickle
from sklearn.utils import resample

pkl_data = 'data_race.pkl'
sample = 'sample_race.pkl'
num = 20000

### Resample
def resam(sample_list, X_train, y_train):
    index = []
    summ = []
    total = 0
    sample_num = []
    for i in range(0, len(sample_list)):
        total += len(sample_list[i])
        if i % 2 == 0:
            summ.extend([len(sample_list[i]) + len(sample_list[i + 1])] * 2)
        sample_num.append(len(sample_list[i]))
    
    print(sample_num)
    sample_num = list(map(lambda x : int(x[0] / x[1] * num), zip(sample_num, summ)))
    print(sample_num)
    
    for i in range(0, len(sample_num)):
        index.extend(resample(sample_list[i], n_samples = sample_num[i], random_state = 42 ))
    
    X_train_new = []
    y_train_new = []
    for train in index:
        X_train_new.append(X_train.loc[train])
        y_train_new.append(y_train[train])
    return (X_train_new, y_train_new)


### divde training dataset 
def train_div(y_train, race_train, index, start):
    if y_train[index] == 1:
        race_train[start].append(index)
    else:
        race_train[start].append(index)


### load dataset
with open(pkl_data, 'rb') as file:  
    X_train, X_test, y_train, y_test= pickle.load(file)
    
### bootstrap -- balance
'''
### ---divide to different groups based on age 
age_train = [[] for i in range(6)]
for i in range(0, len(X_train)):
    if X_train.loc[i, 'age'] < 41:
        train_div(y_train, age_train, i, 0)
    elif X_train.loc[i, 'age'] < 61:
        train_div(y_train, age_train, i, 3)
    else:
        train_div(y_train, age_train, i, 4)
'''

'''

### ---divide to different groups based on gender 
gender_train = [[] for i in range(4)]
for i in range(0, len(X_train)):
    if X_train.loc[i, 'issexMale'] == 1:
        train_div(y_train, gender_train, i, 0)
    elif X_train.loc[i, 'issexFemale'] == 1:
        train_div(y_train, gender_train, i, 2)

'''


###race: white, black/africanAmerican, hispanic/latino, Asian, other 
race_train = [[] for i in range(10)]
for i in range(0, len(X_train)):
    race_name = {'isethnicityWhite': 0, 'isethnicityBlackOrAfricanAmerican': 2, 'isethnicityHispanicOrLatino': 4, 'isethnicityAsian': 6}
    flag = 0
    for name, index in race_name.items():
        if X_train.loc[i, name] == 1:
            train_div(y_train, race_train, i, index)
            flag = 1
    if flag == 0:
        train_div(y_train, race_train, i, 8)


(X_train_new, y_train_new) = resam(race_train, X_train, y_train)

data_objects_new = (X_train_new, X_test, y_train_new, y_test)
with open(sample, 'wb') as file:  
    pickle.dump(data_objects_new, file)
