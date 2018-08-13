#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  2 22:47:56 2018

@author: mirandayuan
"""
import pickle
from sklearn.utils import resample

pkl_data = 'data.pkl'
sample = 'sample.pkl'
num = 30000

### Resample
def resam(sample_list, X_train, y_train):
    index = []
    summ = []
    total = 0
    sample_num = []
    for i in range(0, len(sample_list)):
        total += len(sample_list[i])
        if i % 2 == 0:
            summ.append(len(sample_list[i]) + len(sample_list[i + 1]))
        else:
            summ.append(len(sample_list[i]) + len(sample_list[i - 1]))
        sample_num.append(len(sample_list[i]))
    
    print(sample_num)
    sample_num = list(map(lambda x : int(x[0] / x[1] * num), zip(sample_num, summ)))
    print(sample_num)
    
    for i in range(0, len(sample_num)):
        index.extend(resample(sample_list[i], n_samples = sample_num[i], random_state = 1))
    
    X_train_new = []
    y_train_new = []
    for train in index:
        X_train_new.append(X_train.loc[train])
        y_train_new.append(y_train[train])
    return (X_train_new, y_train_new)


###load dataset
with open(pkl_data, 'rb') as file:  
    X_train, X_test, y_train, y_test= pickle.load(file)
    
### bootstrap -- balance

### ---divide to different groups based on age 20000*3
age_train = [[],[],[],[],[],[]]
for i in range(0, len(X_train)):
    if X_train.loc[i, 'age'] < 41:
        if y_train[i] == 1:
            age_train[0].append(i)
        else:
            age_train[1].append(i)
    elif X_train.loc[i, 'age'] < 61:
        if y_train[i] == 1:
            age_train[2].append(i)
        else:
            age_train[3].append(i)
    else:
        if y_train[i] == 1:
            age_train[4].append(i)
        else:
            age_train[5].append(i)



'''
### ---divide to different groups based on gender 30000*2
gender_train = [[],[],[],[]]
for i in range(0, len(X_train)):
    if X_train.loc[i, 'issexMale'] == 1:
        if y_train[i] == 1:
            gender_train[0].append(i)
        else:
            gender_train[1].append(i)
    elif X_train.loc[i, 'issexFemale'] == 1:
        if y_train[i] == 1:
            gender_train[2].append(i)
        else:
            gender_train[3].append(i)


'''
'''
###race: white, black/africanAmerican, hispanic/latino, Asian, other 12000*5
race_train = [[],[],[],[],[],[],[],[],[],[]]
for i in range(0, len(X_train)):
    if X_train.loc[i, 'isethnicityWhite'] == 1:
        if y_train[i] == 1:
            race_train[0].append(i)
        else:
            race_train[1].append(i)
    elif X_train.loc[i, 'isethnicityBlackOrAfricanAmerican'] == 1:
        if y_train[i] == 1:
            race_train[2].append(i)
        else:
            race_train[3].append(i)
    elif X_train.loc[i, 'isethnicityHispanicOrLatino'] == 1:
        if y_train[i] == 1:
            race_train[4].append(i)
        else:
            race_train[5].append(i)
    elif X_train.loc[i, 'isethnicityAsian'] == 1:
        if y_train[i] == 1:
            race_train[6].append(i)
        else:
            race_train[7].append(i)
    else:
        if y_train[i] == 1:
            race_train[8].append(i)
        else:
            race_train[9].append(i)
'''


(X_train_new, y_train_new) = resam(age_train, X_train, y_train)

data_objects_new = (X_train_new, X_test, y_train_new, y_test)
with open(sample, 'wb') as file:  
    pickle.dump(data_objects_new, file)