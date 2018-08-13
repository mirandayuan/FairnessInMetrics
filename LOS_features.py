#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 21 21:29:48 2018

@author: mirandayuan
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import pickle
import json

pkl_data = 'data.pkl'
feature = 'model_training_ADMIT.json'

#read json file
def read(name):
    with open(name, 'r') as file:
        return file.read()
    
feature_p = json.loads(read(feature))
features = feature_p['features']

### Read the dataset into a dataframe and map the labels to numbers
df = pd.read_csv('mimic_admit_03052018.csv')


for i in range(len(df)):
    if df.loc[i, 'sex'] == 'male':
        df.loc[i, 'issexMale'] = 1
        df.loc[i, 'issexFemale'] = 0
    elif df.loc[i, 'sex'] == 'female':
        df.loc[i, 'issexMale'] = 0
        df.loc[i, 'issexFemale'] = 1
    
    df.loc[i, 'isethnicityWhite'] = 0
    df.loc[i, 'isethnicityBlackOrAfricanAmerican'] = 0
    df.loc[i, 'isethnicityHispanicOrLatino'] = 0
    df.loc[i, 'isethnicityAsian'] = 0
    if df.loc[i, 'ethnicity'].startswith('WHITE'):
        df.loc[i, 'isethnicityWhite'] = 1
    elif df.loc[i, 'ethnicity'] == 'BLACK/AFRICAN AMERICAN':
        df.loc[i, 'isethnicityBlackOrAfricanAmerican'] = 1
    elif df.loc[i, 'ethnicity'] == 'HISPANIC OR LATINO':
        df.loc[i, 'isethnicityHispanicOrLatino'] = 1
    elif df.loc[i, 'ethnicity'] == 'ASIAN':
        df.loc[i, 'isethnicityAsian'] = 1
        
    df.loc[i, 'isadmitSourceEMD'] = 0
    df.loc[i, 'isadmitSourceMP'] = 0
    df.loc[i, 'isadmitSourceOther'] = 1
    df.loc[i, 'isadmitSourceHospTrans'] = 0
    if df.loc[i, 'admitSource'] == 'hosp-trans':
        df.loc[i, 'isadmitSourceHospTrans'] = 1
        df.loc[i, 'isadmitSourceOther'] = 0
    elif df.loc[i, 'admitSource'] == 'emd':
        df.loc[i, 'isadmitSourceEMD'] = 1
        df.loc[i, 'isadmitSourceOther'] = 0
    elif df.loc[i, 'admitSource'] == 'mp':
        df.loc[i, 'isadmitSourceMP'] = 1
        df.loc[i, 'isadmitSourceOther'] = 0
    
    
    
    df.loc[i, 'ismaritalStatusMarried'] = 0
    df.loc[i, 'ismaritalStatusNeverMarried'] = 0
    df.loc[i, 'ismaritalStatusWidowed'] = 0
    df.loc[i, 'ismaritalStatusDivorced'] = 0
    if df.loc[i, 'maritalStatus'] == 'married':
        df.loc[i, 'ismaritalStatusMarried'] = 1
    elif df.loc[i, 'maritalStatus'] == 'never-married':
        df.loc[i, 'ismaritalStatusNeverMarried'] = 1
    elif df.loc[i, 'maritalStatus'] == 'widowed':
        df.loc[i, 'ismaritalStatusWidowed'] = 1
    elif df.loc[i, 'maritalStatus'] == 'divorced': 
        df.loc[i, 'ismaritalStatusDivorced'] = 1
        

    
### convert string/boolean to int & delete the features whose value is missing
features_new = features[:]
for name in features_new:
    if name in df.columns:
        if isinstance(df.loc[0, name], str) or isinstance(df.loc[0, name], bool) or isinstance(df.loc[0, name], np.bool):
            #print('\n feature: ',name)
            i = 0
            for ele in set(df[name]):
                #print(ele)
                #print(i)
                df.replace(ele, i, inplace = True)
                i += 1
    else:        
        features.remove(name)
        
y = df["actualLOS"]
X = df[features]

###binarize actualLOS
y_new = y[:]
print(len(y_new))
X_age = X.loc[:,'age']
print(len(X_age))
count = [0,0,0,0,0,0]
for i in range(len(y_new)):
    if y_new[i] > 20:
        y.drop(i, inplace = True)
        X.drop(X.index[i], inplace = True)
    else:
        if y_new[i] >= 4:
            #y.loc[i] = 1
            if X_age[i] < 41:
                y.loc[i] = 0
                count[0] += 1
            elif X_age[i] < 61:    
                y.loc[i] = 2
                count[2] += 1
            else:
                y.loc[i] = 4
                count[4] += 1
        else:
            #y.loc[i] = 0
            if X_age[i] < 41:
                y.loc[i] = 1
                count[1] += 1
            elif X_age[i] < 61:    
                y.loc[i] = 3
                count[3] += 1
            else:
                y.loc[i] = 5
                count[5] += 1
 
print(count)


### divide to training and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0, stratify = y)
X_train.reset_index(drop=True, inplace=True)
y_train.reset_index(drop=True, inplace=True)
X_test.reset_index(drop=True, inplace=True)
y_test.reset_index(drop=True, inplace=True)

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

count_test = [0,0,0,0,0,0]
for i in range(len(y_test)):
    if y_test[i] == 1:
        if X_test.loc[i,'age'] < 41:
            count_test[0] += 1
        elif X_test.loc[i,'age'] < 61:    
            count_test[2] += 1
        else:
            count_test[4] += 1
    else:
        if X_test.loc[i,'age'] < 41:
            count_test[1] += 1
        elif X_test.loc[i,'age'] < 61:    
            count_test[3] += 1
        else:
            count_test[5] += 1

print(count_test)

### save -> pickle file
data_objects = (X_train, X_test, y_train, y_test)

with open(pkl_data, 'wb') as file:  
    pickle.dump(data_objects, file)
