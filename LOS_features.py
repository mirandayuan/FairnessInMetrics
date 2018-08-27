#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 21 21:29:48 2018

@author: mirandayuan
"""
import numpy as np
import pandas as pd
import pickle
import json

pkl_data = 'data_pre_0.pkl'
feature = 'model_training_ADMIT.json'


# read json file
def read(name):
    with open(name, 'r') as file:
        return file.read()

# divide based on target value
def div_tar(df_target, index, name, start, df):
    if df_target[index] >= 4:
        df.loc[index, name] = start
    else:
        df.loc[index, name] = start + 1


# load features needed
feature = json.loads(read(feature))
features = feature['features']

### Read the dataset into a dataframe and map the labels to numbers
df = pd.read_csv('mimic_admit_03052018.csv')

### Standardize         
df_sex = df['sex'].copy()
df_age = df['age'].copy()
df_race = df['ethnicity'].copy()
df_target = df['actualLOS'].copy()

for i in range(len(df_target)):
    if df_target[i] > 20:
        df.drop(i, inplace=True)
        continue
    
    ### binarize target
    if df_target[i] > 4:
        df.loc[i, 'bin_actualLOS'] = 1
    else:
        df.loc[i, 'bin_actualLOS'] = 0

    ### gender: male, female
    gender = {'male': ['issexMale', 0], 'female': ['issexFemale', 2]}
    for key, value in gender.items():
        df.loc[i, value[0]] = 0
        if df_sex[i] == key:
            df.loc[i, 'sex'] = value[1] // 2
            df.loc[i, value[0]] = 1
            div_tar(df_target, i, 'sex_target', value[1], df)                         


    ### ethnicity: white, black/african american, hipanic/latino, asian, other
    ethnicity = {'WHITE': ['isethnicityWhite', 0], 'BLACK/AFRICAN AMERICAN': ['isethnicityBlackOrAfricanAmerican',2], 'HISPANIC OR LATINO': ['isethnicityHispanicOrLatino', 4], 'ASIAN': ['isethnicityAsian', 6]}
    df.loc[i, 'ethnicity'] = 4
    for key, value in ethnicity.items():
        df.loc[i, value[0]] = 0
        if df_race[i].startswith(key):
            df.loc[i, 'ethnicity'] = value[1] // 2
            df.loc[i, value[0]] = 1
            div_tar(df_target, i, 'race_target', value[1], df)           
#    if isinstance(df.loc[i, 'ethnicity'], str):
        

    ### admit source: emd, mp, other, hosp trans    
    admit = {'hosp-trans': 'isadmitSourceHospTrans', 'emd': 'isadmitSourceEMD', 'mp': 'isadmitSourceMP'}
    for key, value in admit.items():
        df.loc[i, value] = 0
        df.loc[i, 'isadmitSourceOther'] = 1
        if df.loc[i, 'admitSource'] == key:
            df.loc[i, value] = 1
            df.loc[i, 'isadmitSourceOther'] = 0


    ### marital status: married, never married, widowed, divorced
    marital = {'married': 'ismaritalStatusMarried', 'never-married': 'ismaritalStatusNeverMarried', 'widowed':  'ismaritalStatusWidowed', 'divorced': 'ismaritalStatusDivorced'}
    for key, value in marital.items():
        df.loc[i, value] = 0
        if df.loc[i, 'maritalStatus'] == key:
            df.loc[i, value] = 1


    ### age: xx-40, 41-60, 61-xx
    if df_age[i] < 41:
        div_tar(df_target, i, 'age_target', 0, df)
    elif df_age[i] < 61:
        div_tar(df_target, i, 'age_target', 2, df)
    elif df_age[i] > 60:
        div_tar(df_target, i, 'age_target', 4, df)


### convert string/boolean to int & delete the features whose value is missing
features_new = features[:]
for name in features_new:
    if name in df.columns:
        if isinstance(df.loc[0, name], bool) or isinstance(df.loc[0, name], np.bool_) or isinstance(df.loc[0, name], str):
            # print('\n feature: ',name)
            i = 0
            for ele in set(df[name]):
                df[name].replace(ele, int(i), inplace=True)
                i += 1
        if df.loc[:,name].isnull().any():
            #df.loc[:,name].fillna(df.loc[:,name].median(), inplace = True)
            df.loc[:,name].fillna(0, inplace = True)

    else:
        features.remove(name)
        # print(name)


### save -> pickle file
data_objects = (features, df)

with open(pkl_data, 'wb') as file:
    pickle.dump(data_objects, file)