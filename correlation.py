#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 22 22:06:01 2018

@author: mirandayuan
"""

import pandas as pd
import pickle
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import r2_score
import numpy as np

pkl_data = 'data_pre_0.pkl'

###load dataset
with open(pkl_data, 'rb') as file:  
    features, df = pickle.load(file)


#calculate correlation
def get_corr(features, name, model):
    ### divide to X and y
    X = df[features]
    y = pd.factorize(df[name].values)[0].reshape(-1, 1)
    
    ### divide to training and test
    #from sklearn.model_selection import train_test_split
    #X_train, X_test, y_train, y_test = train_test_split(X, y_actual, test_size=0.2, random_state=42)
    
    r2_actual = {}

    for name in X.columns:
        X_name = X[name].values.reshape(-1, 1)
        if model == 'logreg':
            logreg = LogisticRegression(solver='sag', random_state=42)
            logreg.fit(X_name, y)
            y_pred = logreg.predict(X_name)
        elif model == 'linear':
            lin = LinearRegression()
            lin.fit(X_name, y)
            y_pred = lin.predict(X_name)
        r2_actual[name] = abs(r2_score(y, y_pred))
    
    actual_sort = [(k, r2_actual[k]) for k in sorted(r2_actual, key=r2_actual.get, reverse=True)]
    for k, v in actual_sort[:10]:
        print(k, v,'\n')
    
    return actual_sort


### add sex and ethnicity to features
features.extend(['sex', 'ethnicity'])

print('correlation with actual LOS')
actual_sort = get_corr(features, 'actualLOS', 'linear')

print('correlation with binarized LOS')
actual_sort = get_corr(features, 'bin_actualLOS', 'logreg')

print('correlation with feature: age')
features.remove('age')
actual_sort = get_corr(features, 'age', 'linear')

print('correlation with feature: sex')
features = [ele for ele in features if ele not in ('sex', 'issexMale', 'issexFemale')]
features.append('age')
actual_sort = get_corr(features, 'sex', 'logreg')

print('correlation with feature: race')
features = [ele for ele in features if ele not in ('ethnicity', 'isethnicityWhite', 'isethnicityBlackOrAfricanAmerican', 'isethnicityHispanicOrLatino', 'isethnicityAsian')]
features.extend(['sex', 'issexMale', 'issexFemale'])
actual_sort = get_corr(features, 'ethnicity', 'logreg')
