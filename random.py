#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 21 21:38:08 2018

@author: mirandayuan
"""
from sklearn.ensemble import RandomForestClassifier
import pickle

pkl_data = 'data.pkl'
pkl_model = 'model.pkl'
sample = 'sample.pkl'

###load dataset
with open(sample, 'rb') as file:  
    X_train, X_test, y_train, y_test= pickle.load(file)

###random forest
rf = RandomForestClassifier( n_estimators = 1000, oob_score = True, random_state = 0)
rf.fit(X_train,y_train)
prepro = rf.predict_proba(X_test)
pre = rf.predict(X_test)

###save -> pickle file
rf_objects = (rf, X_test, y_test, pre, prepro)
with open(pkl_model, 'wb') as file:  
    pickle.dump(rf_objects, file)