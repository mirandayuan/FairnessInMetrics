#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 21 21:38:08 2018

@author: mirandayuan
"""
from sklearn.ensemble import GradientBoostingClassifier
import pickle

pkl_data = 'data.pkl'
pkl_model = 'model.pkl'

###load dataset
with open(pkl_data, 'rb') as file:  
    X_train, X_test, y_train, y_test= pickle.load(file)

###random forest
gbm = GradientBoostingClassifier()
gbm.fit(X_train,y_train)
prepro = gbm.predict_proba(X_test)
pre = gbm.predict(X_test)

###save -> pickle file
gbm_objects = (gbm, X_test, y_test, pre, prepro)
with open(pkl_model, 'wb') as file:  
    pickle.dump(gbm_objects, file)