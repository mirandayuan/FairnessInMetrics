#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 21 21:38:08 2018

@author: mirandayuan
"""
from sklearn.linear_model import LogisticRegression
import pickle

pkl_data = 'data.pkl'
pkl_model = 'model.pkl'
sample = 'sample.pkl'

###load dataset
with open(pkl_data, 'rb') as file:  
    X_train, X_test, y_train, y_test= pickle.load(file)


###logistic regression
logreg = LogisticRegression(solver='sag', max_iter=5000, random_state=42)
logreg.fit(X_train, y_train)
prepro = logreg.predict_proba(X_test)
pre = logreg.predict(X_test)

###save -> pickle file
logistic_objects = (logreg, X_test, y_test, pre, prepro)
with open(pkl_model, 'wb') as file:  
    pickle.dump(logistic_objects, file)
