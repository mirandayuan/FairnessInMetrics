#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 21 21:38:08 2018

@author: mirandayuan
"""
from sklearn.ensemble import RandomForestClassifier
import json

Data = 'data.json'
Curve_Info = 'curve.json'

###read and save json file
def read(name):
    with open(name, 'r') as file:
        return file.read()

def save(data, name, typ):
	with open(name, typ) as file:
		file.write(data)

###load dataset
data = json.loads(read(Data))
X_train = data['X_train']
X_test = data['X_test']
y_train = data['y_train']
y_test = data['y_test']

###random forest
rf = RandomForestClassifier(random_state=0)
rf.fit(X_train,y_train)
prepro = rf.predict_proba(X_test)
pre = rf.predict(X_test)

curve = {}
curve['y_test'] = y_test
curve['prepro'] = prepro.tolist()
curve['pre'] = pre.tolist()
curve['X_test'] = X_test
save(json.dumps(curve), Curve_Info, 'w')