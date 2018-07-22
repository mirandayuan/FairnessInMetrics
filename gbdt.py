#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 21 21:38:08 2018

@author: mirandayuan
"""
from sklearn.ensemble import GradientBoostingClassifier
import json

Data = 'diabetes.json'
Curve_Info = 'curve.json'

###read and save json file
def read(name):
    with open(name, 'r') as file:
        return file.read()

def save(data, name, typ):
	with open(name, typ) as file:
		file.write(data)

###load dataset
diabetes = json.loads(read(Data))
X_train = diabetes['X_train']
X_test = diabetes['X_test']
y_train = diabetes['y_train']
y_test = diabetes['y_test']

###random forest
gbm = GradientBoostingClassifier()
gbm.fit(X_train,y_train)
prepro = gbm.predict_proba(X_test)[:,1]
pre = gbm.predict(X_test)

curve = {}
curve['y_test'] = y_test
curve['prepro'] = prepro.tolist()
curve['pre'] = pre.tolist()

###separate to different groups based on age: XX~35, 36~50, 51~XX
age_test = [[],[],[]]
age_pre = [[],[],[]]
age_prepro = [[],[],[]]

for i in range(0,len(X_test)):
    if X_test[i][7] < 36:
        age_pre[0].append(int(pre[i]))
        age_test[0].append(y_test[i])
        age_prepro[0].append(prepro[i])
    elif X_test[i][7] < 51:
        age_pre[1].append(int(pre[i]))
        age_test[1].append(y_test[i])
        age_prepro[1].append(prepro[i])
    else:
        age_pre[2].append(int(pre[i]))
        age_test[2].append(y_test[i])
        age_prepro[2].append(prepro[i])
        
curve['age_test'] = age_test
curve['age_pre'] = age_pre
curve['age_prepro'] = age_prepro
save(json.dumps(curve), Curve_Info, 'w')