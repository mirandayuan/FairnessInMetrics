#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  8 14:24:33 2018

@author: mirandayuan
"""
import xgboost as xgb
import pickle
import numpy as np
#from sklearn.model_selection import GridSearchCV


pkl_data = 'data.pkl'
pkl_model = 'model.pkl'
sample = 'sample.pkl'

###load dataset
with open(pkl_data, 'rb') as file:  
    X_train, X_test, y_train, y_test= pickle.load(file)

X_train = np.asarray(X_train)

'''
# grid search
model = xgb.XGBClassifier(n_estimators = 280, max_depth = 5)
n_estimators = [100, 200, 300, 400, 500, 600, 700 ,800]
n_estimators = [210, 220, 230, 240, 250, 260, 270, 280,290]
max_depth = [3,4,5,6,7,8,9,10]
param_grid = dict(n_estimators=n_estimators)
grid_search = GridSearchCV(model, param_grid, scoring="neg_log_loss", n_jobs=-1)
grid_result = grid_search.fit(X_train, y_train)
# summarize results
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
	print("%f (%f) with: %r" % (mean, stdev, param))
    
'''

###gbm regression
gbm =  xgb.XGBClassifier(n_estimators = 10000, max_depth = 30)
gbm.fit(X_train, y_train)
prepro = gbm.predict_proba(np.asarray(X_test))
pre = gbm.predict(np.asarray(X_test))


###save -> pickle file
gbm_objects = (gbm, X_test, y_test, pre, prepro)
with open(pkl_model, 'wb') as file:  
    pickle.dump(gbm_objects, file)

