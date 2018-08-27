#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  8 14:24:33 2018

@author: mirandayuan
"""
import xgboost as xgb
import pickle
import numpy as np
import pandas as pd
#from sklearn.model_selection import GridSearchCV


pkl_data = 'data_race.pkl'
pkl_model = 'model_n_race_s.pkl'
sample = 'sample_race.pkl'

def drop_fea(df, fea):
    df.drop(columns = fea, inplace = True)

###load dataset
with open(sample, 'rb') as file:  
    X_train, X_test, y_train, y_test= pickle.load(file)

X_train_pd = pd.DataFrame(X_train, columns = X_test.columns)
entire = [X_train_pd, X_test]
'''
#delete age
for dataset in entire:
    drop_fea(dataset, 'age')

#delete gender
for dataset in entire:
    drop_fea(dataset, ['issexMale', 'issexFemale'])

'''
#delete race
for dataset in entire:
    drop_fea(dataset, ['isethnicityWhite','isethnicityBlackOrAfricanAmerican', 'isethnicityHispanicOrLatino', 'isethnicityAsian'])


###gbm classifier
gbm =  xgb.XGBClassifier(n_estimators = 1000, max_depth = 10, min_child_weight = 9, gamma = 0.3, subsample = 0.9, colsample_bytree = 0.6, reg_alpha = 3, reg_lambda = 2, learning_rate = 0.051)
gbm.fit(np.asarray(X_train_pd), y_train)
prepro = gbm.predict_proba(np.asarray(X_test))
pre = gbm.predict(np.asarray(X_test))

###save -> pickle file
gbm_objects = (gbm, X_test, y_test, pre, prepro)
with open(pkl_model, 'wb') as file:  
    pickle.dump(gbm_objects, file)



'''
# grid search
model = xgb.XGBClassifier(n_estimators = 280, max_depth = 5, min_child_weight = 9, gamma = 0.3, subsample = 0.9, colsample_bytree = 0.6, reg_alpha = 3, reg_lambda = 2, learning_rate = 0.05)
#n_estimators = [100, 200, 300, 400, 500, 600, 700, 800]
#n_estimators = [210, 220, 230, 240, 250, 260, 270, 280, 290]
#max_depth = [3,4,5,6,7,8,9,10]
#min_child_weight = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
#gamma = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
#subsample = [0.6, 0.7, 0.8, 0.9]
#colsample_bytree = [0.6, 0.7, 0.8, 0.9]
#reg_alpha = [0.05, 0.1, 1, 2, 3]
#reg_lambda = [0.05, 0.1, 1, 2, 3]
learning_rate = [0.01, 0.05, 0.07, 0.1, 0.2]
param_grid = dict(learning_rate = learning_rate)
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


