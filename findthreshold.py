#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 24 17:34:05 2018

@author: mirandayuan
"""
import pickle
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import confusion_matrix

pkl_thre = 'thre_file.pkl'
with open(pkl_thre, 'rb') as file:
    opt_thre, model, y_test, age_test, age_X, X_test = pickle.load(file)
    

class CustomThreshold(BaseEstimator, ClassifierMixin):
    ###Custom threshold wrapper for binary classification
    def __init__(self, base, threshold=0.5):
        self.base = base
        self.threshold = threshold
    def fit(self, *args, **kwargs):
        self.base.fit(*args, **kwargs)
        return self
    def predict(self, X):
        return (self.base.predict_proba(X)[:, 1] > self.threshold).astype(int)

def get_new_score(y_test, y_pre):
    tn, fp, fn, tp = confusion_matrix(y_test, y_pre).ravel()
    precision = tp / (tp + fp)
    print('precision: ', precision)
    recall = tp / (tp + fn)
    print('recall: ', recall)
    f1score = 2 * precision * recall / (precision + recall)
    print('f1score: ', f1score)
    score = [precision, recall, f1score]
    return score

def get_error(new_score):
    error = 0
    t = 0
    for i in range(len(new_score) - 1):
        j = i + 1
        while(j < len(new_score)):
            error = np.sum(np.absolute(np.array(new_score[j]) - np.array(new_score[i])))/3
            j += 1
            t += 1
    error /= t
    return error

###new scores for new threshold
clf = [CustomThreshold(model, threshold) for threshold in opt_thre]

title = ['entire dataset', 'age_group 1: xx-35', 'age_group 2: 36-50', 'age_group 3: 51-xx', 'average of the groups', 'median of the groups', 'maximum of the groups', 'minimum of the groups']
i = 0
error = []
for model in clf:
    print('threshold based on: ',title[i], 'which is: ', opt_thre[i])
    new_score = []
    
    print('scores for entire dataset')
    new_score.append(get_new_score(y_test, model.predict(X_test)))
    print('\n')
    print('scores for group: age xx-35')
    new_score.append(get_new_score(age_test[0], model.predict(age_X[0])))
    print('\n')
    print('scores for group: age 36-50')
    new_score.append(get_new_score(age_test[1], model.predict(age_X[1])))
    print('\n')
    print('scores for group: age 51-xx')
    new_score.append(get_new_score(age_test[2], model.predict(age_X[2])))
    print('\n')
    
    error.append(get_error(new_score))
    print('average error: ', error[i])
    
    i += 1
    print('\n')

print('list of error: ', error, '\n')

idx = np.where(error == np.min(error))[0]
print('the threshold with the minimal error is: ') 
for index in idx:
    print(opt_thre[int(index)], 'which is got based on ', title[int(index)])
