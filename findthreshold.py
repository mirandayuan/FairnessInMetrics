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
import csv

fileHeader = ['Threshold Based On', 'Group', 'Threshold', 'Precision', 'Recall', 'f1score']
pkl_thre = 'thre_file.pkl'
with open(pkl_thre, 'rb') as file:
    opt_thre, model, y_test, group_test, group_X, X_test = pickle.load(file)
    

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
    precision = round(tp / (tp + fp),4)
#    print('precision: ', precision)
    recall = round(tp / (tp + fn),4)
#    print('recall: ', recall)
    f1score = round(2 * precision * recall / (precision + recall),4)
#    print('f1score: ', f1score)
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
    return round(error,4)

def reco(title, y_test, X_test, group_title, threshold):
    record = [title]
#    print(group_title)
    entire = get_new_score(y_test, model.predict(np.asarray(X_test)))
    new_score.append(entire)
    record.append(group_title)
    record.append(threshold)
    record.extend(entire)
    writer.writerow(record)
#    print('\n')

csvFile = open('score_4.csv', 'w')
writer = csv.writer(csvFile, dialect='excel')
writer.writerow(fileHeader)

###new scores for new threshold
clf = [CustomThreshold(model, threshold) for threshold in opt_thre]

#title = ['entire dataset', 'age_group 1: xx-50', 'age_group 2: 51-70', 'age_group 3: 71-xx', 'average of the groups', 'median of the groups', 'maximum of the groups', 'minimum of the groups']
#title = ['entire dataset', 'male', 'female', 'average of the groups', 'median of the groups', 'maximum of the groups', 'minimum of the groups']
title = ['entire dataset', 'white', 'black/african american','hispanic/latino','asian','other',  'average of the groups', 'median of the groups', 'maximum of the groups', 'minimum of the groups']

i = 0
error = []
for model in clf:
    print('threshold based on: ',title[i], 'which is: ', opt_thre[i])
    new_score = []
    '''
    ###age
    reco(title[i], y_test, X_test, 'entire dataset', opt_thre[i])
    reco(title[i], group_test[0], group_X[0], 'group: age xx-50', opt_thre[i])
    reco(title[i], group_test[1], group_X[1], 'group: age 51-70', opt_thre[i])    
    reco(title[i], group_test[2], group_X[2], 'group: age 71-xx', opt_thre[i])    
    '''
    '''
    ###gender
    reco(title[i], y_test, X_test, 'entire dataset', opt_thre[i])
    reco(title[i], group_test[0], group_X[0], 'group: male', opt_thre[i])
    reco(title[i], group_test[1], group_X[1], 'group: female', opt_thre[i])    
    
    '''
    ###race
    reco(title[i], y_test, X_test, 'entire dataset', opt_thre[i])
    race_ti = ['white', 'black/african american','hispanic/latino','asian','other']
    for j in range(len(race_ti)):
        reco(title[i], group_test[j], group_X[j], race_ti[j], opt_thre[i])
    
    
    writer.writerow([])
    error.append(get_error(new_score))
    print('average error: ', error[i])
    
    i += 1
    print('\n')
    
csvFile.close()

print('list of error: ', error, '\n')

idx = np.where(error == np.min(error))[0]
print('the threshold with the minimal error is: ') 
for index in idx:
    print(opt_thre[int(index)], 'which is got based on ', title[int(index)])
