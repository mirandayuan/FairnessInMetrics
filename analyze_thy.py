#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 22 15:32:22 2018

@author: mirandayuan
"""
import json
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.metrics import precision_recall_curve
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt
import seaborn as sns

Curve_Info = 'curve.json'

###read and save json file
def read(name):
    with open(name, 'r') as file:
        return file.read()

def save(data, name, typ):
	with open(name, typ) as file:
		file.write(data)

###calculate precision, recall, f1score, and corresponding thresholds
def getvalue(true, prepro):
    precision, recall, thresholds = precision_recall_curve(true, prepro)
    thresholds = np.append(thresholds, 1) 
    precision = np.array(precision)
    recall = np.array(recall)
    f1score = 2 * precision * recall / (precision + recall)
    return (precision, recall, f1score, thresholds)

###plot threshold curves
def thre_plot(thresholds, para, legend, title):
    plt.figure()
    color = ['steelblue', 'darkgoldenrod', 'maroon', 'green','gold']
    for i in range(len(para)):
        plt.plot(thresholds[i], para[i], color[i])
        i += 1
    leg = plt.legend(legend, loc = "center left", bbox_to_anchor=(1, 0.5), frameon=True) 
    leg.get_frame().set_edgecolor('k') 
    plt.xlabel('thresholds') 
    plt.ylabel('%')
    plt.title(title)
    plt.show()
    plt.close()

###plot roc curves
def roc_plot(tpr, fpr, label, roc):
    plt.figure()
    color = ['steelblue', 'darkgoldenrod', 'maroon', 'green','gold']
    for i in range(len(tpr)):
        plt.plot(fpr[i], tpr[i], color[i], label = label[i] % roc[i])
        i += 1
    plt.plot([0, 1], [0, 1], color='k', linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.title('ROC curve')
    leg = plt.legend(frameon=True, loc="lower right") 
    leg.get_frame().set_edgecolor('k') 
    plt.show()
    plt.close()

###binarize the label
def label(array, n_class):
    array_c = pd.Categorical(array).codes
    array_c = label_binarize(array, np.arange(n_class)).ravel()
    return array_c


###load information needed to plot curves
curve = json.loads(read(Curve_Info))
y_test = curve['y_test']
X_test = curve['X_test']
prepro = curve['prepro']
pre = curve['pre']

###separate to different groups based on age
age_test = [[],[],[],[],[]]
age_pre = [[],[],[],[],[]]
age_prepro = [[],[],[],[],[]]

for i in range(0,len(X_test)):
    if X_test[i][0] < 21:
        age_pre[0].append(int(pre[i]))
        age_test[0].append(y_test[i])
        age_prepro[0].append(prepro[i])
    elif X_test[i][0] < 41:
        age_pre[1].append(int(pre[i]))
        age_test[1].append(y_test[i])
        age_prepro[1].append(prepro[i])
    elif X_test[i][0] < 61:
        age_pre[2].append(int(pre[i]))
        age_test[2].append(y_test[i])
        age_prepro[2].append(prepro[i])
    elif X_test[i][0] < 81:
        age_pre[3].append(int(pre[i]))
        age_test[3].append(y_test[i])
        age_prepro[3].append(prepro[i])
    else:
        age_pre[4].append(int(pre[i]))
        age_test[4].append(y_test[i])
        age_prepro[4].append(prepro[i])

###separate to different groups based on gender
gender_test = [[],[]]
gender_pre = [[],[]]
gender_prepro = [[],[]]

for i in range(0,len(X_test)):
    if X_test[i][1] == 0:
        gender_pre[0].append(int(pre[i]))
        gender_test[0].append(y_test[i])
        gender_prepro[0].append(prepro[i])
    else:
        gender_pre[1].append(int(pre[i]))
        gender_test[1].append(y_test[i])
        gender_prepro[1].append(prepro[i])




###entire dataset:
pre_c = label(pre, 3)
y_test_c = label(y_test, 3)
prepro_c = np.array(prepro).ravel()

###---print confusion matrix and calculate auc socre
target_names = ['negative', 'increased binding protein', 'decreased binding protein']
print(classification_report(y_test, pre, target_names=target_names))
if pre.count(0) == len(pre) or pre.count(1) == len(pre) or pre.count(2) == len(pre):
    print("only one class present")
else:
    print("auc score: ")
    roc_auc = roc_auc_score(y_test_c, prepro_c)
    print(roc_auc)


###---plot:
###------roc curve
sns.set('talk', 'whitegrid', 'dark', font_scale=0.8, font='Ricty',
        rc={"lines.linewidth": 2, 'grid.linestyle': '--'})

fpr, tpr, thre = roc_curve(y_test_c, prepro_c)
tpr_e = [tpr]
fpr_e = [fpr]
label_e = ['ROC curve (AUC = %0.4f)']
roc_e = [roc_auc]
roc_plot(tpr_e, fpr_e, label_e, roc_e)
    
###------threshold curves
(precision, recall, f1score, thresholds) = getvalue(y_test_c, prepro_c)
legend_e = ('precision', 'recall', 'f1score')
title_e = 'thresholds: entire dataset'
thresholds_e = [thresholds, thresholds, thresholds]
para_e = [precision, recall, f1score]
thre_plot(thresholds_e, para_e, legend_e, title_e)
print("\n")
print("\n")



###age(age: 0): xx~20, 21~40, 41~60, 61~80, 81~xx
age_test_c = [[],[],[],[],[]]
age_pre_c = [[],[],[],[],[]]
age_prepro_c = [[],[],[],[],[]]
for i in range(len(age_test)):
    n_class = 3
    age_test_c[i] = label(age_test[i],3)
    age_pre_c[i] = label(age_pre[i],3)
    age_prepro_c[i] = np.array(age_prepro[i]).ravel()

###---print confusion matrix, calculate auc socre, get tpr and fpr, get precision and recall and f1score
age_group = ['Metrics - age - XX~20: ', 'Metrics - age - 21~40: ','Metrics - age - 41~60: ','Metrics - age - 61~80: ', 'Metrics - age - 81~XX: ']
age_auc = []
age_tpr = []
age_fpr = []
age_thre = []
age_precision = []
age_recall = []
age_f1score = []
age_thre_prf = []
for i in range(len(age_test_c)):
    print(age_group[i])
    print(classification_report(age_test[i], age_pre[i], target_names=target_names))
    if pre.count(0) == len(pre) or pre.count(1) == len(pre) or pre.count(2) == len(pre):
        print("only one class present")
    else:
        print("auc score: ")
        age_auc.append(roc_auc_score(age_test_c[i], age_prepro_c[i]))
        print(age_auc[i])
    print("\n")

    fpr, tpr, thre = roc_curve(age_test_c[i], age_prepro_c[i])
    age_tpr.append(tpr)
    age_fpr.append(fpr)
    age_thre.append(thre)    
    (precision, recall, f1score, thresholds) = getvalue(age_test_c[i], age_prepro_c[i])
    age_precision.append(precision)
    age_recall.append(recall)
    age_f1score.append(f1score)
    age_thre_prf.append(thresholds)

###---plot:
###------roc curve
sns.set('talk', 'whitegrid', 'dark', font_scale=0.8, font='Ricty',rc={"lines.linewidth": 2, 'grid.linestyle': '--'})
age_label = ['age XX~20 (AUC = %0.4f)', 'age 21~40 (AUC = %0.4f)','age 41~60 (AUC = %0.4f)','age 61~80 (AUC = %0.4f)', 'age 81~XX (AUC = %0.4f)' ]
roc_plot(age_tpr, age_fpr, age_label, age_auc)

###------threshold curves
age_legend = ['age XX~20: ', 'age 21~40: ','age 41~60: ','age 61~80: ', 'age 81~XX: ']
title_p = 'thresholds-precision(age)'
thre_plot(age_thre_prf, age_precision, age_legend, title_p)
print("\n")
title_r = 'thresholds-recall(age)'
thre_plot(age_thre_prf, age_recall, age_legend, title_r)
print("\n")
title_f = 'thresholds-f1score(age)'
thre_plot(age_thre_prf, age_f1score, age_legend, title_f)
print("\n")
print("\n")






###gender(sex: 1): M(0), F(1)
gender_test_c = [[],[]]
gender_pre_c = [[],[]]
gender_prepro_c = [[],[]]
for i in range(len(gender_test)):
    n_class = 3
    gender_test_c[i] = label(gender_test[i],3)
    gender_pre_c[i] = label(gender_pre[i],3)
    gender_prepro_c[i] = np.array(gender_prepro[i]).ravel()
    
###---print confusion matrix, calculate auc socre, get tpr and fpr, get precision and recall and f1score
gender_group = ['Metrics - male ', 'Metrics - female ']
gender_auc = []
gender_tpr = []
gender_fpr = []
gender_thre = []
gender_precision = []
gender_recall = []
gender_f1score = []
gender_thre_prf = []
for i in range(len(gender_test_c)):
    print(gender_group[i])
    print(classification_report(gender_test[i], gender_pre[i], target_names=target_names))
    if pre.count(0) == len(pre) or pre.count(1) == len(pre) or pre.count(2) == len(pre):
        print("only one class present")
    else:
        print("auc score: ")
        gender_auc.append(roc_auc_score(gender_test_c[i], gender_prepro_c[i]))
        print(gender_auc[i])
    print("\n")

    fpr, tpr, thre = roc_curve(gender_test_c[i], gender_prepro_c[i])
    gender_tpr.append(tpr)
    gender_fpr.append(fpr)
    gender_thre.append(thre)    
    (precision, recall, f1score, thresholds) = getvalue(gender_test_c[i], gender_prepro_c[i])
    gender_precision.append(precision)
    gender_recall.append(recall)
    gender_f1score.append(f1score)
    gender_thre_prf.append(thresholds)

###---plot:
###------roc curve
sns.set('talk', 'whitegrid', 'dark', font_scale=0.8, font='Ricty',rc={"lines.linewidth": 2, 'grid.linestyle': '--'})
gender_label = ['male (AUC = %0.4f)', 'female (AUC = %0.4f)' ]
roc_plot(gender_tpr, gender_fpr, gender_label, gender_auc)

###------threshold curves
title_p = 'thresholds-precision(gender)'
thre_plot(gender_thre_prf, gender_precision, gender_group, title_p)
print("\n")
title_r = 'thresholds-recall(gender)'
thre_plot(gender_thre_prf, gender_recall, gender_group, title_r)
print("\n")
title_f = 'thresholds-f1score(gender)'
thre_plot(gender_thre_prf, gender_f1score, gender_group, title_f)
print("\n")
print("\n")