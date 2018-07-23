#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 21 20:03:52 2018

@author: mirandayuan
"""
import json
import numpy as np
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.metrics import precision_recall_curve
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


###load information needed to plot curves
curve = json.loads(read(Curve_Info))
y_test = curve['y_test']
X_test = curve['X_test']
prepro = np.array(curve['prepro'])[:,1]
pre = curve['pre']
        
###entire dataset:
###---print confusion matrix and calculate auc socre
target_names = ['No', 'Yes']
conf = confusion_matrix(y_test, pre, labels = [0,1])
print(conf)
print(classification_report(y_test, pre, target_names=target_names))
if 0 in pre and 1 in pre:
    print("auc score: ")
    roc_auc = roc_auc_score(y_test, prepro)
    print(roc_auc)
else:
    print("only one class present")

###---plot:
###------roc curve
sns.set('talk', 'whitegrid', 'dark', font_scale=0.8, font='Ricty',
        rc={"lines.linewidth": 2, 'grid.linestyle': '--'})

fpr, tpr, thre = roc_curve(y_test, prepro)
tpr_e = [tpr]
fpr_e = [fpr]
label_e = ['ROC curve (AUC = %0.2f)']
roc_e = [roc_auc]
roc_plot(tpr_e, fpr_e, label_e, roc_e)
    
###------threshold curves
(precision, recall, f1score, thresholds) = getvalue(y_test, prepro)
legend_e = ('precision', 'recall', 'f1score')
title_e = 'thresholds: entire dataset'
thresholds_e = [thresholds, thresholds, thresholds]
para_e = [precision, recall, f1score]
thre_plot(thresholds_e, para_e, legend_e, title_e)
print("\n")
print("\n")



###age: XX~35, 36~50, 51~XX
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


###---print confusion matrix, calculate auc socre, get tpr and fpr, get precision and recall and f1score
age_group = ['Metrics - age - XX~35: ', 'Metrics - age - 36~50: ', 'Metrics - age - 51~XX: ']
age_auc = []
age_tpr = []
age_fpr = []
age_thre = []
age_precision = []
age_recall = []
age_f1score = []
age_thre_prf = []
for i in range(len(age_test)):
    print(age_group[i])
    print(classification_report(age_test[i], age_pre[i], target_names=target_names))
    if 0 in age_pre[i] and 1 in age_pre[i]:
        print("auc score: ")
        age_auc.append(roc_auc_score(age_test[i], age_prepro[i]))
        print(age_auc[i])
    else:
        print("only one class present")
    print("\n")

    fpr, tpr, thre = roc_curve(age_test[i], age_prepro[i])
    age_tpr.append(tpr)
    age_fpr.append(fpr)
    age_thre.append(thre)
    (precision, recall, f1score, thresholds) = getvalue(age_test[i], age_prepro[i])
    age_precision.append(precision)
    age_recall.append(recall)
    age_f1score.append(f1score)
    age_thre_prf.append(thresholds)

 
###---plot:
###------roc curve
sns.set('talk', 'whitegrid', 'dark', font_scale=0.8, font='Ricty',rc={"lines.linewidth": 2, 'grid.linestyle': '--'})
age_label = ['age XX~35 (AUC = %0.2f)', 'age 36~50 (AUC = %0.2f)','age 51~xx (AUC = %0.2f)' ]
roc_plot(age_tpr, age_fpr, age_label, age_auc)

###------threshold curves
age_legend = ['age XX~35', 'age 36~50','age 51~xx']
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
