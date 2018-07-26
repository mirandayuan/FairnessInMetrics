#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 21 20:03:52 2018

@author: mirandayuan
"""
import pickle
import numpy as np
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.metrics import precision_recall_curve
import matplotlib.pyplot as plt
import seaborn as sns

pkl_model = 'model.pkl'
pkl_thre = 'thre_file.pkl'

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


def cutoff_youdens_j(fpr,tpr,thresholds):
    j_scores = tpr-fpr
    j_ordered = sorted(zip(j_scores,thresholds))
    return j_ordered[-1][1]
 


#########################################
####main
###load information needed to plot curves 
with open(pkl_model, 'rb') as file:  
    model, X_test, y_test, pre, prepro_p= pickle.load(file)

prepro = prepro_p[:,1]

opt_thre = []
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

###optimal threshold based on entire dataset
opt_thre.append(cutoff_youdens_j(fpr,tpr,thre))

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
age_X = [[],[],[]]

for i in range(0,len(X_test)):
    if X_test[i][7] < 36:
        age_pre[0].append(int(pre[i]))
        age_test[0].append(y_test[i])
        age_prepro[0].append(prepro[i])
        age_X[0].append(X_test[i])
    elif X_test[i][7] < 51:
        age_pre[1].append(int(pre[i]))
        age_test[1].append(y_test[i])
        age_prepro[1].append(prepro[i])
        age_X[1].append(X_test[i])
    else:
        age_pre[2].append(int(pre[i]))
        age_test[2].append(y_test[i])
        age_prepro[2].append(prepro[i])
        age_X[2].append(X_test[i])


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
    ###optimal threshold based on the group
    opt_thre.append(cutoff_youdens_j(fpr,tpr,thre))
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


opt_thre.extend([np.mean(opt_thre[1:4]), np.median(opt_thre[1:4]), np.max(opt_thre[1:4]), np.min(opt_thre[1:4])])
###save -> pickle file
thre_objects = (opt_thre, model, y_test, age_test, age_X, X_test)
with open(pkl_thre, 'wb') as file:  
    pickle.dump(thre_objects, file)

