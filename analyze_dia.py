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
    color = ['steelblue', 'darkgoldenrod', 'maroon', 'green','gold', 'darkorchid', 'darkcyan']
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
    color = ['steelblue', 'darkgoldenrod', 'maroon', 'green','gold', 'darkorchid', 'darkcyan']
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

###calculate the thresholds
def cutoff_youdens_j(fpr,tpr,thresholds):
    j_scores = tpr-fpr
    j_ordered = sorted(zip(j_scores,thresholds))
    return j_ordered[-1][1]

###classify
def group_(group_pre,pre,group_test,y_test,group_prepro,prepro,group_X,X_test):
    group_pre.append(int(pre))
    group_test.append(y_test)
    group_prepro.append(prepro)
    group_X.append(X_test)


###---print confusion matrix, calculate auc socre, get tpr and fpr, get precision and recall and f1score
def result(group, group_test, target_names, group_pre, group_prepro, group_label, group_legend, title_p, title_r, title_f, n):
    group_auc = []
    group_tpr = []
    group_fpr = []
    group_thre = []
    group_precision = []
    group_recall = []
    group_f1score = []
    group_thre_prf = []
    for i in range(len(group_test)):
        print(group[i])
        print(classification_report(group_test[i], group_pre[i], target_names=target_names))
        if 0 in group_pre[i] and 1 in group_pre[i]:
            print("auc score: ")
            group_auc.append(roc_auc_score(group_test[i], group_prepro[i]))
            print(group_auc[i])
        else:
            print("only one class present")
        print("\n")

        fpr, tpr, thre = roc_curve(group_test[i], group_prepro[i])
        ###optimal threshold based on the group
        opt_thre.append(cutoff_youdens_j(fpr,tpr,thre))
        group_tpr.append(tpr)
        group_fpr.append(fpr)
        group_thre.append(thre)
        (precision, recall, f1score, thresholds) = getvalue(group_test[i], group_prepro[i])
        group_precision.append(precision)
        group_recall.append(recall)
        group_f1score.append(f1score)
        group_thre_prf.append(thresholds)
    
    ###---plot:
    ###------roc curve
    sns.set('talk', 'whitegrid', 'dark', font_scale=0.8, font='Ricty',rc={"lines.linewidth": 2, 'grid.linestyle': '--'})
    roc_plot(group_tpr, group_fpr, group_label, group_auc)

    ###------threshold curves
    thre_plot(group_thre_prf, group_precision, group_legend, title_p)
    print("\n")
    thre_plot(group_thre_prf, group_recall, group_legend, title_r)
    print("\n")
    thre_plot(group_thre_prf, group_f1score, group_legend, title_f)
    print("\n")
    print("\n")

    opt_thre.extend([np.mean(opt_thre[1:n]), np.median(opt_thre[1:n]), np.max(opt_thre[1:n]), np.min(opt_thre[1:n])])
    
    return opt_thre

#########################################
####main
###load information needed to plot curves 
with open(pkl_model, 'rb') as file:  
    model, X_test, y_test, pre, prepro_p= pickle.load(file)




count_test = [0,0,0,0,0,0]
for i in range(len(y_test)):
    if y_test[i] == 1:
        if X_test.loc[i,'age'] < 41:
            count_test[0] += 1
        elif X_test.loc[i,'age'] < 61:    
            count_test[2] += 1
        else:
            count_test[4] += 1
    else:
        if X_test.loc[i,'age'] < 41:
            count_test[1] += 1
        elif X_test.loc[i,'age'] < 61:    
            count_test[3] += 1
        else:
            count_test[5] += 1

print(count_test)



prepro = prepro_p[:,1]

opt_thre = []
###entire dataset:
###---print confusion matrix and calculate auc socre
target_names = ['shorter stay', 'longer stay']
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


###age: XX~40, 41~60, 61~XX
age_test = [[],[],[]]
age_pre = [[],[],[]]
age_prepro = [[],[],[]]
age_X = [[],[],[]]
for i in range(len(X_test)):
    if X_test.loc[i, 'age'] < 41:
        group_(age_pre[0], pre[i], age_test[0], y_test[i], age_prepro[0], prepro[i], age_X[0], X_test.loc[i])
    elif X_test.loc[i, 'age'] < 71:
        group_(age_pre[1], pre[i], age_test[1], y_test[i], age_prepro[1], prepro[i], age_X[1], X_test.loc[i])
    elif X_test.loc[i, 'age'] > 70:
        group_(age_pre[2], pre[i], age_test[2], y_test[i], age_prepro[2], prepro[i], age_X[2], X_test.loc[i])

###result of age
age_group = ['Metrics - age - XX~40: ', 'Metrics - age - 41~70: ', 'Metrics - age - 71~XX: ']
age_label = ['age XX~40 (AUC = %0.2f)', 'age 41~70 (AUC = %0.2f)','age 71~xx (AUC = %0.2f)' ]
age_legend = ['age XX~40', 'age 41~70','age 71~xx']
title_p = 'thresholds-precision(age)'
title_r = 'thresholds-recall(age)'
title_f = 'thresholds-f1score(age)'
n = 4
opt_thre = result(age_group, age_test, target_names, age_pre, age_prepro, age_label, age_legend, title_p, title_r, title_f, n)



'''
###gender: male, female
gender_test = [[],[]]
gender_pre = [[],[]]
gender_prepro = [[],[]]
gender_X = [[],[]]
for i in range(len(X_test)):
    if X_test.loc[i, 'issexMale'] == 1:
        group_(gender_pre[0], pre[i], gender_test[0], y_test[i], gender_prepro[0], prepro[i], gender_X[0], X_test.loc[i])
    elif X_test.loc[i, 'issexFemale'] == 1:
        group_(gender_pre[1], pre[i], gender_test[1], y_test[i], gender_prepro[1], prepro[i], gender_X[1], X_test.loc[i])

###result of gender
gender_group = ['Metrics - gender - Male: ', 'Metrics - gender - Female: ']
gender_label = ['male (AUC = %0.2f)', 'female (AUC = %0.2f)']
gender_legend = ['gender: male', 'gender: female']
title_p = 'thresholds-precision(gender)'
title_r = 'thresholds-recall(gender)'
title_f = 'thresholds-f1score(gender)'
n = 3
opt_thre = result(gender_group, gender_test, target_names, gender_pre, gender_prepro, gender_label, gender_legend, title_p, title_r, title_f, n)
'''



'''
###race: white, black/africanAmerican, hispanic/latino, Asian, other
race_test = [[],[],[],[],[]]
race_pre = [[],[],[],[],[]]
race_prepro = [[],[],[],[],[]]
race_X = [[],[],[],[],[]]
for i in range(len(X_test)):
    if X_test.loc[i, 'isethnicityWhite'] == 1:
        group_(race_pre[0], pre[i], race_test[0], y_test[i], race_prepro[0], prepro[i], race_X[0], X_test.loc[i])
    elif X_test.loc[i, 'isethnicityBlackOrAfricanAmerican'] == 1:
        group_(race_pre[1], pre[i], race_test[1], y_test[i], race_prepro[1], prepro[i], race_X[1], X_test.loc[i])
    elif X_test.loc[i, 'isethnicityHispanicOrLatino'] == 1:
        group_(race_pre[2], pre[i], race_test[2], y_test[i], race_prepro[2], prepro[i], race_X[2], X_test.loc[i])
    elif X_test.loc[i, 'isethnicityAsian'] == 1:
        group_(race_pre[3], pre[i], race_test[3], y_test[i], race_prepro[3], prepro[i], race_X[3], X_test.loc[i])
    else:
        group_(race_pre[4], pre[i], race_test[4], y_test[i], race_prepro[4], prepro[i], race_X[4], X_test.loc[i])

###result of race
race_group = ['Metrics - race - white: ', 'Metrics - race - black/african american: ', 'Metrics - race - hispanic/latino: ', 'asian', 'other']
race_label = ['white (AUC = %0.2f)', 'black/african american 41~70 (AUC = %0.2f)','hispanic/latino (AUC = %0.2f)', 'asian (AUC = %0.2f)', 'other (AUC = %0.2f)' ]
race_legend = ['white', 'black/african american','hispanic/latino','asian','other']
title_p = 'thresholds-precision(race)'
title_r = 'thresholds-recall(race)'
title_f = 'thresholds-f1score(race)'
n = 6
opt_thre = result(race_group, race_test, target_names, race_pre, race_prepro, race_label, race_legend, title_p, title_r, title_f, n)

print(opt_thre)
'''
###save -> pickle file
thre_objects = (opt_thre, model, y_test, age_test, age_X, X_test)
with open(pkl_thre, 'wb') as file:  
    pickle.dump(thre_objects, file)
