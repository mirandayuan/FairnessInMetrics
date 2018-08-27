#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 21 20:03:52 2018

@author: mirandayuan
"""
import pickle
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from utils import getvalue, thre_plot, roc_plot, threshold_youden, group_, result
import seaborn as sns

pkl_model = 'model_n_race_s.pkl'
pkl_thre = 'thre_file.pkl'
protected = 'model_race_ns.pkl'


###load information needed to plot curves 
with open(pkl_model, 'rb') as file:  
    model, X_test, y_test, pre, prepro_p= pickle.load(file)

'''
with open(protected, 'rb') as file:  
    model_p, X_test_p, y_test_p, pre_p, prepro_p_p= pickle.load(file)
X_test = X_test_p
'''

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
label_e = ['ROC curve (AUC = %0.4f)']
roc_e = [roc_auc]
roc_plot(tpr_e, fpr_e, label_e, roc_e)

###optimal threshold based on entire dataset
opt_thre.append(threshold_youden(fpr,tpr,thre))

###------threshold curves
(precision, recall, f1score, thresholds) = getvalue(y_test, prepro)
legend_e = ('precision', 'recall', 'f1score')
title_e = 'thresholds: entire dataset'
thresholds_e = [thresholds, thresholds, thresholds]
para_e = [precision, recall, f1score]
thre_plot(thresholds_e, para_e, legend_e, title_e)
print("\n")
print("\n")

'''
###age: XX~40, 41~60, 61~XX
group_test, age_pre, age_prepro, group_X = [[[] for _ in range(3)] for _ in range(4)]
for i in range(len(X_test)):
    if X_test.loc[i, 'age'] < 41:
        group_(age_pre[0], pre[i], group_test[0], y_test[i], age_prepro[0], prepro[i], group_X[0], X_test.loc[i])
    elif X_test.loc[i, 'age'] < 71:
        group_(age_pre[1], pre[i], group_test[1], y_test[i], age_prepro[1], prepro[i], group_X[1], X_test.loc[i])
    elif X_test.loc[i, 'age'] > 70:
        group_(age_pre[2], pre[i], group_test[2], y_test[i], age_prepro[2], prepro[i], group_X[2], X_test.loc[i])

###result of age
age_group = ['Metrics - age - XX~40: ', 'Metrics - age - 41~70: ', 'Metrics - age - 71~XX: ']
age_label = ['age XX~40 (AUC = %0.4f)', 'age 41~70 (AUC = %0.4f)','age 71~xx (AUC = %0.4f)' ]
age_legend = ['age XX~40', 'age 41~70','age 71~xx']
title_p, title_r, title_f = 'thresholds-precision(age)', 'thresholds-recall(age)', 'thresholds-f1score(age)'
n = 4
opt_thre = result(opt_thre, age_group, group_test, target_names, age_pre, age_prepro, age_label, age_legend, title_p, title_r, title_f, n)
'''


'''
###gender: male, female
group_test, gender_pre, gender_prepro, group_X = [[[] for _ in range(2)] for _ in range(4)]
for i in range(len(X_test)):
    if X_test.loc[i, 'issexMale'] == 1:
        group_(gender_pre[0], pre[i], group_test[0], y_test[i], gender_prepro[0], prepro[i], group_X[0], X_test.loc[i])
    elif X_test.loc[i, 'issexFemale'] == 1:
        group_(gender_pre[1], pre[i], group_test[1], y_test[i], gender_prepro[1], prepro[i], group_X[1], X_test.loc[i])

###result of gender
gender_group = ['Metrics - gender - Male: ', 'Metrics - gender - Female: ']
gender_label = ['male (AUC = %0.4f)', 'female (AUC = %0.4f)']
gender_legend = ['gender: male', 'gender: female']
title_p, title_r, title_f = 'thresholds-precision(gender)', 'thresholds-recall(gender)', 'thresholds-f1score(gender)'
n = 3
opt_thre = result(opt_thre, gender_group, group_test, target_names, gender_pre, gender_prepro, gender_label, gender_legend, title_p, title_r, title_f, n)




'''
###race: white, black/africanAmerican, hispanic/latino, Asian, other
group_test, race_pre, race_prepro, group_X = [[[] for _ in range(5)] for _ in range(4)]
race_list = {'isethnicityWhite': 0, 'isethnicityBlackOrAfricanAmerican': 1, 'isethnicityHispanicOrLatino': 2, 'isethnicityAsian': 3}
for i in range(len(X_test)):
    seted = False
    for race, ind in race_list.items():
        if X_test.loc[i, race] == 1:
            group_(race_pre[ind], pre[i], group_test[ind], y_test[i], race_prepro[ind], prepro[i], group_X[ind], X_test.loc[i])
            seted = True
    if seted == False:
        group_(race_pre[4], pre[i], group_test[4], y_test[i], race_prepro[4], prepro[i], group_X[4], X_test.loc[i])


###result of race
race_group = ['Metrics - race - white: ', 'Metrics - race - black/african american: ', 'Metrics - race - hispanic/latino: ', 'asian', 'other']
race_label = ['white (AUC = %0.4f)', 'black/african american 41~70 (AUC = %0.4f)','hispanic/latino (AUC = %0.4f)', 'asian (AUC = %0.4f)', 'other (AUC = %0.4f)' ]
race_legend = ['white', 'black/african american','hispanic/latino','asian','other']
title_p, title_r, title_f = 'thresholds-precision(race)', 'thresholds-recall(race)', 'thresholds-f1score(race)'
n = 6
opt_thre = result(opt_thre, race_group, group_test, target_names, race_pre, race_prepro, race_label, race_legend, title_p, title_r, title_f, n)

print(opt_thre)

###save -> pickle file
thre_objects = (opt_thre, model, y_test, group_test, group_X, X_test)
with open(pkl_thre, 'wb') as file:  
    pickle.dump(thre_objects, file)
