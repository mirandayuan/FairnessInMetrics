#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 26 21:20:17 2018

@author: mirandayuan
"""
import numpy as np
from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_curve, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

###calculate precision, recall, f1score, and corresponding thresholds
def getvalue(true, prepro):
    precision, recall, thresholds = precision_recall_curve(true, prepro)
    thresholds = np.append(thresholds, 1) 
    precision = np.array(precision).round(4)
    recall = np.array(recall).round(4)
    f1score = (2 * precision * recall / (precision + recall)).round(4)
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

###calculate the thresholds by youden threshold method
def threshold_youden(fpr,tpr,thresholds):
    diff = tpr-fpr
    order = sorted(zip(diff, thresholds))
    return round(order[-1][1],4)

###classify the dataset and according results bassed on protected group
def group_(group_pre,pre,group_test,y_test,group_prepro,prepro,group_X,X_test):
    group_pre.append(int(pre))
    group_test.append(y_test)
    group_prepro.append(prepro)
    group_X.append(X_test)


###---print confusion matrix, calculate auc socre, get tpr and fpr, get precision and recall and f1score
def result(opt_thre, group, group_test, target_names, group_pre, group_prepro, group_label, group_legend, title_p, title_r, title_f, n):
    group_auc, group_tpr, group_fpr, group_thre, group_precision, group_recall, group_f1score, group_thre_prf = [[] for _ in range(8)]
    for i in range(len(group_test)):
        print(group[i])
        print(classification_report(group_test[i], group_pre[i], target_names=target_names))
        if 0 in group_pre[i] and 1 in group_pre[i]:
            print("auc score: ")
            group_auc.append(round(roc_auc_score(group_test[i], group_prepro[i]),4))
            print(group_auc[i])
        else:
            print("only one class present")
        print("\n")

        fpr, tpr, thre = roc_curve(group_test[i], group_prepro[i])
        ###optimal threshold based on the group
        opt_thre.append(threshold_youden(fpr,tpr,thre))
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

    opt_thre.extend([np.mean(opt_thre[1:n]).round(4), np.median(opt_thre[1:n]).round(4), np.max(opt_thre[1:n]), np.min(opt_thre[1:n])])
    
    return opt_thre


###get precision, recall and f1 scores based on the threshold chosen
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

###calculate the average error of precision, recall and f1 scores across each group
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
