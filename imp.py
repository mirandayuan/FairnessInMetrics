#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 21 20:12:32 2018

@author: mirandayuan
"""
from matplotlib import pyplot
import pandas as pd
import pickle

pkl_model = 'model_age_s.pkl'

with open(pkl_model, 'rb') as file:  
    gbm, X_test, y_test, pre, prepro_p= pickle.load(file)

def plot_imp(importance, wid, tit):
    pyplot.figure()
    importance.plot()
    importance.plot(kind='bar',alpha=0.75, figsize=(wid, 10))
    pyplot.title(tit)
    pyplot.xlabel('importance')

# feature importance
importance = gbm.feature_importances_
df = pd.Series(importance, index = X_test.columns)

plot_imp(df, 50, 'XGBoost Feature Importance')

df.sort_values(ascending = False, inplace = True)
df.to_csv('importance_race.csv')
top10 = df.nlargest(10)
plot_imp(top10, 15, 'XGBoost top 10 Feature Importance')

protect = ['age', 'issexMale', 'issexFemale', 'isethnicityWhite', 'isethnicityBlackOrAfricanAmerican', 'isethnicityHispanicOrLatino', 'isethnicityAsian']
for name in protect:
    print(name, df[name])