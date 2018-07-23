#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 11 16:04:36 2018

@author: mirandayuan
"""

import pandas as pd
import numpy as np
import json

JSON_FILE = 'data.json'



def save(data, name, typ):
	with open(name, typ) as file:
		file.write(data)

def feature(in_data):
    out_data = pd.read_csv(in_data, sep=",", header=None, names=["age","sex","thyro","query_thy","anti_medi","sick","preg","thy_sur","I131","hypo","hyper","lith","goi","tumor","hypopi","psych","TSHM","TSH","T3M","T3","TT3M","TT4","T4UM","T4U","FTIM","FTI","TBGM","TBG","referal","class"])
    #print(train_th.columns)
    for i in range(0, len(out_data)):
        for name in out_data.columns:
            #print(name)
            value = out_data.loc[i, name]
            if isinstance(value,str):
                if value == "M" or value == "f":
                    out_data.loc[i, name] = 0
                elif value == "F" or value == "t":
                    out_data.loc[i, name] = 1
                elif value.startswith("negative"):
                    out_data.loc[i, name] = 0
                elif value.startswith("increased"):
                    out_data.loc[i, name] = 1
                elif value.startswith("decreased"):
                    out_data.loc[i, name] = 2
                elif value == "WEST":
                    out_data.loc[i, name] = 0
                elif value == "STMW":
                    out_data.loc[i, name] = 1
                elif value == "SVHC":
                    out_data.loc[i, name] = 2
                elif value == "SVI":
                    out_data.loc[i, name] = 3
                elif value == "SVHD":
                    out_data.loc[i, name] = 4
                elif value == "other":
                    out_data.loc[i, name] = 5

    #TBG : all "?"...thus delete
    out_data.drop(['TBG'], axis = 1, inplace = True)

    for name in out_data.columns:
        temp = out_data[name].tolist()
        temp = list(filter(("?").__ne__, temp))
        temp = [float(i) for i in temp]
        ave = np.mean(temp)
        for i in range(0, len(out_data)):
            if out_data.loc[i,name] == "?":
                out_data.loc[i,name] = ave
    return out_data

########main###########
train_th = feature("allbp.data.txt")
test_th = feature("allbp.test.txt")


### bootstrap - age
### ---divide to different groups
age0 = [[],[],[]]
age1 = [[],[],[]]
age2 = [[],[],[]]
age3 = [[],[],[]]
age4 = [[],[],[]]
def classes(i, ele, index):
    if ele == 0:
        index[0].append(i)
    elif ele == 1:
        index[1].append(i)
    else: 
        index[2].append(i)

for i in range(0,len(train_th)):
    if int(train_th.iloc[i]['age']) < 21:
        classes(i, train_th.iloc[i]['class'], age0)
    elif int(train_th.iloc[i]['age']) < 41:
        classes(i, train_th.iloc[i]['class'], age1)
    elif int(train_th.iloc[i]['age']) < 61:
        classes(i, train_th.iloc[i]['class'], age2)
    elif int(train_th.iloc[i]['age']) < 81:
        classes(i, train_th.iloc[i]['class'], age3)
    else:
        classes(i, train_th.iloc[i]['class'], age4)

"""         
length = [age0, age1, age2, age3, age4]
for it in length:
    for i in range(len(age0)):
        print(len(it[i]))


### ---resample
index = []
sample_num = [66, 184, 131, 119, 117, 133]
sample_list = [age2y, age2n, age4y, age4n, age6y, age6n]
for i in range(len(sample_num)):
    index.append(resample(sample_list[i], n_samples = sample_num[i]))
"""





data = {}

data["y_train"] = train_th["class"].values.tolist()
data["y_train"] = [int(i) for i in data["y_train"]]
#data["y_train"] = pd.Categorical(data["y_train"]).codes.tolist()

data["y_test"] = test_th["class"].values.tolist()
data["y_test"] = [int(i) for i in data["y_test"]]
#data["y_test"] = pd.Categorical(data["y_test"]).codes
#n_class = 3
#data["y_test"] = label_binarize(data["y_test"], np.arange(n_class)).tolist()


train_th.drop(['class'], axis = 1, inplace = True)
data["X_train"] = train_th.values.tolist()
for i in range(0, len(data["X_train"])):
    data["X_train"][i] = [float(j) for j in data["X_train"][i]]
    
test_th.drop(['class'], axis = 1, inplace = True)
data["X_test"] = test_th.values.tolist()
for i in range(0, len(data["X_test"])):
    data["X_test"][i] = [float(j) for j in data["X_test"][i]]


save(json.dumps(data), JSON_FILE, 'w')

 
 
            