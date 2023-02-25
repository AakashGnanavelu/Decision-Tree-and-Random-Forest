# -*- coding: utf-8 -*-
"""
Created on Sat Feb 20 15:48:28 2021

@author: Aakash
"""

import pandas as pd
import numpy as np

data = pd.read_csv(r"C:\Users\Aakash\Desktop\AAKASH\Coding Stuff\Full Data Science\Desicon Tree and Random Forest\Assginment\diabetes.csv")

dict = {' Class variable':   {'YES':1, 'NO' :0}}
data = data.replace(dict)

X = np.array(data.iloc[:,:-1])
Y = np.array(data.iloc[:,-1:])

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3)

from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(n_jobs=2, n_estimators=15, criterion="entropy")

oob_score=True,

rf.fit(X_train, Y_train) 
pred = rf.predict(X_test)

from sklearn.metrics import confusion_matrix

cnf_test_matrix = confusion_matrix(Y_test, pred)
cnf_test_matrix

test_acc = np.mean(rf.predict(X_test)==Y_test)
test_acc

train_acc = np.mean(rf.predict(X_train)==Y_train)
train_acc
