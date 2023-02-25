# -*- coding: utf-8 -*-
"""
Created on Mon Feb 22 14:50:49 2021

@author: Aakash
"""


import pandas as pd
import numpy as np

data = pd.read_csv("fraud.csv")

data.columns = ['grad','marry','tax_income','population','exp','urban']

urban_dict = {'urban':   {'YES':1, 'NO' :0}}
data = data.replace(urban_dict)

grad_dict = {'grad':   {'YES':1, 'NO' :0}}
data = data.replace(grad_dict)

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

labelencoder = LabelEncoder()
enc = OneHotEncoder(handle_unknown='ignore')

data['marry'] = labelencoder.fit_transform(data['marry'])
enc_df = pd.DataFrame(enc.fit_transform(data[['marry']]).toarray())

enc_df.columns = ['divored','married','single']

data = data.join(enc_df)

del data['marry']

for x in range (0,len(data['tax_income'])):
    if data['tax_income'][x] <= 30000:
        data['tax_income'][x] = 0
    else:
        data['tax_income'][x] = 1

X = data.columns[data.columns != 'tax_income']
Y = data.columns[1]

from sklearn.model_selection import train_test_split
train,test = train_test_split(data, test_size = 0.3)

from sklearn.tree import DecisionTreeClassifier as DT
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

model = DT(criterion = 'entropy')
model.fit(train[X], train[Y])

test_preds = model.predict(test[X])
cnf_test_matrix = confusion_matrix(test[Y], test_preds)
cnf_test_matrix

accuracy_score(test[Y],test_preds)

train_preds = model.predict(train[X])
cnf_train_matrix = confusion_matrix(train[Y], train_preds)
cnf_train_matrix

accuracy_score(train[Y],train_preds)
