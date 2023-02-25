# -*- coding: utf-8 -*-
"""
Created on Sat Feb 20 13:41:20 2021

@author: Aakash
"""

import pandas as pd
import numpy as np

data = pd.read_csv("company.csv")

data.columns = ['sales','comp_price','income','advert','population','price','shelve_loc','age','edu','urban','us']

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

urban_dict = {'urban':   {'Yes':1, 'No' :0}}
us_dict = {'us':   {'Yes':1, 'No' :0}}

data = data.replace(urban_dict)
data = data.replace(us_dict)

labelencoder = LabelEncoder()
enc = OneHotEncoder(handle_unknown='ignore')

data['shelve_loc'] = labelencoder.fit_transform(data['shelve_loc'])
enc_df = pd.DataFrame(enc.fit_transform(data[['shelve_loc']]).toarray())

enc_df.columns = ['bad','good','med']

data = data.join(enc_df)

del data['shelve_loc']

for x in range (0,len(data['sales'])):
    if data['sales'][x] > 9:
        data['sales'][x] = 1
    elif data['sales'][x] < 9:
        data['sales'][x] = 0

X = data.columns[:-1]
Y = data.columns[:1]

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
