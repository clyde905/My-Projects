# -*- coding: utf-8 -*-
"""
Created on Tue Feb 18 11:41:21 2020

@author: clyde
"""

import pandas as pd

loandata = pd.read_csv(r'C:\Users\clyde\Downloads\01Exercise1.csv')
df = pd.DataFrame(loandata)
df.isna().sum(axis=0) #count null values
df2 = df.dropna() #clean data

df2 = df2.drop(['gender'],axis=1)
df2 = pd.get_dummies(df2, drop_first=True)

from sklearn.preprocessing import StandardScaler
scalar = StandardScaler()

#use column names so we don't have to worry about shifts
df2['income'] = scalar.fit_transform(df2[['income']]) #normalize values for income
df2['loanamt'] = scalar.fit_transform(df2[['loanamt']]) #normalize values for income

#create the x and y
y= df2[['status_Y']]
x = df2.drop(['status_Y'], axis=1)

from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest = \
train_test_split(x,y,test_size=0.3, random_state=1254, stratify=y)
#stratify ensures the training data is diverse 

#create logistic regressor
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()
lr.fit(xtrain,ytrain)
ypred = lr.predict(xtest)

from sklearn.metrics import confusion_matrix
matrix = confusion_matrix(ytest,ypred)
score = lr.score(xtest,ytest)