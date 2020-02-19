# -*- coding: utf-8 -*-
"""
Created on Sun Feb  2 15:57:59 2020

@author: clyde
"""

#import libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import math
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

#load data from csv
df = pd.read_csv('Salary_Data.csv')
df = df.copy()
df.info()
df.describe()
df.isna().sum()
print(df.corr())

#seperate data into indivdual columns
x = df.iloc[:,:-1]
y = df.iloc[:,-1]

#Create model parameters
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.3,random_state=0)
sal_reg = LinearRegression() #create regressor
fit = sal_reg.fit(x_train,y_train)

#predicted salary from test data
y_pred = sal_reg.predict(x_test)

#find r-square, int, and slope, root mean square
sal_r2 = sal_reg.score(x_test,y_test)
sal_slope = sal_reg.coef_
sal_int = sal_reg.intercept_
sal_rms = math.sqrt(mean_squared_error(y_test,y_pred))

#create plot
plt.figure(1)
plt.scatter(x_test,y_test, color='blue', label='data points')
plt.plot(x_test, y_pred, label='regression line')
plt.ylim(ymin=0)
plt.xlabel('Years of Experience')
plt.ylabel('Predicted Salary ($)')
plt.title('Trend of the Predicted Salary Compared to Years of Employment')
plt.legend()
plt.show()

print('R^2 is calculated to be: ', sal_r2)
print('RMS is calculated to be: ', sal_rms)

#Give an estimated salary for inputted years of experience
def f(m,x,b):
    return m*x+b

while True:
    try:
        exp = input('Please enter the number of years of experience you have: ')
        exp = float(exp)
        sal = f(sal_slope, exp, sal_int)
        sal = round(float(f(sal_slope, exp, sal_int)),2)
        print('The estimated salary is: $', sal)
        break
    except:
        print("Sorry, this value is not valid please enter an integer value")
        continue

