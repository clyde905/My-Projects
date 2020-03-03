# --------------------------------------------------------------
# Simple Linear Regression
# Predict the marks obtained by a student based on hours of study
# --------------------------------------------------------------


# Import Pandas for data processing
import pandas as pd


# Read the CSV file
dataset = pd.read_csv('01Students.csv')
df = dataset.copy()

#Split by column X and Y variable
x = df.iloc[:,:-1]
y = df.iloc[:,-1]

#split by rows for training and test datasets
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test =  \
train_test_split(x,y,test_size=0.3, random_state=1234)

from sklearn.linear_model import LinearRegression

#create regressor
std_reg = LinearRegression()
#need to train the regressor based on training data
std_reg.fit(x_train, y_train, sample_weight=None)

y_pred = std_reg.predict(x_test)

#R-Squared and calculate the coefficient and intercept (matrix)
slr_score = std_reg.score(x_test,y_test) #outputs r-squared

#coefficient and intercept
slr_coef = std_reg.coef_
slr_int = std_reg.intercept_
print(slr_int,slr_coef,slr_score)

#Root mean squared value
from sklearn.metrics import mean_squared_error
import math

slr_rmse = math.sqrt(mean_squared_error(y_test, y_pred))
print(slr_rmse)

#plot the results
import matplotlib.pyplot as plt

plt.figure(1)
plt.scatter(x_test,y_test)
plt.plot(x_test, y_pred)
plt.ylim(ymin=0)
plt.show()