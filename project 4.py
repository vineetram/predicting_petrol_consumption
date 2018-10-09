#Predicting Petrol Consumption

# Importing the libraries
import pandas as pd
import numpy as np

# Importing the dataset
dataset = pd.read_csv('petrol_consumption.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 4].values

# Fitting Multiple Linear Regression to the data set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X, y)


# Predicting the Test set results
y_pred = regressor.predict(X)

#Building the Optimal Model using Bakward Elimination 
import statsmodels.formula.api as sm
X = np.append(arr = np.ones((46,1)).astype(int), values = X ,axis=1)
X_opt = X[:, [0,1,2,3,4]]
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()
X_opt = X[:, [0,1,2,4]]
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()


