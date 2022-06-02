# Multiple Linear Regression

# ----- PREPROCESSING -----

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('FINAL2_ma_real_estate.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 5].values
print(X)
print(y)

# Encoding categorical data
# from sklearn.compose import ColumnTransformer
# from sklearn.preprocessing import OneHotEncoder
# ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [3])], remainder='passthrough')
# X = np.array(ct.fit_transform(X))
# print(X)

#Avoiding the dummy variable trap
#X = X[:, 1:]

#taking care of missing data
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values = np.nan, strategy = 'mean')
imputer.fit(X[:, 1:5])
X[:, 1:5] = imputer.transform(X[:, 1:5])

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# ----- TRAINING -----

# Training the Multiple Linear Regression model on the Training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

print(X_train)
# Predicting the Test set results
y_pred = regressor.predict(X_test)

#compute P-values and find stastical significance of values
import statsmodels.formula.api as sm
import statsmodels.api as sm2
X = np.append(arr = np.ones((42, 1)).astype(int), values = X, axis = 1)
X_opt = np.array(X[:, [0, 1, 2, 3, 4, 5]], dtype=float)
regressor_OLS = sm2.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()

X_opt = np.array(X[:, [1, 2, 3, 4, 5]], dtype=float)
regressor_OLS = sm2.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()

X_opt = np.array(X[:, [1, 3, 4, 5]], dtype=float)
regressor_OLS = sm2.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()

X_opt = np.array(X[:, [1, 3, 4]], dtype=float)
regressor_OLS = sm2.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()

y_pred = regressor.predict(X_test)

y_pred = regressor.predict(X_test)

# Visualising the Training set results
plt.scatter(X_train, y_train, color = 'red')
plt.plot(X_train, regressor.predict(X_train), color = 'blue')
plt.title('Real Estate Value Evaluator (Training set)')
plt.xlabel('Beds, Square Feet, & Acre Lots')
plt.ylabel('Property Value')
plt.show()

# Visualising the Test set results
plt.scatter(X_test, y_test, color = 'red')
plt.plot(X_train, regressor_OLS.predict(X_train), color = 'blue')
plt.title('Real Estate Value Evaluator (Test set)')
plt.xlabel('Beds, Square Feet, & Acre Lots')
plt.ylabel('Property Value')
plt.show()

y_pred = regressor.predict(X_test)
pickle.dump(regressor_OLS, open('property_value_predictor.pkl','wb'))

import pickle



#np.set_printoptions(precision=2)
#print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))