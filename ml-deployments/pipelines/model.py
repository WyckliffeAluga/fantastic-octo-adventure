#!/usr/bin/env python3

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error , r2_score
from math import  sqrt

# load train and test data
x_train = pd.read_csv('datasets/x_train.csv')
x_test  = pd.read_csv('datasets/x_test.csv')

#print(x_train.head(5))

# capture the target
y_train = x_train['SalePrice']
y_test = x_test['SalePrice']

# load the pre-selected features
features = pd.read_csv('datasets/selected_features.csv')
#print(features.columns)
features = features['0'].tolist()

# add another key feature
features = features + ['LotFrontage']

#print(features)

# reduce the train and test to the selected features
x_train = x_train[features]
x_test = x_test[features]
print(x_train.shape, x_test.shape)

"""
Part 1
Regularised linear regression: Lasso
"""
model = Lasso(alpha=0.005, random_state=123)
model.fit(x_train, y_train)

# evaluate the model
pred = model.predict(x_train)

# determine mse and rmse
print('train mse: {}'.format(int(
    mean_squared_error(np.exp(y_train), np.exp(pred)))))
print('train rmse: {}'.format(int(
    sqrt(mean_squared_error(np.exp(y_train), np.exp(pred))))))
print('train r2: {}'.format(
    r2_score(np.exp(y_train), np.exp(pred))))
print()

# make predictions for test set
pred = model.predict(x_test)

# determine mse and rmse
print('test mse: {}'.format(int(
    mean_squared_error(np.exp(y_test), np.exp(pred)))))
print('test rmse: {}'.format(int(
    sqrt(mean_squared_error(np.exp(y_test), np.exp(pred))))))
print('test r2: {}'.format(
    r2_score(np.exp(y_test), np.exp(pred))))
print()

print('Average house price: ', int(np.exp(y_train).median()))


plt.scatter(y_test, model.predict(x_test))
plt.xlabel('True House Price')
plt.ylabel('Predicted House Price')
plt.title('Evaluation of Lasso Predictions')
plt.show()

# check the distribution of errors
errors = y_test - model.predict(x_test)
errors.hist(bins=30)
plt.show()

# feature importance
importance = pd.Series(np.abs(model.coef_.ravel()))
importance.index = features
importance.sort_values(inplace=True, ascending=False)
importance.plot.bar(figsize=(18,6))
plt.ylabel('Lasso Coefficients')
plt.title('Feature Importance')
plt.show()
