#!/usr/bin/env python3

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import  Lasso
from sklearn.feature_selection import SelectFromModel

# load train and test set with engineered variables
x_train = pd.read_csv('datasets/x_train.csv')
x_test  = pd.read_csv('datasets/x_test.csv')

print(x_train.head(2))

# capture the log transfroemd target
y_train = x_train['SalePrice']
y_test  = x_test['SalePrice']

# drop unnessary variables from our training and testing sets
x_train.drop(['Id', 'SalePrice'], axis=1, inplace=True)
x_test.drop(['Id', 'SalePrice'], axis=1, inplace=True)

"""
Feature Selection
"""
selection = SelectFromModel(Lasso(alpha=0.005, random_state=123))

# train Lasso model
selection.fit(x_train, y_train)

# let us visualize the features that we selected
selection.get_support()

# pritn the number of total and selected features
selected_features = x_train.columns[(selection.get_support())]

print("Total Features: {}".format((x_train.shape[1])))
print("Selcted Features: {}".format(len(selected_features)))
print("Features with coeffients shrank to zero: {}".format(np.sum(selection.estimator_.coef_ == 0)))
print("Selected Features: {}".format(selected_features))

selected_features = x_train.columns[(selection.estimator_.coef_ !=0).ravel().tolist()]
print(selected_features)

pd.Series(selected_features).to_csv('datasets/selected_features.csv', index=False, header="False")
