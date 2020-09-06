#!/usr/bin/env python3
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import Lasso
from sklearn.metrics import  mean_squared_error, r2_score
from math import sqrt

import joblib
import warnings
warnings.simplefilter(action='ignore')

# load data
data = pd.read_csv('datasets/train.csv')
print(data.shape)

X_train, X_test, y_train, y_test = train_test_split(
    data,
    data['SalePrice'],
    test_size=0.1,
    random_state=123)

print("x_train shape: {} \t x_test.shape: {}".format(X_train.shape, X_test.shape))

# load selected features
features = pd.read_csv('datasets/selected_features.csv')
# Added the extra feature, LotFrontage
features = features['0'].to_list() + ['LotFrontage']

print("Number of features: {}".format(len(features)))

"""
Engineer missing values
"""

# categorical variables
# make a list of the categorical variables
variables_with_na = [
    var for var in features
    if X_train[var].isnull().sum() > 0 and X_train[var].dtypes == "O"
    ]

print('categorical variables: {}'.format(variables_with_na))

X_train[variables_with_na] = X_train[variables_with_na].fillna('Missing')
X_test[variables_with_na] = X_test[variables_with_na].fillna('Missing')

# check that we have no missing information in the engineered variables
print(X_train[variables_with_na].isnull().sum())


"""
Numerical Variables
"""

# make a list of the numerical variables that contain missing values:
variables_with_na = [
    var for var in features
    if X_train[var].isnull().sum() > 0 and X_train[var].dtypes != 'O'
]

print("Numerical variables: {}".format(variables_with_na))

var = 'LotFrontage'

# calculate the mode
mode_val = X_train[var].mode()[0]
print('mode of LotFrontage: {}'.format(mode_val))

# replace missing values by the mode
# (in train and test)
X_train[var] = X_train[var].fillna(mode_val)
X_test[var] = X_test[var].fillna(mode_val)

"""
Temporal Variables
"""

def elapsed_years(df, var):
    # capture difference between year variable
    # and year in which the house was sold

    df[var] = df['YrSold'] - df[var]

    return df

X_train = elapsed_years(X_train, 'YearRemodAdd')
X_test = elapsed_years(X_test, 'YearRemodAdd')

for var in ['LotFrontage', '1stFlrSF', 'GrLivArea', 'SalePrice']:
    X_train[var] = np.log(X_train[var])
    X_test[var] = np.log(X_test[var])

# categorical variables
categorical_variables = [var for var in features if X_train[var].dtype == 'O']

def find_frequent_labels(df, var, rare_perc):

    # function finds the labels that are shared by more than
    # a certain % of the houses in the dataset

    df = df.copy()

    tmp = df.groupby(var)['SalePrice'].count() / len(df)

    return tmp[tmp > rare_perc].index


for var in categorical_variables:

    # find the frequent categories
    frequent_ls = find_frequent_labels(X_train, var, 0.01)
    print(var)
    print(frequent_ls)
    print()

    # replace rare categories by the string "Rare"
    X_train[var] = np.where(X_train[var].isin(
        frequent_ls), X_train[var], 'Rare')

    X_test[var] = np.where(X_test[var].isin(
        frequent_ls), X_test[var], 'Rare')


def replace_categories(train, test, var, target):

    # order the categories in a variable from that with the lowest
    # house sale price, to that with the highest
    ordered_labels = train.groupby([var])[target].mean().sort_values().index

    # create a dictionary of ordered categories to integer values
    ordinal_label = {k: i for i, k in enumerate(ordered_labels, 0)}

    # use the dictionary to replace the categorical strings by integers
    train[var] = train[var].map(ordinal_label)
    test[var] = test[var].map(ordinal_label)

    print(var)
    print(ordinal_label)
    print()

for var in categorical_variables:
    replace_categories(X_train, X_test, var, 'SalePrice')

print([var for var in features if X_train[var].isnull().sum() > 0])
print([var for var in features if X_test[var].isnull().sum() > 0])

"""
Feature Scaling
"""
y_train = X_train['SalePrice']
y_test = X_test['SalePrice']

# set up scaler
scaler = MinMaxScaler()

# train scaler
scaler.fit(X_train[features])


# transform the train and test set, and add on the Id and SalePrice variables
X_train = scaler.transform(X_train[features])
X_test = scaler.transform(X_test[features])



model = Lasso(alpha=0.005, random_state=123)

# train the model
model.fit(X_train, y_train)
joblib.dump(model, 'datasets/lasso_regression.pkl')

# make predictions for train set
pred = model.predict(X_train)

# determine mse and rmse
print('train mse: {}'.format(int(
    mean_squared_error(np.exp(y_train), np.exp(pred)))))
print('train rmse: {}'.format(int(
    sqrt(mean_squared_error(np.exp(y_train), np.exp(pred))))))
print('train r2: {}'.format(
    r2_score(np.exp(y_train), np.exp(pred))))
print()

# make predictions for test set
pred = model.predict(X_test)

# determine mse and rmse
print('test mse: {}'.format(int(
    mean_squared_error(np.exp(y_test), np.exp(pred)))))
print('test rmse: {}'.format(int(
    sqrt(mean_squared_error(np.exp(y_test), np.exp(pred))))))
print('test r2: {}'.format(
    r2_score(np.exp(y_test), np.exp(pred))))
print()

print('Average house price: ', int(np.exp(y_train).median()))
