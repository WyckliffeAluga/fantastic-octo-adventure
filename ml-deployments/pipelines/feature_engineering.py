#!/usr/bin/env python3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import warnings

warnings.simplefilter(action='ignore')

# load data
data = pd.read_csv('datasets/train.csv')
print('Data Shape: {}'.format(data.shape))

# separate dataset
x_train , x_test, y_train, y_test = train_test_split(data,
                                    data['SalePrice'],
                                    test_size=0.1,
                                    random_state=123)

print("x-train shape: {} \t x_test.shape: {}".format(x_train.shape, x_test.shape))

"""
Part 1
Missing Values
"""
variables_with_na = [var for var in data.columns
                    if x_train[var].isnull().sum() > 0 and x_train[var].dtypes == "O"]

# print percentage of missing per variable
print(x_train[variables_with_na].isnull().mean())

# replace missing values with new label : "Missing"
x_train[variables_with_na] = x_train[variables_with_na].fillna("missing")
x_test[variables_with_na]  = x_test[variables_with_na].fillna("missing")

# check that we have missing information in the engineered variables
print(x_train[variables_with_na].isnull().sum())

# check that test set does not contain null values in the engineered variables
print([var for  var in variables_with_na if x_test[var].isnull().sum() > 0])

"""
Part 2
Numerical Variables
"""
# make a list with numerical variables that contain missing values
variables_with_na = [
var for var in data.columns if x_train[var].isnull().sum() >0 and x_train[var].dtypes != "O"
]

# print percentage missing values per variable
print(x_train[variables_with_na].isnull().mean())

# replace engineer missing values
for var in variables_with_na :

    # calculate the mode using the train set
    mode = x_train[var].mode()[0]

    # add binary missing indicator (in train and test)
    x_train[var+'_na'] = np.where(x_train[var].isnull(), 1, 0)
    x_test[var+'_na'] = np.where(x_test[var].isnull(), 1, 0)

    # replace missing avrues by the mode
    x_train[var] = x_train[var].fillna(mode)
    x_test[var] = x_test[var].fillna(mode)

print(x_train[variables_with_na].isnull().sum())

# check that test set deos not contain null values in the engineered variables
print([vr for var in variables_with_na if x_test[var].isnull().sum() > 0])

# check the binary missing indicator variables
print(x_train[['LotFrontage_na', 'MasVnrArea_na', 'GarageYrBlt_na']].head())

"""
Part 3
Temporal variables
"""

def elapsed_years(df, var):
    # capture difference between the year variable adn the year in which the house was sold
    df[var] = df['YrSold'] - df[var]
    return df

for var in ['YearBuilt', 'YearRemodAdd', 'GarageYrBlt'] :
    x_train = elapsed_years(x_train, var)
    x_test = elapsed_years(x_test, var)

"""
Part 4:
Numerical variable transformation
"""

for var in ['LotFrontage', 'LotArea', '1stFlrSF', 'GrLivArea', 'SalePrice'] :
    x_train[var] = np.log(x_train[var])
    x_test[var] = np.log(x_test[var])

# check that train set does not contain null values in the engineered variable
print([var for var in ['LotFrontage', 'LotArea', '1stFlrSF', 'GrLivArea',
        'SalePrice'] if x_train[var].isnull().sum() > 0])

# check that the test set does not contain null values in the engineered variable
print([var for var in ['LotFrontage', 'LotArea', '1stFlrSF', 'GrLivArea',
        'SalePrice'] if x_test[var].isnull().sum() > 0])

"""
Part 5
Catergorical variable
"""

# capute Catergorical variables
catergorical_variables = [var for var in x_train.columns if x_train[var].dtype == 'O']

def find_frequent_variables(df, var, rare_perc) :

    # find the labels that are shared by nmore thatn a certain %
    df = df.copy()
    tmp = df.groupby(var)['SalePrice'].count() / len(df)
    return tmp[tmp > rare_perc].index

for var in catergorical_variables:
    frequent_ls = find_frequent_variables(x_train, var, 0.01)

    # replace rare categories with the string 'rare'
    x_train[var] = np.where(x_train[var].isin(frequent_ls), x_train[var], 'Rare')

    x_test[var] = np.where(x_test[var].isin(frequent_ls), x_test[var], 'Rare')

"""
Encoding caterogical variables
"""

def replace_catefories(train, test, var, target) :
    # order the categories in a variable from that with the loswest sale price to that with the highest
    ordered_labels = train.groupby([var])[target].mean().sort_values().index

    # create a dictionary of ordered catergories to integer calues
    ordinal_labels = {k: i for i, k in enumerate(ordered_labels, 0)}

    # use the dictionary to replace thr categorical strings with intergers
    train[var] = train[var].map(ordinal_labels)
    test[var] = test[var].map(ordinal_labels)

for var in catergorical_variables:
    replace_catefories(x_train, x_test, var, 'SalePrice')

# check absence of na in the train set
print([var for var in x_train.columns if x_train[var].isnull().sum() > 0])

# check for absence of n in the test set
print([var for var in x_test.columns if x_train[var].isnull().sum() > 0])

# check the monotonic relationship between lables and target
def analyse_vars(df, var) :
    # plots median house sale price per encoded category
    df = df.copy()
    df.groupby(var)['SalePrice'].median().plot.bar()
    plt.title(var)
    plt.ylabel('SalePrice')
    plt.show()

for var in catergorical_variables :
    analyse_vars(x_train, var)

"""
Part 6
Feature Scaling
"""

# capture all variables in a list except the target and ID
training_variables = [var for var in x_train.columns if var not in ['Id', 'SalePrice']]

print('Length of variables : {}'.format(len(training_variables)))

# create a scaler
scaler = MinMaxScaler()
# fit the scaler to the train set
scaler.fit(x_train[training_variables])
# transform the train and test set
x_train[training_variables] = scaler.transform(x_train[training_variables])
x_test[training_variables] = scaler.transform(x_test[training_variables])

print(x_train.head(1))

# save the transformed
x_train.to_csv('datasets/x_train.csv', index=False)
x_test.to_csv('datasets/x_test.csv', index=False)
