#!/usr/bin/env python3

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
pd.pandas.set_option('display.max_columns', None)

# load the data
data = pd.read_csv('datasets/train.csv')

# rows and columns of the data
#print(data.shape)

# visualize the dataset
#print(data.head())

"""
Part 1
Missing Values
"""
# make a list of the variables taht contain missing values
variables_with_na = [var for var in data.columns if data[var].isnull().sum()]

# determine the percentage of missing values
#print(data[variables_with_na].isnull().mean())

# evaluate the price of the house in the observatons where infomation is missing
def analyse_null_value(df, var) :

    df = df.copy()

    # make a variable that indicates 1 if the observaton was Missing
    df[var] = np.where(df[var].isnull(), 1 , 0)

    # compare the median sales price in the observation where data is missing
    # vs observations where a value is available
    df.groupby(var)['SalePrice'].median().plot.bar()
    plt.title(var)
    plt.show()

# run the function for each variable with missing data
#for var in variables_with_na :
    #analyse_null_value(data, var)

"""
Part 2
Numerical Variables
"""

# make a list of numerical variables
numerical_variables = [var for var in data.columns if data[var].dtypes != 'O']
print("Number of numerical variables: {}".format(numerical_variables))
# visualize the numerical variables
#print(data[numerical_variables].head())
print("Number of House ID labels: {}".format(data.Id.unique()))
print("Number of Houses in the dataset: {}".format(len(data)))

"""
Part 3
Temporal variables
"""

# list of variabless that contain year information
year_variables = [var for var in numerical_variables if 'Yr' in var or 'Year' in var]
print(year_variables)

# explore the values of the Temporal variables
for var in year_variables :
    print(var, data[var].unique())
    print()

# explore the evolution of the sale price with years in which the house was sold
data.groupby('YrSold')['SalePrice'].median().plot()
plt.ylabel("Median House Price")
plt.title("Change in House Price with the years")
plt.show()

# Explore the relationship between the year variables
# and the house price in a bit of more detail

def analyse_year_vars(df, var) :
    df = df.copy()

    # capture difference between year variable and year
    # in which the house was sold
    df[var] = df['YrSold'] - df[var]

    plt.scatter(df[var], df['SalePrice'])
    plt.ylabel('Sale Price')
    plt.xlabel(var)
    plt.show()

for var in year_variables :
    if var != 'YrSold':
        analyse_year_vars(data, var)

"""
Part 3
Discrete variables
"""
# make a list of discrete variables
discrete_variables = [var for var in numerical_variables if len(
    data[var].unique()) < 20 and var not in year_variables + ['Id']]
print("Number of discrete variables: {}".format(len(discrete_variables)))

# let us visualize the discrete variables
print(data[discrete_variables].head())

def analyse_discrete(df, var) :
    df = df.copy()
    df.groupby(var)['SalePrice'].median().plot.bar()
    plt.title(var)
    plt.ylabel('Median SalePrice')
    plt.show()

for var in discrete_variables :
    analyse_discrete(data, var)

"""
Part 4
Continous variables
"""
# make a list of continous variables
continous_variables = [var for var in numerical_variables if var not in discrete_variables + year_variables + ['Id']]
print('Number of continous variables: {}'.format(len(continous_variables)))

def analyse_continous(df, var) :
    df = df.copy()
    df[var].hist(bins=30)
    plt.ylabel("Number of houses")
    plt.xlabel(var)
    plt.title(var)
    plt.show()

for var in continous_variables :
    analyse_continous(data, var)


# analyse the distributions of these variables after applying a  logarithmic transformation

def analyse_transformed_continous(df, var) :
    df = df.copy()

    # skip 0 or negative values
    if any(data[var] <= 0) :
        pass
    else:
        # log transform the variable
        df[var] = np.log(df[var])
        df[var].hist(bins=30)
        plt.ylabel("Number of houses")
        plt.xlabel(var)
        plt.title(var)
        plt.show()

for var in continous_variables:
    analyse_transformed_continous(data, var)

# explore the relationship between the house price and the transformed variables with more detail

def transform_analyse_continuous(df, var) :
    df = df.copy()

    if any(data[var] <= 0) :
        pass
    else:
        df[var] = np.log(df[var])
        df['SalePrice'] = np.log(df['SalePrice'])

        # plot
        plt.scatter(df[var], df['SalePrice'])
        plt.ylabel('SalePrice')
        plt.xlabel(var)
        plt.show()

for var in continous_variables :
    if var != 'SalePrice' :
        transform_analyse_continuous(data, var)

"""
Part 5
Outliers
"""

# make boxplots to visualize outliers int he continous variables
def find_outliers(df, var) :
    df = df.copy()

    if any(data[var] <= 0) :
        pass
    else:
        df[var] = np.log(df[var])
        df.boxplot(column=var)
        plt.title(var)
        plt.ylabel(var)
        plt.show()

#for var in continous_variables :
#    find_outliers(data, var)

"""
Part 6
Categorical variables
"""
# capture categorical variables
categorical_variables = [var for var in data.columns if data[var].dtypes == 'O']
print("Number of categorical variables: {}".format(len(categorical_variables)))

def analyse_rare_labels(df, var, rare_perc) :
    df = df.copy()
    tmp = df.groupby(var)['SalePrice'].count() / len(df)
    # return categories that are rare
    return tmp[tmp < rare_perc]

# print categories that are present in less than
# 1 % of the observations
for  var in categorical_variables :
    print(analyse_rare_labels(data , var, 0.02))
    print()

for variables in categorical_variables :
    analyse_discrete(data, var)
