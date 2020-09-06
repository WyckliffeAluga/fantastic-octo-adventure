#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd
import pickle

iris = load_iris()
x = iris.data
y = iris.target

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.5, random_state=123)
clf = RandomForestClassifier(n_estimators=10)
clf.fit(x_train, y_train)
predicted = clf.predict(x_test)
print(accuracy_score(predicted, y_test))

with open('rf.pkl', 'wb') as model :
    pickle.dump(clf, model, protocol=2)
