#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from flask import Flask, render_template, url_for, request
import pandas as pd
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

import joblib

app = Flask(__name__)
cv = CountVectorizer()
with open('datasets/rf.pkl', 'rb') as model :
    model = pickle.load(model)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict',methods=['POST'])
def predict():

	if request.method == 'POST':
		message = request.form['message']
		data = [message]
		vect = cv.transform(data).toarray()
		my_prediction = model.predict(vect)
	return render_template('result.html',prediction = my_prediction)


if __name__ == '__main__' :
    app.run(host='127.0.0.1', port=5000, debug='on')
