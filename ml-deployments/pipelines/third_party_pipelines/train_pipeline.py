#!/usr/bin/env python3

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
import joblib

import pipeline
import config

def run_training():

    # read training data
    data = pd.read_csv(config.TRAINING_DATA_FILE)

    # divide train and test
    x_train, x_test, y_train, y_test = train_test_split(
            data[config.FEATURES],
            data[config.TARGET],
            test_size = 0.1,
            random_state=123
    )

    # transform the target
    y_train = np.log(y_train)
    y_test = np.log(y_test)

    pipeline.price_pipe.fit(x_train[config.FEATURES], y_train)
    joblib.dump(pipeline.price_pipe, config.PIPELINE_NAME)

if __name__ == "__main__" :
    run_training()
