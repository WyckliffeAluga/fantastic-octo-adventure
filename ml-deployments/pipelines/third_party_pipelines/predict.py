#!/usr/bin/env python3
import pandas as pd
import joblib
import config


def make_prediction(input_data) :
    _pipe_price = joblib.load(filename=config.PIPELINE_NAME)
    results = _pipe_price.predict(input_data)
    return results

if __name__ == "__main__" :

    # test pipeline
    import numpy as np
    from sklearn.model_selection import  train_test_split
    from sklearn.metrics import mean_squared_error, r2_score

    data = pd.read_csv(config.TRAINING_DATA_FILE)

    x_train, x_test, y_train, y_test = train_test_split(
            data[config.FEATURES],
            data[config.TARGET],
            test_size = 0.1 ,
            random_state = 123
    )
    pred = make_prediction(x_test)

    # determine mse and rmse
    print("test mse: {}".format(int(mean_squared_error(y_test, np.exp(pred)))))
    print("test rmse: {}".format(int(np.sqrt(mean_squared_error(y_test, np.exp(pred))))))
    print("test r2: {}".format(y_test, np.exp(pred)))
    print()
