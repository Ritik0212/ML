import numpy as np
import pandas as pd
from mlutility import train_test_split
from preprocessing import StandardScaler, MinMaxScaler
from metrics import mean_squared_error
from sgdregressor import SGDRegressor
import mlutility
from sklearn.datasets import load_boston


if __name__ == '__main__':
    boston_data = load_boston()

    feature_names = boston_data['feature_names']
    # ['CRIM' 'ZN' 'INDUS' 'CHAS' 'NOX' 'RM' 'AGE' 'DIS' 'RAD' 'TAX' 'PTRATIO', 'B' 'LSTAT']

    feature_indices = list(range(len(feature_names)))

    # B, DIS, CHAS has lower correlation with price
    # lets remove them
    # feature_indices = [0, 1, 2, 4, 5, 6, 7, 9, 10, 11]

    f, t = boston_data['data'], boston_data['target']
    f = f[:, feature_indices]
    t = t.reshape((t.shape[0], -1))

    f_train, f_test, t_train, t_test = train_test_split(f, t, test_size=0.2)
    # from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()
    f_train = scaler.fit_transform(f_train)
    f_test = scaler.transform(f_test)

    lin_model1 = SGDRegressor(lr=0.01, regu=0.1, epochs=500, batch_size=1)
    lin_model1.fit(f_train, t_train)

    # lin_model1.print_predict_vs_actual(f, t)
    y_hat = lin_model1.predict(f_train)
    print(lin_model1.score(f_train, t_train))
    print(lin_model1.rms_error(f_train, t_train))
    print(mean_squared_error(t_train, y_hat))

    y_hat = lin_model1.predict(f_test)
    print(lin_model1.score(f_test, t_test))
    print(lin_model1.rms_error(f_test, t_test))
    print(mean_squared_error(t_test, y_hat))

    lin_model1.plot_predicted_vs_actual(f_train, t_train)
    lin_model1.plot_predicted_vs_actual(f_test, t_test)
    # from sklearn import linear_model.S