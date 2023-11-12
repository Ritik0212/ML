import numpy as np
import pandas as pd
from mlutility import train_test_split
from preprocessing import StandardScaler, MinMaxScaler
from metrics import mean_squared_error, confusion_matrix, precision_recall_fscore_support
from sgdregressor import SGDRegressor
from sgdclassifier import SGDClassifier
from naive_bayes import GaussianNB
import mlutility
from sklearn.datasets import load_boston
from sklearn.datasets import load_diabetes, load_breast_cancer, make_classification
import os

DATA_DIR = 'data'

if __name__ == '__main__':
    # boston_data = load_boston()
    #
    # feature_names = boston_data['feature_names']
    # # ['CRIM' 'ZN' 'INDUS' 'CHAS' 'NOX' 'RM' 'AGE' 'DIS' 'RAD' 'TAX' 'PTRATIO', 'B' 'LSTAT']
    #
    # feature_indices = list(range(len(feature_names)))
    #
    # # B, DIS, CHAS has lower correlation with price
    # # lets remove them
    # # feature_indices = [0, 1, 2, 4, 5, 6, 7, 9, 10, 11]
    #
    # f, t = boston_data['data'], boston_data['target']
    # f = f[:, feature_indices]
    # t = t.reshape((t.shape[0], -1))
    #
    # f_train, f_test, t_train, t_test = train_test_split(f, t, test_size=0.2)
    # # from sklearn.preprocessing import StandardScaler
    #
    # scaler = StandardScaler()
    # f_train = scaler.fit_transform(f_train)
    # f_test = scaler.transform(f_test)
    #
    # lin_model1 = SGDRegressor(lr=0.01, regu=0.1, epochs=500, batch_size=1)
    # lin_model1.fit(f_train, t_train)
    #
    # # lin_model1.print_predict_vs_actual(f, t)
    # y_hat = lin_model1.predict(f_train)
    # print(lin_model1.score(f_train, t_train))
    # print(lin_model1.rms_error(f_train, t_train))
    # print(mean_squared_error(t_train, y_hat))
    #
    # y_hat = lin_model1.predict(f_test)
    # print(lin_model1.score(f_test, t_test))
    # print(lin_model1.rms_error(f_test, t_test))
    # print(mean_squared_error(t_test, y_hat))
    #
    # lin_model1.plot_predicted_vs_actual(f_train, t_train)
    # lin_model1.plot_predicted_vs_actual(f_test, t_test)
    # # from sklearn import linear_model.S


    # ------------ for sgdclassifier -------------------

    # breast_cancer_data = load_breast_cancer()
    #
    # # print(diabetes_data)
    # feature_names = breast_cancer_data['feature_names']
    # # print(feature_names)
    # # print(breast_cancer_data['DESCR'])
    # feature_indices = list(range(len(feature_names)))
    #
    #
    # f, t = breast_cancer_data['data'], breast_cancer_data['target']
    # # print(f)
    # # print(t)
    # f = f[:, feature_indices]
    # t = t.reshape((t.shape[0], -1))
    #
    # f_train, f_test, t_train, t_test = train_test_split(f, t, test_size=0.2)
    # # from sklearn.preprocessing import StandardScaler
    #
    # scaler = StandardScaler()
    # f_train = scaler.fit_transform(f_train)
    # f_test = scaler.transform(f_test)
    #
    # c1 = SGDClassifier(lr=0.01, regu=0, epochs=500, batch_size=1)
    # c1.fit(f_train, t_train)
    #
    # # c1.print_predict_vs_actual(f_train, t_train)
    # y_pred_train = c1.predict(f_train)
    # print(c1.score(f_train, t_train))
    #
    # y_pred_test = c1.predict(f_test)
    # print(c1.score(f_test, t_test))
    #
    # print(confusion_matrix(t_test, y_pred_test))
    # print(precision_recall_fscore_support(t_test, y_pred_test))

    # y_true = np.array(["cat", "ant", "cat", "cat", "ant", "bird"])
    # y_pred = np.array(["ant", "ant", "cat", "cat", "ant", "cat"])
    # print(confusion_matrix(y_true, y_pred, labels=np.array(["ant", "bird", "cat"])))
    #
    # y_true = np.array(['cat', 'dog', 'pig', 'cat', 'dog', 'pig'])
    # y_pred = np.array(['cat', 'pig', 'dog', 'cat', 'cat', 'dog'])
    # print(precision_recall_fscore_support(y_true, y_pred, labels=np.array(['pig', 'dog', 'cat'])))



    # naive bayes
    # loan_data_file_path = os.path.join(DATA_DIR, 'loan_data.csv')
    # loan_data = pd.read_csv(loan_data_file_path)
    #
    # pre_df = pd.get_dummies(loan_data, columns=['purpose'], drop_first=True)
    #
    # print(loan_data.head())
    # print(pre_df.head())

    X, y = make_classification(
        n_features=,
        n_classes=3,
        n_samples=800,
        n_informative=2,
        random_state=1,
        n_clusters_per_class=1,
    )
    gnb = GaussianNB()
    gnb.fit(X, y)


