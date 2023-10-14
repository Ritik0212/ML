import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')


class SGDRegressor:
    """
        x (nd-array): Shape (m, n) input features to the model
        y (nd-array): Shape (m, 1) target variable or label
        w (nd-array): Shape (n, 1) Parameters (weights) of the model
        b (scalar):  Parameter of the model
        m:           number of training examples
        n:           number of features
    """

    def __init__(self, lr=0.1, regu=0, epochs=1500, batch_size=1):
        self.lr = lr
        self.regu = regu
        self.epochs = epochs
        self.batch_size = batch_size

    def init_weights(self, n):
        self.w = np.random.randn(n, 1)
        # self.b = np.ones((1, 1))
        # self.w = np.zeros((n, 1))
        # self.b = np.zeros((1, 1))

    def compute_cost(self, x, y, w):
        """
        Returns:
            total cost of the model for given dataset
        """
        m = x.shape[0]
        y_hat = x.dot(w)
        cost = np.square(y_hat - y)
        cost = np.sum(cost)
        # cost += self.regu * np.sum(np.square(w))
        cost = cost / (m)
        return cost

    def compute_gradient(self, x, y, w):
        m = x.shape[0]
        y_hat = x.dot(w)
        loss = y_hat - y
        dj_dw = x.T.dot(loss) + self.regu * w
        # dj_dw = np.sum(loss.T.dot(x)) + self.regu * w
        # dj_db = np.sum(loss, axis=0)
        # dj_dw = np.vstack((dj_db, dj_dw))
        dj_dw = dj_dw / m
        # dj_db = np.sum(loss, axis=0) / m

        return dj_dw

    def gradient_descent(self, x, y):
        bias_col = np.ones((x.shape[0], 1))
        x = np.hstack((bias_col, x))  # adding bias feature

        cost_hist = []
        m, n = x.shape
        self.init_weights(n)
        w = self.w
        epochs = self.epochs

        for i in range(epochs):
            n_iters = int(m/self.batch_size)
            for j in range(n_iters):
                indices_for_batch = np.random.randint(m, size=self.batch_size)
                batch_x = x[indices_for_batch]
                batch_y = y[indices_for_batch]
                dj_dw = self.compute_gradient(batch_x, batch_y, w)
                w = w - self.lr * dj_dw
                # b = b - self.lr * dj_db

            cost = self.compute_cost(x, y, w)
            cost_hist.append(cost)

            if i % math.ceil(epochs / 10) == 0 or i == epochs - 1:
                print(f"Epoch {i:4}: Cost {float(cost_hist[-1]):8.2f}   ")

        return w


    def fit(self, x, y):

        self.w = self.gradient_descent(x, y)

    def predict(self, x):
        """
        shape of nd-array x: (m, n)

        """
        bias_col = np.ones((x.shape[0], 1))
        x = np.hstack((bias_col, x))  # adding bias feature

        y_hat = x.dot(self.w)
        return y_hat

    def rms_error(self, x, y):
        m = x.shape[0]
        y_hat = self.predict(x)
        rmse = np.sqrt((np.sum((y - y_hat) ** 2))/m)
        return rmse

    def score(self, x, y):
        y_hat = self.predict(x)
        rss = (np.sum((y - y_hat) ** 2))     # residual sum of squares
        tss = (np.sum((y - y.mean()) ** 2))  # total sum of squares
        cod = 1 - (rss/tss)                  # coefficient of determination
        return cod

    def print_predict_vs_actual(self, x, y):
        df = pd.DataFrame()
        df['predictions'] = self.predict(x).ravel()
        df['actual'] = y.ravel()
        print(df.to_string())

    def plot_predicted_vs_actual(self, x, y):
        y_hat = self.predict(x)
        plt.scatter(y, y_hat, c='green')
        plt.plot(y, y, c='blue')
        plt.xlabel("Price: in $1000's")
        plt.ylabel("Predicted value")
        plt.title("True value vs predicted value : Linear Regression")
        plt.show()

