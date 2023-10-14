import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')


class SGDClassifier:

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

    def sigmoid(self, v):
        return 1/(1 + np.exp(-v))

    def hypothesis(self, x, w):
        v = x.dot(w)
        v = self.sigmoid(v)
        return v

    def compute_cost(self, x, y, w):
        """
        Returns:
            total cost of the model for given dataset
        """
        m = x.shape[0]
        y_hat = self.hypothesis(x, w)
        cost = y * np.log(y_hat) + (1 - y) * (np.log(1 - y_hat))
        cost = - np.sum(cost) / (m)
        return cost

    def compute_gradient(self, x, y, w):
        m = x.shape[0]
        y_hat = self.hypothesis(x, w)
        loss = y_hat - y
        dj_dw = x.T.dot(loss) + self.regu * w
        dj_dw = dj_dw / m

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
        w = self.w
        y_hat = self.hypothesis(x, w)

        return np.array(y_hat > 0.5, dtype=int)

    def score(self, x, y):
        y_hat = self.predict(x)
        v = np.sum(y == y_hat)/len(y)
        return v

    def print_predict_vs_actual(self, x, y):
        df = pd.DataFrame()
        df['predictions'] = self.predict(x).ravel()
        df['actual'] = y.ravel()
        print(df.to_string())
