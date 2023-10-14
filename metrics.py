import numpy as np

def mean_squared_error(y, y_hat):
    m = len(y)
    return (np.sum((y - y_hat) ** 2)) / m