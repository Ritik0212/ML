import numpy as np

# class MlUtility:
#
#     def __init__(self):
#         pass
#
#     @staticmethod

def train_test_split(X, Y, test_size=None, train_size=None, shuffle=True):
    m = X.shape[0]

    if shuffle:
        p = np.random.permutation(m)
        X = X[p]
        Y = Y[p]

    if train_size is None:
        if test_size is not None:
            train_size = 1 - test_size
        else:
            train_size = 0.75

    train_last_idx = int(m * train_size)
    x_train = X[:train_last_idx]
    x_test = X[train_last_idx:]
    y_train = Y[:train_last_idx]
    y_test = Y[train_last_idx:]

    # print(f'X train shape: {x_train.shape}')
    # print(f'X test shape: {x_test.shape}')
    # print(f'Y train shape: {y_train.shape}')
    # print(f'Y test shape: {y_test.shape}')

    return x_train, x_test, y_train, y_test