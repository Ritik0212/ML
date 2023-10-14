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


    def init_weights(self):
        self.w = np.random.randn(n, 1)

