import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')


class DecisionTreeClassifier:

    def __init__(self, criterion='entropy', max_depth=None, min_samples_split=2):
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split

    def compute_entropy_for_node(self, x, y, label):
        total = len(y)
        label_count = len(y[y == label])
        p1 = label_count/total

        h = - p1 * np.log2(p1) - (1 - p1) * (np.log2(1 - p1))

    def compute_ig(self, x, y, parent_node_ig):
        node_ig = parent_node_ig
        all_labels = np.unique(y)
        num_features = x.shape[1]

        # splitting on features one by one
        for f in range(num_features):
            values = x[:, f]
            t_values = values[values == 1]
            ty_values = y[values == 1]
            f_values = values[values == 0]
            fy_values = y[values == 0]




