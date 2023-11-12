import numpy as np
import pandas as pd


def compute_prior_prob(data):
    classes, counts = np.unique(data, return_counts=True)
    prob = counts/np.sum(counts)
    return classes, counts


class GaussianNB:

    def __init__(self):
        pass

    def fit(self, features, target):
        prior_prob = compute_prior_prob(target)
        print(prior_prob)

