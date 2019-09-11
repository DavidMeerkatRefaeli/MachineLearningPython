import numpy as np


def feature_norm(X):
    mu = np.mean(X, axis=0)
    sigma = np.std(X, axis=0)
    return (X - mu)/sigma

