import numpy as np


def compute_centroids(X, idx, k):
    m, n = X.shape
    centroids = np.zeros((k, n))
    for i in range(k):
        centroids[i, :] = X[idx == i].mean(axis=0)
    return centroids