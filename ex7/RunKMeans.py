import numpy as np

from ex7.ComputeCenroids import compute_centroids
from ex7.FindClosestCentroid import find_closest_centroids


def kMeans(X, k, initial, max_iter=10):
    current = initial
    idx = None
    for i in range(max_iter):
        idx = find_closest_centroids(X, current)
        current = compute_centroids(X, idx, k)
    return current, idx


# Random initialization
def kMeans_init_centroids(X, k):
    m = np.size(X, 0)
    idx = np.random.permutation(m)[:k]
    return X[idx]
