import numpy as np


def find_closest_centroids(X, centroids):
    m = np.size(X, 0)
    k = np.size(centroids, 0)
    idx = np.zeros(m)
    for i in range(m):
        x = X[i, :]
        lens = np.zeros(k)
        for j in range(k):
            cent = centroids[j, :]
            lens[j] = np.linalg.norm(x - cent)
        index = np.argmin(lens)
        idx[i] = index
    return idx.astype(int)
