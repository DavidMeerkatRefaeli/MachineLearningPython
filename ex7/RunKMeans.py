from ex7.ComputeCenroids import compute_centroids
from ex7.FindClosestCentroid import find_closest_centroids


def kMeans(X, k, initial, max_iter=10):
    current = initial
    idx = None
    for i in range(max_iter):
        idx = find_closest_centroids(X, current)
        current = compute_centroids(X, idx, k)
    return current, idx