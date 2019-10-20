import numpy as np


def pca(X):
    m, n = X.shape
    sigma = (1/m)*(X.T @ X)  # Covariance matrix
    U, S, V = np.linalg.svd(sigma)
    return U, S