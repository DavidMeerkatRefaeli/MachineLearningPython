import numpy as np


def pca(X_norm):
    m, n = X_norm.shape
    sigma = (1/m)*(X_norm.T @ X_norm)  # Covariance matrix
    U, S, V = np.linalg.svd(sigma)
    return U, S