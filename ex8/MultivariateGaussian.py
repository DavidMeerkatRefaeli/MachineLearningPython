import numpy as np


def multivariate_gaussian(X, mu, sigma2):
    k = len(mu)
    x = X - mu
    if len(sigma2.shape) == 1:
        sigma2 = np.diag(sigma2)
    det = np.linalg.det(sigma2)
    pinv = np.linalg.pinv(sigma2)
    exp = np.sum((x @ pinv) * x, 1)
    p = (2 * np.pi) ** (-k / 2) * det ** (-0.5) * np.exp(-0.5 * exp)
    return p


def estimate_gaussian(X):
    mu = np.mean(X, 0)
    # ddof is set to 1 to accommodate for Matlab sample variance (N-1) vs. python population variance (N)
    sigma2 = np.var(X, 0, ddof=1)
    return mu, sigma2