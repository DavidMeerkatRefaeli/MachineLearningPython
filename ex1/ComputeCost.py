import numpy as np


def compute_cost(X, y, theta):
    m = len(y)
    err = X @ theta - y
    err_sq = err**2
    J = np.sum(err_sq) / (2 * m)
    return J

