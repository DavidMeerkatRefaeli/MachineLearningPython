import numpy as np

def compute_cost(X, y, theta):
    m = len(y)
    err = np.matmul(X, theta) - y
    err_sq = err ** 2
    J = np.sum(err_sq) / (2 * m)
    return J

