import numpy as np

from ex1.ComputeCostMulti import compute_cost_multi


def gradient_descent_multi(X, y, theta, alpha, num_iters):
    m = len(y)
    j_history = np.zeros((num_iters, 1))
    for i in range(num_iters):
        h = X @ theta
        theta = theta - (alpha / m) * (X.T @ (h - y))
        j_history = compute_cost_multi(X, y, theta)
    return theta, j_history