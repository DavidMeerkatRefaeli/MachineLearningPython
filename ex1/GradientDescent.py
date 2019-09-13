import numpy as np
from ex1.ComputeCost import compute_cost as cc


def gradient_descent(X, y, theta, alpha, iterations):
    m = len(y)
    J_history = np.zeros(iterations)
    for it in range(iterations):
        h = X @ theta  # hypothesis
        theta -= alpha * (1 / m) * (X.T @ (h - y))
        J_history[it] = cc(X, y, theta)
    return theta, J_history