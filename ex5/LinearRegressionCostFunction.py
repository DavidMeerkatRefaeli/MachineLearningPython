import numpy as np


def linear_regression_cost_function(X, y, lambd, theta):
    theta = theta.reshape(-1, 1)
    h = X @ theta
    m = np.size(y, 0)
    theta_excluding_intercept = np.insert(theta[1:], 0, 0).reshape(-1, 1)
    J0 = (1 / (2 * m)) * (((h - y)**2).sum())
    J1 = (lambd / (2 * m)) * (theta_excluding_intercept.T @ theta_excluding_intercept)
    J = J0 + J1
    return np.asscalar(J)

def linear_regression_gradient_function(X, y, lambd, theta):
    theta = theta.reshape(-1, 1)
    h = X @ theta
    m = np.size(y, 0)
    theta_excluding_intercept = np.insert(theta[1:], 0, 0).reshape(-1, 1)
    grad = (1 / m) * (X.T @ (h-y) + lambd * theta_excluding_intercept)
    return grad.flatten()

