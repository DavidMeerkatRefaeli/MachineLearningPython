import numpy as np
import ex3.Sigmoid as sg


# Cost and gradient have been separated to use Scipy optimizations functions which requires them separately
def cost_function_reg(X, y, lam, theta):
    theta = theta.reshape(-1, 1)
    h = sg.sigmoid(X @ theta)
    theta_1 = np.copy(theta)
    theta_1[0] = 0
    m = len(y)
    J = ((0 - y.T) @ np.log(h) - (1 - y).T @ np.log(1 - h)) / m + (lam / (2 * m)) * np.sum(theta_1**2)
    return np.asscalar(J)


def gradient_reg(X, y, lam, theta):
    theta = theta.reshape(-1, 1)
    h = sg.sigmoid(X @ theta)
    theta_1 = np.copy(theta)
    theta_1[0] = 0
    m = len(y)
    y_ = (h - y)
    grad = (X.T @ y_) / m + (lam / m) * theta_1
    return grad.flatten()
