import numpy as np
import ex2.Sigmoid as sg


# Cost and gradient have been separated to use Scipy optimizations functions which requires them separately
def cost_function(X, y, theta):
    theta = theta.reshape((-1, 1))
    h = sg.sigmoid(X @ theta)
    m = len(y)
    J = (np.negative(y.T) @ np.log(h) - (1-y).T @ np.log(1-h)) / m
    return np.asscalar(J)


def gradient_function(X, y, theta):
    theta = theta.reshape((-1, 1))
    h = sg.sigmoid(X @ theta)
    m = len(y)
    grad = (X.T @ (h - y)) / m
    return grad.flatten()


def cost_function_reg(X, y, lam, theta):
    theta = theta.reshape((-1, 1))
    h = sg.sigmoid(X @ theta)
    theta_1 = np.copy(theta)
    theta_1[0] = 0
    m = len(y)
    J = (np.negative(y.T) @ np.log(h) - (1-y).T @ np.log(1-h)) / m + (lam / (2*m))*np.sum(theta_1**2)
    return J


def gradient_reg(X, y, lam, theta):
    theta = theta.reshape((-1, 1))
    h = sg.sigmoid(X @ theta)
    theta_1 = np.copy(theta)
    theta_1[0] = 0
    m = len(y)
    grad = (X.T @ (h - y)) / m + (lam / m)*theta_1
    return grad
