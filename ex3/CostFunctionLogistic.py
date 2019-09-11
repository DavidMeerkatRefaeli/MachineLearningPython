import numpy as np
import Sigmoid as sg

def cost_function(theta, X, y):
    h = sg.sigmoid(X @ theta)
    m = len(y)
    J = (np.negative(y.T) @ np.log(h) - (1-y).T @ np.log(1-h)) / m
    return J


def gradient(theta, X, y):
    h = sg.sigmoid(X @ theta)
    m = len(y)
    grad = (np.matmul(X.T, h - y)) / m
    return grad


def cost_function_reg(theta, X, y, lam):
    theta = theta.reshape((-1, 1))
    h = sg.sigmoid(X @ theta)
    theta_1 = np.copy(theta)
    theta_1[0] = 0
    m = len(y)
    J = (~y.T @ np.log(h) - (1 - y).T @ np.log(1 - h)) / m + (lam / (2 * m)) * np.sum(theta_1 ** 2)
    return J


def gradient_reg(theta, X, y, lam):
    theta = theta.reshape((-1, 1))
    h = sg.sigmoid(X @ theta)
    theta_1 = np.copy(theta)
    theta_1[0] = 0
    m = len(y)
    y_ = (h - y)
    grad = (X.T @ y_) / m + (lam / m) * theta_1
    return grad.reshape((-1,))
