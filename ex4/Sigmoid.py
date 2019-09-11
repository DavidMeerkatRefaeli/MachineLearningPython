import numpy as np


def sigmoid(z):
    return np.divide(1, (1 + np.exp(np.negative(z))))


def sigmoid_grad(z):
    return sigmoid(z) * (1 - sigmoid(z))