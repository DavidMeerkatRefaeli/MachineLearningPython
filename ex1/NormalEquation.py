import numpy as np


def normal_equation(X, y):
    xy = X.T @ y
    x_x = np.linalg.inv(X.T @ X)
    return x_x @ xy