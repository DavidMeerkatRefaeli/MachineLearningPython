import numpy as np
import matplotlib.pyplot as plt


def poly_features(X, p):
    """Maps X (1D vector) into the p-th power"""
    X_poly = np.zeros((np.size(X), p))
    for i in range(0, p):
        X_poly[:, i] = (X**(i+1)).flatten()
    return X_poly


def poly_plot(min_x, max_x, mu, sigma, theta, p):
    """Plots a polynomial curve"""
    x = np.arange(min_x - 15, max_x + 25, 0.05).reshape(-1, 1)
    x_poly = poly_features(x, p)
    x_poly = (x_poly - mu) / sigma
    x_poly = np.c_[np.ones(np.size(x, 0)), x_poly]
    plt.plot(x, x_poly @ theta, '--', linewidth=2)
    plt.title('Polynomial fit')

