import numpy as np

from ex5.LinearRegressionCostFunction import linear_regression_cost_function
from ex5.TrainLinearRegression import train_linear_reg


def learning_curves(X, y, Xval, yval, lambd):
    m = np.size(X, 0)
    error_train = np.zeros(m)
    error_val = np.zeros(m)
    for i in range(m):
        Xt = X[:i+1, :]
        Yt = y[:i+1]
        theta_i = train_linear_reg(Xt, Yt, lambd)
        error_train_i = linear_regression_cost_function(Xt, Yt, 0, theta_i)
        error_val_i = linear_regression_cost_function(Xval, yval, 0, theta_i)
        error_train[i] = error_train_i
        error_val[i] = error_val_i
    return error_train, error_val