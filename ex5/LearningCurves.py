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


def learning_curves_random(X, y, Xval, yval, lambd, N=50):
    m = np.size(X, 0)
    error_train = np.zeros(m)
    error_val = np.zeros(m)
    for i in range(m):
        error_train_i = []
        error_val_i = []
        for n in range(N):
            sel = np.random.permutation(m)
            sel = sel[0:i+1]
            Xt = X[sel, :]
            Yt = y[sel]
            Xtval = Xval[sel, :]
            ytval = yval[sel]
            theta_i = train_linear_reg(Xt, Yt, lambd)
            error_train_i.append(linear_regression_cost_function(Xt, Yt, 0, theta_i))
            error_val_i.append(linear_regression_cost_function(Xtval, ytval, 0, theta_i))
        error_train[i] = sum(error_train_i)/len(error_train_i)
        error_val[i] = sum(error_val_i)/len(error_val_i)
    return error_train, error_val