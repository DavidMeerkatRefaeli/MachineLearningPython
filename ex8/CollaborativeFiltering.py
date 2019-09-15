import numpy as np


def collaborative_filtering_cost_function(Y, R, num_users, num_movies, num_features, lambd, params):
    l = num_movies * num_features
    X = params[:l].reshape((num_movies, num_features), order='F')
    Theta = params[l:].reshape((num_users, num_features), order='F')
    err = (X @ Theta.T - Y)*R
    J = np.sum(err**2) / 2 + (lambd / 2) * (np.sum(Theta**2) + np.sum(X**2))
    return J


def collaborative_filtering_gradient(Y, R, num_users, num_movies, num_features, lambd, params):
    l = num_movies * num_features
    X = params[:l].reshape((num_movies, num_features), order='F')
    Theta = params[l:].reshape((num_users, num_features), order='F')
    err = (X @ Theta.T - Y)*R
    X_grad = err @ Theta + lambd * X
    Theta_grad = err.T @ X + lambd * Theta
    grad = np.concatenate([X_grad.ravel(order='F'), Theta_grad.ravel(order='F')])
    return grad
