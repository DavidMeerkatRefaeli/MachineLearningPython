from functools import partial

import numpy as np

from ex8.CollaborativeFiltering import collaborative_filtering_cost_function as cost, \
    collaborative_filtering_gradient as gradient


def check_gradient(lambd=0):
    # Create small problem
    X_t = np.random.rand(4, 3)
    Theta_t = np.random.rand(5, 3)

    # Zap out most entries
    Y = X_t @ Theta_t.T
    zapped = np.random.random(Y.shape) > 0.5
    Y[zapped] = 0
    R = np.zeros_like(Y)
    R[Y != 0] = 1

    # Run Gradient Checking
    X = np.random.randn(X_t.shape[0], X_t.shape[1])
    Theta = np.random.randn(Theta_t.shape[0], Theta_t.shape[1])
    num_users = np.size(Y, 1)
    num_movies = np.size(Y, 0)
    num_features = np.size(Theta_t, 1)

    cost_func = partial(cost, Y, R, num_users, num_movies, num_features, lambd)

    params = np.concatenate([X.ravel(order='F'), Theta.ravel(order='F')])

    numgrad = numerical_grad(cost_func, params)
    grad = gradient(Y, R, num_users, num_movies, num_features, lambd, params)
    diff = np.linalg.norm(numgrad-grad)/np.linalg.norm(numgrad+grad)
    return diff


def numerical_grad(J, theta):
    numgrad = np.zeros_like(theta)
    perturb = np.zeros_like(theta)
    e = 1e-4
    for p in range(theta.size):
        perturb[p] = e
        loss1 = J(theta - perturb)
        loss2 = J(theta + perturb)
        numgrad[p] = (loss2 - loss1) / (2 * e)
        perturb[p] = 0
    return numgrad