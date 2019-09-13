from functools import partial

import numpy as np
from numpy import linalg as LA

from ex4.CostGradientNN import nn_cost_function, nn_gradient


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


def debug_initialize_weights(fan_out, fan_in):
    W = np.sin(np.arange(1, fan_out * (fan_in + 1) + 1)) / 10
    return W.reshape(fan_out, fan_in + 1, order='F')


def check_gradient(lambd=0):
    input_layer_size = 3
    hidden_layer_size = 5
    num_labels = 3
    m = 5

    Theta1 = debug_initialize_weights(hidden_layer_size, input_layer_size)
    Theta2 = debug_initialize_weights(num_labels, hidden_layer_size)
    X = debug_initialize_weights(m, input_layer_size - 1)
    y = 1 + np.mod(np.arange(1, m+1), num_labels).T
    y = y.reshape(-1, 1)

    nn_params = np.concatenate([Theta1.ravel(order='F'), Theta2.ravel(order='F')])

    cost_func = partial(nn_cost_function, input_layer_size, hidden_layer_size, num_labels, X, y, lambd)
    grad_func = partial(nn_gradient, input_layer_size, hidden_layer_size, num_labels, X, y, lambd)

    numgrad = numerical_grad(cost_func, nn_params)
    grad = grad_func(nn_params)
    diff = LA.norm(numgrad-grad)/LA.norm(numgrad+grad)
    return diff
