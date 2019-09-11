import numpy as np
from numpy import linalg as LA


# diff will be small but not as small as in Matlab ... Probably due to rounding errors
def numerical_grad(J, theta):
    numgrad = np.zeros_like(theta)
    e = 1e-4
    perturb = np.full(theta.shape, e)
    for p in range(theta.size):
        loss1 = J(theta - perturb)
        loss2 = J(theta + perturb)
        numgrad[p] = (loss2 - loss1) / (2 * e)
        perturb[p] = 0
    return numgrad


def check_gradient(cost_func, grad_func, nn_params):
    numgrad = numerical_grad(cost_func, nn_params)
    grad = grad_func(nn_params)
    diff = LA.norm(numgrad-grad)/LA.norm(numgrad+grad)
    return diff
