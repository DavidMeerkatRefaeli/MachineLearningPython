from functools import partial

import numpy as np
from scipy import optimize as op

from ex5.LinearRegressionCostFunction import linear_regression_cost_function, linear_regression_gradient_function


def train_linear_reg(X, y, lambd):
    initial_theta = np.zeros(np.size(X, 1))
    cost_func = partial(linear_regression_cost_function, X, y, lambd)
    grad_func = partial(linear_regression_gradient_function, X, y, lambd)
    theta = op.fmin_cg(cost_func, initial_theta, fprime=grad_func, maxiter=20)
    return theta
