from functools import partial

import numpy as np
import scipy.io
import scipy.optimize as op

from ex3.CostFunctionLogistic import cost_function_reg, gradient_reg
from ex3.DisplayData import display_data
from ex3.Sigmoid import sigmoid as sig


# Multi-class Classification

# Load saved matrices from file
mat = scipy.io.loadmat('./Data/ex3data1.mat')
X = mat['X']
y = mat['y'].reshape(-1, 1)

# Display the data
m = X.shape[0]
sel = np.random.permutation(m)
sel = sel[0:100]
display_data(X[sel])


def test_cost_gradient_function():
    theta_t = np.array([-2, -1, 1, 2]).reshape(-1, 1)
    rand_x = (np.arange(15) + 1) / 10
    X_t = np.c_[np.ones(5), rand_x.reshape(3, 5).T]
    y_t = np.array([1, 0, 1, 0, 1]) >= 0.5
    y_t = y_t.reshape(-1, 1)
    lambda_t = 3
    J = cost_function_reg(X_t, y_t, lambda_t, theta_t)
    grad = gradient_reg(X_t, y_t, lambda_t, theta_t)
    return J, grad


# Vectorizing logistic regression
J, grad = test_cost_gradient_function()
print(f'Cost: {J} | Expected cost: 2.534819')
print(f'Gradients: {grad}')
print('Expected gradients:\n 0.146561\n -0.548558\n 0.724722\n 1.398003')


# One-vs-all classification
def one_vs_all(X, y, num_labels, lam):
    m = np.size(X, axis=0)
    n = np.size(X, axis=1)

    all_theta = np.zeros((num_labels, n+1))
    X = np.c_[np.ones(m), X]
    J = []
    for c in range(num_labels):
        theta_0 = np.zeros(n + 1)
        y_vec = y == c + 1
        y_vec = y_vec.reshape(-1, 1)
        # Learning parameters using Conjugated-Gradient
        cost_func = partial(cost_function_reg, X, y_vec, lam)
        grad_func = partial(gradient_reg, X, y_vec, lam)
        result = op.fmin_cg(cost_func, theta_0, fprime=grad_func, maxiter=50)
        all_theta[c, :] = result
        J.append(cost_function_reg(X, y_vec, lam, result))

    return all_theta, J


num_labels = 10
lambd = 0.1
all_theta, J = one_vs_all(X, y, num_labels, lambd)


def predict_one_vs_all(theta, X):
    m = np.size(X, axis=0)
    X = np.c_[np.ones(m), X]
    results = sig(X @ theta.T)
    prediction = np.argmax(results, axis=1) + 1  # indices are 0 to 9, while y is 1 to 10
    return prediction


# One-vs-all prediction
p = predict_one_vs_all(all_theta, X).reshape(-1, 1)
print(f'Training Set Accuracy: {np.mean(p == y)*100}%')
