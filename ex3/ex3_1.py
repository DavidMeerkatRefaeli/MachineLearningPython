import scipy.io
import numpy as np
import scipy.optimize as op
from CostFunctionLogistic import cost_function_reg, gradient_reg
from Sigmoid import sigmoid as sig

mat = scipy.io.loadmat('./Data/ex3data1.mat')
X = mat['X']
y = mat['y']

def one_vs_all(X, y, num_labels, lam):
    m = np.size(X, axis=0)
    n = np.size(X, axis=1)

    all_theta = np.zeros((num_labels, n+1))
    X = np.c_[np.ones((m,)), X]
    J = []
    for c in range(num_labels):
        theta_0 = np.zeros((n+1, 1))
        y_c = y == c + 1
        result = op.fmin_cg(cost_function_reg, theta_0, fprime=gradient_reg, args=(X, y_c, lam))
        all_theta[c, :] = result
        J.append(cost_function_reg(result, X, y, lam))

        # opts = {'maxiter': 400,
        #         'disp': False,
        #         'gtol': 1e-5,
        #         'norm': np.inf,
        #         'eps': 1.4901161193847656e-08}
        # result = op.minimize(cost_function_reg, theta_0, method='CG', jac=gradient_reg, args=(X, y_c, lam), options=opts)
        # all_theta[c, :] = result.x

    return all_theta, J

def predict_one_vs_all(theta, X):
    m = np.size(X, axis=0)
    X = np.c_[np.ones((m, 1)), X]
    results = sig(X @ theta.T)
    prediction = np.argmax(results, axis=1) + 1  # indices are 0 to 9, while y is 1 to 10
    return prediction


def test_cost_function():
    theta_t = np.array([-2, -1, 1, 2])
    X_t = np.c_[np.ones((5, 1)), np.arange(1, 16).reshape((3, 5)).T / 10]
    y_t = np.array([1, 0, 1, 0, 1]).reshape((-1, 1))
    lambda_t = 3
    J_t = cost_function_reg(theta_t, X_t, y_t, lambda_t)
    grad_t = gradient_reg(theta_t, X_t, y_t, lambda_t)
    return J_t, grad_t


all_theta, J = one_vs_all(X, y, 10, 0.1)
p = predict_one_vs_all(all_theta, X).reshape((-1, 1))
print(np.mean(p == y))
