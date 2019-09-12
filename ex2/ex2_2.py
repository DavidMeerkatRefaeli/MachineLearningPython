from functools import partial

import numpy as np
import matplotlib.pyplot as plt

from ex2.Sigmoid import sigmoid as sig
from ex2.CostFunctionLogistic import cost_function_reg, gradient_reg
from scipy import optimize as op


# Visualizing the data
data = np.loadtxt('./Data/ex2data2.txt', delimiter=',')
x = data[:, 0:-1]
y = data[:, -1].reshape((-1, 1))


def scatter(x, y):
    pos = np.where(y == [1])
    neg = np.where(y == [0])
    plt.scatter(x[pos, 0], x[pos, 1], c='g', marker='+')
    plt.scatter(x[neg, 0], x[neg, 1], c='r', marker='x')


scatter(x, y)
plt.show()


def map_features(x1, x2, degree=5):
    size = np.size(x1)
    out = np.ones((size, 1))
    for i in range(1, degree + 1):
        for j in range(i+1):
            out = np.c_[out, x1**(i - j) * x2**(j)]
    return out


# Feature mapping
X = map_features(x[:, 0], x[:, 1])

# Cost function and gradient
m = np.size(X, axis=0)  # data points, rows
d = np.size(X, axis=1)  # variables, cols
theta = np.zeros(d)
lam = 1

cost = cost_function_reg(X, y, lam, theta)
grad = gradient_reg(X, y, lam, theta)
print(f'Cost at initial theta (zeros): {cost}')
print(f'Gradient at initial theta (zeros): {grad}')

# Learning parameters using fminunc
cost_func = partial(cost_function_reg, X, y, lam)
grad_func = partial(gradient_reg, X, y, lam)
theta_opt = op.minimize(cost_func, theta, method='bfgs', jac=grad_func).x
print(f'Optimal theta: {theta_opt}')
J = cost_function_reg(X, y, lam, theta_opt)
print(f'Cost at optimal theta: {J}')


def plot_boundary(l, theta):
    # Plot Boundary
    u = np.linspace(-1, 1.5, 50)
    v = np.linspace(-1, 1.5, 50)
    z = np.zeros(shape=(len(u), len(v)))
    for i in range(len(u)):
        for j in range(len(v)):
            z[i, j] = sig(map_features(np.array(u[i]), np.array(v[j])).dot(np.array(theta)))
    U, V = np.meshgrid(u, v)
    U = U.T
    V = V.T
    plt.contour(U, V, z, 10)
    plt.colorbar()
    plt.title('lambda = %f' % l)
    plt.xlabel('Microchip Test 1')
    plt.ylabel('Microchip Test 2')
    plt.legend(['y = 1', 'y = 0', 'Decision boundary'])


# Plot Decision Boundary
scatter(x, y)
plot_boundary(lam, theta_opt)
plt.show()


def predict(theta, X):
    prob = sig(X @ theta)
    return prob >= 0.5


# Compute accuracy on our training set
p = predict(theta_opt,  X).reshape((-1, 1))
print(f'Train Accuracy: {np.mean(p == y) * 100}%')
