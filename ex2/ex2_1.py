import LoadData as LD
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from Sigmoid import sigmoid as sig
from CostFunctionLogistic import cost_function, gradient
from scipy import optimize as op

# df = LD.load_csv('./ex2/Data/ex2data1.txt')
#
# x = df.iloc[:, :2].values
# y = df.iloc[:, 2:].values

data = np.loadtxt('./Data/ex2data1.txt', delimiter=',')
x = data[:, 0:2]
y = data[:, 2]

def scatter(x, y):
    pos = np.where(y == [1])
    neg = np.where(y == [0])
    plt.scatter(x[pos, 0], x[pos, 1], c='g', marker='+')
    plt.scatter(x[neg, 0], x[neg, 1], c='r', marker='x')


# scatter(x, y)

# print(sig([1, 8, 12]))
m = np.size(y, axis=0)  # data points, rows
d = np.size(x, axis=1)  # variables, cols
theta = np.zeros(d+1).reshape((-1, 1))  # initialize fitting parameters
X = np.c_[np.ones(m), x]  # Add a column of ones to x

# cost = cost_function(theta, X, y)
# gradient = gradient(theta, X, y)

# theta_opt = op.fmin_bfgs(cost_function, theta, fprime=gradient, args=(X, y), maxiter=400)
theta_opt = op.minimize(cost_function, theta, method='bfgs', jac=gradient, args=(X, y)).x
J = cost_function(theta_opt, X, y)
# print(theta_opt)
# print(J)


def plot_decision_boundary():
    x_plot = np.linspace(np.min(X[:, 1]) - 5, np.max(X[:, 1]) + 5, 100)
    y_plot = -(theta_opt[0] + theta_opt[1] * x_plot) / (theta_opt[2])
    plt.plot(x_plot, y_plot)


plot_decision_boundary()

def predict(theta, X):
    prob = sig(X @ theta)
    return prob >= 0.5


p = predict(theta_opt,  X)
print(np.mean(p == y))

# plt.show()
