from functools import partial

import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize as op

from ex2.Sigmoid import sigmoid as sig
from ex2.CostFunctionLogistic import cost_function, gradient_function

# Load data
data = np.loadtxt('./Data/ex2data1.txt', delimiter=',')
x = data[:, :2]
y = data[:, 2].reshape((-1, 1))


def scatter(x, y):
    pos = np.where(y == [1])
    neg = np.where(y == [0])
    plt.scatter(x[pos, 0], x[pos, 1], c='g', marker='+', label='Admitted')
    plt.scatter(x[neg, 0], x[neg, 1], c='r', marker='x', label='Not Admitted')
    plt.legend(loc='upper right')


# Visualizing the data
scatter(x, y)
plt.show()

# Warmup exercise: sigmoid function
print(sig([0, 1, -3]))

# Cost function and gradient
m = np.size(y, axis=0)  # data points, rows
d = np.size(x, axis=1)  # variables, cols
X = np.c_[np.ones(m), x]  # Add a column of ones to x
theta = np.zeros(d+1)  # initialize fitting parameters
cost = cost_function(X, y, theta)
print(f'Cost at initial theta (zeros): {cost}')
gradient = gradient_function(X, y, theta)
print(f'Gradient at initial theta (zeros): {gradient}')

# Learning parameters using fminunc (Quasi-Newton = BFGS)
cost_func = partial(cost_function, X, y)
grad_func = partial(gradient_function, X, y)

# Option 1
theta_opt = op.fmin_bfgs(cost_func, theta, fprime=grad_func, maxiter=400)
J = cost_func(theta_opt)
print(f'Cost at theta found by BFGS: {J}')
print(f'Theta: {theta_opt}')

# Option 2
theta_opt = op.minimize(cost_func, theta, method='bfgs', jac=grad_func).x
J = cost_func(theta_opt)
print(f'Cost at theta found by BFGS: {J}')
print(f'Theta: {theta_opt}')


def plot_decision_boundary():
    x_plot = np.linspace(np.min(X[:, 1]) - 5, np.max(X[:, 1]) + 5, 100)
    y_plot = -(theta_opt[0] + theta_opt[1] * x_plot) / (theta_opt[2])
    plt.plot(x_plot, y_plot)


# Plot Boundary
plot_decision_boundary()
scatter(x, y)
plt.show()

# Predict probability for a student with score 45 on exam 1  and score 85 on exam 2
prob = sig(np.array([1, 45, 85]) @ theta_opt)
print(f'For a student with scores 45 and 85, we predict an admission probability of {prob}')


def predict(theta, X):
    prob = sig(X @ theta)
    return prob >= 0.5


# Compute accuracy on our training set
p = predict(theta_opt,  X).reshape((-1, 1))
print(f'Train Accuracy: {np.mean(p == y)*100} %')
