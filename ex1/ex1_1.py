import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from ex1.ComputeCost import compute_cost as cc
from ex1.GradientDescent import gradient_descent as gd

# Single variable regression

# Warm up
A = np.eye(5)
print(A)

# Plotting the data
data = np.loadtxt('./Data/ex1data1.txt', delimiter=',')
X = data[:, :1]
y = data[:, 1:]


def scatter(X, y):
    plt.scatter(X, y, marker='x', c='r')


scatter(X, y)
plt.show()

# Implementation
m = len(X)  # number of training examples
X_1 = np.c_[np.ones(m), X]  # Add a column of ones to x
theta = np.zeros((2, 1))  # initialize fitting parameters
iterations = 1500
alpha = 0.01

# Compute Cost
print(cc(X_1, y, theta))
print(cc(X_1, y, np.array([[-1], [2]])))

# Gradient Descent
theta, j_history = gd(X_1, y, theta, alpha, iterations)
print(theta)
print(j_history)


def plot_line(X, theta):
    X_plot = np.linspace(np.min(X) - 5, np.max(X) + 5, 100).reshape(-1, 1)
    y_plot = np.c_[np.ones(len(X_plot)), X_plot] @ theta
    plt.plot(X_plot, y_plot)


# Plot the linear fit
plot_line(X_1, theta)
scatter(X, y)
plt.show()

predict1 = np.array([1, 3.5]) @ theta
print(predict1 * 10000)
predict2 = np.array([1, 7]) @ theta
print(predict2 * 10000)

# Visualizing J(theta)
theta0_vals = np.linspace(-10, 10, 100)
theta1_vals = np.linspace(-2, 4, 100)
j_vals = np.zeros((len(theta0_vals), len(theta1_vals)))
for i in range(len(theta0_vals)):
    for j in range(len(theta1_vals)):
        t = np.array([theta0_vals[i], theta1_vals[j]])
        j_vals[i, j] = cc(X_1, y, t)
theta0_vals, theta1_vals = np.meshgrid(theta0_vals, theta1_vals)
# For some reason, meshgrid returns the transposed values, so we need to fix this
j_vals = j_vals.T

def contour(X, Y, Z):
    plt.contour(X, Y, Z, levels=np.logspace(-2, 3, 20))
    plt.colorbar()


def surface(X, Y, Z):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X, Y, Z, cmap='viridis')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')


# Contour graph
contour(theta0_vals, theta1_vals, j_vals)
scatter(theta[0], theta[1])
plt.show()

# Surface graph
surface(theta0_vals, theta1_vals, j_vals)
plt.show()
