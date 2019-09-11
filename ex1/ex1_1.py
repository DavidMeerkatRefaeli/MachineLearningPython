import LoadData as LD
import numpy as np
import matplotlib.pyplot as plt
from ex1.ComputeCost import compute_cost as cc
from ex1.GradientDescent import gradient_descent as gd

df = LD.load_csv('./data/ex1data1.txt')

x = df.iloc[:, :1].values
y = df.iloc[:, 1:].values

# plt.scatter(x, y, marker='x', c='r')

m = len(x)  # number of training examples
X = np.c_[np.ones(m), x]  # Add a column of ones to x
theta = np.zeros((2, 1))  # initialize fitting parameters
iterations = 1500
alpha = 0.01

# print(cc(X, y, theta))
# print(cc(X, y, np.array([-1, 2]).reshape((-1, 1))))
results = gd(X, y, theta, alpha, iterations)
print(results[0][0])
print(results[0][1])

print(np.matmul([1, 3.5], theta)*10000)
print(np.matmul([1, 7], theta)*10000)

# X_plot = np.linspace(np.min(x) - 5, np.max(x) + 5, 100).reshape(-1, 1)
# plt.plot(X_plot, np.matmul(np.c_[np.ones(len(X_plot)), X_plot], theta))

# plt.plot(x, np.matmul(X, theta))

theta0_vals = np.linspace(-10, 10, 100)
theta1_vals = np.linspace(-1, 4, 100)
j_vals = np.zeros((len(theta0_vals), len(theta1_vals)))

for i in range(len(theta0_vals)):
    for j in range(len(theta1_vals)):
        t = np.array([theta0_vals[i], theta1_vals[j]])
        j_vals[i, j] = cc(X, y, t)


plt.contour(theta0_vals, theta1_vals, j_vals, 200, cmap='RdGy')
plt.colorbar()
plt.show()


