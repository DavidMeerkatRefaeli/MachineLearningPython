import numpy as np

from ex1.FeatureNormalization import feature_norm
from ex1.GradientDescentMulti import gradient_descent_multi
from ex1.NormalEquation import normal_equation

np.set_printoptions(suppress=True)

# Linear regression with multiple variables
data = np.loadtxt('./Data/ex1data2.txt', delimiter=',')
X = data[:, :2]
y = data[:, 2:]
m = len(y)

# Print out some data points
print(np.c_[X[0:10, :], y[0:10, :]])

# Feature Normalization
X_norm, mu, sigma = feature_norm(X)

# Add intercept term to normalized X
X_norm = np.c_[np.ones(m), X_norm]

# Gradient Descent
alpha = 0.1
num_iters = 400
theta = np.zeros((3, 1))  # initialize fitting parameters
theta, j_history = gradient_descent_multi(X_norm, y, theta, alpha, num_iters)
print(f'Gradient Descent Theta:\n {theta}')

# Estimate the price of a 1650 sq-ft, 3 br house
x_1 = np.array([1650, 3])
x_1_norm = np.insert((x_1 - mu)/sigma, 0, 1)
print(f'normalized input: {x_1_norm[0]}, {x_1_norm[1]}, {x_1_norm[2]}')
print(f'mu: {mu}, sigma: {sigma}')
price = x_1_norm @ theta
print(f'Predicted price of a 1650 sq-ft, 3 br house (using gradient descent): {price}')

# Normal Equations - use the original X, without normalization
X_1 = np.c_[np.ones(m), X]
theta_norm = normal_equation(X_1, y)
print(f'Normal Equation Theta:\n {theta_norm}')

# Predict
price = np.array([1, 1650, 3]) @ theta_norm
print(f'Predicted price of a 1650 sq-ft, 3 br house (using normal equations): {price}')
