import numpy as np
import scipy.io
import matplotlib.pyplot as plt

# Regularized Linear Regression
# Load the data
from ex1.FeatureNormalization import feature_norm
from ex5.LearningCurves import learning_curves
from ex5.LinearRegressionCostFunction import linear_regression_cost_function, linear_regression_gradient_function
from ex5.PolyStuff import poly_features, poly_plot
from ex5.TrainLinearRegression import train_linear_reg

mat = scipy.io.loadmat('./Data/ex5data1.mat')
X = mat['X']
y = mat['y']
Xval = mat['Xval']
yval = mat['yval']
Xtest = mat['Xtest']
ytest = mat['ytest']

m = np.size(X, 0)


# Plot training data
def scatter(X, y):
    plt.scatter(X, y, marker='x', c='r')
    plt.xlabel('Change in water level (x)')
    plt.ylabel('Water flowing out of the dam (y)')


scatter(X, y)
plt.show()

# Regularized linear regression cost function / gradient
theta = np.array([1, 1])
J = linear_regression_cost_function(np.c_[np.ones(m), X], y, 1, theta)
print(f'Cost at theta = [1, 1] : {J}')
grad = linear_regression_gradient_function(np.c_[np.ones(m), X], y, 1, theta)
print(f'Gradient at theta = [1, 1] : {grad}')


# Fitting linear regression
# Train linear regression with lambda = 0
lambd = 0
theta = train_linear_reg(np.c_[np.ones(m), X], y, lambd)


def plot_line(X, theta):
    X_plot = np.linspace(np.min(X) - 5, np.max(X) + 5, 100).reshape(-1, 1)
    y_plot = np.c_[np.ones(len(X_plot)), X_plot] @ theta
    plt.plot(X_plot, y_plot)


# Plot fit over the data
scatter(X, y)
plot_line(X, theta)
plt.show()


# Bias-variance
def plot_learning_curves(err_train, err_val, m):
    m_vals = np.arange(1, m + 1)
    plt.plot(m_vals, err_train, m_vals, err_val)
    plt.xlabel('Number of training examples')
    plt.ylabel('Error')
    plt.xlim((0, 13))
    plt.ylim((0, 100))

# Learning curves
err_train, err_val = learning_curves(np.c_[np.ones(m), X], y, np.c_[np.ones(np.size(Xval, 0)), Xval], yval, lambd)
plot_learning_curves(err_train, err_val, m)
plt.show()
print('# Training Examples\tTrain Error\tCross Validation Error\n')
for i in range(m):
    print(f'  \t{i}\t\t{err_train[i]}\t{err_val[i]}')


# Polynomial regression
p = 8
# Map X onto Polynomial Features and Normalize
X_poly = poly_features(X, p)
X_poly, mu, sigma = feature_norm(X_poly)    # Normalize
X_poly = np.c_[np.ones(m), X_poly]          # Add ones

X_poly_test = poly_features(Xtest, p)
X_poly_test = X_poly_test - mu
X_poly_test = X_poly_test / sigma
X_poly_test = np.c_[np.ones(np.size(X_poly_test, 0)), X_poly_test]
X_poly_val = poly_features(Xval, p)
X_poly_val = X_poly_val - mu
X_poly_val = X_poly_val / sigma
X_poly_val = np.c_[np.ones(np.size(X_poly_val, 0)), X_poly_val]

lambd = 0
theta = train_linear_reg(X_poly, y, lambd)

# Plot training data and fit
scatter(X, y)
poly_plot(np.min(X), np.max(X), mu, sigma, theta, p)
plt.show()

# Plot learning curves
err_train, err_val = learning_curves(X_poly, y, X_poly_val, yval, lambd)
plot_learning_curves(err_train, err_val, m)
plt.show()

# Selecting lambda using a cross validation set
lambda_vec, error_train, error_val = validation_curve(X_poly, y, X_poly_val, yval)