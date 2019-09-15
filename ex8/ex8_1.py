import numpy as np
import matplotlib.pyplot as plt
import scipy.io

from ex8.MultivariateGaussian import multivariate_gaussian, estimate_gaussian


# Anomaly Detection
data = scipy.io.loadmat('./Data/ex8data1.mat')
X = data['X']
Xval = data['Xval']
yval = data['yval']

plt.scatter(X[:, 0], X[:, 1], marker='x', c='b', linewidth=0.5)
plt.xlabel('Latency (ms)')
plt.ylabel('Throughput (mb/s)')
plt.show()

# Estimating parameters for a Gaussian
mu, sigma2 = estimate_gaussian(X)
p = multivariate_gaussian(X, mu, sigma2)  # probability


def visualize_fit(X, mu, sigma2):
    plt.scatter(X[:, 0], X[:, 1], marker='x', c='b', linewidth=0.5)

    x1 = np.arange(0, 35, 0.5)
    x2 = np.arange(0, 35, 0.5)
    X1, X2 = np.meshgrid(x1, x2)
    Z = multivariate_gaussian(np.c_[X1.flatten(), X2.flatten()], mu, sigma2)
    Z = Z.reshape(X1.shape)
    logspace = np.logspace(-20, 1, 10)
    plt.contour(X1, X2, Z, levels=logspace, cmap='RdGy')


visualize_fit(X, mu, sigma2)
plt.show()

# Selecting the threshold, epsilon
def select_threshold(y, p):
    epsilons = np.arange(np.min(p), np.max(p), (np.max(p) - np.min(p))/1000)
    best_epsilon = 0
    best_F1 = 0
    for eps in epsilons:
        predicted = (p < eps).reshape(-1, 1)
        tp = np.sum((y == 1) & (predicted == 1))
        fp = np.sum((y == 0) & (predicted == 1))
        fn = np.sum((y == 1) & (predicted == 0))
        prec = tp / (tp + fp)
        reca = tp / (tp + fn)
        F1 = (2 * prec * reca) / (prec + reca)
        if F1 > best_F1:
            best_F1 = F1
            best_epsilon = eps
    return best_epsilon, best_F1

pval = multivariate_gaussian(Xval, mu, sigma2)
epsilon, F1 = select_threshold(yval, pval)
print(f'Best epsilon found using cross-validation: {epsilon}')
print(f'Best F1 on cross-validation: {F1}')

# Draw a red circle around those outliers
outliers = np.where(p < epsilon)
visualize_fit(X, mu, sigma2)
plt.scatter(X[outliers, 0], X[outliers, 1], marker='o', facecolors='none', edgecolors='r', s=100)
plt.show()

# High dimensional dataset
data = scipy.io.loadmat('./Data/ex8data2.mat')
X = data['X']
Xval = data['Xval']
yval = data['yval']

mu, sigma2 = estimate_gaussian(X)
p = multivariate_gaussian(X, mu, sigma2)
pval = multivariate_gaussian(Xval, mu, sigma2)
epsilon, F1 = select_threshold(yval, pval)
print(f'Best epsilon found using cross-validation: {epsilon}')
print(f'Best F1 on cross-validation: {F1}')
print(f'Number of  outliers found: {np.sum(p < epsilon)}')
