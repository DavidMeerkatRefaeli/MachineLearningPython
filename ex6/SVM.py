import scipy.io
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC


# Load the data
data = scipy.io.loadmat('./Data/ex6data1.mat')
X = data['X']
y = data['y']


def scatter(x, y):
    pos = np.where(y == [1])
    neg = np.where(y == [0])
    plt.scatter(x[pos, 0], x[pos, 1], c='k', marker='+')
    plt.scatter(x[neg, 0], x[neg, 1], c='y', marker='o')
    plt.title('Data')


scatter(X, y)
plt.show()

C = 100
clf_100 = SVC(C, 'linear')
# https://stackoverflow.com/questions/34165731/a-column-vector-y-was-passed-when-a-1d-array-was-expected
clf_100.fit(X, y.ravel())


def plot_boundary(X, clf):
    # Plot Boundary
    u = np.linspace(min(X[:, 0]), max(X[:, 0]), 100)
    v = np.linspace(min(X[:, 1]), max(X[:, 1]), 100)
    Z = np.zeros(shape=(len(u), len(v)))
    for i in range(len(u)):
        for j in range(len(v)):
            Z[i, j] = clf.predict([[u[i], v[j]]])
    U, V = np.meshgrid(u, v)
    Z = Z.T
    plt.contour(U, V, Z, levels=[0.5])
    plt.title(f'SVM classification with kernel={clf.kernel} C={clf.C}')


scatter(X, y)
plot_boundary(X, clf_100)
plt.show()

# When , you should find that the SVM puts the decision boundary in the gap between the two datasets
# and misclassifies the data point on the far left
C = 1
clf_1 = SVC(C, 'linear')
clf_1.fit(X, y.ravel())
scatter(X, y)
plot_boundary(X, clf_1)
plt.show()


# SVM with gaussian kernels
def gaussian_kernel(x1, x2, sigma):
    d = x1 - x2
    return np.exp(-(d.T @ d)/(2*sigma**2))


x1 = np.array([1, 2, 1])
x2 = np.array([0, 4, - 1])
sigma = 2
sim = gaussian_kernel(x1, x2, sigma)
print(f'Gaussian Kernel: {sim}')

# Example dataset 2 - with Gaussian Kernel
data = scipy.io.loadmat('./Data/ex6data2.mat')
X = data['X']
y = data['y']
scatter(X, y)
plt.show()

C = 1
# The Matlab SVM uses the sigma representation, the Sklearn uses the gamma - so we must translate
sigma = 0.1
gamma = 1/(2*sigma**2)
clf_rgb = SVC(C, kernel='rbf', gamma=gamma)
clf_rgb.fit(X, y.ravel())

scatter(X, y)
plot_boundary(X, clf_rgb)
plt.show()

# Example dataset 3
data = scipy.io.loadmat('./Data/ex6data3.mat')
X = data['X']
y = data['y']
scatter(X, y)
plt.show()

# Note - I use GridSearchCrossValidation instead of implementing the dataset3Params function by hand.
# It essentially does the same thing: creates a table of all possible values for the hyperparameters,
# calculates the (cross-)validation error for each cell (in dataset3Params regular validation is used),
# and chooses the parameters that minimizes the error.
values = [1e-2, 3e-2, 1e-1, 3e-1, 1, 3, 1e1, 3e1, 1e2, 3e2]
param_grid = {'C': values, 'gamma': values}
gscv = GridSearchCV(SVC(kernel='rbf', class_weight='balanced'), param_grid, cv=5, iid=False)
gscv.fit(X, y.ravel())
# Re-fit so we can pass the regular classifier to plot_boundary
best_params = gscv.best_params_
clf_best = SVC(kernel='rbf', gamma=best_params['gamma'], C=best_params['C'], class_weight='balanced')
clf_best.fit(X, y.ravel())
scatter(X, y)
plot_boundary(X, clf_best)
plt.show()

