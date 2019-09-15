import numpy as np
import scipy.io
import matplotlib.pyplot as plt

from ex1.FeatureNormalization import feature_norm
from ex3.DisplayData import display_data
from ex7.Projections import project_data, recover_data
from ex7.RunPCA import pca


# Load and display data
data = scipy.io.loadmat('./Data/ex7data1.mat')
X = data['X']


def scatter(X, marker='D', edgecolor='b'):
    plt.scatter(X[:, 0], X[:, 1], marker=marker, facecolors='none', edgecolors=edgecolor)
    plt.title('Data')


scatter(X)
plt.show()

# Implementing PCA
X_norm, mu, sigma = feature_norm(X)


def plot_line(p1, p2, linestyle='-', color='r', size=2):
    x1, x2 = p1[0], p2[0]
    y1, y2 = p1[1], p2[1]
    plt.plot([x1, x2], [y1, y2], linestyle=linestyle, color=color, linewidth=size)


U, S = pca(X_norm)
plot_line(mu, mu + U[:, 0].T)
plot_line(mu, mu + U[:, 1].T)
scatter(X)
plt.axis('square')
plt.title('Data with Eigenvectors')
plt.show()
print(f'Top eigenvector U[0, :] = {U[0, :]}')

# Dimensionality reduction with PCA
# Project Data into 1 dimension
K = 1
Z = project_data(X_norm, U, K)
print(f'Projection of the first example: {Z[0]}')
X_rec = recover_data(Z, U, K)
print(f'Approximation of the first example:{X_rec[0, :]}')

# Visualizing the projections
scatter(X_norm)
scatter(X_rec, 'o', 'r')
for i in range(np.size(X_norm, 0)):
    plot_line(X_norm[i, :], X_rec[i, :], '--', 'k', 1)
plt.axis('square')
plt.title('Data with projection')
plt.show()

# Face image dataset
data = scipy.io.loadmat('./Data/ex7faces.mat')
X = data['X']
plt.title('First 100 faces')
display_data(X[:100, :])

# PCA on faces
X_norm, _, _ = feature_norm(X)
U, _ = pca(X_norm)
plt.title('First 36 Eigenvectors')
display_data(U[:, :36].T)

K = 100
Z = project_data(X_norm, U, K)
print(f'The projected data Z has a size of: {Z.shape}')

X_rec = recover_data(Z, U, K)

# Show original image with reconstructed image
fig = plt.figure(figsize=(8, 4))
fig.add_subplot(1, 2, 1)
plt.title("Original")
display_data(X[:100, :], False)
fig.add_subplot(1, 2, 2)
plt.title(f"Reconstructed with K={K}")
display_data(X_rec[:100, :])

