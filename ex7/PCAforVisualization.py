import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

from ex1.FeatureNormalization import feature_norm
from ex7.Projections import project_data
from ex7.RunKMeans import kMeans_init_centroids, kMeans
from ex7.RunPCA import pca


# Optional (ungraded) exercise: PCA for visualization
A = plt.imread('./Data/bird_small.png')
img_size = A.shape
m = img_size[0] * img_size[1]
X = A.reshape((m, img_size[2]))
k = 16
max_iter = 10
initial_centroids = kMeans_init_centroids(X, k)
centroids, idx = kMeans(X, k, initial_centroids, max_iter)

# Choose random 1000 pixels to show
m, n = X.shape
indices = np.random.choice(m, 1000)

# 1st show the real colors
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X[indices, 0], X[indices, 1], X[indices, 2], c=X[indices, :], marker='o')
plt.title('3D (real) colors of selected 1000 pixels')
plt.show()


def get_colors(indices, idx, centroids):
    m = len(indices)
    n = np.size(centroids, 1)
    colors = np.zeros((m, n))
    for i in range(m):
        colors[i, :] = centroids[idx[indices[i]], :]
    return colors


# Now show k-means colors
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
# Make the colors the actual colors assigned to each original pixel
colors = get_colors(indices, idx, centroids)
ax.scatter(X[indices, 0], X[indices, 1], X[indices, 2], c=colors, marker='o')
plt.title('3D (k-means) colors of selected 1000 pixels')
plt.show()

# PCA and project the data to 2D
X_norm, _, _ = feature_norm(X)
U, _ = pca(X_norm)
Z = project_data(X_norm, U, 2)
plt.scatter(Z[indices, 0], Z[indices, 1], c=colors)
plt.title('Projected colors into 2D space')
plt.show()
