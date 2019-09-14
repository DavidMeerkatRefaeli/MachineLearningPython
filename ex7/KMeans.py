import numpy as np
import scipy.io
import matplotlib.pyplot as plt

from ex7.ComputeCenroids import compute_centroids
from ex7.FindClosestCentroid import find_closest_centroids


# K-Means Clustering
from ex7.RunKMeans import kMeans

data = scipy.io.loadmat('./Data/ex7data2.mat')
X = data['X']

# Select an initial set of centroids
k = 3
initial_centroids = np.array([[3, 3], [6, 2], [8, 5]])

# Find the closest centroids for the examples using the initial_centroids
idx = find_closest_centroids(X, initial_centroids)
print(f'Closest centroids for the first 3 examples: {idx[0:3]}')

# Compute means based on the closest centroids found in the previous part
centroids = compute_centroids(X, idx, k)
print(f'Centroids computed after initial finding of closest centroids:\n{centroids}')


# Random initialization
def kMeans_init_centroids(X, k):
    m = np.size(X, 0)
    idx = np.random.permutation(m)[:k]
    return X[idx]


random_centroids = kMeans_init_centroids(X, 3)
print(f'Random centroids:\n{random_centroids}')

# Image compression with K-means
A = plt.imread('./Data/bird_small.png')
img_size = A.shape
m = img_size[0] * img_size[1]
X = A.reshape((m, img_size[2]))
k = 16
max_iter = 10
initial_centroids = kMeans_init_centroids(X, k)
centroids, idx = kMeans(X, k, initial_centroids, max_iter)


X_recovered = np.zeros_like(X)
for i in range(m):
    X_recovered[i, :] = centroids[idx[i], :]

X_recovered = X_recovered.reshape((img_size[0], img_size[1], img_size[2]))
plt.imshow(X_recovered)
plt.show()
