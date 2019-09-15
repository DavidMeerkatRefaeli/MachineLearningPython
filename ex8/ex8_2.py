import numpy as np
import matplotlib.pyplot as plt
import scipy.io

from ex8.CheckGradient import check_gradient
from ex8.CollaborativeFiltering import collaborative_filtering_cost_function as cost

# Recommender Systems
data = scipy.io.loadmat('./Data/ex8_movies.mat')
R = data['R']  # 1682 x 943 matrix, where R(i,j) = 1 if and only if user j gave a rating to movie i
Y = data['Y']  # 682 x 943 matrix, containing ratings (1 - 5) of 1682 movies on 943 users
plt.imshow(Y)
plt.xlabel('Users')
plt.ylabel('Movies')
plt.title('Visualize movies ratings by users')
plt.show()

# Collaborative filtering learning algorithm
# Cost function
data = scipy.io.loadmat('./Data/ex8_movieParams.mat')
X = data['X']
Theta = data['Theta']

# Reduce the data set size so that this runs faster
num_users = 4
num_movies = 5
num_features = 3
X = X[:num_movies, :num_features]
Theta = Theta[:num_users, :num_features]
Y = Y[:num_movies, :num_users]
R = R[:num_movies, :num_users]

# Calculate cost
params = np.concatenate([X.ravel(order='F'), Theta.ravel(order='F')])
J = cost(Y, R, num_users, num_movies, num_features, 0, params)
print(f'Cost of loaded parameters (Reduced data): {J}')

# Check gradients
diff = check_gradient()
print(f'Relative Difference: {diff}')

# Regularized cost function
J = cost(Y, R, num_users, num_movies, num_features, 1.5, params)
print(f'Cost at loaded parameters (lambda = 1.5): {J}')

# Check regularized gradients
diff = check_gradient(1.5)
print(f'Relative Difference: {diff}')


# Learning movie recommendations
