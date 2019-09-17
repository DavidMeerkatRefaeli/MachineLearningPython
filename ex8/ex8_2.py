from functools import partial

import numpy as np
import matplotlib.pyplot as plt
import scipy.io
import scipy.optimize as op

from ex8.CheckGradient import check_gradient
from ex8.CollaborativeFiltering import collaborative_filtering_cost_function as cost
from ex8.CollaborativeFiltering import collaborative_filtering_gradient as gradient

# Recommender Systems
from ex8.LoadMovieList import load_movie_list
from ex8.NormalizeRatings import normalize_ratings

data = scipy.io.loadmat('./Data/ex8_movies.mat')
R = data['R']  # 1682 x 943 matrix, where R(i,j) = 1 if and only if user j gave a rating to movie i
Y = data['Y']  # 682 x 943 matrix, containing ratings (1 - 5) of 1682 movies on 943 users
mean = Y[0, R[0, :] != 0].mean()
print(f'Average rating for movie 1 (Toy Story): {mean}')
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
movie_list = load_movie_list()
my_ratings = np.zeros(len(movie_list))
my_ratings[0] = 4
my_ratings[97] = 2
my_ratings[6] = 3
my_ratings[11] = 5
my_ratings[53] = 4
my_ratings[63] = 5
my_ratings[65] = 3
my_ratings[68] = 5
my_ratings[182] = 4
my_ratings[225] = 5
my_ratings[354] = 5
my_ratings = my_ratings.reshape(-1, 1)

# Recommendations
data = scipy.io.loadmat('./Data/ex8_movies.mat')
R = data['R']  # 1682 x 943 matrix, where R(i,j) = 1 if and only if user j gave a rating to movie i
Y = data['Y']  # 682 x 943 matrix, containing ratings (1 - 5) of 1682 movies on 943 users
Y = np.c_[my_ratings, Y]
R = np.c_[my_ratings != 0, R]

# Normalize ratings and set the parameters
Ynorm, Ymean = normalize_ratings(Y, R)
num_movies = np.size(Y, 0)
num_users = np.size(Y, 1)
num_features = 10
X = np.random.randn(num_movies, num_features)
Theta = np.random.randn(num_users, num_features)
params = np.concatenate([X.ravel(order='F'), Theta.ravel(order='F')])
lambd = 10
cost_func = partial(cost, Ynorm, R, num_users, num_movies, num_features, lambd)
grad_func = partial(gradient, Ynorm, R, num_users, num_movies, num_features, lambd)

# Use Conjugated-Gradient optimization to find best params
op_params = op.fmin_cg(cost_func, params, fprime=grad_func, maxiter=100)
print(f'Cost of start parameters: {cost_func(params)}')
# As we can see - the cost is super high... but lower than initial
print(f'Cost of optimized parameters: {cost_func(op_params)}')

# Unfold the returned parameters
op_X = params[:num_movies * num_features].reshape((num_movies, num_features), order='F')
op_Theta = params[num_movies * num_features:].reshape((num_users, num_features), order='F')

# Get predictions
p = op_X @ op_Theta.T
my_predictions = p[:, 0] + Ymean

# Display top recommendations
idx = my_predictions.argsort()
print('Top recommendations for you:')
for i in range(1, 10):
    j = idx[-i]
    # Top prediction come with really high ratings (14+) which is a bit strange
    # Though example given in Matlab also shows ratings of 9, 8, etc.
    print(f'Predicting rating {my_predictions[j]} for movie {movie_list[j]}')
