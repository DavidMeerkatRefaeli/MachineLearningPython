import scipy.io
import numpy as np
import scipy.optimize as op
from functools import partial

from ex4.CostGradientNN import nn_cost_function, nn_gradient
from ex4.Predict import predict
from ex4.Utilities import rand_initialize_weights
from ex4.DisplayData import display_data as display
from ex4.Sigmoid import sigmoid_grad
from ex4.CheckGradient import check_gradient as check

# This module consists of the 2nd part of exercise 3 (feed-forward) and the whole of exercise 4

# Ex. 3 - part 2
# Load data
#     - X is a (5000,400) matrix with 5000 images of 20x20 pixels (=400)
#     - Y is a (5000,1) vector, with output from 1 to 10, where 10 means 0
mat = scipy.io.loadmat('./Data/ex3data1.mat')
X = mat['X']
y = mat['y']

# Display random 100 images
m = X.shape[0]
sel = np.random.permutation(m)
sel = sel[0:100]
display(X[sel])

# Load some preprocessed weights for initial feed-forward exploration
#     - Theta1 - (25,401) - the extra 1 is for the bias term, reduces to 25 dimensions in the latent space
#     - Theta2 - (10,26) - the extra 1 is for the bias term, reduces to 10 dimensions in the output space
mat_w = scipy.io.loadmat('./Data/ex3weights.mat')
Theta1 = mat_w['Theta1']
Theta2 = mat_w['Theta2']
pred = predict(Theta1, Theta2, X)
pred_y = pred == y
print(f'Training Set Accuracy: {np.mean(pred_y)*100}%')

# Show a single random prediction
i = np.random.randint(m)
X_i = X[i, :].reshape((1, -1))
single_pred = predict(Theta1, Theta2, X_i)
print(f"single prediction: {single_pred % 10}")
display(X_i)


# Ex. 4
mat = scipy.io.loadmat('./Data/ex4data1.mat')
X = mat['X']
y = mat['y']
mat_w = scipy.io.loadmat('./Data/ex4weights.mat')
Theta1 = mat_w['Theta1']
Theta2 = mat_w['Theta2']

# Setup the hyper-parameters for the actual functions
input_layer_size = 400
hidden_layer_size = 25
num_labels = 10
lambd = 0  # regularization term, reduce to over-fit better, increase to generalize better

# Some general tests
nn_params = np.concatenate([Theta1.ravel(order='F'), Theta2.ravel(order='F')])
J = nn_cost_function(input_layer_size, hidden_layer_size, num_labels, X, y, lambd, nn_params)
print(f'Cost at parameters (loaded from ex4weights): {J}')

lambd = 1
J = nn_cost_function(input_layer_size, hidden_layer_size, num_labels, X, y, lambd, nn_params)
print(f'Cost at parameters (loaded from ex4weights): {J}')

# Backpropagation

# Sigmoid gradient
print(sigmoid_grad(0))

# Random initialization - Initialize with random parameters
initial_Theta1 = rand_initialize_weights(input_layer_size, hidden_layer_size)
initial_Theta2 = rand_initialize_weights(hidden_layer_size, num_labels)
initial_nn_params = np.concatenate([initial_Theta1.ravel(), initial_Theta2.ravel()])

# Define a wrapper around the cost and gradient functions
cost_func = partial(nn_cost_function, input_layer_size, hidden_layer_size, num_labels, X, y, lambd)
grad_func = partial(nn_gradient, input_layer_size, hidden_layer_size, num_labels, X, y, lambd)

# Check Gradient
diff = check()
print(f'Difference between numerical gradient and analytical gradient: {diff}')

# Cost function of debugging value
lambd = 3
debug_J = nn_cost_function(input_layer_size, hidden_layer_size, num_labels, X, y, lambd, nn_params)
print(f'Cost at (fixed) debugging parameters (w/ lambda = 3): {debug_J}')

# Optimize the parameters
lambd = 1
result = op.fmin_cg(cost_func, initial_nn_params, fprime=grad_func, maxiter=50)
Theta1 = result[:hidden_layer_size * (input_layer_size + 1)].reshape(hidden_layer_size, (input_layer_size + 1), order='F')
Theta2 = result[hidden_layer_size * (input_layer_size + 1):].reshape(num_labels, (hidden_layer_size + 1), order='F')

pred = predict(Theta1, Theta2, X)
pred_y = pred == y
print(f'Training Set Accuracy: {np.mean(pred_y)*100}%')

# Visualizing the hidden layer
display(Theta1[:, 1:])
