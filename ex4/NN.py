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
print(np.mean(pred_y))

# Show a single random prediction
i = np.random.randint(m)
X_i = X[i, :].reshape((1, -1))
single_pred = predict(Theta1, Theta2, X_i)
print(f"single prediction: {single_pred % 10}")
display(X_i)

# Setup the hyper-parameters for the actual functions
input_layer_size = 400
hidden_layer_size = 25
num_labels = 10
lambd = 1  # regularization term, reduce to over-fit better, increase to generalize better

# Some general tests
nn_params = np.concatenate([Theta1.ravel(), Theta2.ravel()])
J = nn_cost_function(input_layer_size, hidden_layer_size, num_labels, X, y, lambd, nn_params)
print(J)
print(sigmoid_grad(0))

# Define a wrapper around the cost and gradient functions
cost_func = partial(nn_cost_function, input_layer_size, hidden_layer_size, num_labels, X, y, lambd)
grad_func = partial(nn_gradient, input_layer_size, hidden_layer_size, num_labels, X, y, lambd)

# Check Gradient
check(cost_func, grad_func, nn_params)

# Initialize with random parameters
initial_Theta1 = rand_initialize_weights(input_layer_size, hidden_layer_size)
initial_Theta2 = rand_initialize_weights(hidden_layer_size, num_labels)
initial_nn_params = np.concatenate([initial_Theta1.ravel(), initial_Theta2.ravel()])

# Optimize the parameters
result = op.fmin_cg(cost_func, initial_nn_params, fprime=grad_func, maxiter=20)
Theta1 = result[:hidden_layer_size * (input_layer_size + 1)].reshape((hidden_layer_size, (input_layer_size + 1)))
Theta2 = result[hidden_layer_size * (input_layer_size + 1):].reshape((num_labels, (hidden_layer_size + 1)))

pred = predict(Theta1, Theta2, X)
pred_y = pred == y
print(np.mean(pred_y))

