import numpy as np

from ex4.Utilities import indices_to_one_hot
from ex4.Sigmoid import sigmoid, sigmoid_grad

# Cost and gradient have been separated to use Scipy optimizations functions which requires them separately


def unroll_thetas(hidden_layer_size, input_layer_size, nn_params, num_labels):
    # Numpy reshape works by default in row-wise order (C-type), while Matlab works in col-wise order (F/Fortran-type)
    Theta1 = nn_params[:hidden_layer_size * (input_layer_size + 1)] \
        .reshape(hidden_layer_size, (input_layer_size + 1), order='F')
    Theta2 = nn_params[hidden_layer_size * (input_layer_size + 1):] \
        .reshape(num_labels, (hidden_layer_size + 1), order='F')
    return Theta1, Theta2


def calculate_outputs(Theta1, Theta2, X, m):
    a1 = np.c_[np.ones(m), X]  # input of 1st layer - i.e. initial input
    z1 = a1 @ Theta1.T  # output of 1st layer to 2nd; also marked as s1
    theta_z1 = sigmoid(z1)  # output of 1st layer after activation function
    a2 = np.c_[np.ones(m), theta_z1]  # input of 2nd layer (hidden)
    z2 = a2 @ Theta2.T  # output of 2nd layer to final output
    h = sigmoid(z2)  # final hypothesis h_theta
    return a1, a2, h


def calculate_deltas(Theta1, Theta2, a1, a2, delta1, delta2, h, m, yvec):
    for t in range(m):
        ht = h[t, :].T
        yvect = yvec[t, :].T
        delta_3t = ht - yvect  # (10,1)
        delta_3t = delta_3t[:, np.newaxis]

        a2t = a2[t, :][np.newaxis, :]  # (1, 26)
        delta2 = delta2 + delta_3t @ a2t  # (10, 26)

        a1t = a1[t, :].T  # (401, 1)
        z1t = Theta1 @ a1t  # (25, 1)
        delta_2t = (Theta2.T @ delta_3t) * (np.insert(sigmoid_grad(z1t), 0, 1)[:, np.newaxis])  # (26, 1)
        delta1 = delta1 + delta_2t[1:] @ a1t.T[np.newaxis, :]
    return delta1, delta2


# A feed-forward algorithm to find the cost given parameters
def nn_cost_function(input_layer_size, hidden_layer_size, num_labels, X, y, lambd, nn_params):
    Theta1, Theta2 = unroll_thetas(hidden_layer_size, input_layer_size, nn_params, num_labels)

    m = X.shape[0]

    yvec = indices_to_one_hot(y - 1, num_labels)

    a1, a2, h = calculate_outputs(Theta1, Theta2, X, m)

    Theta1_exc_bias = Theta1[:, 1:]
    Theta2_exc_bias = Theta2[:, 1:]

    regularization = (Theta1_exc_bias**2).sum() + (Theta2_exc_bias**2).sum()
    J = (-1 / m) * (yvec * np.log(h) + (1 - yvec) * np.log(1 - h)).sum() + (lambd / (2 * m)) * regularization
    return J


# A back-propagation algorithm to find derivatives given an error function
def nn_gradient(input_layer_size, hidden_layer_size, num_labels, X, y, lambd, nn_params):
    Theta1, Theta2 = unroll_thetas(hidden_layer_size, input_layer_size, nn_params, num_labels)

    m = X.shape[0]

    delta1 = np.zeros_like(Theta1)
    delta2 = np.zeros_like(Theta2)

    yvec = indices_to_one_hot(y - 1, num_labels)

    a1, a2, h = calculate_outputs(Theta1, Theta2, X, m)

    delta1, delta2 = calculate_deltas(Theta1, Theta2, a1, a2, delta1, delta2, h, m, yvec)

    Theta1_exc_bias = Theta1[:, 1:]
    Theta2_exc_bias = Theta2[:, 1:]

    regularization1 = np.c_[np.zeros(Theta1_exc_bias.shape[0]), Theta1_exc_bias]
    regularization2 = np.c_[np.zeros(Theta2_exc_bias.shape[0]), Theta2_exc_bias]

    Theta1_grad = delta1 / m + (lambd / m) * regularization1
    Theta2_grad = delta2 / m + (lambd / m) * regularization2

    # Numpy ravel works by default in row-wise order (C-type), while Matlab works in col-wise order (F/Fortran-type)
    return np.concatenate([Theta1_grad.ravel(order='F'), Theta2_grad.ravel(order='F')])
