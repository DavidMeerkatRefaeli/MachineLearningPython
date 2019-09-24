from functools import partial

import numpy as np
import scipy.io
import scipy.optimize as op


# Here is a softmax implementation of the same logistic regression exercise. (w/o regularization)
# Note that loss and gradient are computed separately due to optimization function requirement.
# Also note that W is flattened and reshaped due to optimization function requirement (can only process 1d).

# Load the data
mat = scipy.io.loadmat('./Data/ex3data1.mat')
X = mat['X']
y = mat['y']
y = np.where(y == 10, 0, y).squeeze()  # fix 10 to 0, make 1d array
m = X.shape[0]
n = X.shape[1]


def softmax(z):
    # Assumes z is a nd-array; will throw if it's 1d
    e = np.exp(z)
    s = np.sum(e, axis=1, keepdims=True)
    return e/s


def loss(X, y, W):
    m = X.shape[0]
    n = X.shape[1]

    # Restore W form
    W = W.reshape(n, -1)

    f = X @ W
    f -= np.max(f, axis=1, keepdims=True)  # this is to avoid inf numbers in the sum, and doesn't change results
    h = softmax(f)

    # No need to multiply by y, it's as if it's in 1 hot form and we only care about the index that is 1, rest are 0's.
    # In a sense there is a "difference" from logistic-regression, as you don't penalize how much wrong you are, though
    # this is already internalized by the softmax function.
    loss = (1 / m) * np.sum(-np.log(h[np.arange(m), y]))
    return loss


def gradient(X, y, W):
    # Gradient is a bit tricky, should follow this explanation:
    # https://math.stackexchange.com/a/945918/342736
    #
    # In short - the softmax function has C outputs, when C is the number of classes.
    # Denote zi as the softmax input of the i'th class (X@Wi), and hi as the softmax output of the i'th class.
    # Each output is exp(zi)/sum, so when you differentiate with respect to zi, there are 2 situations:
    #   1. You differentiate the output i - which gives: hi(1-hi) - same as regular sigmoid
    #   2. You differentiate the output j (not equal i) - which gives: -hj*hi
    # y is either i (yi = 1) or not (yi = 0), but it doesn't matter - the general formula for the dL/dzi is (hi - yi)
    # or for the vector form dL/dz = h - y.
    # To further differentiate by W, you get X.T @ (h - y). (in our example X is (5000,401) and h-y is (5000,10))

    m = X.shape[0]
    n = X.shape[1]

    # Restore W form
    W = W.reshape(n, -1)

    f = X @ W
    f -= np.max(f, axis=1, keepdims=True)  # this is to avoid inf numbers in the sum, and doesn't change results
    h = softmax(f)

    y_hat = np.zeros_like(h)
    y_hat[np.arange(m), y] = 1
    dW = (1 / m) * (X.T @ (h - y_hat))
    return dW.ravel()  # return to 1D


def predict(X, W):
    Z = X @ W
    soft = softmax(Z)
    return np.argmax(soft, axis=1)


# Add bias terms
X1 = np.c_[np.ones(m), X]
W0 = np.zeros((n + 1, 10))

# Train - Use optimization to find optimal weights
cost_func = partial(loss, X1, y)
grad_func = partial(gradient, X1, y)
result = op.fmin_cg(cost_func, W0.ravel(), fprime=grad_func, maxiter=50)

# Predict
W_opt = result.reshape(X1.shape[1], -1)
pred = predict(X1, W_opt)
print(f'Training Set Accuracy: {np.mean(pred == y)*100}%')
