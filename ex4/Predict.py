import numpy as np

from ex4.Sigmoid import sigmoid


def predict(Theta1, Theta2, X):
    m = X.shape[0]
    X = np.c_[np.ones(m), X]
    hidden = sigmoid(X @ Theta1.T)
    hidden = np.c_[np.ones(m), hidden]
    output = sigmoid(hidden @ Theta2.T)
    result = np.argmax(output, axis=1) + 1
    return result.reshape((-1, 1))
