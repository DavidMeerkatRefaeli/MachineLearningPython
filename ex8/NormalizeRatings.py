import numpy as np


def normalize_ratings(Y, R):
    Ymean = np.zeros(Y.shape[0])
    Ynorm = np.zeros_like(Y)
    for i in range(Y.shape[0]):
        idx = R[i, :] != 0
        Ymean[i] = Y[i, idx].mean()
        Ynorm[i, idx] = Y[i, idx] - Ymean[i]
    return Ynorm, Ymean
