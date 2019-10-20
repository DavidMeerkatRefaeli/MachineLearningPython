import numpy as np


def feature_norm(X):
    '''
    Preforms feature normalization on X, i.e. standardizes X to have mu=o and std=1
    '''
    mu = np.mean(X, axis=0)
    sigma = np.std(X, axis=0)
    new_X = (X - mu) / sigma
    return new_X, mu, sigma

