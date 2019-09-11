import numpy as np


def rand_initialize_weights(lin, lout):
    epsilon_init = 0.12
    return np.random.rand(lout, 1 + lin) * 2 * epsilon_init - epsilon_init


def indices_to_one_hot(data, nb_classes):
    """Convert an iterable of indices to one-hot encoded labels."""
    targets = np.array(data).reshape(-1)
    return np.eye(nb_classes)[targets]