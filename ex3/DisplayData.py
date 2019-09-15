import numpy as np
import math
from matplotlib import pyplot as plt


def display_data(X, show=True):
    m, n = X.shape
    width = round(math.sqrt(n))
    height = (n // width)

    display_rows = math.floor(math.sqrt(m))
    display_cols = math.ceil(m / display_rows)
    pad = 1
    display_array = np.ones((pad + display_rows * (height + pad),
                             pad + display_cols * (width + pad)))

    curr = 0
    for j in range(display_rows):
        for i in range(display_cols):
            if curr > m:
                break
            x = X[curr, :].reshape(height, width)
            for k in range(height):
                for h in range(width):
                    # max_val = max(abs(X[curr, :]))
                    display_array[pad + j * (height + pad) + k,
                                  pad + i * (width + pad) + h] = x[k, h] # / max_val
            curr += 1
        if curr > m:
            break
    plt.imshow(display_array.T, interpolation='nearest')
    if show:
        plt.show()
