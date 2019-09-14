import random

import numpy as np


# todo: go over algorithm and replace sklearn version with it
def svm_train(X, y, C, kernel_func, tol=1e-3, max_iter=5):
    """ Trains an SVM classifier using a simplified version of the SMO (Sequential minimal optimization) algorithm. """
    m = np.size(X, 0)

    # Map y = 0 to -1
    y[y == 0] = -1

    # Variables
    alphas = np.zeros(m)
    b = 0
    E = np.zeros(m)
    passes = 0

    K = kernel_func(X)

    dots = 12
    while passes < max_iter:
        num_changed_alphas = 0
        for i in range(m):
            # Ei = f(x(i)) - y(i)
            k = K[:, i].reshape(-1, 1)
            E[i] = b + np.sum(alphas * y * k) - y[i]
            if (((y[i] * E[i]) < -tol) & (alphas[i] < C)) | (((y[i] * E[i]) > tol) & (alphas[i] > 0)):
                j = round(m * random.random())
                while j == i:
                    j = round(m * random.random())

                k = K[:, j].reshape(-1, 1)
                E[j] = b + np.sum(alphas * y * k) - y[j]

                alpha_i_old = alphas[i]
                alpha_j_old = alphas[j]

                if y[i] == y[j]:
                    L = max(0, alphas[j] + alphas[i] - C)
                    H = min(C, alphas[j] + alphas[i])
                else:
                    L = max(0, alphas[j] - alphas[i])
                    H = min(C, C + alphas[j] - alphas[i])

                if L == H:
                    continue

                eta = 2 * K[i, j] - K[i, i] - K[j, j]
                if eta >= 0:
                    continue

                # Compute and clip new value for alpha j
                alphas[j] = alphas[j] - (y[j] * (E[i] - E[j])) / eta

                alphas[j] = min(H, alphas[j])
                alphas[j] = max(L, alphas[j])

                # Check if change in alpha is significant
                if abs(alphas[j] - alpha_j_old) < tol:
                    alphas[j] = alpha_j_old
                    continue

                # Determine value for alpha i
                alphas[i] = alphas[i] + y[i] * y[j] * (alpha_j_old - alphas[j])

                # Compute b1 and b2
                b1 = b - E[i] - y[i] * (alphas[i] - alpha_i_old) * K(i, j) - y[j] * (alphas[j] - alpha_j_old) * K(i, j)
                b2 = b - E[j] - y[i] * (alphas[i] - alpha_i_old) * K(i, j) - y[j] * (alphas[j] - alpha_j_old) * K(j, j)

                if (0 < alphas[i]) & (alphas[i] < C):
                    b = b1
                elif (0 < alphas[j]) & (alphas[j] < C):
                    b = b2
                else:
                    b = (b1 + b2) / 2

                num_changed_alphas = num_changed_alphas + 1

            if num_changed_alphas == 0:
                passes = passes + 1
            else:
                passes = 0

            print('.', end='')
            dots += 1
            if dots > 78:
                dots = 0
                print()

    print(' Done! \n')

    # Save the model
    idx = alphas > 0
    model = {}
    model.X = X[idx, :]
    model.y = y[idx]
    model.kernel = kernel_func
    model.b = b
    model.alphas = alphas[idx]
    model.w = ((alphas * y).T @ X).T