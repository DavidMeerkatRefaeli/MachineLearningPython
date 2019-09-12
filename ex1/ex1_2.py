import numpy as np
import pandas as pd

df = pd.read_csv('./Data/ex1data2.txt')

x = df.iloc[:, :2].values
y = df.iloc[:, 2:].values
# X = np.c_[np.ones((len(y), 1)), fn(x)]
X = np.c_[np.ones((len(y), 1)), x]


def normal_equation(X, y):
    xy = X.T @ y
    x_x = np.linalg.inv(X.T @ X)
    return x_x @ xy

print(normal_equation(X, y))