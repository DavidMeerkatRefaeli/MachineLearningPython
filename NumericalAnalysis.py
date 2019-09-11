import numpy as np

def newton1d(f, iter=10, x0=0):
    xn = x0
    for i in range(iter):
        gradient1 = np.gradient(f, xn)
        xn = xn - f(xn) / gradient1
    return xn


def f1(x):
    return x**2 - 2*x + 1


def f2(x):
    return x[0]**2 - x[1]**2


# print(newton1d(f1))
print(newton1d(f2, x0=np.array([30, 2.5])))
