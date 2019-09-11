import numpy as np
import matplotlib.pyplot as plt
from Sigmoid import sigmoid as sig
from CostFunctionLogistic import cost_function_reg, gradient_reg
from scipy import optimize as op

data = np.loadtxt('./Data/ex2data2.txt', delimiter=',')
x = data[:, 0:-1]
y = data[:, -1]


def scatter(x, y):
    pos = np.where(y == [1])
    neg = np.where(y == [0])
    plt.scatter(x[pos, 0], x[pos, 1], c='g', marker='+')
    plt.scatter(x[neg, 0], x[neg, 1], c='r', marker='x')


def map_features(x1, x2, degree=5):
    size = np.size(x1)
    out = np.ones((size, 1))
    for i in range(1, degree + 1):
        for j in range(i+1):
            out = np.c_[out, x1**(i - j) * x2**(j)]
    return out


X = map_features(x[:,0], x[:,1])
theta = np.zeros((np.size(X, axis=1), 1))
lam = 1

# cost = cost_function_reg(theta, X, y, lam)
# grad = gradient_reg(theta, X, y, lam)
# print(cost)
# print(grad)

result = op.minimize(cost_function_reg, theta, method='bfgs', jac=gradient_reg, args=(X, y, lam))
# print(result)
theta_opt = result.x
J = cost_function_reg(theta_opt, X, y, lam)
print(J)

def plot_boundary(l, theta):
    # Plot Boundary
    u = np.linspace(-1, 1.5, 50)
    v = np.linspace(-1, 1.5, 50)
    z = np.zeros(shape=(len(u), len(v)))
    for i in range(len(u)):
        for j in range(len(v)):
            z[i, j] = (map_features(np.array(u[i]), np.array(v[j])).dot(np.array(theta)))
    z = z.T
    plt.contour(u, v, z, 50)
    plt.colorbar()
    plt.title('lambda = %f' % l)
    plt.xlabel('Microchip Test 1')
    plt.ylabel('Microchip Test 2')
    plt.legend(['y = 1', 'y = 0', 'Decision boundary'])


scatter(x, y)
plot_boundary(lam, theta_opt)
plt.show()

def predict(theta, X):
    prob = sig(X @ theta)
    return prob >= 0.5


p = predict(theta_opt,  X)
print(np.mean(p == y))


