import numpy as np

from ex4.CostGradientNN import nn_cost_function, nn_gradient


# Test case for the cost/gradient functions
# https://www.coursera.org/learn/machine-learning/discussions/weeks/5/threads/uPd5FJqnEeWWpRIGHRsuuw

il = 2
hl = 2
nl = 4
nn = np.arange(1, 19) / 10
X = np.cos([[1, 2], [3, 4], [5, 6]])  # Matlab gives slightly different results than Python
y = np.array([4, 2, 3]).reshape(-1, 1)
lambd = 4
J = nn_cost_function(il, hl, nl, X, y, lambd, nn)
grad = nn_gradient(il, hl, nl, X, y, lambd, nn)
print(J)
print(grad)
