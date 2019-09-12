

def compute_cost_multi(X, y, theta):
    m = len(y)
    err = X @ theta - y
    return (err.T @ err) / (2*m)