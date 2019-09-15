# Project data into reduced axis
def project_data(X, U, K):
    U_reduce = U[:, 0:K]
    Z = X @ U_reduce
    return Z


# Reconstructing an approximation of the data
def recover_data(Z, U, K):
    U_reduce = U[:, 0:K]
    X = Z @ U_reduce.T
    return X
