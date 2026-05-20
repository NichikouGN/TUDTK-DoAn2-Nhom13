import numpy as np

def matrix_equal(A, B, tol=1e-9):
    if A.shape != B.shape:
        return False
    return np.all(np.abs(A - B) < tol)