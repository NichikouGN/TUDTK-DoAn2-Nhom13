import numpy as np

def transpose(A):
    return np.array([[A[j][i] for j in range(len(A))] for i in range(len(A[0]))])
