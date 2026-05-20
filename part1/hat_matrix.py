import numpy as np
from helper.inverse import inverse
from helper.transpose import transpose
from helper.matrix_equal import matrix_equal
from helper.build_design_matrix import build_design_matrix

def hat_matrix(X):
    # Compute H = X(X^T X)^(-1)X^T
    XT = transpose(X)
    XTX =  XT @ X 
    XTX_inv = inverse(XTX)
    
    H = X @ XTX_inv @ XT

    #Indempotent check: H^2 should equal H
    H_squared = H @ H

    if matrix_equal(H_squared, H):
        print("H is idempotent: H^2 equals H")
    else:       
        print("H is not idempotent: H^2 does not equal H")
    return H

