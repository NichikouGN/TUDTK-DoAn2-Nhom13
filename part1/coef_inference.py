from helper.inverse import inverse
from helper.transpose import transpose
from part1.ols_fit import ols_fit
from helper.t_table import query_t_table
from helper.build_design_matrix import build_design_matrix
import numpy as np
from math import sqrt

CONFIDENCE_LEVELS = 0.95
ALPHA = 1 - CONFIDENCE_LEVELS

def coef_inference(X, y, beta_hat, sigma2):
    if len(X) != len(y):
        raise ValueError("Number of samples in X and y must be the same.")

    sigma2 = float(sigma2)  # Ensure sigma2 is a float for calculations
    n_samples = len(y)
    p_features = len(X[0]) - 1
    df = n_samples - p_features - 1  # Degrees of freedom for residuals
   
    # Compute (X^T X)^(-1)
    XT = transpose(X)
    XTX = XT @ X
    XTX_inv = inverse(XTX)

    var_beta = [[sigma2 * XTX_inv[i][j] for j in range(len(XTX_inv[0]))] for i in range(len(XTX_inv))]
    se_beta = [sqrt(var_beta[i][i]) for i in range(len(var_beta))]

    # t-statistics for coefficients
    t_stats = beta_hat / se_beta
    p_values = [query_t_table(df, t) for t in t_stats]

    return se_beta, t_stats, p_values
