import numpy as np
from helper.inverse import inverse
from helper.transpose import transpose

#Linear regression using Ordinary Least Squares (OLS)
def ols_fit(X, y):
    # Add intercept term to X
    y_col = [[yi] for yi in y]
    n_samples = len(X)
    p_features = len(X[0]) - 1

    # Compute beta = (X^T X)^(-1) X^T y
    XT = transpose(X)
    XTX = XT @ X
    XTX_inv = inverse(XTX)

    XTy = XT @ y_col

    beta = XTX_inv @ XTy

    predictions = X @ beta
    residuals = y_col - predictions

    RSS = sum(residuals[i][0] ** 2 for i in range(n_samples))

    variance = RSS / (n_samples - p_features - 1)
    return beta.flatten(), variance