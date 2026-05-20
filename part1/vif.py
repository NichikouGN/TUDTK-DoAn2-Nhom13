import numpy as np
from part1.ols_fit import ols_fit
from helper.build_design_matrix import build_design_matrix
from part1.model_metrics import model_metrics

def VIF(X):
    n = len(X)
    p = len(X[0]) - 1  # Exclude intercept

    print(X)
    vif_values = []
    for j in range(1, p + 1):  # Start from 1 to skip intercept
        y_j = np.array([X[i][j] for i in range(n)])
        y_j_col = [[y] for y in y_j]

        X_j = np.array([[X[i][k] for k in range(len(X[0])) if k != j] for i in range(n)])
        beta_j, _ = ols_fit(X_j, y_j)
        beta_j = [[b] for b in beta_j]  # Flatten beta_j

        y_j_hat = X_j @ beta_j

        residual_j = y_j_col - y_j_hat
        RSS_j = sum(residual_j[i][0] ** 2 for i in range(n))

        mean_y_j = sum(y_j[i] for i in range(n)) / n
        TSS_j = sum((y_j[i] - mean_y_j) ** 2 for i in range(n))

        R2_j = 1 - (RSS_j / TSS_j)
        vif_j = 1 / (1 - R2_j)
        vif_values.append(vif_j)


    return vif_values