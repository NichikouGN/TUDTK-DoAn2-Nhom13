import pandas as pd
import numpy as np
from helper.build_design_matrix import build_design_matrix

def main():
    df = pd.read_csv('day.csv')
    contributors=["season", "yr", "mnth", "holiday", "weekday", "workingday", "weathersit", "temp", "atemp", "hum", "windspeed"] 

    x = df[[
        "season",
        "yr",
        "mnth",
        "holiday",
        "weekday",
        "workingday",
        "weathersit",
        "temp",
        "atemp",
        "hum",
        "windspeed"
    ]]

    y = df["cnt"]


    x = x.to_numpy()
    y = y.to_numpy()

    design_matrix = build_design_matrix(x)

    #OLS fit
    from part1.ols_fit import ols_fit
    beta_hat, sigma2_hat = ols_fit(design_matrix, y)
    # print("Beta coefficients:", beta_hat)
    # print("Residuals:", variance)

    #Hat matrix
    from part1.hat_matrix import hat_matrix
    from helper.matrix_equal import matrix_equal
    H = hat_matrix(design_matrix)
    # print("Hat matrix H:\n", H)
    # print("Checking if xB equals Hy: ", matrix_equal(design_matrix @ [[bi] for bi in beta_hat], H @ [[yi] for yi in y])) 

    #Model metrics
    from part1.model_metrics import model_metrics

    p = len(x[0])
    RSS, TSS, R_squared, adj_R_squared, F_statistic = model_metrics(y,  H @ [[yi] for yi in y], p)
    # print("RSS:", RSS)
    # print("TSS:", TSS)
    # print("R-squared:", R_squared)
    # print("Adjusted R-squared:", adj_R_squared)
    # print("F-statistic:", F_statistic)

    #Coefficient inference
    from part1.coef_inference import coef_inference
    se_beta, t_stats, p_values = coef_inference(design_matrix, y, beta_hat, sigma2_hat)
    # print("Standard Errors of Coefficients:", se_beta)
    # print("t-statistics for Coefficients:", t_stats)
    # print("p values: ", p_values)

    #VIF
    from part1.vif import VIF
    vif_values = VIF(design_matrix)
    # print("VIF values for each feature:", vif_values)



if __name__ == "__main__":
    main()