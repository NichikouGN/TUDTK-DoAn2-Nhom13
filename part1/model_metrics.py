def model_metrics(y, y_hat, p):
    y_col = [[yi] for yi in y]
    n = len(y_col)

    residuals = y_col - y_hat

    RSS = sum(residuals[i][0] ** 2 for i in range(n))
    mean = sum(yi[0] for yi in y_col) / n
    TSS = sum((yi[0] - mean) ** 2 for yi in y_col)

    R_squared = 1 - (RSS / TSS)
    adj_R_squared = 1 - (1 - R_squared) * (n - 1) / (n - p - 1)

    F_statistic = (R_squared / p) / ((1 - R_squared) / (n - p - 1))
    
    return RSS, TSS, R_squared, adj_R_squared, F_statistic

