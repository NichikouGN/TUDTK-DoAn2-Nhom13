import numpy as np
import matplotlib.pyplot as plt
from math import sqrt, erf

# --- CÁC HÀM BỔ TRỢ TÍNH TOÁN MA TRẬN & THỐNG KÊ ---

def transpose(A):
    return np.array([[A[j][i] for j in range(len(A))] for i in range(len(A[0]))])

def matrix_equal(A, B, tol=1e-9):
    if A.shape != B.shape:
        return False
    return np.all(np.abs(A - B) < tol)

def build_design_matrix(x):
    n_samples = len(x)
    design_matrix = []
    for i in range(n_samples):
        row = [1] + list(x[i])
        design_matrix.append(row)
    return np.array(design_matrix)

epsilon = 1e-9

def clean_value(x):
    if isinstance(x, complex) or type(x).__name__ in ('complex128', 'complex64'):
        r = clean_value(x.real)
        i = clean_value(x.imag)
        if i == 0.0:
            return r
        return complex(r, i)
    if abs(x) < epsilon:
        return 0.0
    return float(x)

def gaussian_elimination(A, b=None, to_rref=False, silent=False):
    if len(A) == 0:
        raise ValueError("Ma trận A không được rỗng")
    for row in A:
        if len(row) != len(A[0]):
            raise ValueError("Tất cả các dòng của ma trận A phải có cùng độ dài")
    A = [[x if isinstance(x, complex) else float(x) for x in row] for row in A]
    if b is not None:
        if len(A) != len(b):
            raise ValueError("Ma trận A và vector b phải có cùng số dòng")
        b = [float(x) for x in b]
        M = [A[i] + [b[i]] for i in range(len(A))]
    else:
        M = [row[:] for row in A]
    nrows = len(M)
    ncols = len(M[0])
    swap_count = 0
    cur_row = 0
    for k in range(ncols-1):
        if cur_row >= nrows:
            break
        pivot = (cur_row, k)
        for row in range(cur_row + 1, nrows):
            if abs(M[row][k]) > abs(M[pivot[0]][pivot[1]]):
                pivot = (row, k)
        pivot_val = abs(M[pivot[0]][pivot[1]])
        if pivot_val < epsilon:
            M[pivot[0]][pivot[1]] = 0
            continue
        if pivot[0] != cur_row:
            M[cur_row], M[pivot[0]] = M[pivot[0]], M[cur_row]
            swap_count += 1
        if not to_rref:
            for row in range(cur_row + 1, nrows):
                factor = M[row][k] / M[cur_row][k]
                for col in range(k, ncols):
                    M[row][col] -= factor * M[cur_row][col]
                    if abs(M[row][col]) < epsilon:
                        M[row][col] = 0.0
        else:
            pivot_val = M[cur_row][k]
            for col in range(k, ncols):
                M[cur_row][col] /= pivot_val
                if abs(M[cur_row][col]) < epsilon:
                    M[cur_row][col] = 0.0
            for row in range(nrows):
                if row != cur_row:
                    factor = M[row][k]
                    if abs(factor) >= epsilon:
                        for col in range(k, ncols):
                            M[row][col] -= factor * M[cur_row][col]
                            if abs(M[row][col]) < epsilon:
                                M[row][col] = 0.0
        cur_row += 1
    if b is not None:
        U = [row[:-1] for row in M]
        c = [row[-1] for row in M]
        for row in range(nrows):
            if all(abs(M[row][col]) < epsilon for col in range(ncols - 1)) and abs(M[row][ncols-1]) >= epsilon:
                return U, None, swap_count
        x = back_substitution(U, c)
        U = [[clean_value(val) for val in row] for row in U]
    else:
        U = [[clean_value(val) for val in row] for row in M]
        x = None
    return U, x, swap_count

def back_substitution(U, c):
    nrows = len(U)
    ncols = len(U[0])
    pivot_cols = []
    cur_row = 0 
    for k in range(ncols):
        if cur_row >= nrows:
            break
        if abs(U[cur_row][k]) < epsilon:
           continue
        pivot_cols.append((cur_row, k))
        cur_row += 1
    pivot_only_cols = [c_idx for (r, c_idx) in pivot_cols]
    free_vars = [col for col in range(ncols) if col not in pivot_only_cols]
    sol = [{} for _ in range(ncols)]
    if free_vars:
        for i, col in enumerate(free_vars):
            sol[col] = {'const': 0.0, f"t{i+1}": 1.0}
    for row, col in reversed(pivot_cols):
        expr = {'const': c[row]}
        for j in range(ncols - 1, col, -1):
            if sol[j]:
                coeff = U[row][j]
                if abs(coeff) > epsilon:
                    for key, val in sol[j].items():
                        expr[key] = expr.get(key, 0.0) - (coeff * val)
        pivot_val = U[row][col]
        for key in expr:
            expr[key] = expr[key] / pivot_val
        sol[col] = expr
    if not free_vars:
        return [clean_value(sol[i].get('const', 0.0)) for i in range(ncols)]
    final_sol = []
    for i in range(ncols):
        terms = []
        d = sol[i]
        const_val = round(d.get('const', 0.0), 6)
        if abs(const_val) > epsilon or not d:
            terms.append(str(const_val))
        for key, val in d.items():
            if key != 'const':
                val_rounded = round(val, 6)
                if abs(val_rounded) > epsilon:
                    is_one = abs(abs(val_rounded) - 1.0) < epsilon
                    coeff_str = "" if is_one else f"{abs(val_rounded)}*"
                    if val_rounded > 0 and terms:
                        terms.append(f"+ {coeff_str}{key}")
                    elif val_rounded > 0:
                        terms.append(f"{coeff_str}{key}")
                    else: 
                        terms.append(f"- {coeff_str}{key}")
        if not terms:
            final_sol.append("0")
        else:
            final_sol.append(" ".join(terms))
    return final_sol

def inverse(A):
    n = len(A)
    if n == 0 or n != len(A[0]):
        raise ValueError("Chỉ ma trận vuông mới có ma trận nghịch đảo.")
    inv_A_cols = []
    for i in range(n):
        e_i = [1.0 if j == i else 0.0 for j in range(n)]
        U, x, _ = gaussian_elimination(A, b=e_i, silent=True)
        if x is None or (len(x) > 0 and isinstance(x[0], str)):
            return None
        inv_A_cols.append(x)
    return [[inv_A_cols[j][i] for j in range(n)] for i in range(n)]

T_TABLE = {
    1: {1.00: 0.000, 0.50: 1.000, 0.40: 1.376, 0.30: 1.963, 0.20: 3.078, 0.10: 6.314, 0.05: 12.71, 0.02: 31.82, 0.01: 63.66, 0.002: 318.31, 0.001: 636.62},
    2: {1.00: 0.000, 0.50: 0.816, 0.40: 1.061, 0.30: 1.386, 0.20: 1.886, 0.10: 2.920, 0.05: 4.303, 0.02: 6.965, 0.01: 9.925, 0.002: 22.327, 0.001: 31.599},
    3: {1.00: 0.000, 0.50: 0.765, 0.40: 0.978, 0.30: 1.250, 0.20: 1.638, 0.10: 2.353, 0.05: 3.182, 0.02: 4.541, 0.01: 5.841, 0.002: 10.215, 0.001: 12.924},
    4: {1.00: 0.000, 0.50: 0.741, 0.40: 0.941, 0.30: 1.190, 0.20: 1.533, 0.10: 2.132, 0.05: 2.776, 0.02: 3.747, 0.01: 4.604, 0.002: 7.173, 0.001: 8.610},
    5: {1.00: 0.000, 0.50: 0.727, 0.40: 0.920, 0.30: 1.156, 0.20: 1.476, 0.10: 2.015, 0.05: 2.571, 0.02: 3.365, 0.01: 4.032, 0.002: 5.893, 0.001: 6.869},
    6: {1.00: 0.000, 0.50: 0.718, 0.40: 0.906, 0.30: 1.134, 0.20: 1.440, 0.10: 1.943, 0.05: 2.447, 0.02: 3.143, 0.01: 3.707, 0.002: 5.208, 0.001: 5.959},
    7: {1.00: 0.000, 0.50: 0.711, 0.40: 0.896, 0.30: 1.119, 0.20: 1.415, 0.10: 1.895, 0.05: 2.365, 0.02: 2.998, 0.01: 3.499, 0.002: 4.785, 0.001: 5.408},
    8: {1.00: 0.000, 0.50: 0.706, 0.40: 0.889, 0.30: 1.108, 0.20: 1.397, 0.10: 1.860, 0.05: 2.306, 0.02: 2.896, 0.01: 3.355, 0.002: 4.501, 0.001: 5.041},
    9: {1.00: 0.000, 0.50: 0.703, 0.40: 0.883, 0.30: 1.100, 0.20: 1.383, 0.10: 1.833, 0.05: 2.262, 0.02: 2.821, 0.01: 3.250, 0.002: 4.297, 0.001: 4.781},
    10: {1.00: 0.000, 0.50: 0.700, 0.40: 0.879, 0.30: 1.093, 0.20: 1.372, 0.10: 1.812, 0.05: 2.228, 0.02: 2.764, 0.01: 3.169, 0.002: 4.144, 0.001: 4.587},
    11: {1.00: 0.000, 0.50: 0.697, 0.40: 0.876, 0.30: 1.088, 0.20: 1.363, 0.10: 1.796, 0.05: 2.201, 0.02: 2.718, 0.01: 3.106, 0.002: 4.025, 0.001: 4.437},
    12: {1.00: 0.000, 0.50: 0.695, 0.40: 0.873, 0.30: 1.083, 0.20: 1.356, 0.10: 1.782, 0.05: 2.179, 0.02: 2.681, 0.01: 3.055, 0.002: 3.930, 0.001: 4.318},
    13: {1.00: 0.000, 0.50: 0.694, 0.40: 0.870, 0.30: 1.079, 0.20: 1.350, 0.10: 1.771, 0.05: 2.160, 0.02: 2.650, 0.01: 3.012, 0.002: 3.852, 0.001: 4.221},
    14: {1.00: 0.000, 0.50: 0.692, 0.40: 0.868, 0.30: 1.076, 0.20: 1.345, 0.10: 1.761, 0.05: 2.145, 0.02: 2.624, 0.01: 2.977, 0.002: 3.787, 0.001: 4.140},
    15: {1.00: 0.000, 0.50: 0.691, 0.40: 0.866, 0.30: 1.074, 0.20: 1.341, 0.10: 1.753, 0.05: 2.131, 0.02: 2.602, 0.01: 2.947, 0.002: 3.733, 0.001: 4.073},
    16: {1.00: 0.000, 0.50: 0.690, 0.40: 0.865, 0.30: 1.071, 0.20: 1.337, 0.10: 1.746, 0.05: 2.120, 0.02: 2.583, 0.01: 2.921, 0.002: 3.686, 0.001: 4.015},
    17: {1.00: 0.000, 0.50: 0.689, 0.40: 0.863, 0.30: 1.069, 0.20: 1.333, 0.10: 1.740, 0.05: 2.110, 0.02: 2.567, 0.01: 2.898, 0.002: 3.646, 0.001: 3.965},
    18: {1.00: 0.000, 0.50: 0.688, 0.40: 0.862, 0.30: 1.067, 0.20: 1.330, 0.10: 1.734, 0.05: 2.101, 0.02: 2.552, 0.01: 2.878, 0.002: 3.610, 0.001: 3.922},
    19: {1.00: 0.000, 0.50: 0.688, 0.40: 0.861, 0.30: 1.066, 0.20: 1.328, 0.10: 1.729, 0.05: 2.093, 0.02: 2.539, 0.01: 2.861, 0.002: 3.579, 0.001: 3.883},
    20: {1.00: 0.000, 0.50: 0.687, 0.40: 0.860, 0.30: 1.064, 0.20: 1.325, 0.10: 1.725, 0.05: 2.086, 0.02: 2.528, 0.01: 2.845, 0.002: 3.552, 0.001: 3.850},
    21: {1.00: 0.000, 0.50: 0.686, 0.40: 0.859, 0.30: 1.063, 0.20: 1.323, 0.10: 1.721, 0.05: 2.080, 0.02: 2.518, 0.01: 2.831, 0.002: 3.527, 0.001: 3.819},
    22: {1.00: 0.000, 0.50: 0.686, 0.40: 0.858, 0.30: 1.061, 0.20: 1.321, 0.10: 1.717, 0.05: 2.074, 0.02: 2.508, 0.01: 2.819, 0.002: 3.505, 0.001: 3.792},
    23: {1.00: 0.000, 0.50: 0.685, 0.40: 0.858, 0.30: 1.060, 0.20: 1.319, 0.10: 1.714, 0.05: 2.069, 0.02: 2.500, 0.01: 2.807, 0.002: 3.485, 0.001: 3.768},
    24: {1.00: 0.000, 0.50: 0.685, 0.40: 0.857, 0.30: 1.059, 0.20: 1.318, 0.10: 1.711, 0.05: 2.064, 0.02: 2.492, 0.01: 2.797, 0.002: 3.467, 0.001: 3.745},
    25: {1.00: 0.000, 0.50: 0.684, 0.40: 0.856, 0.30: 1.058, 0.20: 1.316, 0.10: 1.708, 0.05: 2.060, 0.02: 2.485, 0.01: 2.787, 0.002: 3.450, 0.001: 3.725},
    26: {1.00: 0.000, 0.50: 0.684, 0.40: 0.856, 0.30: 1.058, 0.20: 1.315, 0.10: 1.706, 0.05: 2.056, 0.02: 2.479, 0.01: 2.779, 0.002: 3.435, 0.001: 3.707},
    27: {1.00: 0.000, 0.50: 0.684, 0.40: 0.855, 0.30: 1.057, 0.20: 1.314, 0.10: 1.703, 0.05: 2.052, 0.02: 2.473, 0.01: 2.771, 0.002: 3.421, 0.001: 3.690},
    28: {1.00: 0.000, 0.50: 0.683, 0.40: 0.855, 0.30: 1.056, 0.20: 1.313, 0.10: 1.701, 0.05: 2.048, 0.02: 2.467, 0.01: 2.763, 0.002: 3.408, 0.001: 3.674},
    29: {1.00: 0.000, 0.50: 0.683, 0.40: 0.854, 0.30: 1.055, 0.20: 1.311, 0.10: 1.699, 0.05: 2.045, 0.02: 2.462, 0.01: 2.756, 0.002: 3.396, 0.001: 3.659},
    30: {1.00: 0.000, 0.50: 0.683, 0.40: 0.854, 0.30: 1.055, 0.20: 1.310, 0.10: 1.697, 0.05: 2.042, 0.02: 2.457, 0.01: 2.750, 0.002: 3.385, 0.001: 3.646}
}

def normal_cdf(x):
    return 0.5 * (1 + erf(x / sqrt(2)))

def query_t_table(df, t_value):
    if df <= 30:
        t_value = abs(float(t_value))
        p_values = T_TABLE[df]
        lower_bound = None
        upper_bound = None
        for p, critical_t in sorted(p_values.items(), reverse=True):
            if critical_t <= t_value:
                lower_bound = (p, critical_t)
            elif critical_t > t_value and upper_bound is None:
                upper_bound = (p, critical_t)
                break
        if upper_bound is None:
            return 0.0
        slope = (upper_bound[0] - lower_bound[0]) / (upper_bound[1] - lower_bound[1])
        estimated_p = lower_bound[0] + slope * (t_value - lower_bound[1])
        return round(estimated_p, 4)
    else:
        z_value = abs(float(t_value))
        p_value = 2 * (1 - normal_cdf(z_value))
        return round(p_value, 4)

# 1. Hàm ols_fit: khớp mô hình OLS tính beta và variance
def ols_fit(X, y):
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


# 2. Hàm hat_matrix: tính ma trận Hat H = X(X^T X)^(-1)X^T
def hat_matrix(X):
    XT = transpose(X)
    XTX =  XT @ X 
    XTX_inv = inverse(XTX)
    
    H = X @ XTX_inv @ XT

    # Idempotent check: H^2 should equal H
    H_squared = H @ H

    if matrix_equal(H_squared, H):
        print("H is idempotent: H^2 equals H")
    else:       
        print("H is not idempotent: H^2 does not equal H")
    return H


# 3. Hàm model_metrics: tính RSS, TSS, R^2, R^2 hiệu chỉnh và F-statistic
def model_metrics(y, y_hat, p):
    y_col = [[yi] for yi in y]
    n = len(y_col)

    residuals = y_col - y_hat

    RSS = sum(residuals[i][0] ** 2 for i in range(n))
    mean = sum(yi[0] for yi in y_col) / n
    TSS = sum((yi[0] - mean) ** 2 for yi in y_col)

    R_squared = 1 - (RSS / TSS)
    adj_R_squared = 1 - (1 - R_squared) * (n - 1) / (n - p - 1)

    if abs(1.0 - R_squared) < 1e-12:
        F_statistic = float('inf')
    else:
        F_statistic = (R_squared / p) / ((1 - R_squared) / (n - p - 1))
    
    return RSS, TSS, R_squared, adj_R_squared, F_statistic


# 4. Hàm coef_inference: kiểm định hệ số hồi quy (se, t-stat, p-value, CI 95%)
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

    # Get t critical value for 95% confidence interval (two-tailed alpha = 0.05)
    if df <= 30:
        t_critical = T_TABLE[df][0.05]
    else:
        try:
            import scipy.stats as stats
            t_critical = stats.t.ppf(0.975, df)
        except ImportError:
            t_critical = 1.95996  # Standard Normal critical value fallback

    confidence_intervals = []
    for j in range(len(beta_hat)):
        margin_of_error = t_critical * se_beta[j]
        confidence_intervals.append((beta_hat[j] - margin_of_error, beta_hat[j] + margin_of_error))

    return se_beta, t_stats, p_values, confidence_intervals


# 5. Hàm vif: tính hệ số phóng đại phương sai (VIF) kiểm tra đa cộng tuyến
def vif(X):
    n = len(X)
    p = len(X[0]) - 1  # Exclude intercept

    vif_values = []
    for j in range(1, p + 1):  # Start from 1 to skip intercept
        y_j = np.array([X[i][j] for i in range(n)])
        y_j_col = [[y] for y in y_j]

        X_j = np.array([[X[i][k] for k in range(len(X[0])) if k != j] for i in range(n)])
        beta_j, _ = ols_fit(X_j, y_j)
        beta_j = [[b] for b in beta_j]  # Convert to column vector

        y_j_hat = X_j @ beta_j

        residual_j = y_j_col - y_j_hat
        RSS_j = sum(residual_j[i][0] ** 2 for i in range(n))

        mean_y_j = sum(y_j[i] for i in range(n)) / n
        TSS_j = sum((y_j[i] - mean_y_j) ** 2 for i in range(n))

        R2_j = 1 - (RSS_j / TSS_j)
        if abs(1.0 - R2_j) < 1e-12:
            vif_j = float('inf')
        else:
            vif_j = 1 / (1 - R2_j)
        vif_values.append(vif_j)

    return vif_values

VIF = vif  # Expose capitalized version for backward compatibility


# 9. Hàm gauss_markov_demo: chạy mô phỏng Monte Carlo để kiểm chứng Gauss-Markov (OLS là BLUE)
def gauss_markov_demo(n_samples=100, n_simulations=1000, seed=42):
    np.random.seed(seed)

    # True parameters
    beta_true = np.array([3.0, 1.5, -2.0, 0.5])  # [β₀, β₁, β₂, β₃]
    sigma = 2.0  # Noise standard deviation
    p = len(beta_true) - 1

    # Feature matrix X (fixed across simulations)
    X_features = np.random.randn(n_samples, p)
    X = np.column_stack([np.ones(n_samples), X_features])

    # Alternative estimator: WLS with non-optimal fixed weights
    weights = np.array([1.0 + 0.8 * np.sin(2 * np.pi * i / n_samples)
                        for i in range(n_samples)])
    weights = np.maximum(weights, 0.2)
    W = np.diag(weights)

    XtWX = X.T @ W @ X
    XtWX_inv = np.linalg.inv(XtWX)

    betas_ols = []
    betas_alt = []

    for sim in range(n_simulations):
        # Generate noise
        epsilon = np.random.normal(0, sigma, n_samples)
        y = X @ beta_true + epsilon

        # OLS estimation
        beta_ols, _ = ols_fit(X, y)
        betas_ols.append(beta_ols)

        # Alternative WLS estimation
        XtWy = X.T @ W @ y
        beta_alt = XtWX_inv @ XtWy
        betas_alt.append(beta_alt)

    betas_ols = np.array(betas_ols)
    betas_alt = np.array(betas_alt)

    print("=" * 65)
    print("  KIỂM CHỨNG ĐỊNH LÝ GAUSS-MARKOV (Monte Carlo Simulation)")
    print("=" * 65)
    print(f"  Số mẫu mỗi simulation: n = {n_samples}")
    print(f"  Số lần mô phỏng: {n_simulations}")
    print(f"  β thật = {beta_true}")
    print(f"  σ (noise) = {sigma}")

    # 1. Unbiasedness check
    mean_ols = np.mean(betas_ols, axis=0)
    bias_ols = mean_ols - beta_true

    print(f"\n{'─'*65}")
    print(f"  1. TÍNH KHÔNG CHỆCH: E[β̂_OLS] ≈ β")
    print(f"{'─'*65}")
    print(f"  {'Hệ số':<8} {'β thật':<10} {'E[β̂_OLS]':<12} {'Bias':<12}")
    for j in range(len(beta_true)):
        print(f"  β{j:<5}  {beta_true[j]:<10.4f} {mean_ols[j]:<12.4f} {bias_ols[j]:<12.6f}")

    # 2. Efficiency comparison (BLUE)
    print(f"\n{'─'*65}")
    print(f"  2. PHƯƠNG SAI NHỎ NHẤT (BLUE): Var(β̂_OLS) ≤ Var(β̃_WLS)")
    print(f"{'─'*65}")
    print(f"  {'Hệ số':<8} {'Var(OLS)':<14} {'Var(WLS)':<14} {'Tỉ số':<10} {'OLS tốt hơn?'}")
    all_better = True
    for j in range(len(beta_true)):
        var_ols = np.var(betas_ols[:, j])
        var_alt = np.var(betas_alt[:, j])
        ratio = var_ols / var_alt if var_alt > 0 else 0
        is_better = "✓" if var_ols <= var_alt else "✗"
        if var_ols > var_alt:
            all_better = False
        print(f"  β{j:<5}  {var_ols:<14.6f} {var_alt:<14.6f} {ratio:<10.4f} {is_better}")

    print(f"\n  → Kết luận: OLS có phương sai nhỏ nhất cho TẤT CẢ hệ số: "
          f"{'✓ ĐÚNG (Gauss-Markov verified)' if all_better else '✗ Cần kiểm tra lại'}")

    # Plotting
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    colors_ols = '#2E86AB'
    colors_alt = '#F6AE2D'

    for j in range(len(beta_true)):
        ax = axes[j // 2, j % 2]
        ax.hist(betas_ols[:, j], bins=40, alpha=0.65, density=True,
                color=colors_ols, edgecolor='white', linewidth=0.5,
                label=f'OLS β̂{j} (Var={np.var(betas_ols[:, j]):.4f})')
        ax.hist(betas_alt[:, j], bins=40, alpha=0.45, density=True,
                color=colors_alt, edgecolor='white', linewidth=0.5,
                label=f'WLS β̃{j} (Var={np.var(betas_alt[:, j]):.4f})')
        ax.axvline(beta_true[j], color='red', linestyle='--', linewidth=2,
                   label=f'β{j} thật = {beta_true[j]}')
        ax.axvline(np.mean(betas_ols[:, j]), color=colors_ols,
                   linestyle='-', linewidth=1.5, alpha=0.8,
                   label=f'E[β̂_OLS] = {np.mean(betas_ols[:, j]):.3f}')
        ax.set_xlabel(f'β{j}', fontsize=11)
        ax.set_ylabel('Mật độ (Density)', fontsize=10)
        ax.set_title(f'Phân phối β̂{j}', fontsize=12, fontweight='bold')
        ax.legend(fontsize=8, loc='upper right')
        ax.grid(True, alpha=0.2)

    plt.suptitle(
        f'Kiểm chứng Gauss-Markov: {n_simulations} simulations, n={n_samples}',
        fontsize=14, fontweight='bold', y=1.01
    )
    plt.tight_layout()
    plt.show()
