import numpy as np
import matplotlib.pyplot as plt
from part1.ols_implementation import inverse, transpose


# Soft Thresholding Operator
# S(z, γ) = sign(z) * max(|z| - γ, 0)
# Dùng trong Lasso Regression (coordinate descent)
def soft_threshold(z, gamma):
    if z > gamma:
        return z - gamma
    elif z < -gamma:
        return z + gamma
    else:
        return 0.0


# Ridge Regression (L2 Regularization)
# β_ridge = (X^T X + λI)^(-1) X^T y
# Không penalize intercept (β₀)
def ridge_fit(X, y, lam):
    # Khớp mô hình Ridge: beta = (X^T X + lam * I)^(-1) X^T y (không phạt hệ số tự do)
    y_col = [[yi] for yi in y]
    n = len(X)
    p = len(X[0])

    # Tính X^T X
    XT = transpose(X)
    XTX = XT @ X

    # Tạo ma trận penalty: λI nhưng KHÔNG phạt intercept (vị trí [0,0] = 0)
    penalty = [[0.0] * p for _ in range(p)]
    for i in range(1, p):
        penalty[i][i] = lam

    # (X^T X + λI)
    XTX_reg = np.array(XTX) + np.array(penalty)

    # Nghịch đảo bằng hàm tự cài đặt (Gaussian elimination)
    XTX_reg_inv = inverse(XTX_reg)

    # β = (X^TX + λI)^(-1) X^T y
    XTy = XT @ y_col
    beta_ridge = np.array(XTX_reg_inv) @ np.array(XTy)

    return np.array(beta_ridge).flatten()


# Lasso Regression (L1 Regularization)
# Giải bằng Coordinate Descent (không có closed-form)
# Objective: min ||y - Xβ||² + λ Σ|βⱼ| (j = 1..p, không phạt β₀)
def lasso_fit(X, y, lam, max_iter=1000, tol=1e-6):
    # Khớp mô hình Lasso bằng coordinate descent (không phạt hệ số tự do)
    X = np.array(X, dtype=float)
    y = np.array(y, dtype=float)
    n, p = X.shape

    # Khởi tạo β = 0
    beta = np.zeros(p)

    for iteration in range(max_iter):
        beta_old = beta.copy()

        for j in range(p):
            # Tính partial residual: r_j = y - Σ_{k≠j} X_k β_k
            # = y - Xβ + X_j β_j
            r_j = y - X @ beta + X[:, j] * beta[j]

            # ρ_j = X_j^T r_j
            rho_j = np.dot(X[:, j], r_j)

            # z_j = X_j^T X_j
            z_j = np.dot(X[:, j], X[:, j])

            if j == 0:
                # Không phạt intercept
                beta[j] = rho_j / z_j
            else:
                # Soft thresholding: β_j = S(ρ_j, λ/2) / z_j
                beta[j] = soft_threshold(rho_j, lam / 2.0) / z_j

        # Kiểm tra hội tụ
        max_change = max(abs(beta[j] - beta_old[j]) for j in range(p))
        if max_change < tol:
            break

    return beta


# Ridge Trace Plot
# Vẽ biểu đồ hệ số β theo λ (trục log)
def plot_ridge_trace(X, y, lambdas=None, feature_names=None):
    # Vẽ biểu đồ hệ số beta biến thiên theo lambda (trục log)
    if lambdas is None:
        lambdas = np.logspace(-2, 6, 100)

    p = len(X[0])
    coefs = []

    for lam in lambdas:
        beta = ridge_fit(X, y, lam)
        coefs.append(beta[1:])  # Bỏ intercept, chỉ lấy hệ số đặc trưng

    coefs = np.array(coefs)

    plt.figure(figsize=(10, 6))
    for j in range(coefs.shape[1]):
        label = feature_names[j] if feature_names else f"β{j+1}"
        plt.plot(lambdas, coefs[:, j], label=label, linewidth=1.5)

    plt.xscale('log')
    plt.xlabel('λ (Regularization Parameter)', fontsize=12)
    plt.ylabel('Coefficient Value', fontsize=12)
    plt.title('Ridge Trace', fontsize=14, fontweight='bold')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
    plt.axhline(y=0, color='black', linestyle='--', linewidth=0.5)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
