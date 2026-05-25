import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from part1.ols_implementation import inverse, transpose


# 4 Biểu Đồ Chẩn Đoán Phần Dư (Residual Diagnostic Plots)
#   1. Residuals vs Fitted
#   2. Normal Q-Q Plot
#   3. Scale-Location Plot
#   4. Cook's Distance

def residual_plots(X, y, beta_hat):
    # Vẽ 4 biểu đồ chẩn đoán phần dư để kiểm tra giả thuyết mô hình hồi quy
    X = np.array(X, dtype=float)
    y = np.array(y, dtype=float).flatten()
    beta_hat = np.array(beta_hat, dtype=float).flatten()

    n = len(y)
    p = len(X[0]) - 1  # Số đặc trưng (không tính intercept)

    # Tính các đại lượng cần thiết

    # Giá trị fitted: ŷ = Xβ̂
    y_hat = X @ beta_hat

    # Phần dư: e = y - ŷ
    residuals = y - y_hat

    # MSE = RSS / (n - p - 1)
    RSS = float(sum(residuals[i] ** 2 for i in range(n)))
    MSE = RSS / (n - p - 1)

    # Hat matrix H = X(X^TX)^(-1)X^T để lấy leverage hᵢᵢ
    XT = transpose(X)
    XTX = np.array(XT) @ X
    XTX_inv = np.array(inverse(XTX))
    H = X @ XTX_inv @ np.array(XT)

    # Leverage: hᵢᵢ = đường chéo của H
    h = np.array([H[i][i] for i in range(n)])

    # Standardized residuals: rᵢ = eᵢ / (s √(1 - hᵢᵢ))
    std_residuals = np.zeros(n)
    for i in range(n):
        denom = np.sqrt(MSE * (1 - h[i]))
        if denom > 1e-12:
            std_residuals[i] = residuals[i] / denom
        else:
            std_residuals[i] = 0.0

    # Vẽ 4 biểu đồ
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))

    # 1. Residuals vs Fitted (Kiểm tra tính tuyến tính, đồng phương sai)
    ax1 = axes[0, 0]
    ax1.scatter(y_hat, residuals, alpha=0.5, edgecolors='steelblue',
                facecolors='lightblue', linewidth=0.8, s=30)
    ax1.axhline(y=0, color='red', linestyle='--', linewidth=1)

    # Đường xu hướng (moving average)
    sorted_idx = np.argsort(y_hat)
    window = max(n // 20, 5)
    if n > window:
        smoothed = np.convolve(residuals[sorted_idx],
                               np.ones(window) / window, mode='valid')
        x_smooth = y_hat[sorted_idx][window//2 : window//2 + len(smoothed)]
        ax1.plot(x_smooth, smoothed, color='red', linewidth=1.5, alpha=0.7)

    ax1.set_xlabel('Fitted Values (ŷ)')
    ax1.set_ylabel('Residuals (e)')
    ax1.set_title('1. Residuals vs Fitted')
    ax1.grid(True, alpha=0.2)

    # 2. Normal Q-Q Plot (Kiểm tra tính chuẩn của phần dư)
    ax2 = axes[0, 1]
    sorted_std_res = np.sort(std_residuals)

    # Tính quantile lý thuyết từ phân phối chuẩn N(0,1)
    theoretical_q = stats.norm.ppf(np.arange(1, n + 1) / (n + 1))

    ax2.scatter(theoretical_q, sorted_std_res, alpha=0.5,
                edgecolors='steelblue', facecolors='lightblue',
                linewidth=0.8, s=30)

    # Đường tham chiếu y = x
    min_val = min(theoretical_q.min(), sorted_std_res.min())
    max_val = max(theoretical_q.max(), sorted_std_res.max())
    ax2.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=1.5)

    ax2.set_xlabel('Theoretical Quantiles (N(0,1))')
    ax2.set_ylabel('Standardized Residuals')
    ax2.set_title('2. Normal Q-Q Plot')
    ax2.grid(True, alpha=0.2)

    # 3. Scale-Location Plot (Kiểm tra tính đồng phương sai)
    ax3 = axes[1, 0]
    sqrt_abs_std_res = np.sqrt(np.abs(std_residuals))

    ax3.scatter(y_hat, sqrt_abs_std_res, alpha=0.5,
                edgecolors='steelblue', facecolors='lightblue',
                linewidth=0.8, s=30)

    # Đường xu hướng
    if n > window:
        smoothed_scale = np.convolve(sqrt_abs_std_res[sorted_idx],
                                      np.ones(window) / window, mode='valid')
        x_smooth = y_hat[sorted_idx][window//2 : window//2 + len(smoothed_scale)]
        ax3.plot(x_smooth, smoothed_scale, color='red', linewidth=1.5, alpha=0.7)

    ax3.set_xlabel('Fitted Values (ŷ)')
    ax3.set_ylabel('√|Standardized Residuals|')
    ax3.set_title('3. Scale-Location')
    ax3.grid(True, alpha=0.2)

    # 4. Cook's Distance (Phát hiện điểm ảnh hưởng lớn)
    ax4 = axes[1, 1]
    cooks_d = np.zeros(n)
    for i in range(n):
        if abs(1 - h[i]) > 1e-12:
            cooks_d[i] = (std_residuals[i] ** 2 / (p + 1)) * (h[i] / (1 - h[i]) ** 2)

    markerline, stemlines, baseline = ax4.stem(
        range(n), cooks_d, linefmt='steelblue', markerfmt='o', basefmt='gray'
    )
    plt.setp(stemlines, linewidth=0.8, alpha=0.7)
    plt.setp(markerline, markersize=3)

    # Ngưỡng Cook's Distance = 4/n
    threshold = 4.0 / n
    ax4.axhline(y=threshold, color='red', linestyle='--', linewidth=1.2,
                label=f'Ngưỡng 4/n = {threshold:.4f}')

    # Đánh dấu các điểm vượt ngưỡng
    influential = [i for i in range(n) if cooks_d[i] > threshold]
    if influential:
        ax4.scatter(influential, cooks_d[influential], color='red', s=40, zorder=5,
                    label=f'{len(influential)} điểm influential')

    ax4.set_xlabel('Observation Index (i)')
    ax4.set_ylabel("Cook's Distance (Dᵢ)")
    ax4.set_title("4. Cook's Distance")
    ax4.legend(fontsize=9)
    ax4.grid(True, alpha=0.2)

    plt.suptitle('Residual Diagnostic Plots', fontsize=15, fontweight='bold', y=1.01)
    plt.tight_layout()
    plt.show()

    # In tóm tắt
    print(f"\n{'='*50}")
    print(f"TÓM TẮT PHÂN TÍCH PHẦN DƯ")
    print(f"{'='*50}")
    print(f"  Số quan sát: {n}")
    print(f"  Số đặc trưng: {p}")
    print(f"  RSS = {RSS:.4f}")
    print(f"  MSE = {MSE:.4f}")
    print(f"  Ngưỡng Cook's Distance (4/n) = {threshold:.4f}")
    print(f"  Số điểm influential (Dᵢ > 4/n): {len(influential)}")
    if influential:
        top_influential = sorted(influential, key=lambda i: cooks_d[i], reverse=True)[:5]
        print(f"  Top influential points: {top_influential}")
