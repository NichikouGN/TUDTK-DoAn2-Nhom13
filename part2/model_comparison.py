# Phân công: Thành viên 2 & 3 - So sánh các mô hình (Bàn giao trước 28/05)
# Công việc:
# - Thành viên 2: Viết OLS Baseline, OLS Selection (loại biến đa cộng tuyến).
# - Thành viên 3: Viết Ridge, Lasso Regression & Bảng so sánh kết quả.

import numpy as np
import matplotlib.pyplot as plt
import sys
import os

sys.stdout.reconfigure(encoding='utf-8')

# Thêm thư mục gốc vào path để import part1
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from part2.data_pipeline import load_data
from part1.ols_implementation import ols_fit, hat_matrix, model_metrics, coef_inference, vif
from part1.ridge_lasso import ridge_fit, lasso_fit
from part1.cross_validation import cv_select_lambda
from part1.residual_analysis import residual_plots

def compute_metrics(y_true, y_pred):
    # Tính các chỉ số MAE, RMSE và R-squared để so sánh mô hình
    n = len(y_true)
    mae = sum(abs(y_true[i] - y_pred[i]) for i in range(n)) / n
    rmse = (sum((y_true[i] - y_pred[i])**2 for i in range(n)) / n) ** 0.5
    
    ss_res = sum((y_true[i] - y_pred[i])**2 for i in range(n))
    y_mean = sum(y_true) / n
    ss_tot = sum((y_true[i] - y_mean)**2 for i in range(n))
    r2 = 1 - ss_res / ss_tot
    return mae, rmse, r2


def run_ols_baseline(X_train, X_test, y_train, y_test, feature_names):
    # Chạy mô hình OLS chuẩn trên tập huấn luyện và in kết quả kiểm định hệ số
    print("\n=== OLS BASELINE MODEL ===")
    beta, sigma2 = ols_fit(X_train, y_train)
    y_pred = X_test @ beta
    mae, rmse, r2 = compute_metrics(y_test, y_pred)
    
    print(f"MAE:  {mae:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"R2:   {r2:.4f}")
    
    # Kiểm định ý nghĩa hệ số hồi quy
    se_beta, t_stats, p_values, confidence_intervals = coef_inference(X_train, y_train, beta, sigma2)
    print("\nChi tiết hệ số hồi quy OLS Baseline:")
    print(f"{'Feature':15s} | {'Coefficient':12s} | {'Std Error':10s} | {'t-stat':8s} | {'p-value':8s} | {'95% Conf. Interval':22s}")
    print("-" * 90)
    for i, name in enumerate(feature_names):
        ci_str = f"[{confidence_intervals[i][0]:.3f}, {confidence_intervals[i][1]:.3f}]"
        print(f"{name:15s} | {beta[i]:12.4f} | {se_beta[i]:10.4f} | {t_stats[i]:8.3f} | {p_values[i]:8.5f} | {ci_str:22s}")
        
    return beta, mae, rmse, r2


def run_ols_selection(X_train, X_test, y_train, y_test, feature_names):
    # Tính VIF phát hiện đa cộng tuyến, loại bỏ biến thừa (ở đây là 'atemp') và fit lại OLS
    print("\n=== OLS VARIABLE SELECTION ===")
    
    # 1. Tính VIF của các đặc trưng (trừ cột intercept)
    vif_values = vif(X_train)
    print("\nGiá trị VIF của các biến:")
    for name, v_val in zip(feature_names[1:], vif_values):
        print(f"  {name:12s}: {v_val:.4f}")
        
    # 2. Thực hiện lựa chọn biến: loại bỏ 'atemp' vì đa cộng tuyến cao với 'temp' (VIF > 10)
    cols_to_remove = ['atemp'] 
    print(f"\nLoại bỏ các đặc trưng: {cols_to_remove}")
    
    indices_to_keep = [i for i, name in enumerate(feature_names) if name not in cols_to_remove]
    X_train_reduced = X_train[:, indices_to_keep]
    X_test_reduced = X_test[:, indices_to_keep]
    features_reduced = [feature_names[i] for i in indices_to_keep]
    
    # Fit mô hình OLS rút gọn
    beta_reduced, sigma2_reduced = ols_fit(X_train_reduced, y_train)
    y_pred_reduced = X_test_reduced @ beta_reduced
    mae_red, rmse_red, r2_red = compute_metrics(y_test, y_pred_reduced)
    
    print(f"MAE (Reduced):  {mae_red:.4f}")
    print(f"RMSE (Reduced): {rmse_red:.4f}")
    print(f"R2 (Reduced):   {r2_red:.4f}")
    
    return beta_reduced, mae_red, rmse_red, r2_red, features_reduced


def plot_feature_importance(beta, feature_names, save_path=None):
    # Vẽ biểu đồ thanh thể hiện độ quan trọng của các đặc trưng (bỏ intercept)
    coefs = beta[1:]
    names = feature_names[1:]
    
    # Sắp xếp theo độ lớn trị tuyệt đối
    indices = np.argsort(np.abs(coefs))
    
    plt.figure(figsize=(10, 6))
    plt.barh(range(len(indices)), coefs[indices], align='center', color='steelblue', edgecolor='black', alpha=0.8)
    plt.yticks(range(len(indices)), [names[i] for i in indices])
    plt.xlabel('Hệ số hồi quy (Coefficient Value)', fontsize=12)
    plt.title('Độ quan trọng đặc trưng (Feature Importance) - OLS Baseline', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.show()


def plot_residuals(X_train, y_train, beta):
    # Gọi hàm vẽ 4 biểu đồ kiểm tra các giả định của phần dư
    print("\nĐang vẽ biểu đồ chẩn đoán phần dư...")
    residual_plots(X_train, y_train, beta)


def plot_cv_curve(cv_scores, model_name, best_lambda):
    # Vẽ đồ thị sai số CV (MSE) qua các lambda khác nhau trên trục log
    lambdas_val = [score[0] for score in cv_scores]
    mses_val = [score[1] for score in cv_scores]
    
    plt.figure(figsize=(8, 5))
    plt.plot(lambdas_val, mses_val, 'o-', color='darkorange' if model_name == 'Lasso' else 'steelblue', markersize=4, label='CV MSE')
    plt.axvline(best_lambda, color='red', linestyle='--', label=rf'$\lambda$ tối ưu = {best_lambda:.6f}')
    plt.xscale('log')
    plt.xlabel(r'$\lambda$ (Regularization Parameter)', fontsize=12)
    plt.ylabel('CV Score (MSE)', fontsize=12)
    plt.title(f'Cross-Validation Curve for {model_name}', fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def run_ridge(X_train, X_test, y_train, y_test, feature_names):
    # Tìm lambda tối ưu bằng 5-fold CV, vẽ đồ thị CV rồi khớp mô hình Ridge
    print("\n=== RIDGE REGRESSION ===")
    
    # 1. Tìm lambda tối ưu
    best_lambda, cv_scores = cv_select_lambda(X_train, y_train, k=5, model='ridge', seed=42)
    print(f"Lambda tốt nhất (Ridge): {best_lambda:.6f}")
    
    # 2. Vẽ đồ thị biểu diễn CV score qua các lambda
    plot_cv_curve(cv_scores, "Ridge", best_lambda)
    
    # 3. Fit mô hình với lambda tốt nhất
    beta_ridge = ridge_fit(X_train, y_train, best_lambda)
    
    # 4. Dự đoán và tính metrics trên test set
    y_pred = X_test @ beta_ridge
    mae, rmse, r2 = compute_metrics(y_test, y_pred)
    print(f"MAE:  {mae:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"R2:   {r2:.4f}")
    
    return beta_ridge, best_lambda, mae, rmse, r2


def run_lasso(X_train, X_test, y_train, y_test, feature_names):
    # Tìm lambda tối ưu cho Lasso bằng 5-fold CV, vẽ đồ thị và liệt kê biến bị triệt tiêu về 0
    print("\n=== LASSO REGRESSION ===")
    
    # 1. Tìm lambda tối ưu
    best_lambda, cv_scores = cv_select_lambda(X_train, y_train, k=5, model='lasso', seed=42)
    print(f"Lambda tốt nhất (Lasso): {best_lambda:.6f}")
    
    # 2. Vẽ đồ thị biểu diễn CV score qua các lambda
    plot_cv_curve(cv_scores, "Lasso", best_lambda)
    
    # 3. Fit mô hình với lambda tốt nhất
    beta_lasso = lasso_fit(X_train, y_train, best_lambda)
    
    # 4. Dự đoán và tính metrics trên test set
    y_pred = X_test @ beta_lasso
    mae, rmse, r2 = compute_metrics(y_test, y_pred)
    print(f"MAE:  {mae:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"R2:   {r2:.4f}")
    
    # 5. Kiểm tra các đặc trưng bị triệt tiêu về 0
    zero_features = [feature_names[i] for i, b in enumerate(beta_lasso) if abs(b) < 1e-6]
    print(f"Các đặc trưng bị triệt tiêu về 0: {zero_features}")
    
    return beta_lasso, best_lambda, mae, rmse, r2


def print_comparison_table(ols_res, ols_select_res, ridge_res, lasso_res):
    # In bảng so sánh kết quả MAE, RMSE, R-squared giữa các mô hình hồi quy
    print("\n=== BẢNG SO SÁNH KẾT QUẢ ===")
    print(f"{'Mô hình':20s} | {'MAE':10s} | {'RMSE':10s} | {'R2 (test)':10s}")
    print("-" * 60)
    
    def print_row(name, metrics):
        if metrics is not None:
            mae, rmse, r2 = metrics
            print(f"{name:20s} | {mae:10.4f} | {rmse:10.4f} | {r2:10.4f}")
        else:
            print(f"{name:20s} | {'N/A':10s} | {'N/A':10s} | {'N/A':10s}")
            
    print_row("OLS Baseline", ols_res)
    print_row("OLS Selection", ols_select_res)
    print_row("Ridge", ridge_res)
    print_row("Lasso", lasso_res)


if __name__ == "__main__":
    X_train, X_test, y_train, y_test, feature_names = load_data()
    
    beta_ols, mae_o, rmse_o, r2_o = run_ols_baseline(X_train, X_test, y_train, y_test, feature_names)
    beta_red, mae_s, rmse_s, r2_s, feat_red = run_ols_selection(X_train, X_test, y_train, y_test, feature_names)
    beta_ridge, lam_r, mae_r, rmse_r, r2_r = run_ridge(X_train, X_test, y_train, y_test, feature_names)
    beta_lasso, lam_l, mae_l, rmse_l, r2_l = run_lasso(X_train, X_test, y_train, y_test, feature_names)
    
    print_comparison_table(
        ols_res=(mae_o, rmse_o, r2_o),
        ols_select_res=(mae_s, rmse_s, r2_s),
        ridge_res=(mae_r, rmse_r, r2_r),
        lasso_res=(mae_l, rmse_l, r2_l)
    )
