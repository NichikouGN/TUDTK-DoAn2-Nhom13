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
    """
    Tính toán các chỉ số đánh giá sai số của mô hình.
    Dựa trên công thức: MAE, RMSE và R-squared.
    """
    n = len(y_true)
    
    # Tính Mean Absolute Error (MAE)
    mae = np.sum(np.abs(y_true - y_pred)) / n
    
    # Tính Root Mean Squared Error (RMSE)
    rmse = np.sqrt(np.sum((y_true - y_pred) ** 2) / n)
    
    # Tính R-squared (R2)
    tss = np.sum((y_true - np.mean(y_true)) ** 2)
    rss = np.sum((y_true - y_pred) ** 2)
    r2 = 1 - (rss / tss) if tss != 0 else 0.0
    
    return float(mae), float(rmse), float(r2)


def run_ols_baseline(X_train, X_test, y_train, y_test, feature_names):
    """
    Xây dựng mô hình OLS cơ bản trên tất cả đặc trưng và kiểm định hệ số.
    """
    print("\n" + "="*50)
    print("🚀 MÔ HÌNH OLS BASELINE (ĐẦY ĐỦ ĐẶC TRƯNG)")
    print("="*50)
    
    # Huấn luyện mô hình từ code Phần 1
    beta, sigma2 = ols_fit(X_train, y_train)
    
    # Dự đoán trên tập Test
    y_pred = X_test @ beta
    mae, rmse, r2 = compute_metrics(y_test, y_pred)
    
    # Kiểm định giả thuyết t cho từng hệ số hồi quy (t-test)
    se, t_stat, p_values, ci_lower, ci_upper = coef_inference(X_train, y_train, beta, sigma2)
    
    # In báo cáo thống kê
    print(f"Hiệu năng trên tập Test -> MAE: {mae:.4f} | RMSE: {rmse:.4f} | R²: {r2:.4f}\n")
    print(f"{'Đặc trưng':<15} | {'Hệ số Beta':<12} | {'Sai số chuẩn':<12} | {'t-stat':<10} | {'Khoảng tin cậy 95%':<20}")
    print("-" * 80)
    
    # Intercept luôn nằm ở index 0
    ci_int_str = f"[{ci_lower[0]:.3f}, {ci_upper[0]:.3f}]"
    print(f"{'Intercept':<15} | {beta[0]:<12.4f} | {se[0]:<12.4f} | {t_stat[0]:<10.4f} | {ci_int_str:<20}")
    
    for i, name in enumerate(feature_names):
        ci_str = f"[{ci_lower[i+1]:.3f}, {ci_upper[i+1]:.3f}]"
        print(f"{name:<15} | {beta[i+1]:<12.4f} | {se[i+1]:<12.4f} | {t_stat[i+1]:<10.4f} | {ci_str:<20}")
        
    return beta, mae, rmse, r2


def run_ols_selection(X_train, X_test, y_train, y_test, feature_names):
    """
    Lựa chọn biến dựa trên chỉ số VIF. Loại bỏ đặc trưng bị đa cộng tuyến.
    """
    print("\n" + "="*50)
    print("🔍 MÔ HÌNH OLS SELECTION (XỬ LÝ ĐA CỘNG TUYẾN)")
    print("="*50)
    
    # Tính chỉ số phóng đại phương sai (VIF)
    vifs = vif(X_train)
    print("Kiểm tra VIF trước khi loại bỏ:")
    for name, v in zip(feature_names, vifs):
        print(f" - {name:<10}: VIF = {v:.4f} {'(CẢNH BÁO)' if v > 10 else ''}")
        
    # Tìm index của 'atemp' để loại bỏ vì VIF > 10
    idx_to_drop = -1
    for i, name in enumerate(feature_names):
        if name == 'atemp':
            idx_to_drop = i
            break
            
    if idx_to_drop != -1:
        print(f"\n[!] Thực hiện loại bỏ đặc trưng: 'atemp'")
        # Chú ý: Index trong feature_names cần +1 khi đối chiếu với X_train vì X_train có cột Intercept ở đầu
        col_to_drop_in_X = idx_to_drop + 1 
        
        X_train_red = np.delete(X_train, col_to_drop_in_X, axis=1)
        X_test_red = np.delete(X_test, col_to_drop_in_X, axis=1)
        features_reduced = [name for name in feature_names if name != 'atemp']
    else:
        print("\nKhông tìm thấy biến cần loại bỏ.")
        X_train_red, X_test_red, features_reduced = X_train, X_test, feature_names
        
    # Huấn luyện lại mô hình OLS trên tập dữ liệu đã rút gọn
    beta_reduced, sigma2_red = ols_fit(X_train_red, y_train)
    y_pred_red = X_test_red @ beta_reduced
    mae_red, rmse_red, r2_red = compute_metrics(y_test, y_pred_red)
    
    print(f"Hiệu năng (Rút gọn) -> MAE: {mae_red:.4f} | RMSE: {rmse_red:.4f} | R²: {r2_red:.4f}")
    
    return beta_reduced, mae_red, rmse_red, r2_red, features_reduced


def plot_feature_importance(beta, feature_names, save_path=None):
    """
    Trực quan hóa độ lớn các hệ số hồi quy (Feature Importance).
    """
    # Bỏ qua hệ số Intercept ở vị trí 0
    coefs = beta[1:]
    
    # Sắp xếp các biến theo giá trị tuyệt đối của hệ số hồi quy để dễ nhìn
    sorted_indices = np.argsort(np.abs(coefs))
    sorted_coefs = coefs[sorted_indices]
    sorted_names = [feature_names[i] for i in sorted_indices]
    
    plt.figure(figsize=(10, 6))
    bars = plt.barh(range(len(sorted_coefs)), sorted_coefs, color='royalblue', edgecolor='black', alpha=0.85)
    plt.yticks(range(len(sorted_coefs)), sorted_names)
    
    plt.axvline(x=0, color='red', linestyle='--', linewidth=1.5)
    plt.xlabel('Trọng số hồi quy (Beta Coefficient)', fontsize=11)
    plt.title('Mức Độ Đóng Góp Của Các Đặc Trưng Trong Mô Hình OLS', fontsize=14, fontweight='bold')
    plt.grid(axis='x', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300)
    plt.show()


def plot_residuals(X_train, y_train, beta):
    """
    Gọi hàm vẽ 4 biểu đồ chẩn đoán phần dư từ Part 1.
    """
    print("\n[INFO] Đang kết xuất 4 biểu đồ chẩn đoán phần dư (Residual Analysis)...")
    try:
        residual_plots(X_train, y_train, beta)
    except NameError:
        print("[LỖI] Chưa import hàm `residual_plots` từ Phần 1. Vui lòng đảm bảo module được liên kết đúng.")


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
