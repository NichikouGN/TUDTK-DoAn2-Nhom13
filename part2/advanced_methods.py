# Phân công: Thành viên 4 - Kỹ thuật nâng cao & Tích hợp (Bàn giao trước 29/05)
# Công việc: Xây dựng Kernel Ridge phi tuyến, hồi quy Bayesian, viết unit tests và tích hợp vào notebook.

import numpy as np
import sys
import os

sys.stdout.reconfigure(encoding='utf-8')

# Thêm thư mục gốc vào path để import part1
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from part2.data_pipeline import load_data
from part2.model_comparison import compute_metrics
from part1.ols_implementation import inverse

def rbf_kernel(X1, X2, length_scale=1.0):
    # Tính ma trận Gram dùng nhân RBF (Radial Basis Function)
    n1 = len(X1)
    n2 = len(X2)
    K = np.zeros((n1, n2))
    for i in range(n1):
        for j in range(n2):
            dist_sq = np.sum((X1[i] - X2[j])**2)
            K[i, j] = np.exp(-dist_sq / (2 * length_scale**2))
    return K

def kernel_ridge_predict(X_train, y_train, X_test, lam=1.0, length_scale=1.0):
    # Huấn luyện và dự đoán bằng mô hình Kernel Ridge Regression phi tuyến
    print("Running Kernel Ridge Regression...")
    K = rbf_kernel(X_train, X_train, length_scale)
    k_test = rbf_kernel(X_test, X_train, length_scale)
    
    n = len(K)
    A = K + lam * np.eye(n)
    
    # Có thể dùng hàm nghịch đảo tự viết hoặc linalg.inv (để tăng tốc độ tính toán)
    try:
        A_inv = np.linalg.inv(A)
    except:
        A_inv = inverse(A)
        
    alpha = A_inv @ y_train
    y_pred = k_test @ alpha
    return y_pred

def bayesian_regression(X_train, y_train, X_test, sigma2=1.0, prior_mean=None, prior_cov=None):
    # Hồi quy tuyến tính Bayesian, tính phân phối hậu nghiệm (mean m_n, cov S_n) và phương sai dự đoán
    print("Running Bayesian Linear Regression...")
    p = X_train.shape[1]
    
    if prior_mean is None:
        prior_mean = np.zeros(p)
    if prior_cov is None:
        prior_cov = np.eye(p) * 10.0  # Vague prior (giả định phân phối tiên nghiệm rộng)
        
    # Tính nghịch đảo ma trận hiệp phương sai tiên nghiệm
    S0_inv = np.linalg.inv(prior_cov)
    
    # Cập nhật Posterior Covariance (S_n) và Mean (m_n)
    S_n = np.linalg.inv(S0_inv + (1.0 / sigma2) * X_train.T @ X_train)
    m_n = S_n @ (S0_inv @ prior_mean + (1.0 / sigma2) * X_train.T @ y_train)
    
    # Dự đoán giá trị trung bình trên test set
    y_pred = X_test @ m_n
    
    # Tính phương sai của dự đoán (để vẽ khoảng tin cậy 95%)
    y_var = np.array([x @ S_n @ x * sigma2 for x in X_test])
    
    return y_pred, m_n, S_n, y_var

# --- UNIT TESTS ---
def test_compute_metrics():
    y_true = np.array([100.0, 200.0, 300.0])
    y_pred = np.array([100.0, 200.0, 300.0])
    mae, rmse, r2 = compute_metrics(y_true, y_pred)
    assert mae == 0.0, "MAE test failed"
    assert rmse == 0.0, "RMSE test failed"
    assert abs(r2 - 1.0) < 1e-6, "R2 test failed"
    print("test_compute_metrics: PASSED")

def test_rbf_kernel():
    X1 = np.array([[1.0, 2.0]])
    X2 = np.array([[1.0, 2.0]])
    K = rbf_kernel(X1, X2, length_scale=1.0)
    assert abs(K[0, 0] - 1.0) < 1e-6, "RBF Kernel same point test failed"
    print("test_rbf_kernel: PASSED")

if __name__ == "__main__":
    # Chạy các unit test
    print("Chạy Unit Tests...")
    test_compute_metrics()
    test_rbf_kernel()
    
    # Chạy thử mô hình
    X_train, X_test, y_train, y_test, feature_names = load_data()
    y_pred_kr = kernel_ridge_predict(X_train, y_train, X_test)
    mae_kr, rmse_kr, r2_kr = compute_metrics(y_test, y_pred_kr)
    print(f"Kernel Ridge -> MAE: {mae_kr:.4f}, RMSE: {rmse_kr:.4f}, R2: {r2_kr:.4f}")
    
    y_pred_bay, m_n, S_n, y_var = bayesian_regression(X_train, y_train, X_test)
    mae_bay, rmse_bay, r2_bay = compute_metrics(y_test, y_pred_bay)
    print(f"Bayesian -> MAE: {mae_bay:.4f}, RMSE: {rmse_bay:.4f}, R2: {r2_bay:.4f}")
