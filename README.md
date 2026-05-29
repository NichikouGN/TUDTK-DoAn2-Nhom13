# Đồ Án 2: Data Fitting và Phương Pháp OLS - Nhóm 13

Repository này chứa mã nguồn và tài liệu cho Đồ án 2 môn Toán ứng dụng thống kê.

## 1. Cấu trúc nội dung

Đồ án được chia thành các phần trọng tâm:

- **`part1`**: Lý thuyết Data Fitting và Phương pháp OLS. Cài đặt các module hồi quy (OLS, Ridge, Lasso), kiểm định hệ số, tính toán khoảng tin cậy 95%, kiểm tra đa cộng tuyến (VIF), phân tích phần dư, k-fold cross-validation từ đầu (from scratch) và mô phỏng Monte Carlo kiểm chứng định lý Gauss-Markov.
- **`part2`**: Ứng dụng hồi quy tuyến tính OLS và Regularization vào dữ liệu thực tế (Bike Sharing Dataset), bao gồm phân tích EDA, xử lý dữ liệu khuyết thiếu, outliers, xây dựng pipeline chuẩn hóa dữ liệu, và các thuật toán nâng cao (Kernel Ridge Regression, Bayesian Linear Regression).

## 2. Yêu cầu hệ thống

- **Python**: `>= 3.10` (Khuyến khích sử dụng bản 3.12+).
- **Quản lý thư viện**: `pip`.
- **Công cụ hỗ trợ**: Jupyter Notebook hoặc VS Code (cài đặt Python/Jupyter extensions).

## 3. Cài đặt môi trường

Thực hiện các lệnh sau để thiết lập môi trường làm việc:

```bash
git clone https://github.com/vmq-16/TUDTK-DoAn2-Nhom13.git
cd TUDTK-DoAn2-Nhom13
python -m venv venv #Windows
python3 -m venv venv #macOS / Linux
```

**Kích hoạt môi trường và cài đặt thư viện:**

- Windows (Git Bash): `source venv/Scripts/activate`
- macOS / Linux: `source venv/bin/activate`
- Cài đặt: `pip install -r requirements.txt`

---
