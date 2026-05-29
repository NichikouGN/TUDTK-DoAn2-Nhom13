# BẢN PHÂN CÔNG CÔNG VIỆC PHẦN 2 (NEW)
## Đồ Án 2: Data Fitting và Phương Pháp OLS - Nhóm 13

Dưới đây là bảng phân công chi tiết công việc cho 4 thành viên nhóm 13 thực hiện Phần 2. Nhiệm vụ được chia rõ ràng theo từng file, từng hàm kèm theo đặc tả Input/Output và ngày bàn giao cụ thể.

---

## 1. THÀNH VIÊN 1: KỸ SƯ DỮ LIỆU & TIỀN XỬ LÝ (DATA PIPELINE ENGINEER)
* **File phụ trách:** [data_pipeline.py](file:///c:/Users/NAM/Downloads/Part2_UDTK/TUDTK-DoAn2-Nhom13/part2/data_pipeline.py)
* **Hạn bàn giao:** **27/05/2026**
* **Nhiệm vụ chi tiết:**

### Hàm 1: `__init__(self, imputation_method='median', seed=42)`
* **Mô tả:** Khởi tạo lớp xử lý dữ liệu.
* **Input:** 
  * `imputation_method` (str): Phương pháp điền khuyết (`'mean'`, `'median'`).
  * `seed` (int): Random seed để đảm bảo tính tái lập kết quả.
* **Output:** Không có (gán giá trị vào `self`).

### Hàm 2: `fit(self, df_train)`
* **Mô tả:** Học các tham số thống kê từ tập huấn luyện (train set).
* **Input:** `df_train` (pandas.DataFrame) - Dữ liệu thô của tập huấn luyện.
* **Output:** Trả về chính đối tượng `self` sau khi đã học tham số.
* **Yêu cầu chi tiết:** Loại bỏ cột thừa (`instant`, `dteday`, `casual`, `registered`). Học các giá trị điền khuyết (Mean/Median) cho từng đặc trưng và học vector `mean` & `std` của tập train phục vụ cho chuẩn hóa Z-score.

### Hàm 3: `transform(self, df)`
* **Mô tả:** Áp dụng tiền xử lý (imputation + scaling + intercept) lên tập dữ liệu bất kỳ.
* **Input:** `df` (pandas.DataFrame) - Dữ liệu thô cần biến đổi.
* **Output:** 
  * `X_final` (numpy.ndarray) - Ma trận đặc trưng cỡ $n \times (p+1)$ có cột đầu tiên toàn số 1 (intercept).
  * `y` (numpy.ndarray) - Vector biến mục tiêu cỡ $n$ (nếu có trong DataFrame).

### Hàm 4: `run(self, df, seed=42)`
* **Mô tả:** Thực hiện chia dữ liệu Train/Test và chạy toàn bộ pipeline.
* **Input:** `df` (pandas.DataFrame), `seed` (int).
* **Output:** `X_train`, `X_test`, `y_train`, `y_test` (numpy arrays) và `feature_names` (danh sách tên biến).
* **Yêu cầu:** Chia dữ liệu ngẫu nhiên theo tỷ lệ 80% Train và 20% Test trước khi thực hiện fit & transform.

### Hàm 5: `load_data()`
* **Mô tả:** Hàm helper hỗ trợ đọc dữ liệu thực tế và chạy pipeline.
* **Input:** Không có.
* **Output:** `X_train`, `X_test`, `y_train`, `y_test` (numpy arrays) và `feature_names`.

---

## 2. THÀNH VIÊN 2: NHÀ PHÂN TÍCH MÔ HÌNH OLS & SAI SỐ (OLS & RESIDUAL ANALYST)
* **File phụ trách:** [model_comparison.py](file:///c:/Users/NAM/Downloads/Part2_UDTK/TUDTK-DoAn2-Nhom13/part2/model_comparison.py) (Phần OLS)
* **Hạn bàn giao:** **28/05/2026**
* **Nhiệm vụ chi tiết:**

### Hàm 1: `compute_metrics(y_true, y_pred)`
* **Mô tả:** Tính toán các chỉ số đánh giá sai số của mô hình.
* **Input:** `y_true` (array-like), `y_pred` (array-like).
* **Output:** Trả về bộ ba giá trị `(mae, rmse, r2)` kiểu float.

### Hàm 2: `run_ols_baseline(X_train, X_test, y_train, y_test, feature_names)`
* **Mô tả:** Xây dựng mô hình OLS cơ bản trên tất cả đặc trưng và kiểm định hệ số.
* **Input:** Các tập train/test và danh sách tên đặc trưng.
* **Output:** `(beta, mae, rmse, r2)`.
* **Yêu cầu:** Dự đoán trên tập test, tính chỉ số đánh giá. Gọi hàm `coef_inference` từ Part 1 để thực hiện kiểm định giả thuyết $t$ cho từng hệ số hồi quy (p-value, khoảng tin cậy 95%).

### Hàm 3: `run_ols_selection(X_train, X_test, y_train, y_test, feature_names)`
* **Mô tả:** Lựa chọn biến dựa trên chỉ số phóng đại sai số VIF.
* **Input:** Tương tự OLS Baseline.
* **Output:** `(beta_reduced, mae_red, rmse_red, r2_red, features_reduced)`.
* **Yêu cầu:** Tính VIF cho các biến. Loại bỏ đặc trưng bị đa cộng tuyến nghiêm trọng (`atemp` vì VIF > 10). Thực hiện huấn luyện lại OLS trên tập đặc trưng rút gọn.

### Hàm 4: `plot_feature_importance(beta, feature_names, save_path=None)`
* **Mô tả:** Trực quan hóa độ lớn các hệ số hồi quy OLS Baseline.
* **Input:** Vector hệ số `beta`, `feature_names`.
* **Output:** Hiển thị biểu đồ dạng thanh ngang (không vẽ intercept).

### Hàm 5: `plot_residuals(X_train, y_train, beta)`
* **Mô tả:** Gọi hàm vẽ 4 biểu đồ chẩn đoán phần dư.
* **Input:** Ma trận `X_train`, nhãn `y_train`, vector hệ số OLS `beta`.

---

## 3. THÀNH VIÊN 3: CHUYÊN GIA ĐIỀU PHỐI REGULARIZATION (REGULARIZATION SPECIALIST)
* **File phụ trách:** [model_comparison.py](file:///c:/Users/NAM/Downloads/Part2_UDTK/TUDTK-DoAn2-Nhom13/part2/model_comparison.py) (Phần Ridge/Lasso)
* **Hạn bàn giao:** **28/05/2026**
* **Nhiệm vụ chi tiết:**

### Hàm 1: `plot_cv_curve(cv_scores, model_name, best_lambda)`
* **Mô tả:** Vẽ đồ thị biến thiên của sai số CV theo giá trị $\lambda$.
* **Input:** `cv_scores` (danh sách tuple `(lambda, score)` từ CV), `model_name` (str), `best_lambda` (float).
* **Output:** Vẽ đồ thị (trục hoành log-scale) thể hiện điểm $\lambda$ tối ưu.

### Hàm 2: `run_ridge(X_train, X_test, y_train, y_test, feature_names)`
* **Mô tả:** Huấn luyện mô hình hồi quy Ridge (L2 regularization).
* **Input:** Các tập train/test và danh sách tên đặc trưng.
* **Output:** `(beta_ridge, best_lambda, mae, rmse, r2)`.
* **Yêu cầu:** Gọi `cv_select_lambda` từ Part 1 thực hiện 5-fold CV chọn $\lambda$ tối ưu. Vẽ đường cong CV. Huấn luyện Ridge với $\lambda$ tối ưu và tính hiệu năng trên Test Set.

### Hàm 3: `run_lasso(X_train, X_test, y_train, y_test, feature_names)`
* **Mô tả:** Huấn luyện mô hình hồi quy Lasso (L1 regularization) và phân tích biến bị loại bỏ.
* **Input:** Tương tự Ridge.
* **Output:** `(beta_lasso, best_lambda, mae, rmse, r2)`.
* **Yêu cầu:** Chạy 5-fold CV chọn $\lambda$ tối ưu cho Lasso. Huấn luyện Lasso, tính hiệu năng và thống kê các đặc trưng có hệ số bị triệt tiêu hoàn toàn về 0.

### Hàm 4: `print_comparison_table(ols_res, ols_select_res, ridge_res, lasso_res)`
* **Mô tả:** In bảng tổng hợp so sánh hiệu năng các mô hình trên tập kiểm thử.
* **Input:** Bộ ba kết quả `(mae, rmse, r2)` của 4 mô hình.
* **Output:** In ra bảng có định dạng rõ ràng ở terminal.

---

## 4. THÀNH VIÊN 4: NHÀ NGHIÊN CỨU THUẬT TOÁN NÂNG CAO & TÍCH HỢP (ADVANCED METHODS & NOTEBOOK INTEGRATOR)
* **File phụ trách:** [advanced_methods.py](file:///c:/Users/NAM/Downloads/Part2_UDTK/TUDTK-DoAn2-Nhom13/part2/advanced_methods.py) & [part2_notebook.ipynb](file:///c:/Users/NAM/Downloads/Part2_UDTK/TUDTK-DoAn2-Nhom13/part2/part2_notebook.ipynb)
* **Hạn bàn giao:** **29/05/2026**
* **Nhiệm vụ chi tiết:**

### Hàm 1: `rbf_kernel(X1, X2, length_scale=1.0)`
* **Mô tả:** Tính toán ma trận Gram sử dụng nhân RBF (Gaussian kernel).
* **Input:** Ma trận $X_1$ ($n_1 \times d$), $X_2$ ($n_2 \times d$).
* **Output:** Ma trận Gram ($n_1 \times n_2$).

### Hàm 2: `kernel_ridge_predict(X_train, y_train, X_test, lam=1.0, length_scale=1.0)`
* **Mô tả:** Thực hiện dự báo bằng mô hình Kernel Ridge Regression phi tuyến.
* **Input:** Ma trận đặc trưng và nhãn tập train, ma trận đặc trưng tập test, tham số phạt `lam` ($\lambda$) và độ rộng nhân `length_scale`.
* **Output:** Vector dự đoán `y_pred` trên tập test.

### Hàm 3: `bayesian_regression(X_train, y_train, X_test, sigma2=1.0, prior_mean=None, prior_cov=None)`
* **Mô tả:** Huấn luyện mô hình hồi quy tuyến tính Bayesian và tính khoảng tin cậy.
* **Input:** Tập huấn luyện, tập kiểm thử, phương sai nhiễu $\sigma^2$, vector kỳ vọng tiên nghiệm `prior_mean` và ma trận hiệp phương sai tiên nghiệm `prior_cov`.
* **Output:** Vector dự báo `y_pred`, kỳ vọng hậu nghiệm `m_n`, hiệp phương sai hậu nghiệm `S_n`, phương sai dự báo `y_var`.

### Hàm 4: Các hàm unit tests (`test_compute_metrics`, `test_rbf_kernel`)
* **Mô tả:** Các kiểm thử độc lập cho hàm đánh giá và hàm tính nhân kernel RBF.

### Tích hợp Jupyter Notebook (`part2_notebook.ipynb`):
* **Yêu cầu:** Kết nối và gọi lần lượt các hàm tiền xử lý (Thành viên 1), mô hình OLS (Thành viên 2), mô hình co rút hệ số (Thành viên 3) và mô hình nâng cao (Thành viên 4) vào file notebook. Thực hiện phân tích, trực quan hóa và viết báo cáo nhận xét chi tiết cho từng phần.

---

## TỔNG HỢP LỊCH TRÌNH BÀN GIAO DỰ ÁN

| Ngày bàn giao | Người thực hiện | File bàn giao | Trạng thái |
|:---|:---|:---|:---|
| **27/05/2026** | **Thành viên 1** | `data_pipeline.py` | Hoàn thành tiền xử lý dữ liệu và pipeline. |
| **28/05/2026** | **Thành viên 2** | `model_comparison.py` (OLS) | Hoàn thành OLS Baseline, Selection & vẽ Residuals. |
| **28/05/2026** | **Thành viên 3** | `model_comparison.py` (Regularization) | Hoàn thành Ridge, Lasso, tìm $\lambda$ bằng CV & bảng so sánh. |
| **29/05/2026** | **Thành viên 4** | `advanced_methods.py` & `part2_notebook.ipynb` | Hoàn thành mô hình nâng cao, unit tests và tích hợp notebook. |

---

## 5. ĐÁNH GIÁ ĐỘ ĐẦY ĐỦ SO VỚI YÊU CẦU ĐỀ BÀI (CHECKLIST)

Bản phân công này đã bao phủ **100% các yêu cầu bắt buộc và tự chọn** của Phần 2 trong file PDF đề bài:
* **EDA & Missing Values:** Thành viên 1 phụ trách.
* **Tiền xử lý & Tránh rò rỉ dữ liệu (Data Leakage):** Đã thiết kế cấu trúc `fit` trên Train và `transform` trên Test trong lớp `DataPipeline` (Thành viên 1).
* **Kiểm tra Đa cộng tuyến & Chọn biến:** Sử dụng VIF để loại bỏ đặc trưng (Thành viên 2).
* **So sánh ít nhất 3 mô hình tuyến tính:** OLS Baseline, OLS Selection, Ridge, Lasso (Thành viên 2 & 3).
* **Các chỉ số MAE, RMSE, R² trên Test Set:** Thành viên 2 & 3.
* **4 biểu đồ chẩn đoán phần dư & Feature Importance:** Thành viên 2.
* **Mô hình nâng cao (Bonus):** Kernel Ridge và Bayesian Regression (Thành viên 4).
* **Unit Tests & Báo cáo Jupyter Notebook:** Thành viên 4.

### Gợi ý nâng cao cho nhóm để đạt điểm tối đa (Tùy chọn):
1. **Xử lý biến phân loại (Categorical Encoding) - Thành viên 1:** Bộ dữ liệu Bike Sharing có các biến phân loại như `season`, `weathersit`. Thành viên 1 có thể bổ sung phương pháp One-Hot Encoding (hoặc Dummy Variables) cho các cột này thay vì chuẩn hóa Z-score trực tiếp để mô hình có cơ sở toán học chặt chẽ hơn.
2. **Xử lý giá trị ngoại lai (Outliers) - Thành viên 1:** Bổ sung phương pháp lọc Outliers (IQR hoặc Winsorization) trước khi thực hiện chuẩn hóa Z-score.
3. **Thêm tương tác phi tuyến (Feature Engineering) - Thành viên 2 & 3:** Tạo thêm biến tương tác (ví dụ: `temp` nhân với `hum`) hoặc đặc trưng bậc cao (Polynomial Features) để xem có cải thiện được độ chính xác ($R^2$) hay không.

