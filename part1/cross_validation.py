import numpy as np
from part1.ols_implementation import ols_fit
from part1.ridge_lasso import ridge_fit, lasso_fit


# k-Fold Cross-Validation
# Chia dữ liệu thành k phần, huấn luyện trên k-1 phần,
# đánh giá trên 1 phần, lặp k lần và lấy trung bình MSE.
# CV(k) = (1/k) Σ MSEᵢ
def kfold_cv(X, y, k=5, model='ols', lam=1.0, seed=42):
    # Chia dữ liệu k folds để cross-validation, tính trung bình MSE qua các fold
    X = np.array(X, dtype=float)
    y = np.array(y, dtype=float).flatten()
    n = len(y)

    if k < 2 or k > n:
        raise ValueError(f"k phải thỏa 2 ≤ k ≤ n (n={n}), nhận k={k}")

    # Tạo chỉ số ngẫu nhiên và xáo trộn
    np.random.seed(seed)
    indices = np.arange(n)
    np.random.shuffle(indices)

    # Tính kích thước mỗi fold
    fold_size = n // k
    fold_scores = []

    for i in range(k):
        # Xác định chỉ số test cho fold thứ i
        start = i * fold_size
        if i == k - 1:
            end = n  # Fold cuối lấy hết phần còn lại
        else:
            end = start + fold_size

        test_idx = indices[start:end]
        train_idx = np.concatenate([indices[:start], indices[end:]])

        # Chia dữ liệu train/test
        X_train = X[train_idx]
        y_train = y[train_idx]
        X_test = X[test_idx]
        y_test = y[test_idx]

        # Huấn luyện mô hình trên tập train
        if model == 'ols':
            beta, _ = ols_fit(X_train, y_train)
        elif model == 'ridge':
            beta = ridge_fit(X_train, y_train, lam)
        elif model == 'lasso':
            beta = lasso_fit(X_train, y_train, lam)
        else:
            raise ValueError(f"model phải là 'ols', 'ridge', hoặc 'lasso', nhận '{model}'")

        # Dự đoán trên tập test
        y_pred = X_test @ beta

        # Tính MSE cho fold này
        mse = 0.0
        n_test = len(y_test)
        for j in range(n_test):
            mse += (y_test[j] - y_pred[j]) ** 2
        mse /= n_test

        fold_scores.append(mse)

    # CV score = trung bình MSE qua k folds
    cv_score = sum(fold_scores) / k

    return cv_score, fold_scores


# Chọn λ tối ưu cho Ridge/Lasso bằng k-fold CV
# Thử nhiều giá trị λ, chọn λ có CV score nhỏ nhất
def cv_select_lambda(X, y, k=5, model='ridge', lambdas=None, seed=42):
    # Chạy CV thử các giá trị lambda khác nhau rồi chọn cái tốt nhất (MSE thấp nhất)
    if lambdas is None:
        lambdas = np.logspace(-4, 4, 50)

    cv_scores = []
    best_lambda = None
    best_score = float('inf')

    for lam in lambdas:
        score, _ = kfold_cv(X, y, k=k, model=model, lam=lam, seed=seed)
        cv_scores.append((lam, score))

        if score < best_score:
            best_score = score
            best_lambda = lam

    print(f"\n{'='*50}")
    print(f"KẾT QUẢ CHỌN λ ({model.upper()}, {k}-fold CV)")
    print(f"{'='*50}")
    print(f"  λ tối ưu: {best_lambda:.6f}")
    print(f"  CV Score (MSE) tối ưu: {best_score:.4f}")

    return best_lambda, cv_scores
