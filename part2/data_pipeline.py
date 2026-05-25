# Phân công: Thành viên 1 - DL & Tiền xử lý (Bàn giao trước 27/05)
# Công việc: Hoàn thiện lớp DataPipeline và hàm load_data() để làm sạch, chuẩn hóa dữ liệu thực tế và chia tập train/test 80/20.

import pandas as pd
import numpy as np
import os

class DataPipeline:
    def __init__(self, imputation_method='median', seed=42):
        # Khởi tạo các cấu hình: điền khuyết (mean/median), seed, và các biến lưu mean_, std_
        self.imputation_method = imputation_method
        self.seed = seed
        self.mean_ = None
        self.std_ = None
        self.impute_values_ = {}
        self.features_ = []
        self.target_ = 'cnt'
        self.drop_cols_ = ['instant', 'dteday', 'casual', 'registered']

    def fit(self, df_train):
        # Tính toán các tham số chuẩn hóa (mean, std) và giá trị điền khuyết từ tập train
        np.random.seed(self.seed)
        
        # 1. Xác định danh sách feature sau khi drop
        df_clean = df_train.drop(columns=self.drop_cols_, errors='ignore')
        self.features_ = [col for col in df_clean.columns if col != self.target_]
        
        # 2. Học các giá trị missing value imputation (Mean/Median)
        # TODO: Người 1 bổ sung các phương pháp khác nếu cần thiết (như Regression imputation, k-NN)
        for col in self.features_:
            if df_clean[col].isnull().sum() > 0:
                if self.imputation_method == 'mean':
                    self.impute_values_[col] = df_clean[col].mean()
                elif self.imputation_method == 'median':
                    self.impute_values_[col] = df_clean[col].median()
        
        # Điền missing trước khi tính mean/std để chuẩn hóa
        df_imputed = df_clean.copy()
        for col, val in self.impute_values_.items():
            df_imputed[col] = df_imputed[col].fillna(val)
            
        # 3. Học mean và std cho z-score normalization
        X_train_raw = df_imputed[self.features_].values
        self.mean_ = X_train_raw.mean(axis=0)
        self.std_ = X_train_raw.std(axis=0)
        self.std_[self.std_ == 0] = 1.0  # Tránh chia cho 0
        
        return self

    def transform(self, df):
        # Áp dụng điền khuyết, chuẩn hóa z-score và thêm cột intercept (toàn 1) vào X
        df_clean = df.drop(columns=self.drop_cols_, errors='ignore')
        
        # 1. Điền missing values
        df_imputed = df_clean.copy()
        for col, val in self.impute_values_.items():
            df_imputed[col] = df_imputed[col].fillna(val)
            
        # 2. Tách X, y
        X_raw = df_imputed[self.features_].values
        
        # 3. Chuẩn hóa z-score (sử dụng mean và std đã fit từ train set)
        X_scaled = (X_raw - self.mean_) / self.std_
        
        # 4. Thêm cột intercept vào vị trí cột đầu tiên
        X_final = np.column_stack([np.ones(len(X_scaled)), X_scaled])
        
        if self.target_ in df_clean.columns:
            y = df_clean[self.target_].values
            return X_final, y
        else:
            return X_final

    def run(self, df, seed=42):
        # Chia tập train/test 80/20 ngẫu nhiên rồi thực hiện fit & transform cho cả hai tập
        np.random.seed(seed)
        n = len(df)
        indices = np.arange(n)
        np.random.shuffle(indices)
        split = int(n * 0.8)
        
        train_df = df.iloc[indices[:split]].reset_index(drop=True)
        test_df = df.iloc[indices[split:]].reset_index(drop=True)
        
        # Fit trên train
        self.fit(train_df)
        
        # Transform cả train và test
        X_train, y_train = self.transform(train_df)
        X_test, y_test = self.transform(test_df)
        
        feature_names = ['intercept'] + self.features_
        return X_train, X_test, y_train, y_test, feature_names

def load_data():
    # Đọc file day.csv và áp dụng DataPipeline tiền xử lý để lấy các numpy array huấn luyện
    import os
    import pandas as pd
    script_dir = os.path.dirname(os.path.abspath(__file__))
    df = pd.read_csv(os.path.join(script_dir, 'data', 'day.csv'))
    pipeline = DataPipeline(imputation_method='median')
    return pipeline.run(df)

# Ví dụ kiểm thử chạy pipeline độc lập
if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    df = pd.read_csv(os.path.join(script_dir, 'data', 'day.csv'))
    
    pipeline = DataPipeline(imputation_method='median')
    X_train, X_test, y_train, y_test, feature_names = pipeline.run(df)
    
    print("=== DATA PIPELINE RUN (SUCCESS) ===")
    print(f"X_train shape: {X_train.shape}")
    print(f"X_test shape:  {X_test.shape}")
    print(f"y_train shape: {y_train.shape}")
    print(f"y_test shape:  {y_test.shape}")
    print(f"Features: {feature_names}")
