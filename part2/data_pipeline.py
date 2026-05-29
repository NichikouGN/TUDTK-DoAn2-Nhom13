# Phân công: Thành viên 1 - DL & Tiền xử lý (Bàn giao trước 27/05)
# Công việc: Hoàn thiện lớp DataPipeline và hàm load_data() để làm sạch, chuẩn hóa dữ liệu thực tế và chia tập train/test 80/20.

import pandas as pd
import numpy as np
import os
from sklearn.linear_model import LinearRegression  # Thêm module để làm Regression Imputation

class DataPipeline:
    def __init__(self, imputation_method='median', seed=42):
        self.imputation_method = imputation_method
        self.seed = seed
        self.mean_ = None
        self.std_ = None
        self.impute_values_ = {}
        self.regression_models_ = {}  # Bộ nhớ lưu các mô hình hồi quy để điền khuyết
        self.features_ = []
        self.target_ = 'cnt'
        self.drop_cols_ = ['instant', 'dteday', 'casual', 'registered']

    def fit(self, df_train):
        np.random.seed(self.seed)
        df_clean = df_train.drop(columns=self.drop_cols_, errors='ignore')
        self.features_ = [col for col in df_clean.columns if col != self.target_]
        
        # HỌC CÁC GIÁ TRỊ VÀ MÔ HÌNH ĐỂ ĐIỀN KHUYẾT
        if self.imputation_method in ['mean', 'median']:
            for col in self.features_:
                if df_clean[col].isnull().sum() > 0:
                    self.impute_values_[col] = df_clean[col].mean() if self.imputation_method == 'mean' else df_clean[col].median()
        
        elif self.imputation_method == 'regression':
            # Bước 1: Lấy median làm bệ đỡ tạm thời cho tất cả các cột
            for col in self.features_:
                self.impute_values_[col] = df_clean[col].median()
            
            df_temp = df_clean.copy()
            for col in self.features_:
                df_temp[col] = df_temp[col].fillna(self.impute_values_[col])
            
            # Bước 2: Huấn luyện mô hình dự đoán cho từng cột bị thủng lỗ
            for col in self.features_:
                if df_clean[col].isnull().sum() > 0:
                    model = LinearRegression()
                    
                    # Tách tập train chỉ chứa các hàng mà cột này không bị khuyết
                    not_null_mask = df_clean[col].notnull()
                    
                    # Dùng các đặc trưng khác (đã lấp tạm) làm X để dự đoán chính cột bị khuyết (y)
                    X_train_impute = df_temp.loc[not_null_mask].drop(columns=[col, self.target_], errors='ignore').values
                    y_train_impute = df_clean.loc[not_null_mask, col].values
                    
                    model.fit(X_train_impute, y_train_impute)
                    self.regression_models_[col] = model  # Lưu mô hình lại

        # Áp dụng điền khuyết lên tập Train trước khi tính Z-score
        df_imputed = self._impute_dataframe(df_clean)
            
        # Học mean và std cho z-score normalization
        X_train_raw = df_imputed[self.features_].values
        self.mean_ = X_train_raw.mean(axis=0)
        self.std_ = X_train_raw.std(axis=0)
        self.std_[self.std_ == 0] = 1.0  # Tránh lỗi chia cho 0
        
        return self

    def _impute_dataframe(self, df_clean):
        """Hàm phụ trợ độc quyền để điền khuyết (Dùng chung cho cả Train và Test)"""
        df_imputed = df_clean.copy()
        
        if self.imputation_method in ['mean', 'median']:
            for col, val in self.impute_values_.items():
                if col in df_imputed.columns:
                    df_imputed[col] = df_imputed[col].fillna(val)
                    
        elif self.imputation_method == 'regression':
            # Điền tạm bằng median để có ma trận đầy đủ đưa vào mô hình dự đoán
            df_temp = df_clean.copy()
            for col in self.features_:
                if col in df_temp.columns and col in self.impute_values_:
                    df_temp[col] = df_temp[col].fillna(self.impute_values_[col])
            
            # Kích hoạt các mô hình hồi quy để dự đoán giá trị khuyết
            for col, model in self.regression_models_.items():
                if col in df_clean.columns:
                    missing_mask = df_clean[col].isnull()
                    if missing_mask.sum() > 0:
                        # Dự đoán dựa trên các cột khác
                        X_missing = df_temp.loc[missing_mask].drop(columns=[col, self.target_], errors='ignore').values
                        predicted_values = model.predict(X_missing)
                        
                        # Điền giá trị mô hình đã tính đè lên ô trống
                        df_imputed.loc[missing_mask, col] = predicted_values
        
        return df_imputed

    def transform(self, df):
        df_clean = df.drop(columns=self.drop_cols_, errors='ignore')
        
        # 1. Gọi hàm phụ trợ để lấp lỗ hổng dữ liệu
        df_imputed = self._impute_dataframe(df_clean)
            
        # 2. Tách X, y
        X_raw = df_imputed[self.features_].values
        
        # 3. Chuẩn hóa z-score (dựa trên thông số từ tập Train)
        X_scaled = (X_raw - self.mean_) / self.std_
        
        # 4. Thêm cột intercept (hệ số tự do 1.0)
        X_final = np.column_stack([np.ones(len(X_scaled)), X_scaled])
        
        if self.target_ in df_clean.columns:
            y = df_clean[self.target_].values
            return X_final, y
        else:
            return X_final

    def run(self, df, seed=42):
        np.random.seed(seed)
        n = len(df)
        indices = np.arange(n)
        np.random.shuffle(indices)
        split = int(n * 0.8)
        
        train_df = df.iloc[indices[:split]].reset_index(drop=True)
        test_df = df.iloc[indices[split:]].reset_index(drop=True)
        
        self.fit(train_df)
        
        X_train, y_train = self.transform(train_df)
        X_test, y_test = self.transform(test_df)
        
        feature_names = ['intercept'] + self.features_
        return X_train, X_test, y_train, y_test, feature_names

def load_data():
    import os
    import pandas as pd
    script_dir = os.path.dirname(os.path.abspath(__file__))
    df = pd.read_csv(os.path.join(script_dir, 'data', 'day.csv'))
    
    pipeline = DataPipeline(imputation_method='regression') 
    return pipeline.run(df)

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    df = pd.read_csv(os.path.join(script_dir, 'data', 'day.csv'))
    pipeline = DataPipeline(imputation_method='regression')
    X_train, X_test, y_train, y_test, feature_names = pipeline.run(df)
    
    print("=== DATA PIPELINE TÍCH HỢP REGRESSION IMPUTATION (SUCCESS) ===")
    print(f"X_train shape: {X_train.shape}")
    print(f"X_test shape:  {X_test.shape}")