"""
Helper load data cho Part 2

File này cung cấp hàm load_data() duy nhất.
- Ban đầu: tự động load data tạm từ data/temp/
- Khi Người 1 xong DataPipeline: chỉ cần sửa file NÀY, 
  Người 2, 3, 4 KHÔNG cần sửa gì trong code của họ.

Cách dùng (Người 2, 3, 4 copy dòng này):
    from load_data import load_data
    X_train, X_test, y_train, y_test, feature_names = load_data()
"""

import numpy as np
import pandas as pd
import os
from prepare_dataset import prepare_dataset

# =================================================================
#   ĐỔI use_pipeline = True KHI NGƯỜI 1 XONG DataPipeline
# =================================================================
use_pipeline = True
imputation_method = 'regression'  # 'mean', 'median', 'regression'
using_prepared_dataset = True  # Nếu đã chạy prepare_dataset.py để tạo data mới có missing values
# =================================================================


def load_data():
    """
    Trả về X_train, X_test, y_train, y_test, feature_names
    
    - use_pipeline = False → load data tạm (median + z-score)
    - use_pipeline = True  → dùng DataPipeline chính thức
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    if use_pipeline:
        # ===== NGƯỜI 1: Điền code DataPipeline vào đây =====
        from data_pipeline import DataPipeline
        pipeline = DataPipeline(imputation_method=imputation_method)
        df = pd.read_csv(os.path.join(script_dir, 'data', 'day.csv')) if using_prepared_dataset else prepare_dataset()  # Nếu đã chạy prepare_dataset.py để tạo data mới có missing values
        X_train, X_test, y_train, y_test, feature_names = pipeline.run(df)
        return X_train, X_test, y_train, y_test, feature_names
    else:
        # Load data tạm
        temp_dir = os.path.join(script_dir, 'data', 'temp')
        X_train = np.load(os.path.join(temp_dir, 'X_train.npy'))
        X_test = np.load(os.path.join(temp_dir, 'X_test.npy'))
        y_train = np.load(os.path.join(temp_dir, 'y_train.npy'))
        y_test = np.load(os.path.join(temp_dir, 'y_test.npy'))
        
        with open(os.path.join(temp_dir, 'feature_names.txt')) as f:
            feature_names = f.read().strip().split('\n')
        
        return X_train, X_test, y_train, y_test, feature_names


if __name__ == "__main__":
    X_train, X_test, y_train, y_test, names = load_data()
    print(f"X_train: {X_train.shape}")
    print(f"X_test:  {X_test.shape}")
    print(f"y_train: {y_train.shape}")
    print(f"y_test:  {y_test.shape}")
    print(f"Features: {names}")
    print(f"\nDang dung: {'DataPipeline' if use_pipeline else 'Data tam'}")