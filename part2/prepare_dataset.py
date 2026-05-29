"""
Chuẩn bị Dataset Bike Sharing cho Part 2

Bộ dữ liệu Bike Sharing gốc từ UCI không có missing values.
Tuy nhiên, trong thực tế, dữ liệu thời tiết thường bị thiếu do:
  - Cảm biến gió (anemometer) bị hỏng hoặc bảo trì     → windspeed bị NaN
  - Cảm biến độ ẩm (hygrometer) gặp sự cố              → hum bị NaN
  - Trạm thời tiết không gửi báo cáo                     → weathersit bị NaN
  - Nhiệt kế cảm nhận (apparent temp) lỗi đo            → atemp bị NaN

Script này mô phỏng các tình huống thiếu dữ liệu thực tế (MCAR - Missing
Completely At Random) để đáp ứng yêu cầu đề bài (≥5% missing values).

Seed cố định: 42 → Kết quả hoàn toàn reproducible.

Cách chạy:
    python part2/prepare_dataset.py
"""

import sys
# sys.stdout.reconfigure(encoding='utf-8')
import pandas as pd
import numpy as np
import os
import shutil


def prepare_dataset(seed=42):
    """
    Đọc dataset gốc, tạo missing values, lưu ra file mới.

    Tỉ lệ missing:
        - windspeed  : ~8%  (≈58 dòng)  — lỗi cảm biến gió
        - hum        : ~7%  (≈51 dòng)  — lỗi cảm biến độ ẩm
        - weathersit : ~5%  (≈36 dòng)  — trạm thời tiết không báo cáo
        - atemp      : ~5%  (≈36 dòng)  — lỗi nhiệt kế cảm nhận
    Tổng cộng: ~181 ô bị thiếu trên 731 × 16 = 11,696 ô (~1.5% tổng thể)
    """
    np.random.seed(seed)

    # Xác định đường dẫn
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(script_dir)
    data_dir = os.path.join(script_dir, 'data')
    os.makedirs(data_dir, exist_ok=True)

    input_path = os.path.join(project_dir, 'day.csv')

    # Đọc dữ liệu gốc
    df = pd.read_csv(input_path)
    n = len(df)

    print("=" * 60)
    print("  CHUẨN BỊ DATASET BIKE SHARING CHO PART 2")
    print("=" * 60)
    print(f"  Dữ liệu gốc: {input_path}")
    print(f"  Số dòng: {n}, Số cột: {len(df.columns)}")
    print(f"  Missing gốc: {df.isnull().sum().sum()} (0%)")
    print()

    # Lưu bản gốc (backup)
    original_path = os.path.join(data_dir, 'day_original.csv')
    shutil.copy2(input_path, original_path)
    print(f"  ✓ Đã sao lưu bản gốc → data/day_original.csv")

    # --- Tạo Missing Values ---
    # Mô phỏng lỗi cảm biến thời tiết thực tế (MCAR)
    missing_config = {
        'windspeed':  0.08,   # 8% — cảm biến gió hỏng
        'hum':        0.07,   # 7% — cảm biến độ ẩm lỗi
        'weathersit': 0.05,   # 5% — trạm thời tiết không báo cáo
        'atemp':      0.05,   # 5% — nhiệt kế cảm nhận lỗi
    }

    print()
    print("  Tạo missing values (mô phỏng lỗi cảm biến):")
    print("  " + "-" * 50)

    total_missing = 0
    for col, rate in missing_config.items():
        n_missing = int(n * rate)
        indices = np.random.choice(n, size=n_missing, replace=False)
        df.loc[indices, col] = np.nan
        total_missing += n_missing
        pct = rate * 100
        print(f"    {col:12s}: {n_missing:3d}/{n} dòng bị thiếu ({pct:.0f}%)")

    total_cells = n * len(df.columns)
    print("  " + "-" * 50)
    pct_total = total_missing / total_cells * 100
    print(f"    Tổng ô bị thiếu: {total_missing}/{total_cells} ({pct_total:.1f}%)")

    # Lưu dataset đã có missing values
    output_path = os.path.join(data_dir, 'day.csv')
    df.to_csv(output_path, index=False)
    print()
    print(f"  ✓ Đã lưu dataset (có missing) → data/day.csv")

    # Kiểm tra lại
    print()
    print("  Kiểm tra missing values sau xử lý:")
    print("  " + "-" * 50)
    for col in df.columns:
        n_miss = df[col].isnull().sum()
        if n_miss > 0:
            pct = n_miss / n * 100
            print(f"    {col:12s}: {n_miss:3d} missing ({pct:.1f}%)")

    print()
    print("  ✓ Dataset sẵn sàng cho Part 2!")
    print("=" * 60)
    return df


if __name__ == "__main__":
    prepare_dataset()
