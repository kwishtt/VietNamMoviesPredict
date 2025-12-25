# Phần 1: Chuẩn bị môi trường và dữ liệu cho Modeling (Tuần 5)
# Mục đích: Tải clean_movies_features.csv, chia train/test, chuẩn hóa, xử lý mất cân bằng
# Tác dụng: Chuẩn bị dữ liệu sẵn sàng cho Logistic Regression và Random Forest

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from imblearn.over_sampling import SMOTE
import os

# 1. Tải dữ liệu từ clean_movies_features.csv
data_path = './data/clean_movies_features.csv'
df = pd.read_csv(data_path)
print(f"Dữ liệu tải thành công: {len(df)} hàng, {len(df.columns)} cột")

# 2. Xác định features và target
# Target là 'success' (0: thất bại, 1: thành công)
target_col = 'success'
if target_col not in df.columns:
    raise ValueError(f"Cột target '{target_col}' không có trong dữ liệu.")

# Features: tất cả cột số trừ target và ID nếu có
exclude_cols = [target_col, 'Id', 'Title']  # Loại bỏ cột không cần thiết
features = [col for col in df.columns if col not in exclude_cols and df[col].dtype != 'object']
X = df[features]
y = df[target_col]

print(f"Features: {len(features)} cột")
print(f"Target distribution: {y.value_counts().to_dict()}")

# 3. Chia train/test với stratified sampling (80/20)
test_size = 0.2
random_state = 42
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=test_size, stratify=y, random_state=random_state
)

print(f"Train set: {len(X_train)} mẫu")
print(f"Test set: {len(X_test)} mẫu")
print(f"Tỷ lệ lớp trong train: {y_train.value_counts(normalize=True).to_dict()}")
print(f"Tỷ lệ lớp trong test: {y_test.value_counts(normalize=True).to_dict()}")

# 4. Áp dụng Min-Max Scaler cho features số
# Chỉ scale cột số, bỏ qua categorical nếu có
numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
scaler = MinMaxScaler()
X_train_scaled = X_train.copy()
X_test_scaled = X_test.copy()
X_train_scaled[numeric_cols] = scaler.fit_transform(X_train[numeric_cols])
X_test_scaled[numeric_cols] = scaler.transform(X_test[numeric_cols])

print(f"Đã scale {len(numeric_cols)} cột số: {numeric_cols}")

# Xử lý missing values bằng fillna(0) cho Logistic Regression
X_train_scaled = X_train_scaled.fillna(0)
X_test_scaled = X_test_scaled.fillna(0)
print("Đã xử lý missing values bằng fillna(0).")

# 5. Xử lý mất cân bằng lớp bằng SMOTE (nếu cần)
# Kiểm tra tỷ lệ lớp
class_counts = y_train.value_counts()
min_class = class_counts.min()
max_class = class_counts.max()
ratio = min_class / max_class

if ratio < 0.5:  # Nếu mất cân bằng nghiêm trọng (dưới 50%)
    print("Dữ liệu mất cân bằng, áp dụng SMOTE...")
    smote = SMOTE(random_state=random_state)
    X_train_scaled, y_train = smote.fit_resample(X_train_scaled, y_train)
    print(f"Sau SMOTE: {y_train.value_counts().to_dict()}")
else:
    print("Dữ liệu cân bằng, không cần SMOTE.")

# 6. Lưu dữ liệu đã xử lý (tùy chọn, để dùng cho các phần sau)
# Có thể lưu ra file CSV nếu cần
# X_train_scaled.to_csv('X_train_scaled.csv', index=False)
# y_train.to_csv('y_train.csv', index=False)
# X_test_scaled.to_csv('X_test_scaled.csv', index=False)
# y_test.to_csv('y_test.csv', index=False)

print("Chuẩn bị dữ liệu hoàn thành! Sẵn sàng cho modeling.")
# Lưu bằng pkl để tiện sử dụng, chỉ cần load lại chứ không cần chia lại tỉ lệ train/test
# 7. Lưu dữ liệu đã xử lý để dùng cho các bước sau
import pickle
output_path = './data/pkl/train_test_data.pkl'
data_to_save = {
    'X_train': X_train_scaled,
    'X_test': X_test_scaled,
    'y_train': y_train,
    'y_test': y_test,
    'scaler': scaler,  # Để scale dữ liệu mới nếu cần
    'feature_names': features  # Danh sách tên features
}
with open(output_path, 'wb') as f:
    pickle.dump(data_to_save, f)
print(f"Dữ liệu đã lưu tại: {output_path}")