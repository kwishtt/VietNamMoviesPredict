# Phần 2: Xây dựng mô hình cơ sở (Logistic Regression) - Tuần 5
# Mục đích: Train Logistic Regression làm baseline, đánh giá trên test
# Tác dụng: Cung cấp kết quả ban đầu để so sánh với Random Forest

import pandas as pd
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
import os

# 1. Load dữ liệu đã xử lý từ phần 1
data_path = './data/pkl/train_test_data.pkl'
if not os.path.exists(data_path):
    raise FileNotFoundError(f"File {data_path} không tồn tại. Hãy chạy phần 1 trước.")

with open(data_path, 'rb') as f:
    data = pickle.load(f)

X_train = data['X_train']
X_test = data['X_test']
y_train = data['y_train']
y_test = data['y_test']

print("Dữ liệu load thành công từ pickle.")

# 2. Triển khai Logistic Regression với tham số mặc định + max_iter=1000
model = LogisticRegression(random_state=42, max_iter=1000)
model.fit(X_train, y_train)
print("Logistic Regression đã train xong.")

# 3. Dự đoán trên test set
y_pred = model.predict(X_test)

# 4. Tính các metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

print("\n=== Kết quả Logistic Regression (Baseline) ===")
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-Score: {f1:.4f}")
print("\nConfusion Matrix:")
print(conf_matrix)
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# 5. Lưu kết quả vào file .txt trong week05/
results_path = './progress/week05/Logistic_Regression_Model/logistic_results.txt'
os.makedirs(os.path.dirname(results_path), exist_ok=True)
with open(results_path, 'w') as f:
    f.write("=== Kết quả Logistic Regression (Baseline) ===\n")
    f.write(f"Accuracy: {accuracy:.4f}\n")
    f.write(f"Precision: {precision:.4f}\n")
    f.write(f"Recall: {recall:.4f}\n")
    f.write(f"F1-Score: {f1:.4f}\n")
    f.write("\nConfusion Matrix:\n")
    f.write(str(conf_matrix))
    f.write("\n\nClassification Report:\n")
    f.write(classification_report(y_test, y_pred))

print(f"Kết quả đã lưu vào: {results_path}")

# 6. Lưu model vào data/pkl/logistic_model.pkl
model_path = './data/pkl/logistic_model.pkl'
os.makedirs(os.path.dirname(model_path), exist_ok=True)
with open(model_path, 'wb') as f:
    pickle.dump(model, f)

print(f"Model đã lưu vào: {model_path}")
print("Phần 2 hoàn thành! Sẵn sàng cho phần 3 (Random Forest).")