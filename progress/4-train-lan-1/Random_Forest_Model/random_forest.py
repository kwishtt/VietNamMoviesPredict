# Phần 3: Xây dựng mô hình chính (Random Forest) - Tuần 5
# Mục đích: Train Random Forest, đánh giá và so sánh với Logistic Regression
# Tác dụng: Mô hình chính, giảm overfitting, phân tích feature importance

import pandas as pd
import pickle
from sklearn.ensemble import RandomForestClassifier
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

# 2. Triển khai Random Forest với tham số mặc định
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)
print("Random Forest đã train xong.")

# 3. Dự đoán trên test set
y_pred = model.predict(X_test)

# 4. Tính các metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

print("\n=== Kết quả Random Forest ===")
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-Score: {f1:.4f}")
print("\nConfusion Matrix:")
print(conf_matrix)
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# 5. So sánh với Logistic Regression (nếu có kết quả)
logistic_results_path = './progress/week05/Logistic_Regression_Model/logistic_results.txt'
if os.path.exists(logistic_results_path):
    with open(logistic_results_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            if 'Accuracy:' in line:
                logistic_acc = float(line.split(':')[1].strip())
            elif 'F1-Score:' in line:
                logistic_f1 = float(line.split(':')[1].strip())
    print("\n=== So sánh với Logistic Regression ===")
    print(f"Logistic Accuracy: {logistic_acc:.4f}, Random Forest: {accuracy:.4f} (Cải thiện: {accuracy - logistic_acc:.4f})")
    print(f"Logistic F1-Score: {logistic_f1:.4f}, Random Forest: {f1:.4f} (Cải thiện: {f1 - logistic_f1:.4f})")
else:
    print("Không tìm thấy kết quả Logistic Regression để so sánh.")

# 6. Lưu kết quả vào file .txt trong week05/
results_path = './progress/week05/Random_Forest_Model/random_forest_results.txt'
os.makedirs(os.path.dirname(results_path), exist_ok=True)
with open(results_path, 'w') as f:
    f.write("=== Kết quả Random Forest ===\n")
    f.write(f"Accuracy: {accuracy:.4f}\n")
    f.write(f"Precision: {precision:.4f}\n")
    f.write(f"Recall: {recall:.4f}\n")
    f.write(f"F1-Score: {f1:.4f}\n")
    f.write("\nConfusion Matrix:\n")
    f.write(str(conf_matrix))
    f.write("\n\nClassification Report:\n")
    f.write(classification_report(y_test, y_pred))
    if os.path.exists(logistic_results_path):
        f.write("\n=== So sánh với Logistic Regression ===\n")
        f.write(f"Logistic Accuracy: {logistic_acc:.4f}, Random Forest: {accuracy:.4f}\n")
        f.write(f"Logistic F1-Score: {logistic_f1:.4f}, Random Forest: {f1:.4f}\n")

print(f"Kết quả đã lưu vào: {results_path}")

# 7. Lưu model vào data/pkl/random_forest_model.pkl
model_path = './data/pkl/random_forest_model.pkl'
os.makedirs(os.path.dirname(model_path), exist_ok=True)
with open(model_path, 'wb') as f:
    pickle.dump(model, f)

print(f"Model đã lưu vào: {model_path}")
print("Phần 3 hoàn thành! Sẵn sàng cho phần 4 (Cross-Validation).")