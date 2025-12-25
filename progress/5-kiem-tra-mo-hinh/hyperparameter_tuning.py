# Bước 1.1: Hyperparameter Tuning cho Random Forest - Tuần 6
# Mục đích: Tối ưu hóa Random Forest model để cải thiện hiệu suất và tránh overfitting
# Tác dụng: Tìm best parameters, so sánh với model tuần 5, tạo optimized model

import pandas as pd
import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
import time
import os

print("=== Bước 1.1: Hyperparameter Tuning cho Random Forest ===")

# 1. Load dữ liệu đã xử lý từ tuần 5
data_path = './data/pkl/train_test_data.pkl'
if not os.path.exists(data_path):
    raise FileNotFoundError(f"File {data_path} không tồn tại. Hãy chạy data_split.py từ tuần 5 trước.")

with open(data_path, 'rb') as f:
    data = pickle.load(f)

X_train = data['X_train']
X_test = data['X_test']
y_train = data['y_train']
y_test = data['y_test']
feature_names = data['feature_names']

print(f"Dữ liệu load thành công: Train {len(X_train)}, Test {len(X_test)}")
print(f"Features: {len(feature_names)} cột")

# 2. Load model Random Forest từ tuần 5 để so sánh
baseline_model_path = './data/pkl/random_forest_model.pkl'
if os.path.exists(baseline_model_path):
    with open(baseline_model_path, 'rb') as f:
        baseline_model = pickle.load(f)
    
    # Đánh giá baseline model
    baseline_pred = baseline_model.predict(X_test)
    baseline_accuracy = accuracy_score(y_test, baseline_pred)
    baseline_f1 = f1_score(y_test, baseline_pred)
    
    print(f"\n=== Baseline Random Forest (Tuần 5) ===")
    print(f"Accuracy: {baseline_accuracy:.4f}")
    print(f"F1-Score: {baseline_f1:.4f}")
else:
    print("Không tìm thấy baseline model từ tuần 5. Sẽ tạo mới.")
    baseline_accuracy = 0
    baseline_f1 = 0

# 3. Định nghĩa parameter grid cho GridSearchCV
print("\n=== Thiết lập Parameter Grid ===")
param_grid = {
    'n_estimators': [50, 100, 200, 300],
    'max_depth': [5, 10, 15, 20, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 'log2', None]
}

print(f"Parameter combinations: {np.prod([len(v) for v in param_grid.values()])}")
print("Sử dụng RandomizedSearchCV để tiết kiệm thời gian...")

# 4. Hyperparameter tuning với RandomizedSearchCV (nhanh hơn GridSearchCV)
rf = RandomForestClassifier(random_state=42)

print("\n=== Bắt đầu Hyperparameter Tuning ===")
start_time = time.time()

# Sử dụng RandomizedSearchCV với 50 iterations
random_search = RandomizedSearchCV(
    estimator=rf,
    param_distributions=param_grid,
    n_iter=50,  # Thử 50 combinations ngẫu nhiên
    cv=5,  # 5-fold cross-validation
    scoring='f1',  # Tối ưu F1-score
    n_jobs=-1,  # Sử dụng tất cả CPU cores
    random_state=42,
    verbose=1
)

random_search.fit(X_train, y_train)
tuning_time = time.time() - start_time

print(f"\nHyperparameter tuning hoàn thành trong {tuning_time:.2f} giây")

# 5. Lấy best model và parameters
best_model = random_search.best_estimator_
best_params = random_search.best_params_
best_cv_score = random_search.best_score_

print(f"\n=== Best Parameters ===")
for param, value in best_params.items():
    print(f"{param}: {value}")

print(f"\nBest CV F1-Score: {best_cv_score:.4f}")

# 6. Đánh giá best model trên test set
print("\n=== Đánh giá Best Model trên Test Set ===")
best_pred = best_model.predict(X_test)

# Tính các metrics
tuned_accuracy = accuracy_score(y_test, best_pred)
tuned_precision = precision_score(y_test, best_pred)
tuned_recall = recall_score(y_test, best_pred)
tuned_f1 = f1_score(y_test, best_pred)
tuned_conf_matrix = confusion_matrix(y_test, best_pred)

print(f"Accuracy: {tuned_accuracy:.4f}")
print(f"Precision: {tuned_precision:.4f}")
print(f"Recall: {tuned_recall:.4f}")
print(f"F1-Score: {tuned_f1:.4f}")
print("\nConfusion Matrix:")
print(tuned_conf_matrix)

# 7. So sánh với baseline model
print(f"\n=== So sánh với Baseline Model ===")
if baseline_accuracy > 0:
    accuracy_improvement = tuned_accuracy - baseline_accuracy
    f1_improvement = tuned_f1 - baseline_f1
    
    print(f"Baseline Accuracy: {baseline_accuracy:.4f}")
    print(f"Tuned Accuracy: {tuned_accuracy:.4f}")
    print(f"Accuracy Improvement: {accuracy_improvement:.4f} ({accuracy_improvement/baseline_accuracy*100:+.2f}%)")
    
    print(f"\nBaseline F1-Score: {baseline_f1:.4f}")
    print(f"Tuned F1-Score: {tuned_f1:.4f}")
    print(f"F1 Improvement: {f1_improvement:.4f} ({f1_improvement/baseline_f1*100:+.2f}%)")
    
    if tuned_f1 > baseline_f1:
        print("✅ Hyperparameter tuning cải thiện model!")
    elif tuned_f1 == baseline_f1:
        print("➖ Hyperparameter tuning không cải thiện (bằng baseline)")
    else:
        print("❌ Hyperparameter tuning làm giảm hiệu suất")

# 8. Lưu kết quả vào file txt
results_dir = './progress/week06'
os.makedirs(results_dir, exist_ok=True)

results_path = os.path.join(results_dir, 'hyperparameter_tuning_results.txt')
with open(results_path, 'w', encoding='utf-8') as f:
    f.write("=== Kết quả Hyperparameter Tuning - Tuần 6 ===\n\n")
    
    f.write("=== Best Parameters ===\n")
    for param, value in best_params.items():
        f.write(f"{param}: {value}\n")
    
    f.write(f"\nBest CV F1-Score: {best_cv_score:.4f}\n")
    f.write(f"Tuning Time: {tuning_time:.2f} seconds\n")
    
    f.write(f"\n=== Test Set Performance ===\n")
    f.write(f"Accuracy: {tuned_accuracy:.4f}\n")
    f.write(f"Precision: {tuned_precision:.4f}\n")
    f.write(f"Recall: {tuned_recall:.4f}\n")
    f.write(f"F1-Score: {tuned_f1:.4f}\n")
    
    f.write(f"\nConfusion Matrix:\n")
    f.write(str(tuned_conf_matrix))
    
    f.write(f"\n\nClassification Report:\n")
    f.write(classification_report(y_test, best_pred))
    
    if baseline_accuracy > 0:
        f.write(f"\n=== So sánh với Baseline ===\n")
        f.write(f"Baseline Accuracy: {baseline_accuracy:.4f}\n")
        f.write(f"Tuned Accuracy: {tuned_accuracy:.4f}\n")
        f.write(f"Improvement: {accuracy_improvement:.4f} ({accuracy_improvement/baseline_accuracy*100:+.2f}%)\n")
        
        f.write(f"\nBaseline F1-Score: {baseline_f1:.4f}\n")
        f.write(f"Tuned F1-Score: {tuned_f1:.4f}\n")
        f.write(f"F1 Improvement: {f1_improvement:.4f} ({f1_improvement/baseline_f1*100:+.2f}%)\n")

print(f"\nKết quả đã lưu vào: {results_path}")

# 9. Lưu optimized model
optimized_model_path = './data/pkl/optimized_rf_model.pkl'
os.makedirs(os.path.dirname(optimized_model_path), exist_ok=True)

# Lưu model và metadata
optimized_data = {
    'model': best_model,
    'best_params': best_params,
    'best_cv_score': best_cv_score,
    'test_accuracy': tuned_accuracy,
    'test_f1': tuned_f1,
    'feature_names': feature_names,
    'tuning_time': tuning_time
}

with open(optimized_model_path, 'wb') as f:
    pickle.dump(optimized_data, f)

print(f"Optimized model đã lưu vào: {optimized_model_path}")

# 10. Tóm tắt kết quả
print(f"\n=== TÓM TẮT HYPERPARAMETER TUNING ===")
print(f"✅ Đã thử {random_search.n_iter} parameter combinations")
print(f"✅ Best CV F1-Score: {best_cv_score:.4f}")
print(f"✅ Test F1-Score: {tuned_f1:.4f}")
if baseline_f1 > 0:
    improvement_status = "Cải thiện" if tuned_f1 > baseline_f1 else "Không cải thiện" if tuned_f1 == baseline_f1 else "Giảm"
    print(f"✅ So với baseline: {improvement_status}")
print(f"✅ Thời gian tuning: {tuning_time:.2f}s")
print(f"✅ Model và kết quả đã lưu")

print("\nBước 1.1 hoàn thành! Sẵn sàng cho Bước 1.2 (Overfitting Analysis).")