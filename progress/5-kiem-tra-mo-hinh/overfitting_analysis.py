# Bước 1.2: Overfitting Analysis cho Random Forest - Tuần 6
# Mục đích: Kiểm tra overfitting, tạo learning curves, phân tích train vs validation performance
# Tác dụng: Đảm bảo model robust, không overfit, cải thiện generalization

import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import learning_curve, validation_curve
from sklearn.metrics import accuracy_score, f1_score
import os

print("=== Bước 1.2: Overfitting Analysis cho Random Forest ===")

# 1. Load dữ liệu và optimized model từ Bước 1.1
data_path = './data/pkl/train_test_data.pkl'
model_path = './data/pkl/optimized_rf_model.pkl'

if not os.path.exists(data_path):
    raise FileNotFoundError(f"File {data_path} không tồn tại. Hãy chạy data_split.py từ tuần 5 trước.")

if not os.path.exists(model_path):
    raise FileNotFoundError(f"File {model_path} không tồn tại. Hãy chạy hyperparameter_tuning.py trước.")

# Load data
with open(data_path, 'rb') as f:
    data = pickle.load(f)

X_train = data['X_train']
X_test = data['X_test']
y_train = data['y_train']
y_test = data['y_test']

# Load optimized model
with open(model_path, 'rb') as f:
    model_data = pickle.load(f)

best_model = model_data['model']
best_params = model_data['best_params']

print(f"Dữ liệu load thành công: Train {len(X_train)}, Test {len(X_test)}")
print(f"Best model parameters: {best_params}")

# 2. Đánh giá train vs test performance gap
print("\n=== Train vs Test Performance Gap ===")

# Đánh giá trên train set
train_pred = best_model.predict(X_train)
train_accuracy = accuracy_score(y_train, train_pred)
train_f1 = f1_score(y_train, train_pred)

# Đánh giá trên test set
test_pred = best_model.predict(X_test)
test_accuracy = accuracy_score(y_test, test_pred)
test_f1 = f1_score(y_test, test_pred)

# Tính gap
accuracy_gap = train_accuracy - test_accuracy
f1_gap = train_f1 - test_f1

print(f"Train Accuracy: {train_accuracy:.4f}")
print(f"Test Accuracy: {test_accuracy:.4f}")
print(f"Accuracy Gap: {accuracy_gap:.4f}")

print(f"\nTrain F1-Score: {train_f1:.4f}")
print(f"Test F1-Score: {test_f1:.4f}")
print(f"F1 Gap: {f1_gap:.4f}")

# Đánh giá overfitting
if accuracy_gap > 0.05:
    overfitting_status = "High Overfitting"
    overfitting_recommendation = "Giảm max_depth, tăng min_samples_split/leaf"
elif accuracy_gap > 0.02:
    overfitting_status = "Moderate Overfitting"
    overfitting_recommendation = "Cân nhắc regularization nhẹ"
else:
    overfitting_status = "No Significant Overfitting"
    overfitting_recommendation = "Model ổn định, có thể sử dụng"

print(f"\nOverfitting Status: {overfitting_status}")
print(f"Recommendation: {overfitting_recommendation}")

# 3. Tạo Learning Curves
print("\n=== Tạo Learning Curves ===")

# Tạo thư mục cho charts
charts_dir = './chart/week06'
os.makedirs(charts_dir, exist_ok=True)

# Learning curve với train sizes khác nhau
train_sizes = np.linspace(0.1, 1.0, 10)

print("Đang tính learning curves... (có thể mất vài phút)")
train_sizes_abs, train_scores, val_scores = learning_curve(
    best_model, X_train, y_train, 
    train_sizes=train_sizes,
    cv=5,
    scoring='f1',
    n_jobs=-1,
    random_state=42
)

# Tính mean và std
train_mean = np.mean(train_scores, axis=1)
train_std = np.std(train_scores, axis=1)
val_mean = np.mean(val_scores, axis=1)
val_std = np.std(val_scores, axis=1)

# Vẽ learning curve
plt.figure(figsize=(12, 8))
plt.subplot(2, 2, 1)
plt.plot(train_sizes_abs, train_mean, 'o-', color='blue', label='Training Score')
plt.fill_between(train_sizes_abs, train_mean - train_std, train_mean + train_std, alpha=0.1, color='blue')
plt.plot(train_sizes_abs, val_mean, 'o-', color='red', label='Validation Score')
plt.fill_between(train_sizes_abs, val_mean - val_std, val_mean + val_std, alpha=0.1, color='red')
plt.xlabel('Training Set Size')
plt.ylabel('F1 Score')
plt.title('Learning Curve (F1 Score)')
plt.legend()
plt.grid(True, alpha=0.3)

# 4. Validation Curve cho key parameters
print("Đang tính validation curves...")

# Validation curve cho n_estimators
param_range_estimators = [50, 100, 150, 200, 250, 300]
train_scores_est, val_scores_est = validation_curve(
    RandomForestClassifier(
        max_depth=best_params.get('max_depth'),
        min_samples_split=best_params.get('min_samples_split', 2),
        min_samples_leaf=best_params.get('min_samples_leaf', 1),
        random_state=42
    ),
    X_train, y_train,
    param_name='n_estimators',
    param_range=param_range_estimators,
    cv=5, scoring='f1', n_jobs=-1
)

train_mean_est = np.mean(train_scores_est, axis=1)
val_mean_est = np.mean(val_scores_est, axis=1)

plt.subplot(2, 2, 2)
plt.plot(param_range_estimators, train_mean_est, 'o-', color='blue', label='Training Score')
plt.plot(param_range_estimators, val_mean_est, 'o-', color='red', label='Validation Score')
plt.xlabel('n_estimators')
plt.ylabel('F1 Score')
plt.title('Validation Curve (n_estimators)')
plt.legend()
plt.grid(True, alpha=0.3)

# Validation curve cho max_depth
param_range_depth = [5, 10, 15, 20, 25, None]
param_range_depth_numeric = [5, 10, 15, 20, 25, 30]  # Thay None = 30 để vẽ

train_scores_depth, val_scores_depth = validation_curve(
    RandomForestClassifier(
        n_estimators=best_params.get('n_estimators', 100),
        min_samples_split=best_params.get('min_samples_split', 2),
        min_samples_leaf=best_params.get('min_samples_leaf', 1),
        random_state=42
    ),
    X_train, y_train,
    param_name='max_depth',
    param_range=[5, 10, 15, 20, 25, None],
    cv=5, scoring='f1', n_jobs=-1
)

train_mean_depth = np.mean(train_scores_depth, axis=1)
val_mean_depth = np.mean(val_scores_depth, axis=1)

plt.subplot(2, 2, 3)
plt.plot(param_range_depth_numeric, train_mean_depth, 'o-', color='blue', label='Training Score')
plt.plot(param_range_depth_numeric, val_mean_depth, 'o-', color='red', label='Validation Score')
plt.xlabel('max_depth (None = 30)')
plt.ylabel('F1 Score')
plt.title('Validation Curve (max_depth)')
plt.legend()
plt.grid(True, alpha=0.3)

# Gap analysis plot
plt.subplot(2, 2, 4)
gap_estimators = train_mean_est - val_mean_est
gap_depth = train_mean_depth - val_mean_depth

plt.plot(param_range_estimators, gap_estimators, 'o-', color='green', label='n_estimators gap')
plt.plot(param_range_depth_numeric, gap_depth, 's-', color='orange', label='max_depth gap')
plt.xlabel('Parameter Value')
plt.ylabel('Train - Validation Gap')
plt.title('Overfitting Gap Analysis')
plt.legend()
plt.grid(True, alpha=0.3)
plt.axhline(y=0.05, color='red', linestyle='--', alpha=0.7, label='Overfitting Threshold')

plt.tight_layout()
learning_curve_path = os.path.join(charts_dir, 'overfitting_analysis.png')
plt.savefig(learning_curve_path, dpi=300, bbox_inches='tight')
plt.close()

print(f"Learning curves đã lưu vào: {learning_curve_path}")

# 5. Phân tích chi tiết overfitting
print("\n=== Phân tích Chi Tiết Overfitting ===")

# Tìm optimal point từ learning curve
optimal_size_idx = np.argmin(np.abs(train_mean - val_mean))
optimal_size = train_sizes_abs[optimal_size_idx]
optimal_train_score = train_mean[optimal_size_idx]
optimal_val_score = val_mean[optimal_size_idx]

print(f"Optimal training size: {optimal_size} samples")
print(f"At optimal size - Train F1: {optimal_train_score:.4f}, Val F1: {optimal_val_score:.4f}")

# Recommendations based on analysis
recommendations = []

if accuracy_gap > 0.05:
    recommendations.append("Giảm max_depth từ hiện tại xuống")
    recommendations.append("Tăng min_samples_split lên 10-20")
    recommendations.append("Tăng min_samples_leaf lên 4-8")
    
if f1_gap > 0.05:
    recommendations.append("Cân nhắc giảm n_estimators nếu gap lớn")
    recommendations.append("Thêm regularization (max_features='sqrt')")

if len(recommendations) == 0:
    recommendations.append("Model hiện tại ổn định, không cần điều chỉnh")

# 6. Lưu kết quả phân tích
results_path = './progress/week06/overfitting_analysis.txt'
with open(results_path, 'w', encoding='utf-8') as f:
    f.write("=== Overfitting Analysis - Tuần 6 ===\n\n")
    
    f.write("=== Train vs Test Performance ===\n")
    f.write(f"Train Accuracy: {train_accuracy:.4f}\n")
    f.write(f"Test Accuracy: {test_accuracy:.4f}\n")
    f.write(f"Accuracy Gap: {accuracy_gap:.4f}\n\n")
    
    f.write(f"Train F1-Score: {train_f1:.4f}\n")
    f.write(f"Test F1-Score: {test_f1:.4f}\n")
    f.write(f"F1 Gap: {f1_gap:.4f}\n\n")
    
    f.write(f"Overfitting Status: {overfitting_status}\n")
    f.write(f"Primary Recommendation: {overfitting_recommendation}\n\n")
    
    f.write("=== Learning Curve Analysis ===\n")
    f.write(f"Optimal training size: {optimal_size} samples\n")
    f.write(f"At optimal size - Train F1: {optimal_train_score:.4f}\n")
    f.write(f"At optimal size - Val F1: {optimal_val_score:.4f}\n\n")
    
    f.write("=== Detailed Recommendations ===\n")
    for i, rec in enumerate(recommendations, 1):
        f.write(f"{i}. {rec}\n")
    
    f.write(f"\n=== Files Generated ===\n")
    f.write(f"Learning curves chart: {learning_curve_path}\n")
    f.write(f"Analysis report: {results_path}\n")

print(f"Kết quả phân tích đã lưu vào: {results_path}")

# 7. Tạo improved model nếu cần (dựa trên recommendations)
if accuracy_gap > 0.05 or f1_gap > 0.05:
    print(f"\n=== Tạo Improved Model (Giảm Overfitting) ===")
    
    # Điều chỉnh parameters để giảm overfitting
    improved_params = best_params.copy()
    
    if improved_params.get('max_depth') is None or improved_params.get('max_depth', 20) > 15:
        improved_params['max_depth'] = 15
        
    if improved_params.get('min_samples_split', 2) < 5:
        improved_params['min_samples_split'] = 5
        
    if improved_params.get('min_samples_leaf', 1) < 2:
        improved_params['min_samples_leaf'] = 2
    
    print(f"Improved parameters: {improved_params}")
    
    # Train improved model
    improved_model = RandomForestClassifier(**improved_params, random_state=42)
    improved_model.fit(X_train, y_train)
    
    # Đánh giá improved model
    improved_train_pred = improved_model.predict(X_train)
    improved_test_pred = improved_model.predict(X_test)
    
    improved_train_f1 = f1_score(y_train, improved_train_pred)
    improved_test_f1 = f1_score(y_test, improved_test_pred)
    improved_gap = improved_train_f1 - improved_test_f1
    
    print(f"Improved Train F1: {improved_train_f1:.4f}")
    print(f"Improved Test F1: {improved_test_f1:.4f}")
    print(f"Improved Gap: {improved_gap:.4f}")
    
    if improved_gap < f1_gap:
        print("✅ Improved model giảm overfitting thành công!")
        
        # Lưu improved model
        improved_model_path = './data/pkl/improved_rf_model.pkl'
        improved_data = {
            'model': improved_model,
            'params': improved_params,
            'train_f1': improved_train_f1,
            'test_f1': improved_test_f1,
            'gap': improved_gap,
            'feature_names': data['feature_names']
        }
        
        with open(improved_model_path, 'wb') as f:
            pickle.dump(improved_data, f)
        
        print(f"Improved model đã lưu vào: {improved_model_path}")
    else:
        print("❌ Improved model không cải thiện overfitting")

# 8. Tóm tắt kết quả
print(f"\n=== TÓM TẮT OVERFITTING ANALYSIS ===")
print(f"✅ Train-Test Accuracy Gap: {accuracy_gap:.4f}")
print(f"✅ Train-Test F1 Gap: {f1_gap:.4f}")
print(f"✅ Overfitting Status: {overfitting_status}")
print(f"✅ Learning curves đã tạo")
print(f"✅ Validation curves đã phân tích")
print(f"✅ Recommendations đã đưa ra")

print("\nBước 1.2 hoàn thành! Model đã được phân tích overfitting chi tiết.")
print("Sẵn sàng cho Bước 2: Feature Importance Analysis.")