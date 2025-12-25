# Phần 4: Áp dụng Cross-Validation (5-Fold) - Tuần 5
# Mục đích: Đánh giá ổn định của Logistic Regression và Random Forest
# Tác dụng: Tính confidence interval, phát hiện overfitting

import pandas as pd
import pickle
import numpy as np
from sklearn.model_selection import cross_val_score, cross_validate
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import make_scorer, f1_score
import os

# 1. Load dữ liệu đã xử lý từ phần 1
data_path = './data/pkl/train_test_data.pkl'
if not os.path.exists(data_path):
    raise FileNotFoundError(f"File {data_path} không tồn tại. Hãy chạy phần 1 trước.")

with open(data_path, 'rb') as f:
    data = pickle.load(f)

X = pd.concat([data['X_train'], data['X_test']], ignore_index=True)
y = pd.concat([data['y_train'], data['y_test']], ignore_index=True)

print(f"Dữ liệu CV: {len(X)} mẫu, {len(X.columns)} features.")

# 2. Load models
logistic_path = './data/pkl/logistic_model.pkl'
rf_path = './data/pkl/random_forest_model.pkl'

if not os.path.exists(logistic_path):
    print("Logistic model không tồn tại, train lại...")
    logistic_model = LogisticRegression(random_state=42, max_iter=1000)
    logistic_model.fit(data['X_train'], data['y_train'])
else:
    with open(logistic_path, 'rb') as f:
        logistic_model = pickle.load(f)

if not os.path.exists(rf_path):
    print("Random Forest model không tồn tại, train lại...")
    rf_model = RandomForestClassifier(random_state=42)
    rf_model.fit(data['X_train'], data['y_train'])
else:
    with open(rf_path, 'rb') as f:
        rf_model = pickle.load(f)

models = {'Logistic Regression': logistic_model, 'Random Forest': rf_model}

# 3. Chạy 5-Fold CV cho Accuracy và F1-Score
cv_results = {}
for name, model in models.items():
    print(f"\nĐánh giá {name} với 5-Fold CV...")
    # Accuracy
    acc_scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
    acc_mean = acc_scores.mean()
    acc_std = acc_scores.std()
    acc_ci = 1.96 * acc_std  # 95% CI

    # F1-Score
    f1_scores = cross_val_score(model, X, y, cv=5, scoring='f1')
    f1_mean = f1_scores.mean()
    f1_std = f1_scores.std()
    f1_ci = 1.96 * f1_std

    # Phát hiện overfitting: So sánh train vs validation score
    cv_detail = cross_validate(model, X, y, cv=5, scoring=['accuracy', 'f1'], return_train_score=True)
    train_acc_mean = cv_detail['train_accuracy'].mean()
    val_acc_mean = cv_detail['test_accuracy'].mean()
    overfitting_acc = train_acc_mean - val_acc_mean

    train_f1_mean = cv_detail['train_f1'].mean()
    val_f1_mean = cv_detail['test_f1'].mean()
    overfitting_f1 = train_f1_mean - val_f1_mean

    cv_results[name] = {
        'Accuracy': {'mean': acc_mean, 'std': acc_std, 'ci': acc_ci, 'scores': acc_scores},
        'F1-Score': {'mean': f1_mean, 'std': f1_std, 'ci': f1_ci, 'scores': f1_scores},
        'Overfitting': {'acc_gap': overfitting_acc, 'f1_gap': overfitting_f1}
    }

    print(f"  Accuracy: {acc_mean:.4f} ± {acc_ci:.4f} (std: {acc_std:.4f})")
    print(f"  F1-Score: {f1_mean:.4f} ± {f1_ci:.4f} (std: {f1_std:.4f})")
    print(f"  Overfitting gap (Train - Val): Acc {overfitting_acc:.4f}, F1 {overfitting_f1:.4f}")
    if overfitting_acc > 0.1 or overfitting_f1 > 0.1:
        print("Dấu hiệu overfitting!")
    else:
        print("Không có overfitting rõ rệt.")

# 4. So sánh hai mô hình
print("\n=== So sánh CV Results ===")
for metric in ['Accuracy', 'F1-Score']:
    log_mean = cv_results['Logistic Regression'][metric]['mean']
    rf_mean = cv_results['Random Forest'][metric]['mean']
    diff = rf_mean - log_mean
    print(f"{metric}: Logistic {log_mean:.4f}, Random Forest {rf_mean:.4f} (RF tốt hơn: {diff:.4f})")

# 5. Lưu kết quả vào file .txt trong week05/
results_path = './progress/week05/CV-5Fold/cv_results.txt'
os.makedirs(os.path.dirname(results_path), exist_ok=True)
with open(results_path, 'w') as f:
    f.write("=== 5-Fold Cross-Validation Results ===\n")
    for name, res in cv_results.items():
        f.write(f"\n{name}:\n")
        for metric, vals in res.items():
            if metric == 'Overfitting':
                f.write(f"  {metric}: Acc gap {vals['acc_gap']:.4f}, F1 gap {vals['f1_gap']:.4f}\n")
            else:
                f.write(f"  {metric}: {vals['mean']:.4f} ± {vals['ci']:.4f} (std: {vals['std']:.4f})\n")
                f.write(f"    Scores: {vals['scores']}\n")
    f.write("\n=== So sánh ===\n")
    for metric in ['Accuracy', 'F1-Score']:
        log_mean = cv_results['Logistic Regression'][metric]['mean']
        rf_mean = cv_results['Random Forest'][metric]['mean']
        diff = rf_mean - log_mean
        f.write(f"{metric}: Logistic {log_mean:.4f}, RF {rf_mean:.4f} (RF tốt hơn: {diff:.4f})\n")

print(f"Kết quả CV đã lưu vào: {results_path}")
print("Phần 4 hoàn thành! Sẵn sàng cho phần 5 (Feature Importance).")