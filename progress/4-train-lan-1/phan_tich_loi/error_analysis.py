import pickle
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, f1_score

# Load data and models
with open('./data/pkl/train_test_data.pkl', 'rb') as f:
    data = pickle.load(f)

X_train = data['X_train']
X_test = data['X_test']
y_train = data['y_train']
y_test = data['y_test']
feature_names = data['feature_names']

with open('./data/pkl/random_forest_model.pkl', 'rb') as f:
    rf_model = pickle.load(f)

with open('./data/pkl/logistic_model.pkl', 'rb') as f:
    log_model = pickle.load(f)

# Predictions
y_pred_log = log_model.predict(X_test)
y_pred_rf = rf_model.predict(X_test)

print("=== Error Analysis ===")

# Confusion Matrix for Logistic Regression
cm_log = confusion_matrix(y_test, y_pred_log)
print("\nLogistic Regression Confusion Matrix:")
print(cm_log)

# Confusion Matrix for Random Forest
cm_rf = confusion_matrix(y_test, y_pred_rf)
print("\nRandom Forest Confusion Matrix:")
print(cm_rf)

# Classification Report
print("\nLogistic Regression Classification Report:")
print(classification_report(y_test, y_pred_log))

print("\nRandom Forest Classification Report:")
print(classification_report(y_test, y_pred_rf))

# Misclassified samples for Random Forest (better model)
misclassified = X_test[y_test != y_pred_rf]
misclassified_labels = y_test[y_test != y_pred_rf]
misclassified_preds = y_pred_rf[y_test != y_pred_rf]

print(f"\nRandom Forest Misclassified Samples: {len(misclassified)}")
if len(misclassified) > 0:
    print("Top misclassified features (mean values):")
    misclassified_means = misclassified.mean()
    print(misclassified_means.sort_values(ascending=False).head(5))

# Save to file
with open('./progress/week05/phan_tich_loi/error_analysis.txt', 'w') as f:
    f.write("=== Error Analysis ===\n\n")
    f.write("Logistic Regression Confusion Matrix:\n")
    f.write(str(cm_log))
    f.write("\n\nRandom Forest Confusion Matrix:\n")
    f.write(str(cm_rf))
    f.write("\n\nLogistic Regression Classification Report:\n")
    f.write(classification_report(y_test, y_pred_log))
    f.write("\n\nRandom Forest Classification Report:\n")
    f.write(classification_report(y_test, y_pred_rf))
    f.write(f"\n\nMisclassified Samples (Random Forest): {len(misclassified)}\n")
    if len(misclassified) > 0:
        f.write("Top misclassified features (mean values):\n")
        f.write(str(misclassified_means.sort_values(ascending=False).head(5)))

print("Error analysis đã lưu vào: ./progress/week05/phan_tich_loi/error_analysis.txt")
print("Phần 6 hoàn thành! Sẵn sàng cho phần 7 (Model Selection).")