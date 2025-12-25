import pickle
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, f1_score

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

# Feature Importance from Random Forest
feature_importances = rf_model.feature_importances_
importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': feature_importances
}).sort_values('Importance', ascending=False)

print("=== Top 10 Feature Importances (Random Forest) ===")
print(importance_df.head(10))

# Save to file
with open('./progress/week05/phan_tich_dac_trung/feature_importance.txt', 'w') as f:
    f.write("=== Feature Importance Analysis (Random Forest) ===\n\n")
    f.write("Top 10 Features:\n")
    f.write(importance_df.head(10).to_string(index=False))
    f.write("\n\nAll Features:\n")
    f.write(importance_df.to_string(index=False))

print("Feature importance đã lưu vào: ./progress/week05/phan_tich_dac_trung/feature_importance.txt")
print("Phần 5 hoàn thành! Sẵn sàng cho phần 6 (Error Analysis).")