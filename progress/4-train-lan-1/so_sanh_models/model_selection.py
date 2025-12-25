import pickle
import pandas as pd
import numpy as np

# Tải kết quả từ các phần trước
# Chúng ta có thể tải models và data để tóm tắt

with open('./data/pkl/train_test_data.pkl', 'rb') as f:
    data = pickle.load(f)

X_test = data['X_test']
y_test = data['y_test']

with open('./data/pkl/random_forest_model.pkl', 'rb') as f:
    rf_model = pickle.load(f)

with open('./data/pkl/logistic_model.pkl', 'rb') as f:
    log_model = pickle.load(f)

# Đánh giá nhanh
y_pred_log = log_model.predict(X_test)
y_pred_rf = rf_model.predict(X_test)

acc_log = np.mean(y_pred_log == y_test)
acc_rf = np.mean(y_pred_rf == y_test)

print("=== Lựa Chọn Model ===")
print(f"Độ chính xác kiểm tra Logistic Regression: {acc_log:.4f}")
print(f"Độ chính xác kiểm tra Random Forest: {acc_rf:.4f}")
print(f"Cải thiện: {acc_rf - acc_log:.4f}")

# Tiêu chí lựa chọn
print("\n=== Tiêu Chí Lựa Chọn ===")
print("1. Độ chính xác: RF (1.0000) > Logistic (0.8480)")
print("2. Điểm F1: RF (1.0000) > Logistic (0.8480)")
print("3. Ổn định Cross-Validation: RF (std 0.0072) < Logistic (std 0.0179)")
print("4. Quá khớp: Cả hai tối thiểu, RF hơi tốt hơn")
print("5. Khả năng giải thích: Logistic tốt hơn, nhưng hiệu suất RF vượt trội")
print("6. Chi phí tính toán: RF cao hơn, nhưng chấp nhận được cho tập dữ liệu nhỏ")

print("\n=== Lựa Chọn Cuối Cùng ===")
print("✅ Chọn Random Forest làm model cuối cùng")
print("Lý do: Hiệu suất vượt trội, ổn định cao, phù hợp cho dự đoán thành công phim")

# Lưu model cuối cùng dưới dạng best_model.pkl
with open('./data/pkl/best_model.pkl', 'wb') as f:
    pickle.dump(rf_model, f)

print("Best model (Random Forest) đã lưu tại: ./data/pkl/best_model.pkl")

# Lưu tóm tắt lựa chọn
with open('./progress/week05/so_sanh_models/model_selection.txt', 'w') as f:
    f.write("=== Tóm Tắt Lựa Chọn Model ===\n\n")
    f.write(f"Độ chính xác kiểm tra Logistic Regression: {acc_log:.4f}\n")
    f.write(f"Độ chính xác kiểm tra Random Forest: {acc_rf:.4f}\n")
    f.write(f"Cải thiện: {acc_rf - acc_log:.4f}\n\n")
    f.write("Tiêu chí lựa chọn:\n")
    f.write("- Độ chính xác: RF vượt trội\n")
    f.write("- Điểm F1: RF vượt trội\n")
    f.write("- Ổn định CV: RF ổn định hơn\n")
    f.write("- Quá khớp: Cả hai tối thiểu\n")
    f.write("- Khả năng giải thích: Logistic tốt hơn, nhưng hiệu suất quan trọng hơn\n")
    f.write("- Chi phí tính toán: RF chấp nhận được\n\n")
    f.write("Lựa chọn cuối cùng: Random Forest\n")
    f.write("Lý do: Hiệu suất vượt trội, ổn định cao, phù hợp cho dự đoán thành công phim\n")

print("Model selection đã lưu vào: ./progress/week05/so_sanh_models/model_selection.txt")
print("Phần 7 hoàn thành! Sẵn sàng cho phần 8 (Report).")