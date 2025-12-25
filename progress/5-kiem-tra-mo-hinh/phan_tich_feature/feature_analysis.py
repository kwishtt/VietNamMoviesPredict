# Bước 2.1: Visualize Feature Importance - Tuần 6
# Mục đích: Phân tích sâu feature importance, tạo visualizations và insights
# Tác dụng: Hiểu rõ yếu tố nào ảnh hưởng đến thành công phim, support business decisions

import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
import os

print("=== Bước 2.1: Feature Importance Visualization ===")

# 1. Load dữ liệu và model từ tuần 5
data_path = './data/pkl/train_test_data.pkl'
model_path = './data/pkl/optimized_rf_model.pkl'

if not os.path.exists(data_path):
    raise FileNotFoundError(f"File {data_path} không tồn tại. Hãy chạy data preparation trước.")

if not os.path.exists(model_path):
    raise FileNotFoundError(f"File {model_path} không tồn tại. Hãy chạy hyperparameter tuning trước.")

# Load data và model
with open(data_path, 'rb') as f:
    data = pickle.load(f)

with open(model_path, 'rb') as f:
    model_data = pickle.load(f)

X_train = data['X_train']
X_test = data['X_test']
y_train = data['y_train']
y_test = data['y_test']
feature_names = data['feature_names']
best_model = model_data['model']

print(f"Dữ liệu load thành công: {len(X_train)} training samples, {len(feature_names)} features")

# 2. Trích xuất feature importance từ model
feature_importances = best_model.feature_importances_
feature_importance_df = pd.DataFrame({
    'feature': feature_names,
    'importance': feature_importances
}).sort_values('importance', ascending=False)

print(f"\n=== Top 15 Features Quan Trọng Nhất ===")
print(feature_importance_df.head(15).to_string(index=False))

# 3. Tạo thư mục cho charts
charts_dir = './chart/week06'
os.makedirs(charts_dir, exist_ok=True)

# 4. Visualization 1: Bar Chart Top 15 Features
plt.figure(figsize=(15, 10))

# Plot 1: Top 15 features horizontal bar chart
plt.subplot(2, 2, 1)
top_15 = feature_importance_df.head(15)
colors = plt.cm.viridis(np.linspace(0, 1, 15))

bars = plt.barh(range(len(top_15)), top_15['importance'], color=colors)
plt.yticks(range(len(top_15)), top_15['feature'])
plt.xlabel('Feature Importance')
plt.title('Top 15 Features Quan Trọng Nhất')
plt.gca().invert_yaxis()

# Thêm giá trị trên bars
for i, (bar, value) in enumerate(zip(bars, top_15['importance'])):
    plt.text(value + 0.001, bar.get_y() + bar.get_height()/2, 
             f'{value:.3f}', va='center', ha='left', fontsize=8)

# Plot 2: Cumulative importance
plt.subplot(2, 2, 2)
cumulative_importance = feature_importance_df['importance'].cumsum()
plt.plot(range(1, len(cumulative_importance) + 1), cumulative_importance, 'b-', linewidth=2)
plt.axhline(y=0.8, color='red', linestyle='--', alpha=0.7, label='80% threshold')
plt.axhline(y=0.9, color='orange', linestyle='--', alpha=0.7, label='90% threshold')
plt.xlabel('Số Features')
plt.ylabel('Cumulative Importance')
plt.title('Cumulative Feature Importance')
plt.legend()
plt.grid(True, alpha=0.3)

# Tìm số features cần thiết cho 80% và 90% importance
features_80 = (cumulative_importance >= 0.8).idxmax() + 1
features_90 = (cumulative_importance >= 0.9).idxmax() + 1
print(f"\nSố features cần thiết:")
print(f"- 80% importance: {features_80} features")
print(f"- 90% importance: {features_90} features")

# Plot 3: Feature categories analysis
plt.subplot(2, 2, 3)

# Phân loại features theo categories
def categorize_features(feature_names):
    categories = {
        'Financial': ['roi', 'revenue', 'budget', 'Budget', 'Revenue'],
        'Quality': ['vote', 'Vote'],
        'Time': ['release', 'year', 'month', 'quarter', 'weekday'],
        'Content': ['genre', 'runtime', 'cast', 'num_'],
        'Geographic': ['is_', 'country'],
        'Engineered': ['_vs_', 'interaction', '_log', '_clipped', '_per_']
    }
    
    feature_categories = []
    for feature in feature_names:
        category = 'Other'
        for cat, keywords in categories.items():
            if any(keyword.lower() in feature.lower() for keyword in keywords):
                category = cat
                break
        feature_categories.append(category)
    
    return feature_categories

feature_importance_df['category'] = categorize_features(feature_importance_df['feature'])
category_importance = feature_importance_df.groupby('category')['importance'].sum().sort_values(ascending=False)

colors_cat = plt.cm.Set3(np.linspace(0, 1, len(category_importance)))
wedges, texts, autotexts = plt.pie(category_importance.values, labels=category_importance.index, 
                                  autopct='%1.1f%%', colors=colors_cat, startangle=90)
plt.title('Feature Importance theo Categories')

# Plot 4: Original vs Engineered features
plt.subplot(2, 2, 4)
original_features = ['Vote Average', 'Budget', 'Revenue', 'Runtime', 'Vote Count']
engineered_features = [f for f in feature_names if any(eng in f for eng in ['roi', '_vs_', '_log', '_clipped', '_per_', 'interaction'])]

original_importance = feature_importance_df[feature_importance_df['feature'].isin(original_features)]['importance'].sum()
engineered_importance = feature_importance_df[feature_importance_df['feature'].isin(engineered_features)]['importance'].sum()
other_importance = 1 - original_importance - engineered_importance

comparison_data = [original_importance, engineered_importance, other_importance]
comparison_labels = ['Original Features', 'Engineered Features', 'Other Features']
colors_comp = ['lightblue', 'lightgreen', 'lightcoral']

bars = plt.bar(comparison_labels, comparison_data, color=colors_comp)
plt.ylabel('Total Importance')
plt.title('Original vs Engineered Features')
plt.xticks(rotation=45, ha='right')

# Thêm giá trị trên bars
for bar, value in zip(bars, comparison_data):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
             f'{value:.3f}', ha='center', va='bottom')

plt.tight_layout()
plt.savefig(os.path.join(charts_dir, 'feature_importance_analysis.png'), dpi=300, bbox_inches='tight')
plt.close()

print(f"Biểu đồ feature importance đã lưu vào: {os.path.join(charts_dir, 'feature_importance_analysis.png')}")

# 5. Correlation Heatmap với Top Features và Target
print(f"\n=== Tạo Correlation Heatmap ===")

# Lấy top 15 features và tính correlation với target
top_features = feature_importance_df.head(15)['feature'].tolist()

# Tạo DataFrame với top features và target
correlation_data = X_train[top_features].copy()
correlation_data['success'] = y_train.values

# Tính correlation matrix
correlation_matrix = correlation_data.corr()

# Vẽ heatmap
plt.figure(figsize=(14, 12))
mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
heatmap = sns.heatmap(correlation_matrix, 
                     annot=True, 
                     cmap='RdYlBu_r', 
                     center=0,
                     mask=mask,
                     square=True,
                     fmt='.3f',
                     cbar_kws={"shrink": .8})

plt.title('Correlation Heatmap: Top 15 Features + Target', fontsize=16, pad=20)
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig(os.path.join(charts_dir, 'correlation_heatmap.png'), dpi=300, bbox_inches='tight')
plt.close()

print(f"Correlation heatmap đã lưu vào: {os.path.join(charts_dir, 'correlation_heatmap.png')}")

# 6. Feature Importance vs Correlation Analysis
print(f"\n=== Feature Importance vs Correlation Analysis ===")

# Tính correlation của mỗi feature với target
target_correlations = []
for feature in feature_names:
    if feature in X_train.columns:
        corr = np.corrcoef(X_train[feature], y_train)[0, 1]
        target_correlations.append(abs(corr))  # Lấy absolute value
    else:
        target_correlations.append(0)

feature_importance_df['target_correlation'] = target_correlations

# Vẽ scatter plot
plt.figure(figsize=(12, 8))
plt.scatter(feature_importance_df['target_correlation'], 
           feature_importance_df['importance'],
           alpha=0.6, s=50)

# Highlight top 10 features
top_10 = feature_importance_df.head(10)
plt.scatter(top_10['target_correlation'], 
           top_10['importance'],
           color='red', s=100, alpha=0.8, label='Top 10 Features')

# Thêm labels cho top 5 features
for idx, row in top_10.head(5).iterrows():
    plt.annotate(row['feature'], 
                (row['target_correlation'], row['importance']),
                xytext=(5, 5), textcoords='offset points',
                fontsize=9, alpha=0.8)

plt.xlabel('Absolute Correlation với Target')
plt.ylabel('Random Forest Feature Importance')
plt.title('Feature Importance vs Target Correlation')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(charts_dir, 'importance_vs_correlation.png'), dpi=300, bbox_inches='tight')
plt.close()

print(f"Importance vs Correlation plot đã lưu vào: {os.path.join(charts_dir, 'importance_vs_correlation.png')}")

# 7. Lưu kết quả analysis
results_path = './progress/week06/feature_analysis_results.txt'
with open(results_path, 'w', encoding='utf-8') as f:
    f.write("=== Feature Importance Analysis - Tuần 6 ===\n\n")
    
    f.write("=== Top 15 Features ===\n")
    f.write(feature_importance_df.head(15).to_string(index=False))
    
    f.write(f"\n\n=== Feature Efficiency Analysis ===\n")
    f.write(f"Features cần thiết cho 80% importance: {features_80}/{len(feature_names)} ({features_80/len(feature_names)*100:.1f}%)\n")
    f.write(f"Features cần thiết cho 90% importance: {features_90}/{len(feature_names)} ({features_90/len(feature_names)*100:.1f}%)\n")
    
    f.write(f"\n=== Category Analysis ===\n")
    for category, importance in category_importance.items():
        f.write(f"{category}: {importance:.3f} ({importance*100:.1f}%)\n")
    
    f.write(f"\n=== Original vs Engineered Features ===\n")
    f.write(f"Original Features Importance: {original_importance:.3f} ({original_importance*100:.1f}%)\n")
    f.write(f"Engineered Features Importance: {engineered_importance:.3f} ({engineered_importance*100:.1f}%)\n")
    f.write(f"Other Features Importance: {other_importance:.3f} ({other_importance*100:.1f}%)\n")
    
    f.write(f"\n=== High Correlation + High Importance Features ===\n")
    high_both = feature_importance_df[(feature_importance_df['importance'] > 0.02) & 
                                     (feature_importance_df['target_correlation'] > 0.3)]
    f.write(high_both[['feature', 'importance', 'target_correlation']].to_string(index=False))
    
    f.write(f"\n=== Files Generated ===\n")
    f.write(f"Feature importance analysis: {os.path.join(charts_dir, 'feature_importance_analysis.png')}\n")
    f.write(f"Correlation heatmap: {os.path.join(charts_dir, 'correlation_heatmap.png')}\n")
    f.write(f"Importance vs correlation: {os.path.join(charts_dir, 'importance_vs_correlation.png')}\n")

print(f"Kết quả analysis đã lưu vào: {results_path}")

# 8. Tóm tắt insights
print(f"\n=== TÓM TẮT FEATURE IMPORTANCE INSIGHTS ===")
print(f"✅ Top feature: {feature_importance_df.iloc[0]['feature']} ({feature_importance_df.iloc[0]['importance']:.1%})")
print(f"✅ {features_80} features cover 80% importance")
print(f"✅ Engineered features đóng góp: {engineered_importance:.1%}")
print(f"✅ Quality features (Vote) dominance: {category_importance.get('Quality', 0):.1%}")
print(f"✅ Financial features importance: {category_importance.get('Financial', 0):.1%}")

print("\nBước 2.1 hoàn thành! Sẵn sàng cho Bước 2.2 (Business Insights).")