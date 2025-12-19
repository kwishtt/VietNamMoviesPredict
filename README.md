# 🎬 Movie Success Prediction - Dự Đoán Độ Thành Công Của Phim

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange.svg?style=for-the-badge&logo=jupyter&logoColor=white)](https://jupyter.org/)
[![Pandas](https://img.shields.io/badge/Pandas-Data%20Analysis-150458.svg?style=for-the-badge&logo=pandas&logoColor=white)](https://pandas.pydata.org/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit_Learn-Machine%20Learning-orange.svg?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![Flask](https://img.shields.io/badge/Flask-Web%20App-red.svg?style=for-the-badge&logo=flask&logoColor=white)](https://flask.palletsprojects.com/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge)](LICENSE)

---

## � Mục Lục

- [Giới Thiệu Dự Án](#-giới-thiệu-dự-án)
- [Cấu Trúc Thư Mục](#-cấu-trúc-thư-mục)
- [Dataset](#-dataset)
- [Mô Hình Machine Learning](#-mô-hình-machine-learning)
- [Kết Quả](#-kết-quả)
- [Hướng Dẫn Cài Đặt](#-hướng-dẫn-cài-đặt)
- [Web Application](#-web-application)
- [Thông Tin Liên Hệ](#-thông-tin-liên-hệ)

---

## 🎯 Giới Thiệu Dự Án

Dự án **Movie Success Prediction** xây dựng mô hình Machine Learning để **dự đoán độ thành công của phim chiếu rạp** dựa trên các thông tin có sẵn trước khi phim ra mắt.

### Mục Tiêu Chính:
- 🎥 Hỗ trợ nhà sản xuất đánh giá **rủi ro đầu tư** vào dự án phim
- 📊 Cung cấp **insights** về các yếu tố ảnh hưởng đến thành công của phim
- 🌐 Xây dựng **Web Application** để demo và sử dụng thực tế

### Định Nghĩa Thành Công:
Phim được coi là **thành công** khi thỏa mãn đồng thời:
- `ROI ≥ 1.0` (Return on Investment: Revenue / Budget)
- `Vote Average ≥ 6.5` (Điểm đánh giá từ khán giả)

---

## 📁 Cấu Trúc Thư Mục

```
Do_An/
├── 📂 data/                      # Dữ liệu và Models
│   ├── raw_Movies.csv            # Dữ liệu gốc (2,194 phim)
│   ├── clean_movies.csv          # Dữ liệu đã làm sạch (1,020 phim)
│   ├── clean_movies_with_labels.csv
│   ├── clean_movies_features.csv # Dữ liệu đã Feature Engineering (65 features)
│   └── 📂 pkl/                   # Các model đã train
│       ├── random_forest_model.pkl
│       ├── logistic_model.pkl
│       ├── optimized_rf_model.pkl
│       ├── pre_release_rf_model.pkl     # ⭐ Model chính (Pre-Release)
│       └── train_test_data.pkl
│
├── 📂 progress/                  # Code theo tiến độ tuần
│   ├── week01/                   # Setup project
│   ├── week02/                   # Data Cleaning
│   ├── week03/                   # Label Creation
│   ├── week04/                   # Feature Engineering
│   ├── week05/                   # Baseline Models (Logistic, RF)
│   ├── week06/                   # Hyperparameter Tuning & Overfitting Analysis
│   ├── week07/                   # ⭐ Pre-Release Model (Giải quyết Data Leakage)
│   └── week08-10/                # Refinement & Web Development
│
├── 📂 webs/MoviePredict/         # 🌐 Web Application (Flask)
│   ├── app.py                    # Flask Backend
│   ├── 📂 models/                # Prediction Services
│   │   ├── prediction_service.py
│   │   └── pre_release_service.py
│   ├── 📂 static/                # CSS, JS, Images
│   └── 📂 templates/             # HTML Templates
│
├── 📂 docs/                      # Tài liệu & Báo cáo
│   ├── BaoCaoDoAn1_Nhom04.docx/.md
│   ├── Phan_Tich_Toan_Dien_Du_An.md
│   ├── ThuatNgu_Glossary.md
│   └── ...
│
├── 📂 chart/                     # Biểu đồ & Visualization
├── 📂 craw_data/                 # Scripts thu thập dữ liệu
├── requirements.txt
└── README.md
```

### 📖 Cách Đọc Dự Án:

1. **Bắt đầu:** Đọc `docs/Phan_Tich_Toan_Dien_Du_An.md` để nắm tổng quan
2. **Dữ liệu:** Xem các file CSV trong `data/`
3. **Tiến trình:** Theo dõi `progress/week01` → `week10` để hiểu từng bước
4. **Models:** Xem chi tiết tham số trong mục [Mô Hình Machine Learning](#-mô-hình-machine-learning)
5. **Demo:** Chạy Web App trong `webs/MoviePredict/`

---

## 📊 Dataset

| Thông Số | Giá Trị |
|----------|---------|
| **Nguồn** | TMDB (The Movie Database) |
| **Dữ liệu gốc** | 2,194 phim |
| **Sau làm sạch** | 1,020 phim |
| **Số features** | 65 (sau Feature Engineering) |
| **Phân bố label** | 514 thành công (50.4%) / 506 thất bại (49.6%) |

### Features Chính:
- **Tài chính:** `budget`, `Budget_log`, `roi`, `revenue`
- **Thời gian:** `release_year`, `release_month`, `release_quarter`, `is_holiday_season`
- **Thể loại:** `genre_Action`, `genre_Comedy`, `genre_Drama`, ... (One-Hot Encoded)
- **Địa lý:** `is_usa`, `is_vietnam`, `is_uk`, ...
- **Nội dung:** `num_genres`, `num_main_cast`, `runtime`

---

## 🤖 Mô Hình Machine Learning

### 1️⃣ Logistic Regression (Baseline)

| Tham Số | Giá Trị |
|---------|---------|
| `max_iter` | 1000 |
| `random_state` | 42 |
| `solver` | lbfgs (default) |

| Metric | Giá Trị |
|--------|---------|
| Accuracy | 84.80% |
| F1-Score | 84.88% |

📍 **Vị trí:** `progress/week05/Logistic_Regression_Model/logistic_regression.py`

---

### 2️⃣ Random Forest (Baseline)

| Tham Số | Giá Trị |
|---------|---------|
| `n_estimators` | 100 (default) |
| `random_state` | 42 |
| Các tham số khác | Default |

| Metric | Giá Trị |
|--------|---------|
| Accuracy | 99.51% |
| F1-Score | 99.52% |
| Recall | 100.00% |
| Precision | 99.04% |
| CV F1-Score | 99.88% ± 0.14% |

⚠️ **Lưu ý:** Model này có hiệu suất **cao bất thường** do sử dụng features bị **data leakage** (`vote_average`, `revenue`, `roi`).

📍 **Vị trí:** `progress/week05/Random_Forest_Model/random_forest.py`

---

### 3️⃣ Optimized Random Forest (Hyperparameter Tuned)

| Tham Số | Giá Trị |
|---------|---------|
| `n_estimators` | [50, 100, 200, 300] |
| `max_depth` | [5, 10, 15, 20, None] |
| `min_samples_split` | [2, 5, 10] |
| `min_samples_leaf` | [1, 2, 4] |
| `max_features` | ['sqrt', 'log2', None] |
| **Tuning Method** | RandomizedSearchCV (50 iterations) |
| **CV** | 5-Fold |
| **Scoring** | F1-Score |

� **Vị trí:** 
- Script: `progress/week06/hyperparameter_tuning.py`
- Model: `data/pkl/optimized_rf_model.pkl`

---

### 4️⃣ ⭐ Pre-Release Random Forest (PRODUCTION MODEL)

> **Đây là model chính được sử dụng trong Web Application** vì nó giải quyết vấn đề **Data Leakage**.

#### Tham Số Mô Hình:

| Tham Số | Giá Trị | Mô Tả |
|---------|---------|-------|
| `n_estimators` | 100 | Số lượng cây trong rừng |
| `max_depth` | 10 | Độ sâu tối đa của mỗi cây |
| `min_samples_split` | 5 | Số mẫu tối thiểu để phân chia node |
| `min_samples_leaf` | 2 | Số mẫu tối thiểu ở node lá |
| `random_state` | 42 | Seed cho reproducibility |
| `n_jobs` | -1 | Sử dụng tất cả CPU cores |
| `class_weight` | 'balanced' | Xử lý class imbalance |

#### Features Sử Dụng (Pre-Release Only):

```python
PRE_RELEASE_FEATURES = [
    # Basic
    'budget', 'Budget_log', 'runtime',
    
    # Time features
    'release_year', 'release_month', 'release_weekday', 
    'release_quarter', 'is_holiday_season',
    
    # Genre features (one-hot encoded)
    'num_genres', 'genre_Action', 'genre_Adventure', 'genre_Comedy', 
    'genre_Drama', 'genre_Thriller', 'genre_Science Fiction', ...
    
    # Country features
    'is_usa', 'is_vietnam', 'is_uk', 'is_china', 'is_france', ...
    
    # Cast features
    'num_main_cast'
]
```

#### Features Đã Loại Bỏ (Data Leakage):

```python
# ❌ KHÔNG sử dụng (chỉ biết SAU khi phim ra mắt)
POST_RELEASE_FEATURES = [
    'revenue', 'Revenue_log',
    'vote_average', 'vote_count',
    'roi', 'roi_clipped', 'roi_vs_vote'
]
```

#### Kết Quả:

| Metric | Giá Trị |
|--------|---------|
| Accuracy | ~70-80% |
| F1-Score | ~70-80% |
| CV Mean | ~70-80% |

> 📌 **Giải thích:** Accuracy thấp hơn model cũ (99%) là **hoàn toàn bình thường** và **đúng logic** vì model này chỉ sử dụng thông tin biết trước → dự đoán THỰC SỰ.

📍 **Vị trí:**
- Training Script: `progress/week07/retrain.py`
- Model File: `data/pkl/pre_release_rf_model.pkl`
- Service: `webs/MoviePredict/models/pre_release_service.py`

---

## 📈 Kết Quả

### So Sánh Các Mô Hình:

| Model | Accuracy | F1-Score | Data Leakage? | Production Ready? |
|-------|----------|----------|---------------|-------------------|
| Logistic Regression | 84.80% | 84.88% | ✅ Có | ❌ |
| Random Forest (Baseline) | 99.51% | 99.52% | ✅ Có | ❌ |
| Random Forest (Tuned) | ~99% | ~99% | ✅ Có | ❌ |
| **Pre-Release RF** | **~75%** | **~75%** | **❌ Không** | **✅ Có** |

### Top 5 Features Quan Trọng (Pre-Release Model):

| Rank | Feature | Mô Tả |
|------|---------|-------|
| 1 | `budget` | Ngân sách sản xuất |
| 2 | `release_month` | Tháng phát hành |
| 3 | `num_genres` | Số thể loại phim |
| 4 | `is_usa` | Sản xuất tại Mỹ hay không |
| 5 | `genre_Action` | Có phải phim hành động |

---

## 🛠 Hướng Dẫn Cài Đặt

### Yêu Cầu:
- Python 3.8+
- pip

### Cài Đặt:

```bash
# Clone repository
git clone https://github.com/kwishtt/Do_An_1.git
cd Do_An_1

# Tạo virtual environment
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# hoặc .venv\Scripts\activate  # Windows

# Cài đặt dependencies
pip install -r requirements.txt
```

### Chạy Web Application:

```bash
cd webs/MoviePredict
python app.py
```

Truy cập: `http://localhost:5000`

---

## 🌐 Web Application

Web Application được xây dựng bằng **Flask** với giao diện Modern Dark Sci-Fi Theme.

### Tính Năng:
- 🎬 Dự đoán xác suất thành công của phim
- 📊 Hiển thị Feature Importance 
- 💰 Ước tính ROI và mức độ rủi ro
- 🎯 Gợi ý các phim tương tự trong lịch sử

### Tech Stack:
- **Backend:** Flask (Python)
- **Frontend:** HTML5, CSS3, JavaScript
- **ML Model:** Scikit-Learn Random Forest
- **Visualization:** Chart.js

---

## 👥 Thông Tin Liên Hệ

Dự án được thực hiện bởi **Nhóm 04 - Khoa Học Dữ Liệu - HUMG**.


- 📧 **Email:** kforwork04@gmail.com
- 🔗 **Repository:** [GitHub Link](https://github.com/kwishtt/VietNamMoviesPredict)

---

<div align="center">
  
  **Made with ❤️ by Team 04**
  
  Copyright © 2025 Team 04. All rights reserved.
  
</div>
