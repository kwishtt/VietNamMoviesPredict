# 📚 TÀI LIỆU ÔN TẬP THUYẾT TRÌNH ĐỒ ÁN
## DỰ ĐOÁN ĐỘ THÀNH CÔNG CỦA PHIM

# 🎯 MỤC LỤC

1. [Tổng Quan Dự Án](#1-tổng-quan-dự-án)
2. [Dataset & Tiền Xử Lý](#2-dataset--tiền-xử-lý)
3. [Feature Engineering](#3-feature-engineering)
4. [Mô Hình Machine Learning](#4-mô-hình-machine-learning)
5. [Vấn Đề Data Leakage (QUAN TRỌNG)](#5-vấn-đề-data-leakage-quan-trọng)
6. [Đánh Giá Mô Hình](#6-đánh-giá-mô-hình)
7. [Web Application](#7-web-application)
8. [Thuật Ngữ Cần Nhớ](#8-thuật-ngữ-cần-nhớ)
9. [Câu Hỏi GV Có Thể Hỏi & Cách Trả Lời](#9-câu-hỏi-gv-có-thể-hỏi--cách-trả-lời)
10. [Tài Liệu Tham Khảo](#10-tài-liệu-tham-khảo)

---

# 1. TỔNG QUAN DỰ ÁN

## 1.1 Mục Tiêu

| Mục tiêu | Mô tả |
|----------|-------|
| **Chính** | Dự đoán phim có **thành công** hay không **TRƯỚC KHI** phim ra mắt |
| **Ứng dụng** | Hỗ trợ nhà sản xuất đánh giá **rủi ro đầu tư** |
| **Insight** | Phân tích các yếu tố ảnh hưởng đến thành công của phim |

## 1.2 Định Nghĩa "Thành Công"

> ⭐ **ĐÂY LÀ ĐỊNH NGHĨA QUAN TRỌNG - CẦN THUỘC LÒNG**

Phim được coi là **thành công (success = 1)** khi thỏa mãn **CẢ HAI** điều kiện:

```
┌─────────────────────────────────────────────────┐
│  ROI ≥ 1.0  VÀ  Vote Average ≥ 6.5             │
└─────────────────────────────────────────────────┘
```

### Giải thích:
- **ROI (Return on Investment)** = Revenue / Budget
  - ROI ≥ 1.0 nghĩa là phim **thu hồi vốn** (revenue ≥ budget)
  - Ví dụ: Phim budget $100M, revenue $150M → ROI = 1.5 ✅
  
- **Vote Average ≥ 6.5**: Điểm đánh giá từ khán giả (thang 1-10)
  - Đảm bảo phim **được khán giả yêu thích**, không chỉ may mắn

### Tại sao dùng CẢ HAI?
> "Phim thành công cần cả hai yếu tố: **sinh lời về tài chính** VÀ **được đánh giá cao về chất lượng**. Phim có ROI cao nhưng vote thấp có thể chỉ do marketing tốt, không bền vững."

## 1.3 Công Nghệ Sử Dụng

| Công nghệ | Vai trò | Phiên bản |
|-----------|---------|-----------|
| Python | Ngôn ngữ chính | 3.8+ |
| Pandas | Xử lý dữ liệu | 2.0+ |
| Scikit-Learn | Machine Learning | 1.3+ |
| Flask | Web Framework | 2.0+ |
| Chart.js | Visualization | 4.4 |

## 1.4 Lộ Trình 3 Bước

- **Bước 1: Chuẩn bị & tiền xử lý dữ liệu** — thu thập, làm sạch, gán nhãn, tạo đặc trưng.
- **Bước 2: Huấn luyện** — chia dữ liệu, chọn mô hình, tuning, kiểm soát overfitting, đánh giá.
- **Bước 3: Sản phẩm** — đóng gói model, tích hợp API/web, báo cáo và trình bày.

---

# 2. DATASET & TIỀN XỬ LÝ

## 2.1 Nguồn Dữ Liệu

| Thông số | Giá trị |
|----------|---------|
| **Nguồn** | TMDB (The Movie Database) via Kaggle |
| **Dữ liệu gốc** | 2,194 phim |
| **Sau làm sạch** | 1,020 phim |
| **Số cột gốc** | ~15 cột |
| **Số features cuối** | 65 features (sau Feature Engineering) |

## 2.2 Các Bước Làm Sạch Dữ Liệu

### Bước 1: Loại bỏ giá trị thiếu/không hợp lệ
```python
# Loại bỏ phim có Budget = 0 hoặc Revenue = 0
df = df[(df['Budget'] > 0) & (df['Revenue'] > 0)]
```
**Kết quả**: 2,194 → 1,020 phim (loại 1,174 phim)

### Bước 2: Chuẩn hóa ngày phát hành
```python
df['Release Date'] = pd.to_datetime(df['Release Date'])
df['release_year'] = df['Release Date'].dt.year
df['release_month'] = df['Release Date'].dt.month
```

### Bước 3: Xử lý cột Genres (One-Hot Encoding)
```python
# Từ: "['Action', 'Comedy']"
# Thành: genre_Action=1, genre_Comedy=1, genre_Drama=0, ...
```

## 2.3 Phân Bố Label

```
┌───────────────────────────────────────┐
│  Thành công: 514 phim (50.4%)        │
│  Thất bại:   506 phim (49.6%)        │
└───────────────────────────────────────┘
```

> ⭐ **Dữ liệu khá CÂN BẰNG** → Không cần SMOTE hay oversampling

---

# 3. FEATURE ENGINEERING

## 3.1 Tổng Quan

| Loại Feature | Số lượng | Ví dụ |
|--------------|----------|-------|
| Tài chính | 4 | budget, Budget_log, roi, revenue |
| Thời gian | 5 | release_year, release_month, is_holiday_season |
| Thể loại | 16 | genre_Action, genre_Comedy, num_genres |
| Quốc gia | 11 | is_usa, is_vietnam, is_china |
| Nội dung | 3 | runtime, num_main_cast |

**Tổng: 65 features**

## 3.2 Chi Tiết Từng Nhóm Feature

### A. Features Tài Chính
| Feature | Công thức | Ý nghĩa |
|---------|-----------|---------|
| `budget` | Giá trị gốc | Ngân sách sản xuất (USD) |
| `Budget_log` | log₁₀(budget + 1) | Chuẩn hóa phân phối lệch |
| `revenue` | Giá trị gốc | Doanh thu (USD) |
| `roi` | revenue / budget | Tỷ suất lợi nhuận |

### B. Features Thời Gian
| Feature | Công thức | Ý nghĩa |
|---------|-----------|---------|
| `release_year` | Trích xuất từ date | Năm phát hành |
| `release_month` | Trích xuất từ date | Tháng phát hành (1-12) |
| `release_weekday` | Trích xuất từ date | Ngày trong tuần (0-6) |
| `release_quarter` | (month-1)//3 + 1 | Quý (1-4) |
| `is_holiday_season` | 1 nếu tháng 6,7,11,12 | Mùa lễ/hè |

### C. Features Thể Loại (One-Hot Encoding)
```
genre_Action, genre_Adventure, genre_Comedy, genre_Drama,
genre_Thriller, genre_Science Fiction, genre_Family, 
genre_Fantasy, genre_Crime, genre_Animation, genre_Horror,
genre_Romance, genre_Mystery, genre_History, genre_Music
```

| Feature | Ý nghĩa |
|---------|---------|
| `num_genres` | Số thể loại của phim |
| `main_genre` | Thể loại chính (thể loại đầu tiên) |

### D. Features Quốc Gia
```
is_usa, is_united_kingdom, is_vietnam, is_china, is_france,
is_south_korea, is_japan, is_india, is_australia, is_canada
```

### E. Features Nội Dung
| Feature | Ý nghĩa |
|---------|---------|
| `runtime` | Thời lượng phim (phút) |
| `num_main_cast` | Số diễn viên chính |
| `cast_genre_interaction` | num_cast × num_genres |

## 3.3 Kỹ Thuật Biến Đổi Sử Dụng

| Kỹ thuật | Mục đích | Áp dụng cho |
|----------|----------|-------------|
| **Log Transformation** | Chuẩn hóa phân phối lệch | budget, revenue |
| **One-Hot Encoding** | Chuyển categorical → binary | genres, countries |
| **Min-Max Scaling** | Chuẩn hóa về [0,1] | Tất cả numerical features |
| **Feature Interaction** | Tạo feature mới từ tổ hợp | cast_genre_interaction |

---

# 4. MÔ HÌNH MACHINE LEARNING

## 4.1 Tổng Quan Các Mô Hình Đã Thử

| Model | Accuracy | F1-Score | Data Leakage? | Production Ready? |
|-------|----------|----------|---------------|-------------------|
| Logistic Regression | 84.80% | 84.88% | ✅ CÓ | ❌ |
| Random Forest (cũ) | 99.51% | 99.52% | ✅ CÓ | ❌ |
| **Pre-Release RF** | **67.65%** | **67.96%** | **❌ KHÔNG** | **✅** |

> ⭐ **Model Pre-Release Random Forest được chọn làm model chính**

## 4.2 Tại Sao Chọn Random Forest?

### Ưu điểm của Random Forest:
1. **Xử lý tốt dữ liệu phi tuyến** - không cần giả định linear
2. **Ít bị overfitting** hơn Decision Tree đơn lẻ
3. **Có Feature Importance** - biết feature nào quan trọng
4. **Xử lý được missing values** và outliers tốt
5. **Không cần feature scaling** (nhưng ta vẫn scale để nhất quán)

### Random Forest là gì?
> "Random Forest là thuật toán **Ensemble Learning** kết hợp nhiều Decision Trees. Mỗi tree được train trên một subset ngẫu nhiên của data và features. Kết quả cuối cùng là **voting** (phân loại) hoặc **trung bình** (hồi quy) từ tất cả các trees."

```
        ┌─────────────────────────────────────┐
        │         RANDOM FOREST               │
        │  ┌─────┐ ┌─────┐ ┌─────┐ ... ┌─────┐│
        │  │Tree1│ │Tree2│ │Tree3│     │TreeN││
        │  └──┬──┘ └──┬──┘ └──┬──┘     └──┬──┘│
        │     │       │       │           │   │
        │     └───────┴───┬───┴───────────┘   │
        │                 ▼                   │
        │            MAJORITY VOTE            │
        │                 ▼                   │
        │           Final Prediction          │
        └─────────────────────────────────────┘
```

## 4.3 Tham Số Mô Hình (Hyperparameters)

### Pre-Release Random Forest Model:

| Tham số | Giá trị | Ý nghĩa | Giải thích thêm |
|---------|---------|---------|----------------|
| `n_estimators` | 100 | Số cây trong rừng. Nhiều cây giúp kết quả ổn định | 50 cây cho kết quả dao động hơn; 100 cây tăng ổn định ~không tăng thời gian nhiều, thêm nữa cải thiện rất ít |
| `max_depth` | 10 | Giới hạn độ sâu của cây để không học thuộc lòng dữ liệu | Đủ sâu để học quan hệ phi tuyến, nhưng không quá sâu để nhớ dữ liệu |
| `min_samples_split` | 5 | Phải có tối thiểu 5 mẫu mới cho phép tách nút | Ép cây dừng sớm nếu nút quá ít dữ liệu, giảm chia vụn |
| `min_samples_leaf` | 2 | Mỗi lá phải có ít nhất 2 mẫu | Ngăn lá đơn lẻ gây overfit/nhiễu |
| `random_state` | 42 | Cố định ngẫu nhiên để chạy lại cho ra kết quả giống nhau | Dễ kiểm chứng và tái lập thí nghiệm |
| `n_jobs` | -1 | Dùng tất cả lõi CPU sẵn có để train/predict nhanh hơn | Tận dụng tài nguyên máy, rút ngắn thời gian chạy |
| `class_weight` | 'balanced' | Tự cân bằng trọng số giữa hai lớp | Giảm lệch dự đoán khi dữ liệu lệch lớp |

### Giải thích chi tiết từng tham số (tăng/giảm ảnh hưởng gì?)

**n_estimators = 100**
- Vai trò: số lượng cây trong rừng, càng nhiều càng ổn định.
- Tăng lên: giảm phương sai (kết quả ít dao động) nhưng thời gian/ram tăng nhẹ sau 200 cây ít cải thiện.
- Giảm xuống: train nhanh hơn nhưng kết quả dao động nhiều, dễ kém ổn định.
- Chọn 100: đủ ổn định, thời gian train/predict vẫn nhanh.

**max_depth = 10**
- Vai trò: chặn độ sâu để kiểm soát độ phức tạp.
- Tăng lên: cây học chi tiết hơn nhưng dễ overfit.
- Giảm xuống: giảm overfit nhưng có nguy cơ underfit nếu quá nông.
- Chọn 10: đủ sâu để học quan hệ phi tuyến, nhưng vẫn giữ được generalization.

**min_samples_split = 5**
- Vai trò: yêu cầu tối thiểu 5 mẫu để tách tiếp một nút.
- Tăng lên: cây dừng sớm hơn, ít chia vụn, giảm overfit.
- Giảm xuống: cây tách nhiều hơn, có thể bắt nhiễu.
- Chọn 5: cân bằng giữa việc khám phá phân tách và tránh chia quá nhỏ.

**min_samples_leaf = 2**
- Vai trò: số mẫu tối thiểu trong một lá cuối.
- Tăng lên: lá lớn hơn, mượt hơn, giảm overfit.
- Giảm xuống: có thể tạo lá đơn lẻ, nhạy nhiễu.
- Chọn 2: đủ để tránh lá đơn, vẫn giữ được chi tiết cần thiết.

**class_weight = 'balanced'**
- Vai trò: tự động cân trọng số dựa trên tần suất lớp.
- Bỏ cân bằng: model có thể nghiêng về lớp phổ biến nếu lệch phân bố.
- Chọn 'balanced': giúp Precision/Recall hai lớp tiệm cận, giảm bias.

**n_jobs = -1**
- Vai trò: dùng toàn bộ CPU song song để train/predict.
- Tăng/giảm: không ảnh hưởng chất lượng, chỉ ảnh hưởng tốc độ.
- Chọn -1: tận dụng tối đa tài nguyên máy, rút ngắn thời gian.

**random_state = 42**
- Vai trò: cố định seed cho bootstrap và chọn feature ngẫu nhiên.
- Để ngẫu nhiên: khó tái lập kết quả và so sánh.
- Chọn 42: giúp tái lập và debug dễ dàng.

## 4.4 Pipeline Training

```
┌──────────────┐
│ Load Data    │ clean_movies_features.csv (1,020 phim)
└──────┬───────┘
       ▼
┌──────────────┐
│ Select       │ Chọn 37 Pre-Release features
│ Features     │ Loại bỏ post-release features
└──────┬───────┘
       ▼
┌──────────────┐
│ Train/Test   │ 80% Train (816) / 20% Test (204)
│ Split        │ Stratified Sampling
└──────┬───────┘
       ▼
┌──────────────┐
│ Scale        │ StandardScaler (mean=0, std=1)
│ Features     │
└──────┬───────┘
       ▼
┌──────────────┐
│ Train        │ RandomForestClassifier
│ Model        │
└──────┬───────┘
       ▼
┌──────────────┐
│ Evaluate     │ Accuracy, F1, Precision, Recall
│              │ 5-Fold Cross-Validation
└──────┬───────┘
       ▼
┌──────────────┐
│ Save         │ pre_release_rf_model.pkl
│ Model        │
└──────────────┘
```

---

# 5. VẤN ĐỀ DATA LEAKAGE (QUAN TRỌNG)

> ⭐ **ĐÂY LÀ PHẦN QUAN TRỌNG NHẤT - ĐIỂM KHÁC BIỆT CỦA DỰ ÁN**

## 5.1 Data Leakage Là Gì?

**Định nghĩa**: Data Leakage xảy ra khi model **sử dụng thông tin mà thực tế không có sẵn tại thời điểm dự đoán**.

### Ví dụ trong dự án này:

| Feature | Thời điểm biết | Vấn đề |
|---------|----------------|--------|
| `revenue` | SAU khi phim chiếu | ❌ Không thể dùng để dự đoán TRƯỚC |
| `vote_average` | SAU khi phim chiếu | ❌ Chỉ có SAU khi khán giả đánh giá |
| `vote_count` | SAU khi phim chiếu | ❌ Chỉ có SAU khi người xem rate |
| `roi` | SAU khi phim chiếu | ❌ Cần biết revenue |

### Hậu quả của Data Leakage:
```
Model cũ (có leakage):     Accuracy = 99.51%  ← Giả tạo!
Model mới (không leakage): Accuracy = 67.65%  ← Thực tế!
```

> ⚠️ "Accuracy 99% nghe có vẻ tốt, nhưng thực ra là **vô nghĩa** vì model đang 'gian lận' - sử dụng thông tin từ tương lai."

## 5.2 Cách Giải Quyết

### Features đã LOẠI BỎ (Post-Release):
```python
POST_RELEASE_FEATURES = [
    'revenue', 'Revenue_log',      # Chỉ biết SAU khi chiếu
    'vote_average', 'vote_count',  # Chỉ biết SAU khi đánh giá
    'roi', 'roi_clipped',          # Cần biết revenue
    'roi_vs_vote', 'budget_per_year'
]
```

### Features được SỬ DỤNG (Pre-Release):
```python
PRE_RELEASE_FEATURES = [
    # Basic - biết trước khi phim ra mắt
    'budget', 'Budget_log', 'runtime',
    
    # Time - biết ngay khi lên kế hoạch
    'release_year', 'release_month', 'release_weekday',
    'release_quarter', 'is_holiday_season',
    
    # Genre - biết từ kịch bản
    'num_genres', 'genre_Action', 'genre_Comedy', ...
    
    # Country - biết từ studio
    'is_usa', 'is_vietnam', ...
    
    # Cast - biết khi casting
    'num_main_cast'
]
```

## 5.3 Tại Sao 67% Vẫn Là Kết Quả Tốt?

> "67% accuracy cho Pre-Release prediction là **hợp lý và thực tế** vì:
> 1. Thị trường điện ảnh rất **khó dự đoán** - phụ thuộc nhiều yếu tố không đo được
> 2. Không có thông tin về: chất lượng kịch bản, diễn xuất, đạo diễn...
> 3. Các studio lớn với đội ngũ chuyên gia vẫn có nhiều phim thất bại
> 
> Mô hình này có ý nghĩa thực tiễn: giúp đánh giá rủi ro **trước khi** đầu tư."

---

# 6. ĐÁNH GIÁ MÔ HÌNH

## 6.1 Các Metrics Đánh Giá

### A. Confusion Matrix

```
                    Predicted
                 Fail    Success
         ┌─────────┬─────────┐
Fail     │   TN    │   FP    │
Actual   ├─────────┼─────────┤
Success  │   FN    │   TP    │
         └─────────┴─────────┘
```

| Ký hiệu | Tên đầy đủ | Ý nghĩa |
|---------|------------|---------|
| TP | True Positive | Dự đoán Success, thực tế Success ✅ |
| TN | True Negative | Dự đoán Fail, thực tế Fail ✅ |
| FP | False Positive | Dự đoán Success, thực tế Fail ❌ |
| FN | False Negative | Dự đoán Fail, thực tế Success ❌ |

### B. Các Công Thức Metrics

| Metric | Công thức | Ý nghĩa |
|--------|-----------|---------|
| **Accuracy** | (TP + TN) / (TP + TN + FP + FN) | Tỷ lệ dự đoán đúng tổng thể |
| **Precision** | TP / (TP + FP) | Trong các dự đoán Success, bao nhiêu đúng? |
| **Recall** | TP / (TP + FN) | Trong các phim Success thật, bao nhiêu được phát hiện? |
| **F1-Score** | 2 × (P × R) / (P + R) | Trung bình điều hòa của Precision và Recall |

### Giải thích dễ hiểu:
- **Precision cao**: Khi model nói "thành công", khả năng cao là đúng → Ít đầu tư nhầm
- **Recall cao**: Model phát hiện được nhiều phim thành công → Không bỏ lỡ cơ hội
- **F1 cao**: Cân bằng giữa Precision và Recall

## 6.2 Kết Quả Pre-Release Model

| Metric | Giá trị | Đánh giá |
|--------|---------|----------|
| Accuracy | 67.65% | Khá (thực tế cho bài toán khó) |
| Precision | 68.27% | Khá |
| Recall | 67.96% | Khá |
| F1-Score | 67.96% | Khá |
| CV Mean | 69.31% | Ổn định qua 5 folds |

## 6.3 Cross-Validation

### Tại sao cần Cross-Validation?
> "Cross-Validation đảm bảo model **ổn định** và **không phụ thuộc** vào cách chia train/test cụ thể."

### Quy trình 5-Fold CV:
```
Data: [████████████████████████████████████████]

Fold 1: [Test][████████████████████████████████]
Fold 2: [████][Test][████████████████████████]
Fold 3: [████████][Test][████████████████████]
Fold 4: [████████████████][Test][████████████]
Fold 5: [████████████████████████████████][Test]

Final Score = Mean(Fold1, Fold2, Fold3, Fold4, Fold5)
```

**Kết quả**: 69.31% ± 2.14%
> "Độ lệch chuẩn thấp (2.14%) cho thấy model ổn định."

## 6.4 Feature Importance

### Top 10 Features Quan Trọng Nhất:

| Rank | Feature | Importance | Giải thích |
|------|---------|------------|------------|
| 1 | `Budget_log` | ~20% | Ngân sách là yếu tố quyết định |
| 2 | `budget` | ~15% | Ngân sách gốc |
| 3 | `release_month` | ~8% | Thời điểm phát hành quan trọng |
| 4 | `num_genres` | ~6% | Đa dạng thể loại |
| 5 | `is_usa` | ~5% | Phim Mỹ có lợi thế |
| 6 | `release_year` | ~5% | Xu hướng theo thời gian |
| 7 | `runtime` | ~4% | Thời lượng phù hợp |
| 8 | `num_main_cast` | ~4% | Số sao tham gia |
| 9 | `genre_Action` | ~3% | Thể loại Action phổ biến |
| 10 | `is_holiday_season` | ~3% | Mùa phát hành |

### Insight từ Feature Importance:
1. **Budget là quan trọng nhất** (~35% tổng) → Phim budget cao có nhiều nguồn lực hơn
2. **Thời điểm phát hành quan trọng** → Tháng hè và lễ có lợi thế
3. **Thể loại ảnh hưởng** → Action, Adventure thường thành công hơn

---

# 7. WEB APPLICATION

## 7.1 Kiến Trúc

```
┌─────────────────────────────────────────────────────┐
│                   FRONTEND                          │
│  ┌──────────────────────────────────────────────┐  │
│  │  HTML/CSS/JavaScript (index.html, app.js)    │  │
│  │  - Form nhập thông tin phim                  │  │
│  │  - Chart.js visualization                    │  │
│  │  - Dark Sci-Fi Theme                         │  │
│  └──────────────────────────────────────────────┘  │
│                        │                            │
│                        ▼ API Call (JSON)            │
│  ┌──────────────────────────────────────────────┐  │
│  │               BACKEND (Flask)                 │  │
│  │  - app.py: Route handlers                    │  │
│  │  - pre_release_service.py: ML Service        │  │
│  └──────────────────────────────────────────────┘  │
│                        │                            │
│                        ▼ Load Model                 │
│  ┌──────────────────────────────────────────────┐  │
│  │          ML MODEL (pkl file)                  │  │
│  │  - pre_release_rf_model.pkl                  │  │
│  │  - Scaler, Feature names                     │  │
│  └──────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────┘
```

## 7.2 API Endpoints & Backend Logic

### A. Các Endpoints Chính
| Endpoint | Method | Mô tả | Chi tiết xử lý |
|----------|--------|-------|----------------|
| `/` | GET | Trang chủ | Trả về giao diện `index.html` và độ chính xác mô hình. |
| `/predict` | POST | Dự đoán | Nhận JSON, gọi `PreReleaseService` để xử lý và trả về kết quả. |
| `/api/model-info` | GET | Thông tin model | Trả về metadata: loại model, số lượng features, trạng thái load. |
| `/api/sample-data` | GET | Dữ liệu mẫu | Cung cấp các bộ dữ liệu test nhanh (Action, Indie, Comedy...). |

### B. Cấu Trúc Backend (Flask)
Backend được thiết kế theo mô hình **Service-Oriented**:
- **`app.py`**: Đóng vai trò Controller, tiếp nhận Request, điều phối dữ liệu và trả về Response JSON.
- **`pre_release_service.py`**: Đóng vai trò Logic Layer, chứa toàn bộ quy trình xử lý dữ liệu học máy.

## 7.3 Quy Trình Xử Lý Tại Backend (ML Pipeline)

Khi nhận được một yêu cầu dự đoán, Backend thực hiện 5 bước nghiêm ngặt:

1. **Validation (Xác thực)**:
   - Kiểm tra dữ liệu đầu vào (phải có `title`, `budget`).
   - Gán giá trị mặc định cho các trường thiếu (`runtime=120`, `releaseMonth=tháng hiện tại`).

2. **Feature Preparation (Tiền xử lý đặc trưng)**:
   - **Log Transformation**: Chuyển `budget` sang `Budget_log` (cơ số 10) để giảm độ lệch dữ liệu.
   - **Time Features**: Trích xuất `release_quarter`, `is_holiday_season` (tháng 6,7,11,12).
   - **One-Hot Encoding**: Chuyển danh sách `genres` (ví dụ: ['Action', 'Comedy']) thành các cột binary (0/1) tương ứng với 37 đặc trưng mà model đã học.
   - **Interaction Features**: Tạo `cast_genre_interaction` (số diễn viên × số thể loại).

3. **Feature Scaling (Chuẩn hóa)**:
   - Sử dụng `StandardScaler` đã được lưu từ quá trình training.
   - Đưa tất cả giá trị về cùng một phân phối (mean=0, std=1) để model Random Forest hoạt động ổn định nhất.

4. **Inference (Dự đoán)**:
   - Gọi `model.predict()` để lấy nhãn (0: Thất bại, 1: Thành công).
   - Gọi `model.predict_proba()` để lấy xác suất chính xác (ví dụ: 0.82).

5. **Post-processing (Hậu xử lý)**:
   - Tính toán **Risk Level**: Dựa trên xác suất (Prob < 0.4: HIGH, 0.4-0.6: MEDIUM, > 0.6: LOW).
   - Trích xuất **Feature Importance**: Lấy top các yếu tố ảnh hưởng nhất đến kết quả của bộ phim cụ thể đó.

## 7.4 Chi Tiết ML Service (`pre_release_service.py`)

Đây là "trái tim" của Backend, đảm bảo tính nhất quán giữa môi trường Training và Production.

### A. Cấu trúc Lớp `PreReleaseMoviePredictionService`
Lớp này quản lý vòng đời của mô hình và thực hiện các phép biến đổi dữ liệu:
- **`__init__`**: Khởi tạo các tham số mặc định và gọi hàm load model.
- **`_load_model`**: Sử dụng `pickle` để nạp file `pre_release_rf_model.pkl`. File này thực chất là một `dict` chứa:
    - `model`: Đối tượng `RandomForestClassifier` đã train.
    - `scaler`: Đối tượng `StandardScaler` dùng để chuẩn hóa.
    - `feature_names`: Danh sách tên cột theo đúng thứ tự (37 đặc trưng).
    - `metrics`: Các chỉ số Accuracy, F1-score từ lúc train.

### B. Logic Xử Lý Đặc Trưng (Feature Engineering Logic)
Để tránh **Data Leakage**, Service chỉ thực hiện các phép toán trên dữ liệu "biết trước":
- **Xử lý Budget**: Áp dụng `np.log10(budget + 1)` để nén dải giá trị ngân sách (từ vài nghìn đến hàng trăm triệu USD) về một khoảng nhỏ hơn, giúp mô hình học dễ hơn.
- **Xử lý Thể loại (Genres)**: 
    - Sử dụng một `genre_mapping` cố định để ánh xạ từ tên thể loại người dùng chọn sang tên cột trong model (ví dụ: "Science Fiction" -> `genre_Science Fiction`).
    - Tính toán `num_genres` (tổng số thể loại) làm một đặc trưng định lượng.
- **Xử lý Thời gian**: 
    - Chuyển đổi tháng sang Quý (`release_quarter`).
    - Xác định `is_holiday_season`: Trả về 1 nếu tháng thuộc [6, 7, 11, 12], ngược lại là 0.
- **Xử lý Quốc gia**: Mặc định ưu tiên `is_usa` và `is_united_states_of_america` nếu không có thông tin, vì đây là các thị trường lớn nhất trong dataset.

### C. Quy trình Dự đoán & Trả về (Inference & Response)
```python
# Mã giả logic trong service
feature_vector = [features[name] for name in self.feature_names] # Đảm bảo đúng thứ tự
scaled_vector = self.scaler.transform([feature_vector])
prob = self.model.predict_proba(scaled_vector)[0][1] # Lấy xác suất lớp 1 (Success)
```
Kết quả trả về cho Frontend không chỉ là con số, mà là một gói dữ liệu đầy đủ:
- **`confidence`**: Xác suất thành công (%).
- **`risk_level`**: Phân loại rủi ro dựa trên ngưỡng (Threshold).
- **`feature_importance`**: Danh sách các đặc trưng đóng góp nhiều nhất vào quyết định của cây (lấy từ `model.feature_importances_`).


# 8. THUẬT NGỮ CẦN NHỚ

## 8.1 Machine Learning

| Thuật ngữ | Định nghĩa | Ví dụ/Công thức |
|-----------|------------|-----------------|
| **Accuracy** | Tỷ lệ dự đoán đúng | (TP+TN) / Total |
| **Precision** | Độ chính xác dự đoán positive | TP / (TP+FP) |
| **Recall** | Độ nhạy (tỷ lệ phát hiện) | TP / (TP+FN) |
| **F1-Score** | Trung bình điều hòa P và R | 2×(P×R)/(P+R) |
| **Overfitting** | Model học thuộc training data | Train 99%, Test 60% |
| **Cross-Validation** | Kiểm định chéo | 5-Fold CV |
| **Feature Engineering** | Tạo features mới từ data gốc | log(budget), is_holiday |
| **One-Hot Encoding** | Chuyển categorical → binary | Action → [1,0,0] |
| **Data Leakage** | Sử dụng thông tin từ tương lai | Dùng revenue để dự đoán |
| **Ensemble Learning** | Kết hợp nhiều models | Random Forest |
| **Stratified Sampling** | Giữ tỷ lệ class khi chia | 50/50 ở cả train và test |

## 8.2 Domain Phim Ảnh

| Thuật ngữ | Định nghĩa |
|-----------|------------|
| **Budget** | Ngân sách sản xuất |
| **Revenue** | Doanh thu phòng vé |
| **ROI** | Return on Investment = Revenue / Budget |
| **Box Office** | Doanh thu phòng vé |
| **Vote Average** | Điểm đánh giá trung bình từ khán giả |
| **Genre** | Thể loại phim |
| **Runtime** | Thời lượng phim (phút) |
| **Holiday Season** | Mùa lễ hội (tháng 6,7,11,12) |

## 8.3 Data Science

| Thuật ngữ | Định nghĩa |
|-----------|------------|
| **CRISP-DM** | Cross-Industry Standard Process for Data Mining |
| **EDA** | Exploratory Data Analysis - Phân tích thăm dò |
| **ETL** | Extract, Transform, Load |
| **Missing Values** | Giá trị bị thiếu |
| **Outliers** | Giá trị ngoại lai |
| **Normalization** | Chuẩn hóa dữ liệu |
| **StandardScaler** | Chuẩn hóa về mean=0, std=1 |
| **MinMaxScaler** | Chuẩn hóa về [0,1] |

---

# 9. CÂU HỎI GV CÓ THỂ HỎI & CÁCH TRẢ LỜI

## 9.1 Về Định Nghĩa Thành Công

**Q: "Tại sao định nghĩa thành công lại dùng cả ROI và Vote?"**

> A: "Dạ thưa thầy/cô, chúng em định nghĩa thành công dựa trên CẢ HAI tiêu chí vì:
> 
> 1. **ROI ≥ 1.0** đảm bảo phim **sinh lời về tài chính** - thu hồi được vốn đầu tư
> 2. **Vote Average ≥ 6.5** đảm bảo phim **được khán giả đón nhận** - chất lượng tốt
> 
> Nếu chỉ dùng ROI, phim có thể thành công do marketing mạnh nhưng chất lượng kém, không bền vững. Nếu chỉ dùng Vote, phim được đánh giá cao nhưng không sinh lời thì studio vẫn thua lỗ.
> 
> Kết hợp cả hai đảm bảo phim **vừa có lợi nhuận vừa được yêu thích**."

---

## 9.2 Về Data Leakage

**Q: "Tại sao model cũ 99% mà không dùng?"**

> A: "Dạ thưa thầy/cô, model cũ có **Accuracy 99.51%** nhưng **không thể sử dụng** trong thực tế vì bị **Data Leakage**.
> 
> Model cũ sử dụng các features như `revenue`, `vote_average` - đây là thông tin chỉ biết **SAU KHI** phim đã chiếu. Trong thực tế, khi nhà sản xuất muốn đánh giá rủi ro **TRƯỚC KHI** đầu tư, họ không có những thông tin này.
> 
> Vì vậy, accuracy 99% là **giả tạo** - model đang 'gian lận' bằng cách nhìn vào tương lai.
> 
> Model Pre-Release của chúng em đạt **67.65%** - con số này **thực tế và có ý nghĩa** vì chỉ sử dụng thông tin biết trước như budget, thể loại, thời điểm phát hành."

---

## 9.3 Về Accuracy

**Q: "Accuracy 67% có thấp quá không?"**

> A: "Dạ thưa thầy/cô, con số 67% cho Pre-Release prediction là **hợp lý và thực tế** vì:
> 
> 1. **Thị trường điện ảnh rất khó dự đoán** - ngay cả các studio lớn với đội ngũ chuyên gia vẫn có nhiều phim thất bại
> 
> 2. **Thiếu nhiều yếu tố quan trọng không đo được** như: chất lượng kịch bản, diễn xuất diễn viên, tài năng đạo diễn, chiến lược marketing...
> 
> 3. **So sánh với baseline**: Nếu đoán ngẫu nhiên chỉ được 50%. Model của chúng em cải thiện **17 điểm phần trăm** so với baseline.
> 
> 4. **Có ý nghĩa thực tiễn**: Giúp nhà sản xuất có thêm một công cụ đánh giá rủi ro trước khi quyết định đầu tư."

---

## 9.4 Về Random Forest

**Q: "Random Forest là gì? Tại sao chọn?"**

> A: "Dạ thưa thầy/cô, **Random Forest** là thuật toán **Ensemble Learning** kết hợp nhiều Decision Trees.
> 
> **Cách hoạt động:**
> - Tạo nhiều cây quyết định (100 cây trong model của em)
> - Mỗi cây train trên một subset ngẫu nhiên của data và features
> - Kết quả cuối cùng là **voting** từ tất cả các cây
> 
> **Lý do chọn Random Forest:**
> 1. **Xử lý tốt dữ liệu phi tuyến** - không cần giả định linear
> 2. **Ít bị overfitting** hơn Decision Tree đơn lẻ
> 3. **Có Feature Importance** - biết feature nào quan trọng
> 4. **Robust với outliers và missing values**
> 5. **Không cần nhiều hyperparameter tuning**"

---

## 9.5 Về Feature Engineering

**Q: "Feature Engineering là gì? Cho ví dụ?"**

> A: "Dạ thưa thầy/cô, **Feature Engineering** là quá trình **tạo ra features mới** từ dữ liệu gốc để giúp model học tốt hơn.
> 
> **Ví dụ trong dự án:**
> 
> 1. **Log Transformation:**
>    - Từ `budget` → tạo `Budget_log = log10(budget + 1)`
>    - Mục đích: Chuẩn hóa phân phối lệch của budget
> 
> 2. **Time Features:**
>    - Từ `release_date` → tạo `release_month`, `release_quarter`, `is_holiday_season`
>    - Mục đích: Capture xu hướng theo thời gian
> 
> 3. **One-Hot Encoding:**
>    - Từ `genres = ['Action', 'Comedy']` → `genre_Action=1, genre_Comedy=1, genre_Drama=0`
>    - Mục đích: Chuyển categorical thành numerical
> 
> 4. **Feature Interaction:**
>    - `cast_genre_interaction = num_cast × num_genres`
>    - Mục đích: Capture mối quan hệ giữa các features"

---

## 9.6 Về Tham Số Đầu Vào

**Q: "Các tham số đầu vào ảnh hưởng gì đến mô hình?"**

> A: "Dạ thưa thầy/cô, mô hình sử dụng **37 Pre-Release features**, trong đó quan trọng nhất là:
> 
> | Feature | Importance | Ảnh hưởng |
> |---------|------------|-----------|
> | Budget | ~35% | Phim budget cao thường có nguồn lực marketing và sản xuất tốt hơn |
> | Release timing | ~10% | Tháng 6,7,11,12 là thời điểm vàng (hè, holiday) |
> | Genre | ~10% | Action, Adventure thường có tỷ lệ thành công cao |
> | Runtime | ~5% | Thời lượng phù hợp (90-150 phút) được khán giả ưa thích |
> | Country | ~5% | Phim Mỹ có lợi thế thị trường quốc tế |
> 
> **Về tham số của Random Forest:**
> - `n_estimators=100`: 100 cây để đảm bảo ổn định
> - `max_depth=10`: Giới hạn độ sâu để tránh overfitting
> - `class_weight='balanced'`: Xử lý class imbalance"

---

## 9.7 Về Dữ Liệu Test

**Q: "Dữ liệu test mô hình như thế nào?"**

> A: "Dạ thưa thầy/cô, quy trình test như sau:
> 
> | Thông số | Giá trị |
> |----------|---------|
> | Tổng dataset | 1,020 phim đã làm sạch |
> | Train set | 816 phim (80%) |
> | Test set | 204 phim (20%) |
> | Phương pháp chia | Stratified Sampling (giữ tỷ lệ class) |
> | Cross-Validation | 5-Fold CV |
> 
> **Kết quả trên Test Set:**
> - Accuracy: 67.65%
> - F1-Score: 67.96%
> - CV Mean: 69.31% ± 2.14%
> 
> Dataset bao gồm cả phim Việt Nam như 'Mai', 'Nhà Bà Nữ', 'Bố Già' và phim quốc tế như 'Titanic', 'The Matrix', 'Avengers'..."

---

## 9.8 Về Tài Liệu Tham Khảo

**Q: "Tài liệu tham khảo dự án?"**

> A: "Dạ thưa thầy/cô, dự án sử dụng các nguồn tham khảo:
> 
> 1. **Dataset**: TMDB Movies Dataset từ Kaggle (The Movie Database)
> 2. **Machine Learning**: Scikit-Learn Documentation (https://scikit-learn.org)
> 3. **Web Framework**: Flask Documentation (https://flask.palletsprojects.com)
> 4. **Methodology**: CRISP-DM - Cross-Industry Standard Process for Data Mining
> 5. **Visualization**: Chart.js (https://chartjs.org)
> 
> Nếu thầy/cô yêu cầu thêm tài liệu học thuật, chúng em sẽ bổ sung các bài báo về Movie Box Office Prediction vào báo cáo."

---

## 9.9 Về Việt Hóa

**Q: "Chuyển thể loại, tên sang Tiếng Việt?"**

> A: "Dạ thưa thầy/cô, chúng em đã chuẩn bị sẵn dictionary mapping Tiếng Anh → Tiếng Việt trong code:
> 
> | English | Tiếng Việt |
> |---------|------------|
> | Action | Hành Động |
> | Adventure | Phiêu Lưu |
> | Comedy | Hài Hước |
> | Drama | Chính Kịch |
> | Horror | Kinh Dị |
> | Romance | Lãng Mạn |
> | Thriller | Giật Gân |
> | Sci-Fi | Khoa Học Viễn Tưởng |
> 
> Em có thể thêm code Việt hóa ngay nếu thầy/cô yêu cầu ạ."

---

## 9.10 Về Kỹ Thuật & Chỉnh Sửa Website (LIVE DEMO)

> ⭐ **DÀNH CHO TRƯỜNG HỢP GV YÊU CẦU SỬA CODE TRỰC TIẾP**

**Q1: "Làm sao để đổi tên Website hoặc Logo?"**
- **File**: [webs/MoviePredict/templates/index.html](webs/MoviePredict/templates/index.html)
- **Cách làm**: Tìm thẻ `<title>` (dòng 7) để đổi tên tab trình duyệt, và thẻ `<span class="logo-text">` (dòng 48) để đổi tên hiển thị trên thanh điều hướng.

**Q2: "Làm sao để thêm một thể loại phim mới (ví dụ: 'Documentary') vào danh sách chọn?"**
- **Bước 1 (Frontend)**: Mở [webs/MoviePredict/static/js/app.js](webs/MoviePredict/static/js/app.js), tìm hàm `renderGenreChips` và thêm 'Documentary' vào mảng `genres`.
- **Bước 2 (Backend)**: Mở [webs/MoviePredict/models/pre_release_service.py](webs/MoviePredict/models/pre_release_service.py), thêm 'Documentary' vào `genre_mapping` để model biết cách xử lý đặc trưng này.
- **Lưu ý**: Nếu model chưa được train với thể loại này, nó sẽ mặc định coi là 0.

**Q3: "Làm sao để thay đổi màu sắc của biểu đồ Gauge (xác suất)?"**
- **File**: [webs/MoviePredict/static/js/app.js](webs/MoviePredict/static/js/app.js)
- **Cách làm**: Tìm hàm `renderGauge`. Chỉnh sửa biến `color` trong các điều kiện `if (prob >= 0.6)` (màu xanh) hoặc `else` (màu đỏ/vàng).

**Q4: "Nếu tôi muốn thay đổi ngưỡng thành công (ví dụ: ROI phải ≥ 2.0 mới là thành công), tôi phải sửa ở đâu?"**
- **File**: [progress/week03/crea_label.ipynb](progress/week03/crea_label.ipynb) hoặc file script tạo label.
- **Cách làm**: Sửa dòng code định nghĩa nhãn: `df['success'] = ((df['roi'] >= 2.0) & (df['vote_average'] >= 6.5)).astype(int)`. Sau đó phải chạy lại toàn bộ quy trình training để model học theo tiêu chuẩn mới.

**Q5: "Làm sao để thay đổi giao diện từ Dark Mode sang Light Mode?"**
- **File**: [webs/MoviePredict/static/css/styles.css](webs/MoviePredict/static/css/styles.css)
- **Cách làm**: Chỉnh sửa các biến CSS ở đầu file (`:root`). Thay đổi `--bg-dark` từ màu đen sang màu trắng và `--text-main` từ trắng sang đen.

**Q6: "Làm sao để cập nhật Model mới sau khi đã train lại?"**
- **File**: [webs/MoviePredict/models/pre_release_service.py](webs/MoviePredict/models/pre_release_service.py)
- **Cách làm**: Chỉ cần copy file `.pkl` mới vào thư mục `data/pkl/` và đảm bảo tên file khớp với khai báo trong hàm `_load_model`. Server Flask sẽ tự động load model mới khi khởi động lại.

---

# 10. TÀI LIỆU THAM KHẢO

## 10.1 Dataset & API

| Nguồn | Link | Mô tả |
|-------|------|-------|
| TMDB API | https://www.themoviedb.org/ | Nguồn dữ liệu chính |
| Kaggle TMDB | https://www.kaggle.com/datasets/ | Dataset movies |

## 10.2 Thư Viện Python

| Thư viện | Phiên bản | Công dụng |
|----------|-----------|-----------|
| pandas | 2.0+ | Xử lý dữ liệu |
| numpy | 1.24+ | Tính toán số học |
| scikit-learn | 1.3+ | Machine Learning |
| flask | 2.0+ | Web Framework |

## 10.3 Tài Liệu Học Thuật (Có thể bổ sung)

1. "Predicting Movie Success with Machine Learning" - Various papers
2. "Box Office Revenue Prediction Using Deep Learning"
3. "Feature Engineering for Movie Rating Prediction"

---

# 📝 CHECKLIST TRƯỚC KHI THI

- [ ] Thuộc **định nghĩa thành công**: ROI ≥ 1.0 VÀ Vote ≥ 6.5
- [ ] Hiểu **Data Leakage** và cách giải quyết
- [ ] Nhớ **Top 5 features quan trọng**: budget, release_month, num_genres, is_usa, genre_Action
- [ ] Biết **số liệu**: 1,020 phim, 37 features, 67.65% accuracy
- [ ] Hiểu **Random Forest**: ensemble của nhiều Decision Trees
- [ ] Nhớ **công thức**: Accuracy, Precision, Recall, F1
- [ ] Biết cách **chạy web**: `python app.py` → localhost:8000

---

