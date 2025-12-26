# 📚 TUYỂN TẬP CÂU HỎI & ĐÁP ÁN BẢO VỆ ĐỒ ÁN (FULL VERSION)
## DỰ ĐOÁN ĐỘ THÀNH CÔNG CỦA PHIM (MOVIE SUCCESS PREDICTION)

Tài liệu này bao gồm tất cả các câu hỏi có thể gặp, từ mức độ cơ bản đến chuyên sâu, kèm theo hướng dẫn sửa code trực tiếp (Live Coding).

---

# 🟢 PHẦN 1: TỔNG QUAN & KHÁI NIỆM CƠ BẢN

### Q1: Mục tiêu của đồ án này là gì?
- **Trả lời**: Xây dựng hệ thống dự đoán khả năng thành công của phim dựa trên thông tin **biết trước khi phát hành** (Pre-release), giúp nhà sản xuất đánh giá rủi ro đầu tư.

### Q2: "Thành công" được định nghĩa như thế nào?
- **Trả lời**: Dựa trên 2 tiêu chí đồng thời:
    1.  **Tài chính**: ROI (Return on Investment) ≥ 1.0 (Hòa vốn hoặc có lãi).
    2.  **Chất lượng**: Vote Average ≥ 6.5 (Được khán giả đón nhận).
- **Tại sao?**: Tránh trường hợp phim lãi cao nhưng chất lượng tệ (thành công ngắn hạn) hoặc phim hay nhưng lỗ vốn.

### Q3: Tại sao chọn Random Forest mà không phải thuật toán khác?
- **Trả lời**:
    - **So với Linear/Logistic Regression**: Random Forest bắt được các mối quan hệ phi tuyến (non-linear) phức tạp giữa các đặc trưng.
    - **So với Deep Learning**: Dữ liệu dạng bảng (tabular) với kích thước nhỏ (~1000 mẫu) thì Random Forest thường hiệu quả hơn, ít bị overfitting hơn và dễ giải thích (Feature Importance).
    - **Tính ổn định**: Là thuật toán Ensemble (kết hợp nhiều cây), giảm thiểu phương sai (variance).

---

# 🟡 PHẦN 2: DỮ LIỆU & TIỀN XỬ LÝ (DATA PREPROCESSING)

### Q4: Data Leakage là gì? Em xử lý nó ra sao?
- **Trả lời**: Data Leakage là dùng thông tin từ tương lai để dự đoán quá khứ.
- **Cụ thể**: Các cột `revenue`, `vote_average`, `vote_count` chỉ có SAU khi phim chiếu. Nếu dùng để train, model sẽ "nhìn thấy đáp án".
- **Xử lý**: Loại bỏ hoàn toàn các cột này khỏi tập features (`X`), chỉ giữ lại trong tập target (`y`) để gán nhãn lúc đầu.

### Q5: Tại sao dùng `StandardScaler` mà không phải `MinMaxScaler`?
- **Trả lời**: 
    - `StandardScaler` đưa dữ liệu về phân phối chuẩn (mean=0, std=1).
    - Random Forest thực tế không quá nhạy cảm với scaling như SVM hay KNN, nhưng việc scaling giúp hội tụ tốt hơn và đồng nhất dữ liệu.
    - `MinMaxScaler` (0-1) dễ bị ảnh hưởng bởi outliers (giá trị ngoại lai) hơn `StandardScaler`.

### Q6: Em xử lý dữ liệu thiếu (Missing Values) như thế nào?
- **Trả lời**: (Dựa trên `cleandata.py`)
    - **Cột số (Numeric)**: Điền bằng giá trị trung bình (`mean`).
    - **Cột phân loại (Categorical)**: Điền bằng chuỗi `'Unknown'`.
    - **Dòng lỗi**: Loại bỏ các dòng có `Budget` hoặc `Revenue` bằng 0 vì đây là dữ liệu rác đối với bài toán tài chính.

### Q7: Tại sao chia Train/Test theo tỷ lệ 80/20 và dùng Stratified Sampling?
- **Trả lời**:
    - **80/20**: Tỷ lệ chuẩn để đảm bảo đủ dữ liệu cho model học (80%) và đủ để đánh giá khách quan (20%).
    - **Stratified Sampling**: Đảm bảo tỷ lệ nhãn Success/Fail trong tập Train và Test là **giống nhau** (ví dụ: đều là 40% Success). Nếu chia ngẫu nhiên (Random), có thể tập Test toàn phim Fail -> đánh giá sai lệch.

---

# 🟠 PHẦN 3: FEATURE ENGINEERING (TẠO ĐẶC TRƯNG)

### Q8: Tại sao lại `log10(budget + 1)`?
- **Trả lời**:
    - **Vấn đề**: Budget trải rộng từ vài nghìn $ đến 300 triệu $. Phân phối bị lệch phải (right-skewed).
    - **Giải pháp**: Log transform giúp nén dải giá trị lại, phân phối gần chuẩn hơn.
    - **Tại sao +1**: Để tránh lỗi toán học `log(0)` nếu có phim budget = 0 (dù đã lọc, nhưng an toàn vẫn hơn).

### Q9: One-Hot Encoding là gì? Tại sao dùng cho Genres?
- **Trả lời**:
    - Model không hiểu chữ "Action", "Comedy".
    - One-Hot biến mỗi thể loại thành 1 cột riêng (0/1). Ví dụ: Phim vừa Action vừa Comedy sẽ có `genre_Action=1` và `genre_Comedy=1`.

### Q10: Em có tạo ra feature nào mới (Derived Features) không?
- **Trả lời**: Có, ví dụ:
    - `release_quarter`: Quý phát hành (từ tháng).
    - `is_holiday_season`: Mùa lễ hội (Tháng 6, 7, 11, 12).
    - `cast_genre_interaction`: Tương tác giữa số lượng diễn viên và số lượng thể loại.

---

# 🔴 PHẦN 4: HUẤN LUYỆN & ĐÁNH GIÁ MODEL

### Q11: Giải thích các tham số của Random Forest mà em đã tinh chỉnh?
- **Trả lời**:
    - `n_estimators`: Số lượng cây (chọn 100-200). Càng nhiều càng ổn định nhưng chậm.
    - `max_depth`: Độ sâu tối đa của cây (chọn 10). Giới hạn để tránh Overfitting (học vẹt).
    - `min_samples_split`: Số mẫu tối thiểu để tách nút (chọn 5).
    - `class_weight='balanced'`: Tự động tăng trọng số cho lớp thiểu số (ít mẫu hơn) để model không thiên vị lớp đa số.

### Q12: Kết quả Accuracy 67% có ý nghĩa gì?
- **Trả lời**:
    - Baseline (đoán mò) là 50%. Model cải thiện 17%.
    - Trong ngành điện ảnh đầy rủi ro, việc dự đoán đúng 2/3 số phim là công cụ hỗ trợ ra quyết định cực kỳ giá trị.
    - Model không thay thế con người, mà giúp lọc bớt các dự án rủi ro cao (High Risk).

### Q13: Làm sao em biết model không bị Overfitting?
- **Trả lời**:
    - So sánh Accuracy trên tập Train và tập Test.
    - Nếu Train = 99% mà Test = 60% -> Overfitting nặng.
    - Model của em: Train ~75%, Test ~67% -> Gap nhỏ (<10%), chấp nhận được.
    - Ngoài ra em dùng Cross-Validation 5-Fold để kiểm chứng độ ổn định.

---

# 🟣 PHẦN 5: WEB APP & HỆ THỐNG (FLASK)

### Q14: Luồng dữ liệu (Data Flow) khi chạy Web?
- **Trả lời**:
    1.  **Client**: Người dùng nhập form -> JS đóng gói JSON -> Gửi POST `/predict`.
    2.  **Server (Flask)**: Nhận JSON -> Validate dữ liệu.
    3.  **Service Layer**:
        -   Load model `.pkl` (chỉ load 1 lần khi khởi động app).
        -   Pre-process: Log budget, One-hot genres, Scale dữ liệu.
        -   Predict: Gọi `model.predict_proba()`.
    4.  **Response**: Trả về JSON gồm xác suất, nhãn, và feature importance.
    5.  **Client**: JS nhận kết quả -> Vẽ biểu đồ Gauge và Bar chart.

### Q15: Làm sao để cập nhật model mới mà không sửa code?
- **Trả lời**: Chỉ cần ghi đè file `pre_release_rf_model.pkl` mới vào thư mục `data/pkl/`. Khi restart server, code sẽ tự động load file mới nhất.

---

# 🔥 PHẦN 6: LIVE CODING & XỬ LÝ TÌNH HUỐNG (QUAN TRỌNG)

*Giảng viên có thể yêu cầu sửa code trực tiếp để kiểm tra độ hiểu.*

### 🛠️ Tình huống 1: "Thêm một trường nhập liệu 'Đạo diễn' (Director) vào Web"
*Đây là câu hỏi bẫy. Model hiện tại KHÔNG dùng tên đạo diễn để dự đoán.*
- **Trả lời**: "Thưa thầy, model hiện tại chưa được huấn luyện với feature Đạo diễn. Nếu thêm vào giao diện thì chỉ để hiển thị, không ảnh hưởng kết quả dự đoán. Nếu muốn dùng, em phải train lại model với kỹ thuật Target Encoding cho tên đạo diễn."

### 🛠️ Tình huống 2: "Sửa logic: Chỉ coi là Mùa Lễ Hội (Holiday) nếu là tháng 12"
- **File**: [webs/MoviePredict/models/pre_release_service.py](webs/MoviePredict/models/pre_release_service.py)
- **Tìm dòng**: `features['is_holiday_season'] = 1 if release_month in [6, 7, 11, 12] else 0`
- **Sửa thành**: `features['is_holiday_season'] = 1 if release_month == 12 else 0`

### 🛠️ Tình huống 3: "Thay đổi Port chạy Web từ 5000 sang 8080"
- **File**: [webs/MoviePredict/app.py](webs/MoviePredict/app.py)
- **Cuối file**: Thêm hoặc sửa đoạn `if __name__ == '__main__':`
    ```python
    if __name__ == '__main__':
        app.run(debug=True, port=8080)
    ```

### 🛠️ Tình huống 4: "Hiển thị thêm thông tin 'Độ tin cậy' (Confidence) lên tiêu đề trang Web"
- **File**: [webs/MoviePredict/static/js/app.js](webs/MoviePredict/static/js/app.js)
- **Tìm hàm**: `updateUI(data)`
- **Thêm code**:
    ```javascript
    document.title = `Dự đoán: ${data.prediction.confidence}% - MoviePredict`;
    ```

### 🛠️ Tình huống 5: "Thay đổi ngưỡng Rủi ro (Risk Threshold)"
*Hiện tại: <40% là High Risk. Muốn sửa thành <50% là High Risk.*
- **File**: [webs/MoviePredict/models/pre_release_service.py](webs/MoviePredict/models/pre_release_service.py)
- **Tìm đoạn logic xếp loại (khoảng dòng 250-270)**:
    ```python
    # Code cũ
    if prob < 0.4: risk = 'HIGH'
    
    # Code mới
    if prob < 0.5: risk = 'HIGH'
    ```

### 🛠️ Tình huống 6: "Tắt chế độ Debug của Flask"
- **File**: [webs/MoviePredict/app.py](webs/MoviePredict/app.py)
- **Dòng 22**: Sửa `app.config['DEBUG'] = True` thành `False`.

### 🛠️ Tình huống 7: "Thêm log ghi lại mỗi lần có người dự đoán"
- **File**: [webs/MoviePredict/app.py](webs/MoviePredict/app.py)
- **Trong hàm `predict()`**:
    ```python
    logger.info(f"New prediction request: Title='{data.get('title')}', Budget={data.get('budget')}")
    ```

---

# ⚡ PHẦN 7: CÂU HỎI KHÓ & PHẢN BIỆN

### Q16: Tại sao Feature Importance của 'Budget' lại cao nhất? Có phải cứ nhiều tiền là phim thành công?
- **Trả lời**: 
    - Budget cao thường đi kèm với: Marketing mạnh, Diễn viên nổi tiếng, Kỹ xảo tốt -> Dễ thu hút khán giả ban đầu.
    - Tuy nhiên, model là **phi tuyến**. Không phải cứ tăng budget là xác suất tăng mãi. Random Forest học được ngưỡng bão hòa.
    - Có những phim budget thấp nhưng thành công (Indie movies), model học được điều này qua các feature khác như Genre hay Runtime.

### Q17: Nếu tôi nhập một bộ phim Việt Nam, model có dự đoán đúng không?
- **Trả lời**: 
    - Có, nhưng độ chính xác có thể thấp hơn phim Mỹ.
    - Lý do: Dữ liệu train chủ yếu là phim Mỹ (Hollywood).
    - Tuy nhiên, em đã thêm feature `is_vietnam` (One-hot country) để model nhận biết và điều chỉnh trọng số cho phim Việt.

### Q18: Làm sao để cải thiện model lên 80%?
- **Trả lời**:
    1.  **Thêm dữ liệu**: 1000 phim là hơi ít. Cần 5000+ phim.
    2.  **Thêm feature sâu hơn**: Đạo diễn (Director track record), Diễn viên (Star power score), Kịch bản (NLP analysis tóm tắt phim).
    3.  **Dùng Deep Learning**: Nếu có đủ dữ liệu text (tóm tắt phim) và ảnh (poster), có thể dùng Multi-modal Deep Learning.

---

# 🔵 PHẦN 8: FEATURE ENGINEERING (CHI TIẾT TỪ NOTEBOOK)

### Q19: Em đã trích xuất những đặc trưng thời gian nào từ cột "Release Date"?
- **Trả lời**: Em đã tạo các features:
    - `release_year`: Năm phát hành
    - `release_month`: Tháng phát hành  
    - `release_weekday`: Ngày trong tuần (0=Monday, 6=Sunday)
    - `release_quarter`: Quý phát hành (1-4)
    - `is_holiday_season`: Mùa lễ hội (Tháng 11, 12, 1)
- **Mục đích**: Giúp model học được pattern theo thời gian - ví dụ phim ra mắt mùa hè (blockbuster) thường có doanh thu cao hơn.

### Q20: Tại sao lại nhóm runtime thành các nhóm (runtime_group)?
- **Trả lời**: 
    - Phim được chia thành 5 nhóm: `< 1 hour`, `1-1.5 hours`, `1.5-2 hours`, `2-2.5 hours`, `> 2.5 hours`
    - **Lý do**: Khán giả có xu hướng ưa thích phim có độ dài nhất định. Phim quá ngắn hoặc quá dài đều có thể ảnh hưởng tiêu cực đến trải nghiệm.
    - Phân tích dữ liệu cho thấy: Phim 1.5-2 giờ chiếm đa số (642 phim), tiếp theo là 2-2.5 giờ (255 phim).

### Q21: Giải thích cách em đếm số diễn viên chính (num_main_cast)?
- **Trả lời**: Hàm `count_cast_fixed()` xử lý nhiều trường hợp:
    1. Nếu giá trị là NaN → trả về 0
    2. Nếu là list → đếm trực tiếp các phần tử khác rỗng
    3. Nếu là string → thử parse bằng `ast.literal_eval()`, nếu không được thì tách bằng dấu phẩy, chấm phẩy, hoặc xuống dòng.
- **Ý nghĩa**: Phim có nhiều diễn viên nổi tiếng có thể ảnh hưởng đến thành công.

### Q22: Em xử lý Genres như thế nào?
- **Trả lời**: 
    - **Bước 1**: Tách chuỗi genres thành list (`genres_list`)
    - **Bước 2**: Đếm số lượng genres (`num_genres`)
    - **Bước 3**: Lấy top 15 genres phổ biến nhất
    - **Bước 4**: One-Hot Encoding cho từng genre (ví dụ: `genre_Action`, `genre_Comedy`...)
    - **Bước 5**: Xác định main_genre (genre chính) dựa trên độ phổ biến
- **Top genres**: Action, Adventure, Comedy, Drama, Thriller...

### Q23: Em đã tạo những features kết hợp (derived features) nào?
- **Trả lời**:
    - `budget_per_year`: Budget chia cho số năm kể từ năm phát hành - giúp điều chỉnh lạm phát
    - `roi_vs_vote`: ROI nhân với Vote Average / 10 - kết hợp yếu tố tài chính và chất lượng
    - `cast_genre_interaction`: Số diễn viên × Số genres - capture tương tác phức tạp
- **Lưu ý**: `budget_per_year` và `roi_vs_vote` là POST-RELEASE features, không dùng cho dự đoán Pre-Release!

### Q24: Em xử lý Production Countries như thế nào?
- **Trả lời**:
    - Lấy quốc gia đầu tiên trong danh sách
    - Top 10 quốc gia phổ biến: USA, UK, Canada, Vietnam, China, France, South Korea, Australia, Japan, India
    - Tạo flag `is_usa` cho phim Mỹ (chiếm ~50% dataset)
    - Các quốc gia khác gộp vào nhóm "Other"

### Q25: Tổng cộng em đã tạo được bao nhiêu features?
- **Trả lời**: 46 features, bao gồm:
    - 5 time features
    - 3 runtime features  
    - 2 cast features
    - 16 genre-related features (15 one-hot + num_genres)
    - 11 country-related features
    - 6 numeric features (Budget, Revenue, ROI...)
    - 3 derived features

---

# 🟤 PHẦN 9: QUÁ TRÌNH HUẤN LUYỆN LẠI MODEL (RETRAIN.PY)

### Q26: Tại sao cần phân biệt Pre-Release và Post-Release features?
- **Trả lời**: 
    - **Pre-Release**: Thông tin biết TRƯỚC khi phim chiếu (Budget, Runtime, Genres, Release Date...) → **Dùng để dự đoán**
    - **Post-Release**: Thông tin chỉ có SAU khi phim chiếu (Revenue, Vote Average, Vote Count, ROI...) → **Gây Data Leakage**
- **Data Leakage**: Nếu dùng revenue để dự đoán thành công, model sẽ "nhìn thấy đáp án" → độ chính xác cao giả tạo!

### Q27: Em đã loại bỏ những features nào để tránh Data Leakage?
- **Trả lời** (Theo code `POST_RELEASE_FEATURES`):
    - `revenue`, `Revenue_log`
    - `vote_average`, `Vote Average`, `vote_count`
    - `roi`, `roi_clipped`, `roi_vs_vote`, `budget_per_year`
    - `success` (đây là label, không phải feature)

### Q28: Giải thích cấu hình Random Forest trong code retrain.py?
- **Trả lời**:
```python
RandomForestClassifier(
    n_estimators=100,      # 100 cây quyết định
    max_depth=10,          # Độ sâu tối đa = 10 (tránh overfitting)
    min_samples_split=5,   # Cần ít nhất 5 mẫu để tách nút
    min_samples_leaf=2,    # Mỗi lá cần ít nhất 2 mẫu
    random_state=42,       # Seed để reproducible
    n_jobs=-1,             # Sử dụng tất cả CPU cores
    class_weight='balanced' # Tự động cân bằng class
)
```
- **class_weight='balanced'**: Rất quan trọng! Tự động tăng trọng số cho class thiểu số.

#### 📖 **Giải thích chi tiết từng tham số:**

| Tham số | Giá trị | Ý nghĩa |
|---------|---------|---------|
| `n_estimators` | 100 | Tạo 100 cây quyết định độc lập, sau đó "vote đa số" để đưa ra kết quả |
| `max_depth` | 10 | Mỗi cây chỉ được phép có tối đa 10 level (từ root đến leaf) |
| `min_samples_split` | 5 | Một nút cần ít nhất 5 samples mới được phép chia tiếp |
| `min_samples_leaf` | 2 | Mỗi leaf node phải chứa ít nhất 2 samples |
| `random_state` | 42 | Cố định random seed để kết quả lặp lại được |
| `n_jobs` | -1 | Sử dụng tất cả CPU cores để train song song |
| `class_weight` | 'balanced' | Tự động tính weight cho mỗi class theo tỷ lệ nghịch |

#### 🔹 **`n_estimators=100` - Số lượng cây**
- **Tại sao 100?**: Là baseline chuẩn, đủ để ensemble có hiệu quả. Dataset ~2000 samples → 100 cây là hợp lý.
- **Nếu thay đổi**:
  - `10-50`: Train nhanh hơn, nhưng dễ bị **high variance** (không ổn định)
  - `200-500`: Accuracy tăng nhẹ (~0.5-1%), nhưng train **lâu gấp 2-5 lần**
  - `1000+`: Hiệu quả tăng **không đáng kể** (diminishing returns)

#### 🔹 **`max_depth=10` - Độ sâu tối đa** ⭐ QUAN TRỌNG
- **Tại sao 10?**: Đây là **regularization quan trọng nhất** để chống overfitting!
- **Nếu thay đổi**:
  - `None` (default): Cây phát triển thoải mái → **OVERFITTING NẶNG** (train 100%, test thấp)
  - `3-5`: **Underfitting** - cây quá nông, không học được pattern phức tạp
  - `15-20`: Bắt đầu overfit, đặc biệt với dataset nhỏ

#### 🔹 **`min_samples_split=5` - Số mẫu tối thiểu để tách nút**
- **Tại sao 5?**: Ngăn cây tạo các nhánh quá nhỏ chỉ để fit vài samples → chống overfitting.
- **Nếu thay đổi**:
  - `2` (default): Cây chia quá chi tiết → overfit
  - `50+`: **Underfitting** - bỏ qua nhiều pattern quan trọng

#### 🔹 **`min_samples_leaf=2` - Số mẫu tối thiểu ở mỗi lá**
- **Tại sao 2?**: Đảm bảo không có leaf chỉ chứa 1 sample (noise). Kết hợp với `min_samples_split=5` tạo **double protection** chống overfit.

#### 🔹 **`random_state=42` - Seed cho reproducibility**
- **Tại sao 42?**: Con số huyền thoại từ "The Hitchhiker's Guide to the Galaxy". Thực ra có thể chọn bất kỳ số nào, quan trọng là **cố định** để kết quả lặp lại được.

#### 🔹 **`n_jobs=-1` - Sử dụng đa luồng**
- **Tại sao -1?**: Random Forest train 100 cây độc lập → dễ dàng parallelized. Tận dụng toàn bộ CPU → train nhanh hơn nhiều lần.

#### 🔹 **`class_weight='balanced'` - Cân bằng class** ⭐ RẤT QUAN TRỌNG!
- **Công thức**: `weight = n_samples / (n_classes × n_samples_class_i)`
- **Ví dụ**: Dataset có 1000 phim thất bại, 500 phim thành công:
  - weight(thất bại) = 1500 / (2 × 1000) = **0.75**
  - weight(thành công) = 1500 / (2 × 500) = **1.5**
  - → Sai phim thành công bị phạt **GẤP ĐÔI**!
- **Nếu không dùng**: Model sẽ **BIAS về class đa số** (thất bại) để maximize accuracy, dẫn đến Recall của class thiểu số rất thấp.

#### 📊 **Tổng hợp: Mối quan hệ giữa các tham số**
```
                    OVERFITTING ←─────────────────→ UNDERFITTING
                    
n_estimators:       Thấp (10)                       Cao (500+) - ổn định hơn
max_depth:          Cao (None, 50+)                 Thấp (3-5)
min_samples_split:  Thấp (2)                        Cao (50+)
min_samples_leaf:   Thấp (1)                        Cao (20+)
```

**Cấu hình hiện tại nằm ở SWEET SPOT**: `max_depth=10` + `min_samples_split=5` + `min_samples_leaf=2` → Cân bằng giữa học pattern và chống overfit.

### Q29: Em đã sử dụng kỹ thuật Cross-Validation như thế nào?
- **Trả lời**:
    - Sử dụng **Stratified K-Fold** với K=5
    - **Stratified**: Đảm bảo mỗi fold có tỷ lệ Success/Fail giống nhau
    - **Kết quả**: CV Mean = 69.31% (±2σ), cho thấy model ổn định
- **Code**:
```python
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(model, scaler.transform(X), y, cv=cv, scoring='accuracy')
```

### Q30: Model được lưu ở đâu và chứa những gì?
- **Trả lời**: Model được lưu dưới dạng pickle file tại:
    1. `progress/5-train-lan-2/output/pre_release_rf_model.pkl`
    2. `data/pkl/pre_release_rf_model.pkl` (cho web app)
- **Nội dung file pkl**:
    - `model`: RandomForest classifier đã train
    - `scaler`: StandardScaler đã fit (quan trọng khi predict!)
    - `feature_names`: Danh sách 37 features
    - `metrics`: Accuracy, Precision, Recall, F1-Score, CV Mean
    - `model_type`: "pre_release"
    - `description`: Mô tả model

---

# 🟢 PHẦN 10: PRE-RELEASE PREDICTION SERVICE (WEB APP)

### Q31: Giải thích luồng xử lý khi người dùng gửi request dự đoán?
- **Trả lời**:
1. **Nhận input**: Budget, Runtime, Genres, Release Month, Countries, Num Cast...
2. **Chuẩn bị features** (`prepare_features()`):
   - Tính `Budget_log = log10(budget + 1)`
   - Tính `runtime_hours = runtime / 60`
   - One-hot encode genres và countries
   - Tính `cast_genre_interaction = num_cast × num_genres`
3. **Scale features**: Dùng scaler đã load từ pkl
4. **Predict**: `model.predict()` và `model.predict_proba()`
5. **Xác định Risk Level**: HIGH/MEDIUM/LOW dựa trên xác suất
6. **Ước tính ROI**: Dựa trên xác suất thành công
7. **Trả về JSON**: success, probability, risk, metrics, feature_importance...

### Q32: Ngưỡng phân loại Risk Level như thế nào?
- **Trả lời** (Theo code):
    - `success_prob >= 0.7`: **LOW RISK** - Phim có tiềm năng thành công cao
    - `0.5 <= success_prob < 0.7`: **MEDIUM** - Phim có tiềm năng trung bình
    - `success_prob < 0.5`: **HIGH RISK** - Phim có rủi ro thất bại cao

### Q33: Công thức ước tính ROI trong service là gì?
- **Trả lời**:
```python
if success_prob >= 0.7:
    estimated_roi = 2.0 + (success_prob - 0.7) * 10  # 2.0 - 5.0
elif success_prob >= 0.5:
    estimated_roi = 1.0 + (success_prob - 0.5) * 5   # 1.0 - 2.0
else:
    estimated_roi = 0.3 + success_prob * 1.4         # 0.3 - 1.0
```
- **Giải thích**: Phim thành công thường có ROI 2-5x, phim thất bại 0.3-0.8x.

### Q34: Tại sao service dùng Singleton Pattern?
- **Trả lời**:
```python
_service_instance = None

def get_prediction_service():
    global _service_instance
    if _service_instance is None:
        _service_instance = PreReleaseMoviePredictionService()
    return _service_instance
```
- **Lý do**: 
    - Model chỉ cần load **1 lần** khi server khởi động
    - Tiết kiệm memory và thời gian
    - Tất cả requests dùng chung 1 instance

### Q35: Service xử lý trường hợp không có country nào được chọn như thế nào?
- **Trả lời**: Default là USA
```python
if not countries:
    if 'is_usa' in features:
        features['is_usa'] = 1
    if 'is_united_states_of_america' in features:
        features['is_united_states_of_america'] = 1
```
- **Lý do**: Dataset chủ yếu là phim Mỹ (~50%), nên đây là default hợp lý.

### Q36: Feature Importance được trả về như thế nào trong response?
- **Trả lời**: Top 10 features quan trọng nhất
```python
def _get_top_features(self):
    importances = self.model.feature_importances_
    feature_importance = list(zip(self.feature_names, importances))
    feature_importance.sort(key=lambda x: x[1], reverse=True)
    return [
        {'feature': name, 'importance': round(imp * 100, 2)}
        for name, imp in feature_importance[:10]
    ]
```
- **Ý nghĩa**: Giúp người dùng hiểu yếu tố nào ảnh hưởng nhiều nhất đến kết quả dự đoán.

### Q37: Làm sao để cập nhật model mới mà không thay đổi code service?
- **Trả lời**: 
    1. Train model mới với `retrain.py`
    2. Model tự động lưu vào `data/pkl/pre_release_rf_model.pkl`
    3. Restart Flask server (hoặc chờ lazy loading)
    4. Service sẽ load model mới
- **Lưu ý**: Đảm bảo `feature_names` trong model mới phải tương thích với model cũ!

---
**💡 Mẹo nhỏ khi trả lời:**
- Luôn tự tin, nói to, rõ ràng.
- Nếu không biết, hãy nói: *"Vấn đề này em chưa nghiên cứu sâu, nhưng theo suy luận của em thì..."* (đừng im lặng).
- Nhấn mạnh vào tính **thực tế** và **ứng dụng** của đồ án.
