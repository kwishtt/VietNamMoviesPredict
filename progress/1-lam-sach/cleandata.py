# Mục tiêu: Loại bỏ hoặc ghi chú các lỗi dữ liệu, chuẩn hóa định dạng.
# Việc cần làm:
# Xác định và xử lý hàng có Budget hoặc Revenue = 0 (loại/báo cáo).
# Điền hoặc ghi chú giá trị thiếu (NaN) theo cột.
# Chuyển Release Date sang kiểu ngày, thống kê tỉ lệ thiếu theo cột.
# In ra các bước đã thực hiện và kết quả.
# Clone dataset thành clean_movies.csv sau khi làm sạch.

import pandas as pd
def clean_data(file_path, output_path):
    # Đọc dữ liệu
    df = pd.read_csv(file_path)

    # Xử lý Budget hoặc Revenue = 0
    zero_budget_revenue = df[(df['Budget'] == 0) | (df['Revenue'] == 0)]
    print(f"Số hàng có Budget hoặc Revenue = 0: {len(zero_budget_revenue)}")
    df = df[(df['Budget'] != 0) & (df['Revenue'] != 0)]

    # Điền giá trị thiếu
    missing_values = df.isnull().sum()
    print("Số giá trị thiếu theo cột trước khi điền:")
    print(missing_values)

    for column in df.columns:
        if df[column].dtype == 'object':
            df[column].fillna('Unknown', inplace=True)
        else:
            df[column].fillna(df[column].mean(), inplace=True)

    missing_values_after = df.isnull().sum()
    print("Số giá trị thiếu theo cột sau khi điền:")
    print(missing_values_after)

    # Chuyển Release Date sang kiểu ngày
    df['Release Date'] = pd.to_datetime(df['Release Date'], errors='coerce')
    missing_release_dates = df['Release Date'].isnull().sum()
    print(f"Số giá trị thiếu trong Release Date sau khi chuyển đổi: {missing_release_dates}")

    # Lưu dữ liệu đã làm sạch
    df.to_csv(output_path, index=False)
    print(f"Dữ liệu đã được làm sạch và lưu vào {output_path}")

clean_data('./data/Movies.csv', './data/clean_movies.csv')


