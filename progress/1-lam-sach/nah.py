# kiểm tra dữ liệu
import pandas as pd
df = pd.read_csv('./data/raw_Movies.csv')
print(f"Dữ liệu gốc: {len(df)} hàng, {len(df.columns)} cột")
print("Kiểu dữ liệu mỗi cột:")
print(df.dtypes)
print("\nMột số hàng đầu:")
print(df.head())
print("\nKiểm tra định dạng không nhất quán:")
# Kiểm tra định dạng không nhất quán ở một số cột chính
for col in ['Genres', 'Release Date']:
    print(f"\nCột: {col}")
    print(df[col].head(10))
    print(df[col].apply(type).value_counts())
print("\nKiểm tra giá trị thiếu:")
print(df.isnull().sum())
print("\nKiểm tra giá trị 0 trong Budget và Revenue:")
print(f"Số hàng có Budget = 0: {(df['Budget'] == 0
).sum()}")
print(f"Số hàng có Revenue = 0: {(df['Revenue'] == 0
).sum()}")
print("\nKiểm tra giá trị âm trong Budget và Revenue:")
print(f"Số hàng có Budget < 0: {(df['Budget'] < 0
).sum()}")
print(f"Số hàng có Revenue < 0: {(df['Revenue'] < 0
).sum()}")
print("\nKiểm tra định dạng ngày tháng trong Release Date:")
print(df['Release Date'].head(10))
print(df['Release Date'].apply(lambda x: isinstance(x, str)).value_counts())
print("\nKiểm tra các giá trị duy nhất trong một số cột:")
for col in ['Genres', 'Production Countries']:
    print(f"\nCột: {col}")
    unique_values = df[col].dropna().unique()
    print(f"Số giá trị duy nhất: {len(unique_values)}")
    print(f"Ví dụ giá trị: {unique_values[:5]}")
print("\nKiểm tra phân phối của các cột số:")
print(df[['Budget', 'Revenue', 'Vote Average', 'Runtime']].describe())
print("\nKiểm tra phân phối của cột mục tiêu tiềm năng (nếu có):")
if 'success' in df.columns:
    print(df['success'].value_counts())
else:
    print("Cột 'success' chưa tồn tại trong dữ liệu gốc.")
