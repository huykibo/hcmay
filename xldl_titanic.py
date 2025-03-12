import mlflow
import mlflow.sklearn
import pandas as pd
# Đọc file CSV
df = pd.read_csv("titanic.csv")

# Kiểm tra thông tin tổng quan
print(df.info())  # Xem kiểu dữ liệu, số lượng giá trị null
print(df.head())  # Xem 5 dòng đầu tiên
print(df.isnull().sum())  # Đếm số lượng giá trị NaN trong mỗi cột
# Thay thế giá trị NaN trong cột 'Age' bằng giá trị trung bình
df['Age'] = df['Age'].fillna(df['Age'].mean())

# Thay thế NaN trong 'Embarked' bằng giá trị xuất hiện nhiều nhất
df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])

# Nếu cột 'Cabin' có quá nhiều giá trị NaN, có thể xóa bỏ
df.drop(columns=['Cabin'], inplace=True)

# Kiểm tra lại sau khi xử lý
print(df.isnull().sum())  
# Chuyển cột 'Survived' thành kiểu bool (0/1)
df['Survived'] = df['Survived'].astype(bool)

# Kiểm tra dữ liệu trùng lặp
print(df.duplicated().sum())

# Loại bỏ các dòng trùng lặp (nếu có)
df.drop_duplicates(inplace=True)
df.to_csv("titanic_cleaned.csv", index=False)  # Lưu lại file sạch
#log kết quả vào mlflow
with mlflow.start_run():  
    mlflow.log_param("Rows_after_cleaning", df.shape[0])
    mlflow.log_param("Columns_after_cleaning", df.shape[1])
    mlflow.log_artifact("titanic_cleaned.csv")  # Lưu file CSV vào MLflow

print(" Dữ liệu Titanic đã được xử lý và theo dõi trong MLflow.")

# my_env\Scripts\activate
# rm -r mlruns
# python xldl_titanic.py
# mlflow ui

# #dã thiết lâp git trc đó fff
git add 
git commit -m "Update: Cập nhật code ứng dụng Streamlit MLflow"
git push
