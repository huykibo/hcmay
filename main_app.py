import streamlit as st

# Import các hàm từ ứng dụng con
from Titanic_app import run_titanic_app
from MNIST_app import run_mnist_classification_app
from MNIST_kmean import run_mnist_clustering_app
from MNIST_PCA import run_mnist_dimension_reduction_app
from netw import run_mnist_neural_network_app  # Thêm import cho MNIST Neural Network

# Cấu hình trang chính - phải được gọi ngay đầu file
st.set_page_config(page_title="Multi-App", layout="wide")

# Sidebar chứa menu ứng dụng
st.sidebar.title("Menu Ứng Dụng")
app_choice = st.sidebar.selectbox(
    "Chọn ứng dụng:",
    ["Linear Regression", "Classification", "Clustering", "Dimension Reduction", "Neural Network Classification"]  # Thêm Neural Network
)

# Nội dung chính của trang11
st.title("Chương Trình Ứng Dụng")

# Điều hướng đến ứng dụng được chọn
if app_choice == "Linear Regression":
    run_titanic_app()
elif app_choice == "Classification":
    run_mnist_classification_app()
elif app_choice == "Clustering":
    run_mnist_clustering_app()
elif app_choice == "Dimension Reduction":
    run_mnist_dimension_reduction_app()
elif app_choice == "Neural Network Classification":
    run_mnist_neural_network_app()