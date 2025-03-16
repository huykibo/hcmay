import streamlit as st

# Import các hàm từ ứng dụng con
from Titanic_app import run_titanic_app
from MNIST_app import run_mnist_classification_app
from MNIST_kmean import run_mnist_clustering_app
from MNIST_PCA import run_mnist_dimension_reduction_app
from netw import run_mnist_neural_network_app
from Semisupervised import run_mnist_labelding_neural_network_app

# Cấu hình trang chính - phải được gọi ngay đầu file
st.set_page_config(page_title="Multi-App", layout="wide")

# Khởi tạo session_state để theo dõi ứng dụng hiện tại
if 'current_app' not in st.session_state:
    st.session_state.current_app = None

# Sidebar chứa menu ứng dụng
st.sidebar.title("Menu Ứng Dụng")
app_choice = st.sidebar.selectbox(
    "Chọn ứng dụng:",
    [
        "Linear Regression",
        "Classification",
        "Clustering",
        "Dimension Reduction",
        "Neural Network Classification ",
        "Neural Network Classification (Pseudo-Labeling)"
    ]
)

# Hàm để reset trạng thái của các ứng dụng khác
def reset_other_apps(selected_app):
    # Nếu ứng dụng hiện tại thay đổi, reset trạng thái
    if st.session_state.current_app != selected_app:
        st.session_state.clear()  # Xóa toàn bộ trạng thái
        st.session_state.current_app = selected_app

# Nội dung chính của trang
st.title("Chương Trình Ứng Dụng")

# Điều hướng đến ứng dụng được chọn và reset các ứng dụng khác
if app_choice == "Linear Regression":
    reset_other_apps("Linear Regression")
    run_titanic_app()
elif app_choice == "Classification":
    reset_other_apps("Classification")
    run_mnist_classification_app()
elif app_choice == "Clustering":
    reset_other_apps("Clustering")
    run_mnist_clustering_app()
elif app_choice == "Dimension Reduction":
    reset_other_apps("Dimension Reduction")
    run_mnist_dimension_reduction_app()
elif app_choice == "Neural Network Classification ":
    reset_other_apps("Neural Network Classification ")
    st.header("Neural Network Classification")
    run_mnist_neural_network_app()
elif app_choice == "Neural Network Classification (Pseudo-Labeling)":
    reset_other_apps("Neural Network Classification (Pseudo-Labeling)")
    st.header("Neural Network Classification (Pseudo-Labeling)")
    run_mnist_labelding_neural_network_app()