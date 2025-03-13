import os
import mlflow
import streamlit as st
import openml
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from mlflow.tracking import MlflowClient
from datetime import datetime
from sklearn.impute import SimpleImputer
import time
import plotly.express as px
from PIL import Image

def resize_image(image_path, size=(50, 50)):
    """Resize image to specified size and return as bytes."""
    img = Image.open(image_path)
    img_resized = img.resize(size, Image.Resampling.LANCZOS)  # LANCZOS for better quality
    return img_resized

def run_mnist_dimension_reduction_app():
    st.title("Ứng dụng Trực quan hóa Dữ liệu MNIST với PCA và t-SNE")

    # CSS for inline layout
    st.markdown("""
        <style>
            .inline-container {
                display: inline-flex;
                align-items: center;
                gap: 5px;
            }
            .image-container {
                max-width: 800px; /* Giới hạn chiều rộng tối đa */
                margin: auto;
            }
        </style>
    """, unsafe_allow_html=True)

    # Đường dẫn tới thư mục chứa hình ảnh minh họa PCA và t-SNE
    pca_image_dir = r"pca"
    tsne_image_dir = r"tsne_steps"

    # Khởi tạo và kiểm tra/tạo experiment
    client = MlflowClient()
    experiment_name = "MNIST_PCA"
    experiment = client.get_experiment_by_name(experiment_name)
    if not experiment:
        st.warning(f"Experiment '{experiment_name}' chưa tồn tại. Đang tạo mới...")
        experiment_id = client.create_experiment(experiment_name)
        st.success(f"Đã tạo experiment '{experiment_name}' với ID: {experiment_id}")
    else:
        experiment_id = experiment.experiment_id

    # Tabs for navigation
    tab_info, tab_load, tab_visualize, tab_log_info = st.tabs(["Thông tin", "Tải dữ liệu", "Trực quan hóa", "Theo dõi kết quả"])

    # Tab Thông tin (Cập nhật với thông tin MNIST trong selectbox)
    with tab_info:
        st.header("Giới thiệu về PCA và t-SNE")
        st.markdown("""
        Ứng dụng này giúp bạn trực quan hóa tập dữ liệu **MNIST** bằng cách giảm chiều dữ liệu từ $784$ chiều (28x28 pixel) xuống $2D$ hoặc $3D$ sử dụng **PCA** (Phân tích Thành phần Chính) và **t-SNE** (Nhúng Ngẫu nhiên Láng giềng Phân phối t).  
        - **Mục tiêu**: Khám phá cấu trúc ẩn trong dữ liệu, giúp hiểu cách các chữ số (0-9) phân bố trong không gian thấp chiều.  
        - **Khác biệt chính**: PCA là tuyến tính và nhanh, trong khi t-SNE là phi tuyến tính và tập trung vào cấu trúc cục bộ.  
        """, unsafe_allow_html=True)

        st.subheader("Chọn mục để tìm hiểu:")
        info_option = st.selectbox(
            "",
            ["MNIST là gì?", "PCA là gì?", "t-SNE là gì?", "So sánh PCA và t-SNE"],
            label_visibility="collapsed",
            help="Chọn để xem chi tiết về MNIST, PCA, t-SNE hoặc so sánh PCA và t-SNE."
        )

        if info_option == "MNIST là gì?":
            st.subheader("📚 MNIST – Tập dữ liệu MNIST là gì?")
            st.markdown("""
            **MNIST** (Modified National Institute of Standards and Technology) là một tập dữ liệu nổi tiếng trong lĩnh vực học máy và thị giác máy tính, được sử dụng rộng rãi để huấn luyện và kiểm tra các mô hình nhận diện chữ số viết tay.  
            - **Nguồn gốc**: Được tạo ra bởi Yann LeCun, Corinna Cortes và Christopher Burges, dựa trên dữ liệu từ NIST (National Institute of Standards and Technology).  
            - **Cấu trúc**:  
              - **Số lượng mẫu**: 70,000 hình ảnh, chia thành:  
                - 60,000 mẫu huấn luyện.  
                - 10,000 mẫu kiểm tra.  
              - **Kích thước**: Mỗi hình ảnh là ảnh xám (grayscale) 28x28 pixel, tổng cộng 784 đặc trưng (pixel).  
              - **Nhãn**: 10 lớp tương ứng với các chữ số từ 0 đến 9.  
            - **Đặc điểm**:  
              - Các hình ảnh được chuẩn hóa (centered và normalized) để các chữ số nằm ở giữa khung 28x28.  
              - Dữ liệu đơn giản nhưng đủ thách thức để thử nghiệm các thuật toán học máy cơ bản và nâng cao.  
            - **Ứng dụng**: Được dùng trong giáo dục, nghiên cứu, và là chuẩn mực (benchmark) để so sánh hiệu suất các mô hình học máy.  
            - **Định dạng dữ liệu**: Mỗi pixel có giá trị từ 0 (đen) đến 255 (trắng), biểu thị cường độ sáng. Trong ứng dụng này, dữ liệu được chuẩn hóa về [0, 1] trước khi giảm chiều.  
            - **Thách thức**: Sự biến thiên lớn trong cách viết tay (kích thước, độ dày nét, góc nghiêng) tạo ra thử thách cho các mô hình nhận diện.  
            """, unsafe_allow_html=True)

            # Thêm ảnh minh họa MNIST (mnist.png)
            st.markdown("### Minh họa tập dữ liệu MNIST")
            mnist_image_path = os.path.join("mnist.png")  # Đường dẫn tới ảnh
            try:
                img_resized = resize_image(mnist_image_path, size=(900, 500))
                st.image(img_resized, caption="Ví dụ một số hình ảnh từ tập dữ liệu MNIST", use_container_width=False)
            except FileNotFoundError:
                st.warning("Không tìm thấy ảnh 'mnist.png'. Vui lòng đặt ảnh vào thư mục 'images' và kiểm tra lại đường dẫn.")

        elif info_option == "PCA là gì?":
            st.subheader("📊 PCA – Phân tích Thành phần Chính")
            st.markdown("""
            **PCA** là một kỹ thuật giảm chiều tuyến tính, chuyển đổi dữ liệu từ không gian chiều cao (ví dụ: $784$ chiều của MNIST) sang không gian chiều thấp (như $2D$ hoặc $3D$) bằng cách tìm các hướng (thành phần chính) có phương sai lớn nhất.  
            - **Ý tưởng chính**: Tìm các trục mới sao cho dữ liệu được chiếu lên giữ lại tối đa thông tin (phương sai).  
            - **Ứng dụng**: Trực quan hóa, nén dữ liệu, loại bỏ nhiễu.  
            """, unsafe_allow_html=True)

            st.markdown("### Các bước thực hiện PCA:")

            st.subheader("1. Tìm vector trung bình")
            st.markdown("""
            - Tính trung bình của dữ liệu dọc theo mỗi chiều:  
              $$ \\bar{X} = \\frac{1}{n} \\sum_{i=1}^{n} X_i $$  
            - **Giải thích**:  
              - $X_i$: Vector đặc trưng của mẫu thứ $i$, có kích thước $p$ (số chiều gốc, với MNIST là $784$).  
              - $n$: Tổng số mẫu trong tập dữ liệu (ví dụ: 70,000 với MNIST).  
              - $\\bar{X}$: Vector trung bình, cũng có kích thước $p$, biểu thị "tâm" của đám mây dữ liệu.  
              - $\\sum_{i=1}^{n}$: Tổng cộng giá trị của tất cả các mẫu.  
            - Mục đích: Chuẩn hóa dữ liệu về gốc tọa độ bằng cách tìm điểm trung tâm của tập dữ liệu.
            """, unsafe_allow_html=True)
            img_path = os.path.join(pca_image_dir, "1pca.png")
            img_resized = resize_image(img_path, size=(400, 400))
            st.image(img_resized, caption="Minh họa tìm vector trung bình", use_container_width=False)

            st.subheader("2. Trừ trung bình")
            st.markdown("""
            - Trừ vector trung bình khỏi mỗi mẫu để chuẩn hóa dữ liệu:  
              $$ X_{\\text{centered}} = X - \\bar{X} $$  
            - **Giải thích**:  
              - $X$: Ma trận dữ liệu gốc, kích thước $n \\times p$ (hàng là mẫu, cột là đặc trưng).  
              - $\\bar{X}$: Vector trung bình từ bước 1, được trừ khỏi từng hàng của $X$.  
              - $X_{\\text{centered}}$: Ma trận dữ liệu đã được chuẩn hóa, vẫn có kích thước $n \\times p$, nhưng trung bình bằng 0.  
            - Mục đích: Loại bỏ ảnh hưởng của giá trị trung bình, tập trung vào sự phân tán của dữ liệu.
            """, unsafe_allow_html=True)
            img_path = os.path.join(pca_image_dir, "2pca.png")
            img_resized = resize_image(img_path, size=(400, 400))
            st.image(img_resized, caption="Minh họa trừ trung bình", use_container_width=False)

            st.subheader("3. Tính ma trận hiệp phương sai")
            st.markdown("""
            - Tính ma trận hiệp phương sai để đo lường sự phân tán và mối quan hệ giữa các chiều:  
              $$ S = \\frac{1}{n-1} X_{\\text{centered}}^T X_{\\text{centered}} $$  
            - **Giải thích**:  
              - $X_{\\text{centered}}$: Ma trận dữ liệu đã chuẩn hóa từ bước 2, kích thước $n \\times p$.  
              - $X_{\\text{centered}}^T$: Ma trận chuyển vị của $X_{\\text{centered}}$, kích thước $p \\times n$.  
              - $S$: Ma trận hiệp phương sai, kích thước $p \\times p$, chứa phương sai (trên đường chéo) và hiệp phương sai (ngoài đường chéo) giữa các chiều.  
              - $n-1$: Hiệu chỉnh để ước lượng không chệch (Bessel's correction).  
            - Mục đích: Xác định mức độ biến thiên và tương quan giữa các đặc trưng để tìm hướng quan trọng nhất.
            """, unsafe_allow_html=True)
            img_path = os.path.join(pca_image_dir, "3-4pca.png")
            img_resized = resize_image(img_path, size=(400, 400))
            st.image(img_resized, caption="Minh họa tính ma trận hiệp phương sai", use_container_width=False)

            st.subheader("4. Tính giá trị riêng và vector riêng")
            st.markdown("""
            - Phân rã ma trận hiệp phương sai để tìm các giá trị riêng $\\lambda_i$ và vector riêng $u_i$:  
              $$ S u_i = \\lambda_i u_i $$  
            - **Giải thích**:  
              - $S$: Ma trận hiệp phương sai từ bước 3, kích thước $p \\times p$.  
              - $u_i$: Vector riêng thứ $i$, kích thước $p$, biểu thị hướng của thành phần chính (vuông góc với các $u_j$ khác).  
              - $\\lambda_i$: Giá trị riêng thứ $i$, biểu thị phương sai (độ lớn) của dữ liệu dọc theo hướng $u_i$.  
            - Mục đích: Xác định các hướng chính (thành phần chính) và tầm quan trọng của chúng thông qua phương sai.
            """, unsafe_allow_html=True)
            img_path = os.path.join(pca_image_dir, "3-4pca.png")
            img_resized = resize_image(img_path, size=(400, 400))
            st.image(img_resized, caption="Minh họa tính giá trị riêng và vector riêng", use_container_width=False)

            st.subheader("5. Chọn k vector riêng với giá trị riêng lớn nhất")
            st.markdown("""
            - Sắp xếp các giá trị riêng $\\lambda_i$ theo thứ tự giảm dần và chọn $k$ vector riêng tương ứng (thường $k=2$ hoặc $3$ cho trực quan hóa).  
            - **Giải thích**:  
              - $\\lambda_i$: Các giá trị riêng từ bước 4, đại diện cho phương sai của mỗi thành phần chính.  
              - $k$: Số lượng thành phần chính được chọn (số chiều giảm xuống).  
            - Mục đích: Giới hạn số chiều để giữ lại phần lớn thông tin quan trọng nhất.
            """, unsafe_allow_html=True)
            img_path = os.path.join(pca_image_dir, "5pca.png")
            img_resized = resize_image(img_path, size=(400, 400))
            st.image(img_resized, caption="Minh họa chọn k vector riêng", use_container_width=False)

            st.subheader("6. Chiếu dữ liệu lên vector đã chọn")
            st.markdown("""
            - Chiếu dữ liệu đã chuẩn hóa lên không gian mới được định nghĩa bởi $k$ vector riêng:  
              $$ Z = X_{\\text{centered}} U_k $$  
            - **Giải thích**:  
              - $X_{\\text{centered}}$: Ma trận dữ liệu đã chuẩn hóa từ bước 2, kích thước $n \\times p$.  
              - $U_k$: Ma trận chứa $k$ vector riêng đầu tiên (các cột là $u_i$), kích thước $p \\times k$.  
              - $Z$: Ma trận dữ liệu sau khi giảm chiều, kích thước $n \\times k$, chứa tọa độ trong không gian mới.  
            - Mục đích: Chuyển đổi dữ liệu sang không gian thấp chiều dựa trên các hướng quan trọng nhất.
            """, unsafe_allow_html=True)
            img_path = os.path.join(pca_image_dir, "6pca.png")
            img_resized = resize_image(img_path, size=(400, 400))
            st.image(img_resized, caption="Minh họa chiếu dữ liệu", use_container_width=False)

            st.subheader("7. Lấy điểm chiếu trong không gian thấp")
            st.markdown("""
            - Hiển thị dữ liệu đã giảm chiều trong không gian $k$-chiều (thường là 2D hoặc 3D) để trực quan hóa.  
            - **Giải thích**:  
              - $Z$: Ma trận dữ liệu sau khi chiếu từ bước 6, kích thước $n \\times k$.  
              - $k$: Số chiều của không gian thấp (thường là 2 hoặc 3).  
            - Mục đích: Tạo biểu đồ trực quan để khám phá cấu trúc dữ liệu.
            """, unsafe_allow_html=True)
            img_path = os.path.join(pca_image_dir, "7pca.png")
            img_resized = resize_image(img_path, size=(400, 400))
            st.image(img_resized, caption="Minh họa điểm chiếu trong không gian thấp", use_container_width=False)

            st.markdown("""
            - **Tham số chính**:  
              - $n_{\\text{components}}$: Số chiều giảm xuống (tức là $k$), thường là $2$ hoặc $3$ cho trực quan hóa.  
              - **Tỷ lệ phương sai giải thích (Explained Variance Ratio - EVR)**:  
                $$ \\text{EVR} = \\frac{\\sum_{i=1}^{k} \\lambda_i}{\\sum_{i=1}^{p} \\lambda_i} $$  
                - $\\sum_{i=1}^{k} \\lambda_i$: Tổng phương sai của $k$ thành phần chính được chọn.  
                - $\\sum_{i=1}^{p} \\lambda_i$: Tổng phương sai của tất cả $p$ chiều gốc.  
                - Ý nghĩa: Phản ánh phần trăm thông tin (phương sai) giữ lại sau khi giảm chiều.  

            - **Ưu điểm**:  
              - Tính toán nhanh, hiệu quả ngay cả với dữ liệu lớn (độ phức tạp $O(np^2)$).  
              - Giữ được cấu trúc toàn cục của dữ liệu (khoảng cách lớn giữa các cụm).  
              - Dễ hiểu và triển khai.  
            - **Nhược điểm**:  
              - Chỉ phù hợp với dữ liệu có mối quan hệ tuyến tính.  
              - Không giữ tốt khoảng cách cục bộ giữa các điểm gần nhau.  
              - Có thể bỏ lỡ cấu trúc phi tuyến trong dữ liệu phức tạp như MNIST.  
            """, unsafe_allow_html=True)

        elif info_option == "t-SNE là gì?":
            st.subheader("📈 t-SNE – Nhúng Ngẫu nhiên Láng giềng Phân phối t")
            st.markdown("""
            **t-SNE** là một kỹ thuật giảm chiều phi tuyến tính, tập trung vào việc giữ cấu trúc cục bộ của dữ liệu bằng cách tối ưu hóa sự tương đồng giữa các điểm trong không gian chiều cao và chiều thấp.  
            - **Ý tưởng chính**: Bảo toàn mối quan hệ láng giềng gần trong dữ liệu gốc, làm nổi bật các cụm cục bộ.  
            - **Ứng dụng**: Trực quan hóa dữ liệu phức tạp, phát hiện cụm trong không gian thấp chiều.  
            """, unsafe_allow_html=True)

            st.markdown("### Các bước thực hiện t-SNE:")

            st.subheader("1. Tính độ tương đồng trong không gian gốc")
            st.markdown("""
            - Sử dụng phân phối Gaussian để tính xác suất $p_{ij}$ rằng điểm $x_i$ chọn $x_j$ làm láng giềng:  
              $$ p_{j|i} = \\frac{\\exp(-||x_i - x_j||^2 / 2\\sigma_i^2)}{\\sum_{k \\neq i} \\exp(-||x_i - x_k||^2 / 2\\sigma_i^2)} $$  
              $$ p_{ij} = \\frac{p_{j|i} + p_{i|j}}{2n} $$  
            - **Giải thích tham số và ký tự**:  
              - $x_i, x_j$: Vector đặc trưng của mẫu $i$ và $j$ trong không gian gốc, kích thước $p$ (với MNIST là $784$).  
              - $||x_i - x_j||^2$: Bình phương khoảng cách Euclidean giữa $x_i$ và $x_j$.  
              - $\\sigma_i$: Độ lệch chuẩn của phân phối Gaussian quanh $x_i$, điều chỉnh bởi **perplexity**.  
              - $p_{j|i}$: Xác suất có điều kiện $x_i$ chọn $x_j$ làm láng giềng.  
              - $\\exp()$: Hàm mũ, giúp chuyển đổi khoảng cách thành xác suất (gần thì xác suất cao).  
              - $\\sum_{k \\neq i}$: Tổng chuẩn hóa để đảm bảo tổng xác suất bằng 1.  
              - $p_{ij}$: Xác suất đối xứng giữa $x_i$ và $x_j$, chuẩn hóa bằng $2n$.  
              - $n$: Tổng số mẫu trong tập dữ liệu.  
            - Mục đích: Đo mức độ "gần" giữa các điểm trong không gian gốc để bảo toàn cấu trúc cục bộ.
            """, unsafe_allow_html=True)
            img_path = os.path.join(tsne_image_dir, "1tsne.png")
            img_resized = resize_image(img_path, size=(400, 400))
            st.image(img_resized, caption="Minh họa tính độ tương đồng Gaussian", use_container_width=False)

            st.subheader("2. Khởi tạo không gian chiều thấp")
            st.markdown("""
            - Tạo ngẫu nhiên các điểm $y_i$ trong không gian $k$-chiều (thường $k=2$ hoặc $3$).  
            - **Giải thích**:  
              - $y_i$: Vector tọa độ của mẫu $i$ trong không gian thấp chiều, kích thước $k$.  
              - $k$: Số chiều của không gian mới (thường là 2 hoặc 3 để trực quan hóa).  
            - Mục đích: Tạo điểm khởi đầu ngẫu nhiên để t-SNE tối ưu hóa vị trí trong không gian thấp chiều.
            """, unsafe_allow_html=True)
            img_path = os.path.join(tsne_image_dir, "2tsne.png")
            img_resized = resize_image(img_path, size=(400, 400))
            st.image(img_resized, caption="Minh họa khởi tạo ngẫu nhiên", use_container_width=False)

            st.subheader("3. Tính độ tương đồng trong không gian mới")
            st.markdown("""
            - Sử dụng phân phối t-Student (độ tự do $1$, đuôi dài) để tính xác suất $q_{ij}$ trong không gian thấp chiều:  
              $$ q_{ij} = \\frac{(1 + ||y_i - y_j||^2)^{-1}}{\\sum_{k \\neq l} (1 + ||y_k - y_l||^2)^{-1}} $$  
            - **Giải thích**:  
              - $y_i, y_j$: Vector tọa độ của mẫu $i$ và $j$ trong không gian thấp chiều, kích thước $k$.  
              - $||y_i - y_j||^2$: Bình phương khoảng cách Euclidean giữa $y_i$ và $y_j$ trong không gian mới.  
              - $(1 + ||y_i - y_j||^2)^{-1}$: Nghịch đảo khoảng cách, dựa trên phân phối t-Student, ưu tiên các điểm gần nhau.  
              - $\\sum_{k \\neq l}$: Tổng chuẩn hóa để đảm bảo tổng xác suất $q_{ij}$ bằng 1.  
              - $q_{ij}$: Xác suất đối xứng giữa $y_i$ và $y_j$ trong không gian thấp chiều.  
            - Mục đích: Đo mức độ "gần" trong không gian mới, sử dụng t-Student để giảm vấn đề chen chúc (crowding problem).
            """, unsafe_allow_html=True)
            img_path = os.path.join(tsne_image_dir, "3tsne.png")
            img_resized = resize_image(img_path, size=(400, 400))
            st.image(img_resized, caption="Minh họa tính độ tương đồng t-Student", use_container_width=False)

            st.subheader("4. Tối ưu hóa")
            st.markdown("""
            - Giảm thiểu sai số Kullback-Leibler (KL-divergence) giữa $p_{ij}$ và $q_{ij}$ bằng gradient descent:  
              $$ \\text{KL}(P || Q) = \\sum_{i \\neq j} p_{ij} \\log \\frac{p_{ij}}{q_{ij}} $$  
            - **Giải thích**:  
              - $p_{ij}$: Xác suất đối xứng trong không gian gốc từ bước 1.  
              - $q_{ij}$: Xác suất đối xứng trong không gian thấp chiều từ bước 3.  
              - $\\text{KL}(P || Q)$: Độ đo KL-divergence, biểu thị sự khác biệt giữa phân phối $P$ (không gian gốc) và $Q$ (không gian mới).  
              - $\\sum_{i \\neq j}$: Tổng trên tất cả các cặp điểm (trừ $i=j$).  
              - $\\log \\frac{p_{ij}}{q_{ij}}$: Độ chênh lệch logarit giữa hai xác suất, được tối ưu để $q_{ij}$ gần $p_{ij}$.  
            - Mục đích: Điều chỉnh $y_i$ sao cho cấu trúc láng giềng trong không gian thấp chiều tương đồng với không gian gốc.
            """, unsafe_allow_html=True)
            img_path = os.path.join(tsne_image_dir, "4tsne.png")
            img_resized = resize_image(img_path, size=(900, 400))
            st.image(img_resized, caption="Minh họa quá trình tối ưu hóa", use_container_width=False)

            st.markdown("""
            - **Tham số chính**:  
              - $n_{\\text{components}}$: Số chiều giảm xuống (tức là $k$), thường $2$ hoặc $3$.  
              - **Perplexity**: Số láng giềng hiệu quả, ảnh hưởng đến $\\sigma_i$:  
                $$ \\text{Perplexity} = 2^{H(P_i)}, \\quad H(P_i) = -\\sum_j p_{j|i} \\log_2 p_{j|i} $$  
                - $H(P_i)$: Entropy của phân phối $p_{j|i}$, đo mức độ không chắc chắn trong việc chọn láng giềng.  
                - Thường chọn từ $5$ đến $50$, phụ thuộc vào kích thước dữ liệu.  
              - **Learning Rate**: Tốc độ cập nhật $y_i$ trong gradient descent, thường từ $10$ đến $1000$.  
              - **Early Exaggeration**: Phóng đại $p_{ij}$ ban đầu (mặc định 12.0) để tạo cụm rõ hơn trong giai đoạn đầu tối ưu hóa.  

            - **Ưu điểm**:  
              - Giữ tốt cấu trúc cục bộ, tạo ra các nhóm trực quan rõ ràng (hữu ích với MNIST).  
              - Hiệu quả cho dữ liệu phi tuyến tính, phức tạp.  
              - Tốt trong việc phát hiện cụm nhỏ.  
            - **Nhược điểm**:  
              - Chậm với dữ liệu lớn (độ phức tạp $O(n^2)$).  
              - Không giữ cấu trúc toàn cục tốt (khoảng cách lớn có thể bị bóp méo).  
              - Kết quả nhạy với tham số (perplexity, learning rate).  
            """, unsafe_allow_html=True)

        elif info_option == "So sánh PCA và t-SNE":
            st.subheader("So sánh PCA và t-SNE")
            st.markdown("""
            PCA và t-SNE đều là các phương pháp giảm chiều phổ biến, nhưng chúng khác nhau về mục tiêu, cách tiếp cận và ứng dụng. Dưới đây là so sánh chi tiết:
            """, unsafe_allow_html=True)
            st.markdown("""
            | **Tiêu chí**            | **PCA**                              | **t-SNE**                            |  
            |-------------------------|--------------------------------------|--------------------------------------|  
            | **Loại**               | Tuyến tính                          | Phi tuyến tính                      |  
            | **Mục tiêu**           | Tối đa hóa phương sai toàn cục      | Tối ưu hóa cấu trúc cục bộ          |  
            | **Tốc độ**             | Nhanh ($O(np^2)$)                  | Chậm hơn ($O(n^2)$)                |  
            | **Cấu trúc dữ liệu**   | Giữ toàn cục (khoảng cách lớn)     | Giữ cục bộ (láng giềng gần)         |  
            | **Tham số chính**      | $n_{\\text{components}}$           | $n_{\\text{components}}$, Perplexity, Learning Rate |  
            | **Độ đo hiệu quả**     | Tỷ lệ phương sai giải thích        | KL-divergence                      |  
            | **Ứng dụng MNIST**     | Nhanh, đơn giản, cấu trúc tuyến tính| Chi tiết, cụm rõ ràng, phi tuyến tính|  
            | **Tính tái lập**       | Ổn định (kết quả cố định)          | Ngẫu nhiên (phụ thuộc khởi tạo)     |  

            - **Khi nào dùng PCA?**  
              - Dữ liệu lớn, cần xử lý nhanh.  
              - Muốn giữ cấu trúc toàn cục hoặc phân tích tuyến tính.  
              - Cần kết quả ổn định để phân tích sâu hơn.  
            - **Khi nào dùng t-SNE?**  
              - Dữ liệu phức tạp, phi tuyến tính (như MNIST).  
              - Muốn trực quan hóa các cụm cục bộ rõ ràng.  
              - Chấp nhận thời gian xử lý lâu để có kết quả chi tiết.  
            """, unsafe_allow_html=True)

    # Tab Tải dữ liệu
    with tab_load:
        st.header("Tải Dữ liệu MNIST")
        st.markdown("""
        Phần này cho phép tải dữ liệu MNIST từ OpenML và chọn số lượng mẫu để trực quan hóa. Tổng cộng có $70,000$ mẫu, bạn có thể chọn một phần nhỏ hơn để giảm thời gian xử lý.
        """, unsafe_allow_html=True)

        if st.button("Tải dữ liệu"):
            try:
                with st.spinner("Đang tải dữ liệu..."):
                    progress_bar = st.progress(0)
                    status_text = st.empty()

                    mnist = openml.datasets.get_dataset(554)
                    progress_bar.progress(20)
                    status_text.text("Đã tải 20% - Đang truy xuất dữ liệu...")

                    X, y, _, _ = mnist.get_data(target=mnist.default_target_attribute)
                    progress_bar.progress(50)
                    status_text.text("Đã tải 50% - Đang xử lý dữ liệu...")

                    if X.isnull().values.any():
                        progress_bar.progress(70)
                        status_text.text("Đã tải 70% - Đang xử lý giá trị NaN...")
                        imputer = SimpleImputer(strategy='mean')
                        X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns, index=X.index)

                    st.session_state['full_data'] = (X, y)
                    progress_bar.progress(90)
                    status_text.text(f"Đã tải 90% - Hoàn tất {X.shape[0]} mẫu...")

                    with mlflow.start_run(experiment_id=experiment_id, run_name="Data_Load"):
                        mlflow.log_param("total_samples", X.shape[0])
                    
                    progress_bar.progress(100)
                    status_text.text("Đã tải 100% - Hoàn tất!")
                    time.sleep(1)
                    status_text.empty()
                    progress_bar.empty()
                    st.success("Tải dữ liệu thành công.")
                    st.write("Kích thước dữ liệu gốc:", X.shape)
            except Exception as e:
                st.error(f"Lỗi khi tải dữ liệu: {e}")

        if 'full_data' in st.session_state:
            X_full, y_full = st.session_state['full_data']
            num_samples = st.slider("Chọn số lượng mẫu:", 
                                    min_value=10, max_value=len(X_full), value=min(1000, len(X_full)), step=1)
            if st.button("Xác nhận số lượng mẫu"):
                try:
                    with st.spinner(f"Đang xử lý {num_samples} mẫu..."):
                        progress_bar = st.progress(0)
                        status_text = st.empty()

                        df = pd.concat([X_full, y_full.rename("label")], axis=1)
                        progress_bar.progress(30)
                        status_text.text("Đã xử lý 30% - Đang kết hợp dữ liệu...")

                        sampled_df = df.sample(n=num_samples, random_state=42)
                        progress_bar.progress(70)
                        status_text.text("Đã xử lý 70% - Đang lấy mẫu ngẫu nhiên...")

                        X_sampled = sampled_df.drop(columns=["label"])
                        y_sampled = sampled_df["label"]
                        st.session_state['data'] = (X_sampled, y_sampled)
                        progress_bar.progress(90)
                        status_text.text("Đang xử lý 90% - Đang lưu trữ dữ liệu...")

                        with mlflow.start_run(experiment_id=experiment_id, run_name="Data_Sample"):
                            mlflow.log_param("num_samples", num_samples)
                        
                        progress_bar.progress(100)
                        status_text.text("Đã xử lý 100% - Hoàn tất!")
                        time.sleep(1)
                        status_text.empty()
                        progress_bar.empty()
                        st.success(f"Đã chọn {num_samples} mẫu để trực quan hóa.")
                except Exception as e:
                    st.error(f"Lỗi khi xử lý dữ liệu: {e}")

    # Tab Trực quan hóa
    with tab_visualize:
        st.header("Trực quan hóa Dữ liệu MNIST")
        st.markdown("""
        Phần này giúp bạn giảm chiều dữ liệu MNIST từ $784$ chiều xuống $2D$ hoặc $3D$ bằng PCA hoặc t-SNE, sau đó trực quan hóa kết quả. Nhãn thật ($0$-$9$) sẽ được hiển thị để so sánh.
        """, unsafe_allow_html=True)

        if 'data' not in st.session_state:
            st.info("Vui lòng tải dữ liệu từ tab 'Tải dữ liệu' trước khi thực hiện trực quan hóa.")
        else:
            X, y = st.session_state['data']
            num_samples = X.shape[0]
            st.write(f"Dữ liệu hiện tại: {num_samples} ảnh, mỗi ảnh có {X.shape[1]} đặc trưng.")

            st.subheader("Cấu hình Trực quan hóa")
            col1, col2 = st.columns([1, 1])

            with col1:
                reduce_method = st.selectbox(
                    "Chọn phương pháp giảm chiều:",
                    ["PCA", "t-SNE"],
                    help="PCA nhanh và tuyến tính; t-SNE chậm hơn nhưng giữ cấu trúc cục bộ."
                )

            suggestion_data_pca = {
                "Số lượng mẫu": ["nhỏ hơn 1,000", "1,000–10,000", "lớn hơn 10,000"],
                "n_components": ["2 hoặc 3", "2 hoặc 3", "2 hoặc 3"]
            }
            suggestion_data_tsne = {
                "Số lượng mẫu": ["nhỏ hơn 1,000", "1,000–10,000", "lớn hơn 10,000"],
                "n_components": ["2 hoặc 3", "2 hoặc 3", "2 hoặc 3"],
                "perplexity": ["5–10", "20–30", "30–50"],
                "learning_rate": ["100–200", "200–500", "500–1000"]
            }

            if num_samples < 1000:
                range_idx = 0
                suggested_perplexity = 10
                suggested_learning_rate = 200
            elif num_samples <= 10000:
                range_idx = 1
                suggested_perplexity = 30
                suggested_learning_rate = 200
            else:
                range_idx = 2
                suggested_perplexity = 50
                suggested_learning_rate = 500

            suggested_n_components = 2

            params = {}
            with col2:
                if reduce_method == "PCA":
                    st.markdown("**Số chiều ($n_{\\text{components}}$)**", unsafe_allow_html=True)
                    n_components = st.selectbox(
                        "",
                        [2, 3],
                        index=0,
                        label_visibility="collapsed",
                        help=f"Gợi ý: {suggestion_data_pca['n_components'][range_idx]}. Giá trị tối ưu tự động: {suggested_n_components}"
                    )
                    params["n_components"] = n_components
                else:
                    st.markdown("**Số chiều ($n_{\\text{components}}$)**", unsafe_allow_html=True)
                    n_components = st.selectbox(
                        "",
                        [2, 3],
                        index=0,
                        label_visibility="collapsed",
                        help=f"Gợi ý: {suggestion_data_tsne['n_components'][range_idx]}. Giá trị tối ưu tự động: {suggested_n_components}"
                    )
                    st.markdown("**Perplexity**", unsafe_allow_html=True)
                    perplexity = st.number_input(
                        "",
                        min_value=5.0, max_value=50.0, value=float(suggested_perplexity), step=1.0,
                        label_visibility="collapsed",
                        help=f"Gợi ý: {suggestion_data_tsne['perplexity'][range_idx]}. Giá trị tối ưu tự động: {suggested_perplexity}"
                    )
                    st.markdown("**Learning Rate**", unsafe_allow_html=True)
                    learning_rate = st.number_input(
                        "",
                        min_value=10.0, max_value=1000.0, value=float(suggested_learning_rate), step=10.0,
                        label_visibility="collapsed",
                        help=f"Gợi ý: {suggestion_data_tsne['learning_rate'][range_idx]}. Giá trị tối ưu tự động: {suggested_learning_rate}"
                    )
                    params["n_components"] = n_components
                    params["perplexity"] = perplexity
                    params["learning_rate"] = learning_rate

            st.subheader("Gợi ý tham số tối ưu dựa trên số lượng dữ liệu")
            st.markdown(
                f"Dựa trên số lượng mẫu hiện tại (**{num_samples} mẫu**), dưới đây là gợi ý tham số tối ưu cho {reduce_method}:",
                unsafe_allow_html=True
            )
            if reduce_method == "PCA":
                st.table(suggestion_data_pca)
            else:
                st.table(suggestion_data_tsne)

            if st.button("Bắt đầu giảm chiều"):
                try:
                    with st.spinner("Đang xử lý..."):
                        progress_bar = st.progress(0)
                        status_text = st.empty()
                        start_time = time.time()

                        status_text.text("Đang chuẩn hóa dữ liệu...")
                        X_processed = X / 255.0
                        scaler = StandardScaler()
                        X_processed = scaler.fit_transform(X_processed)
                        progress_bar.progress(20)

                        run_name = f"{reduce_method}_Run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                        with mlflow.start_run(experiment_id=experiment_id, run_name=run_name) as run:
                            if reduce_method == "PCA":
                                status_text.text("Đang chạy PCA...")
                                model = PCA(n_components=n_components)
                                progress_bar.progress(40)
                                X_reduced = model.fit_transform(X_processed)
                                explained_variance_ratio = model.explained_variance_ratio_.sum()
                                progress_bar.progress(80)
                                mlflow.log_metric("explained_variance_ratio", explained_variance_ratio)
                                mlflow.sklearn.log_model(model, "pca_model")
                            else:
                                status_text.text("Đang chạy t-SNE (có thể mất vài phút)...")
                                model = TSNE(n_components=n_components, perplexity=perplexity, learning_rate=learning_rate, random_state=42)
                                progress_bar.progress(40)
                                X_reduced = model.fit_transform(X_processed)
                                progress_bar.progress(80)
                                mlflow.sklearn.log_model(model, "tsne_model")

                            training_time = time.time() - start_time
                            mlflow.log_params(params)
                            mlflow.log_param("reduce_method", reduce_method)
                            mlflow.log_metric("training_time_seconds", training_time)

                            run_id = run.info.run_id
                            st.session_state['latest_run'] = {'run_id': run_id, 'run_name': run_name}
                            st.session_state['X_reduced'] = X_reduced

                        status_text.text("Đang chuẩn bị biểu đồ trực quan...")
                        df_plot = pd.DataFrame(X_reduced, columns=[f"Dim{i+1}" for i in range(n_components)])
                        df_plot['Label'] = y.values
                        progress_bar.progress(95)

                        status_text.text("Đang hoàn tất...")
                        if n_components == 2:
                            fig = px.scatter(
                                df_plot, x="Dim1", y="Dim2", color="Label",
                                title=f"{reduce_method} - Trực quan hóa 2D",
                                width=900, height=600,
                                hover_data=["Label"]
                            )
                        else:
                            fig = px.scatter_3d(
                                df_plot, x="Dim1", y="Dim2", z="Dim3", color="Label",
                                title=f"{reduce_method} - Trực quan hóa 3D",
                                width=900, height=600,
                                hover_data=["Label"]
                            )
                        progress_bar.progress(100)
                        status_text.text("Hoàn tất!")
                        time.sleep(0.5)
                        status_text.empty()
                        progress_bar.empty()

                        st.success(f"Giảm chiều xong! Thời gian: {training_time:.2f} giây.")

                        st.subheader(f"Kết quả Trực quan hóa ({n_components}D)")
                        st.plotly_chart(fig, use_container_width=True)

                        st.subheader("Hiểu biểu đồ này như thế nào?")
                        if reduce_method == "PCA":
                            st.markdown(f"""
                            - **Biểu đồ**: Mỗi điểm là một ảnh chữ số, giảm từ $784$ chiều xuống ${n_components}D$ bằng PCA.  
                            - **Màu sắc**: Mỗi nhãn ($0$-$9$) có một màu riêng.  
                            - **Ý nghĩa**: PCA giữ cấu trúc toàn cục, các điểm cùng nhãn nên nằm gần nhau nếu dữ liệu có tính tuyến tính.  
                            - **Tỷ lệ phương sai giải thích**: ${explained_variance_ratio:.4f}$ (phần dữ liệu được giữ lại).  
                            """, unsafe_allow_html=True)
                        else:
                            st.markdown(f"""
                            - **Biểu đồ**: Mỗi điểm là một ảnh chữ số, giảm từ $784$ chiều xuống ${n_components}D$ bằng t-SNE.  
                            - **Màu sắc**: Mỗi nhãn ($0$-$9$) có một màu riêng.  
                            - **Ý nghĩa**: t-SNE giữ cấu trúc cục bộ, các điểm cùng nhãn thường tạo thành các nhóm rõ ràng hơn PCA.  
                            """, unsafe_allow_html=True)

                        st.subheader("Thông tin chi tiết")
                        with st.expander("Xem chi tiết kết quả", expanded=True):
                            st.markdown("**Thông tin lần chạy:**")
                            st.write(f"- Tên lần chạy: {run_name}")
                            st.write(f"- ID lần chạy: {run_id}")

                            st.markdown("**Cài đặt:**")
                            st.write(f"- Phương pháp: {reduce_method}")
                            st.write(f"- Số chiều: {n_components}")
                            if reduce_method == "t-SNE":
                                st.write(f"- Perplexity: {perplexity}")
                                st.write(f"- Learning Rate: {learning_rate}")
                            st.write(f"- Thời gian chạy: {training_time:.2f} giây")

                            st.markdown("**Kết quả chi tiết:**")
                            if reduce_method == "PCA":
                                st.write(f"- Tỷ lệ phương sai giải thích: {explained_variance_ratio:.4f}")
                            st.write(f"- Số mẫu đã xử lý: {num_samples}")
                except Exception as e:
                    st.error(f"Lỗi khi thực hiện giảm chiều: {e}")

    # Tab Theo dõi kết quả
    with tab_log_info:
        st.header("Theo dõi kết quả")
        st.markdown("""
        Tab này cho phép bạn xem danh sách các lần giảm chiều đã thực hiện. Chọn một lần chạy để xem chi tiết, đổi tên hoặc xóa.
        """, unsafe_allow_html=True)
        
        try:
            client = MlflowClient()
            experiment = client.get_experiment_by_name("MNIST_PCA")
            if not experiment:
                st.error(f"Không tìm thấy experiment 'MNIST_PCA'. Vui lòng kiểm tra lại MLflow tracking URI.")
            else:
                experiment_id = experiment.experiment_id
                runs = client.search_runs(experiment_ids=[experiment_id], order_by=["attributes.start_time DESC"])
                
                if not runs:
                    st.info("Chưa có lần chạy nào được ghi nhận trong experiment 'MNIST_PCA'.")
                else:
                    run_options = {run.info.run_id: run.data.tags.get('mlflow.runName', f"Run_{run.info.run_id}") for run in runs}
                    run_names = list(run_options.values())

                    default_run_name = st.session_state.get('latest_run', {}).get('run_name', run_names[0]) if 'latest_run' in st.session_state else run_names[0]

                    st.subheader("Danh sách run")
                    selected_run_name = st.selectbox(
                        "Chọn run:",
                        options=run_names,
                        index=run_names.index(default_run_name) if default_run_name in run_names else 0,
                        key="main_select",
                        help="Chọn một lần chạy để xem chi tiết, đổi tên hoặc xóa."
                    )
                    selected_run_id = [k for k, v in run_options.items() if v == selected_run_name][0]
                    selected_run = client.get_run(selected_run_id)

                    st.subheader("Đổi tên Run")
                    new_run_name = st.text_input(
                        "Nhập tên mới:",
                        value=selected_run_name,
                        key="rename_input"
                    )
                    if st.button("Cập nhật tên", key="rename_button"):
                        if new_run_name.strip() and new_run_name.strip() != selected_run_name:
                            with st.spinner("Đang cập nhật tên..."):
                                client.set_tag(selected_run_id, "mlflow.runName", new_run_name.strip())
                                if 'latest_run' in st.session_state and st.session_state['latest_run']['run_id'] == selected_run_id:
                                    st.session_state['latest_run']['run_name'] = new_run_name.strip()
                                st.success(f"Đã đổi tên thành: {new_run_name.strip()}")
                                time.sleep(0.5)
                                st.rerun()
                        elif not new_run_name.strip():
                            st.warning("Vui lòng nhập tên hợp lệ.")
                        else:
                            st.info("Tên mới trùng với tên hiện tại.")

                    st.subheader("Xóa Run")
                    if st.button("Xóa lần chạy", key="delete_button"):
                        with st.spinner("Đang xóa lần chạy..."):
                            client.delete_run(selected_run_id)
                            if 'latest_run' in st.session_state and st.session_state['latest_run']['run_id'] == selected_run_id:
                                del st.session_state['latest_run']
                            st.success(f"Đã xóa: {selected_run_name}")
                            time.sleep(0.5)
                            st.rerun()

                    st.subheader("Thông tin chi tiết của Run")
                    st.write(f"**Tên lần chạy:** {selected_run_name}")
                    st.write(f"**ID lần chạy:** {selected_run_id}")
                    st.write(f"**Thời gian bắt đầu:** {datetime.fromtimestamp(selected_run.info.start_time / 1000)}")

                    st.markdown("**Tham số:**", unsafe_allow_html=True)
                    if selected_run.data.params:
                        st.json(selected_run.data.params, expanded=True)
                    else:
                        st.write("Không có tham số được ghi nhận.")

                    st.markdown("**Kết quả:**", unsafe_allow_html=True)
                    if selected_run.data.metrics:
                        reduce_method = selected_run.data.params.get("reduce_method", "Không xác định")
                        metrics_display = {}

                        training_time = selected_run.data.metrics.get("training_time_seconds", "N/A")
                        metrics_display["Thời gian thực hiện (giây)"] = f"{float(training_time):.2f}" if training_time != "N/A" else "N/A"

                        if reduce_method == "PCA":
                            evr = selected_run.data.metrics.get("explained_variance_ratio", "N/A")
                            metrics_display["Tỷ lệ phương sai giải thích"] = f"{float(evr):.4f}" if evr != "N/A" else "N/A"

                        st.json(metrics_display, expanded=True)
                    else:
                        st.write("Không có kết quả được ghi nhận.")

                    # Thêm nút liên kết tới MLflow UI
                    st.subheader("Truy cập MLflow UI")
                    mlflow_url = "https://dagshub.com/huykibo/streamlit_mlflow.mlflow"  # Thay bằng URL MLflow của bạn nếu khác
                    if st.button("Mở MLflow UI trên Dagshub"):
                        st.markdown(f'[Click để mở MLflow UI]({mlflow_url})', unsafe_allow_html=True)
        except Exception as e:
            st.error(f"Lỗi kết nối MLflow: {e}. Vui lòng kiểm tra MLFLOW_TRACKING_URI và thông tin xác thực.")

if __name__ == "__main__":
    run_mnist_dimension_reduction_app()