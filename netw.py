import os
import mlflow
import streamlit as st
import openml
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.neural_network import MLPClassifier
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from mlflow.tracking import MlflowClient
from streamlit_drawable_canvas import st_canvas
from datetime import datetime
import time
import requests

# Hàm tải dữ liệu MNIST
def fetch_mnist_data():
    mnist = openml.datasets.get_dataset(554)
    X, y, _, _ = mnist.get_data(target=mnist.default_target_attribute)
    return X, y

# Hàm kiểm tra và chuẩn hóa dữ liệu pixel về [0, 255]
def validate_and_fix_pixels(X, name="dữ liệu"):
    invalid_mask = (X < 0) | (X > 255)
    if np.any(invalid_mask):
        st.warning(f"Phát hiện giá trị pixel không hợp lệ trong {name} (ngoài [0, 255]). Đang chuẩn hóa...")
        X_fixed = np.clip(X, 0, 255)
        return X_fixed, True
    return X, False

def run_mnist_neural_network_app():
    # Thiết lập MLflow
    mlflow_tracking_uri = "https://dagshub.com/huykibo/streamlit_mlflow.mlflow"
    try:
        os.environ["MLFLOW_TRACKING_USERNAME"] = st.secrets["mlflow"]["MLFLOW_TRACKING_USERNAME"]
        os.environ["MLFLOW_TRACKING_PASSWORD"] = st.secrets["mlflow"]["MLFLOW_TRACKING_PASSWORD"]
        mlflow.set_tracking_uri(mlflow_tracking_uri)
    except KeyError as e:
        st.error(f"Lỗi: Không tìm thấy khóa {e} trong st.secrets.")
        st.stop()

    try:
        response = requests.get(mlflow_tracking_uri, timeout=5)
        if response.status_code != 200:
            st.error(f"Kết nối MLflow thất bại. Mã trạng thái: {response.status_code}.")
            st.stop()
    except requests.exceptions.RequestException as e:
        st.error(f"Không thể kết nối MLflow: {e}.")
        st.stop()

    EXPERIMENT_ID = "5"

    try:
        client = MlflowClient()
        experiment = client.get_experiment(EXPERIMENT_ID)
        if experiment is None:
            st.error(f"Experiment ID {EXPERIMENT_ID} không tồn tại.")
            st.stop()
    except Exception as e:
        st.error(f"Lỗi truy xuất Experiment ID {EXPERIMENT_ID}: {e}.")
        st.stop()

    st.title("Phân loại Chữ số MNIST với Neural Network")

    st.markdown("""
        <script src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.9/MathJax.js?config=TeX-MML-AM_CHTML" async></script>
        <style>
            .tooltip {
                position: relative;
                display: inline-block;
                cursor: pointer;
                color: #1f77b4;
                font-weight: bold;
                margin-left: 5px;
            }
            .tooltip .tooltiptext {
                visibility: hidden;
                width: 400px;
                background-color: #f9f9f9;
                color: #333;
                text-align: left;
                border-radius: 6px;
                padding: 10px;
                position: absolute;
                z-index: 1;
                right: 105%;
                top: 50%;
                transform: translateY(-50%);
                opacity: 0;
                transition: opacity 0.3s;
                border: 1px solid #ccc;
                font-size: 0.9em;
                line-height: 1.4;
            }
            .tooltip:hover .tooltiptext {
                visibility: visible;
                opacity: 1;
            }
        </style>
    """, unsafe_allow_html=True)

    tabs = st.tabs(["Thông tin", "Tải dữ liệu", "Xử lý dữ liệu", "Chia dữ liệu", "Huấn luyện/Đánh giá", "Demo dự đoán", "Thông tin huấn luyện"])
    tab_info, tab_load, tab_preprocess, tab_split, tab_train_eval, tab_demo, tab_log_info = tabs

    with tab_info:
        st.header("Giới thiệu về Ứng dụng và Mạng Neural Network")
        st.markdown("""
        Chào bạn! Đây là ứng dụng phân loại chữ số viết tay từ tập dữ liệu **MNIST** bằng **Mạng nơ-ron nhân tạo (Neural Network)**. Hãy khám phá các tính năng và cách hoạt động của nó nhé!
        """, unsafe_allow_html=True)

        st.subheader("Chọn thông tin để xem")
        info_option = st.selectbox(
            "",
            [
                "Ứng dụng này là gì và mục tiêu của nó?",
                "Tập dữ liệu MNIST: Đặc điểm và ý nghĩa",
                "Neural Network – Mạng nơ-ron nhân tạo",
                "Công thức đánh giá độ chính xác (Accuracy)"
            ],
            label_visibility="collapsed",
            help="Chọn để xem chi tiết về ứng dụng, dữ liệu, hoặc mô hình."
        )

        if info_option == "Ứng dụng này là gì và mục tiêu của nó?":
            with st.spinner("Đang tải thông tin..."):
                progress_bar = st.progress(0)
                status_text = st.empty()
                for i in [20, 30, 40, 50, 60, 70, 80, 90, 100]:
                    progress_bar.progress(i)
                    status_text.text(f"Đang tải {i}%")
                    time.sleep(0.05)
                st.subheader("📘 1. Ứng dụng này là gì và mục tiêu của nó?")
                st.markdown("""
                Đây là một ứng dụng phân loại chữ số viết tay dựa trên tập dữ liệu **MNIST**, sử dụng **Mạng nơ-ron nhân tạo (Neural Network)**.  
                - **MNIST**: Tập dữ liệu gồm $70,000$ ảnh chữ số từ $0$ đến $9$, mỗi ảnh kích thước $28 \\times 28$ pixel (tổng cộng $784$ đặc trưng).  
                - **Mục tiêu**:  
                  - Xây dựng và huấn luyện một mạng nơ-ron để nhận diện chính xác các chữ số.  
                  - Cung cấp công cụ trực quan để học tập và đánh giá hiệu quả của thuật toán.  

                **Thông tin cơ bản**:  
                - **$784$ đặc trưng**: Mỗi ảnh được biểu diễn dưới dạng vector $784$ chiều (giá trị pixel từ $0$ đến $255$).  
                - **$70,000$ mẫu**: Tổng số ảnh, được chia thành tập huấn luyện và kiểm tra.  
                - **Nhiệm vụ**: Dự đoán nhãn ($0$-$9$) dựa trên đặc trưng pixel.  
                """, unsafe_allow_html=True)
                status_text.text("Đã tải 100%")
                time.sleep(0.5)
                status_text.empty()
                progress_bar.empty()

        elif info_option == "Tập dữ liệu MNIST: Đặc điểm và ý nghĩa":
            with st.spinner("Đang tải thông tin..."):
                progress_bar = st.progress(0)
                status_text = st.empty()
                for i in [20, 30, 40, 50, 60, 70, 80, 90, 100]:
                    progress_bar.progress(i)
                    status_text.text(f"Đang tải {i}%")
                    time.sleep(0.05)
                st.subheader("📘 2. Tập dữ liệu MNIST: Đặc điểm và ý nghĩa")
                st.markdown("""
                **MNIST** là tập dữ liệu chuẩn trong học máy, được tạo bởi Yann LeCun và các cộng sự.  
                - **Đặc điểm**:  
                  - Gồm các ảnh chữ số viết tay từ học sinh trung học và nhân viên điều tra dân số Mỹ.  
                  - Chuẩn hóa thành kích thước $28 \\times 28$ pixel, thang độ xám (giá trị từ $0$ đến $255$).  

                **Ý nghĩa**:  
                - Là bài toán cơ bản để kiểm tra khả năng phân loại của các mô hình học máy.  
                - Đơn giản nhưng đủ phức tạp để đánh giá khả năng phân biệt các lớp tương tự (ví dụ: "$4$" và "$9$").  
                - Phù hợp cho cả người mới bắt đầu và nghiên cứu mô hình phức tạp.  
                """, unsafe_allow_html=True)

                st.subheader("📷 Minh họa dữ liệu MNIST")
                st.markdown("""
                Dưới đây là ảnh minh họa $10$ chữ số từ $0$ đến $9$ từ tập dữ liệu MNIST để bạn hình dung. Mỗi chữ số được biểu diễn dưới dạng ma trận $28 \\times 28$ pixel.
                """, unsafe_allow_html=True)
                try:
                    mnist_image = Image.open("mnist.png")
                    st.image(mnist_image, caption="Ảnh minh họa $10$ chữ số từ $0$ đến $9$ trong MNIST", width=800)
                except FileNotFoundError:
                    st.error("Không tìm thấy file `mnist.png`. Vui lòng kiểm tra đường dẫn.")
                except Exception as e:
                    st.error(f"Lỗi khi tải ảnh: {e}")
                status_text.text("Đã tải 100%")
                time.sleep(0.5)
                status_text.empty()
                progress_bar.empty()

        elif info_option == "Neural Network – Mạng nơ-ron nhân tạo":
            with st.spinner("Đang tải thông tin..."):
                progress_bar = st.progress(0)
                status_text = st.empty()
                for i in [20, 30, 40, 50, 60, 70, 80, 90, 100]:
                    progress_bar.progress(i)
                    status_text.text(f"Đang tải {i}%")
                    time.sleep(0.05)
                st.subheader("📊 3. Neural Network – Mạng nơ-ron nhân tạo")
                st.markdown("""
                **Neural Network (Mạng nơ-ron nhân tạo)** là một mô hình học máy mô phỏng cách hoạt động của mạng nơ-ron sinh học trong não người.  
                - **Cấu trúc**: Gồm các **nơ-ron nhân tạo** (nodes) được tổ chức thành các **lớp (layers)**:  
                  - **Lớp đầu vào (Input Layer)**: Nhận dữ liệu ($784$ pixel từ ảnh MNIST).  
                  - **Lớp ẩn (Hidden Layers)**: Xử lý thông tin bằng cách kết hợp tuyến tính và áp dụng hàm kích hoạt phi tuyến.  
                  - **Lớp đầu ra (Output Layer)**: Đưa ra dự đoán (nhãn từ $0$-$9$).  

                Neural Network đặc biệt hiệu quả với bài toán MNIST nhờ khả năng học các đặc trưng phức tạp từ dữ liệu hình ảnh.
                """, unsafe_allow_html=True)

                st.subheader("🛠️ Các bước thực hiện trong Neural Network")
                st.markdown("""
                1. **Khởi tạo mô hình**:  
                   - Xác định cấu trúc mạng (số lớp ẩn, số nơ-ron mỗi lớp).  
                   - Khởi tạo **trọng số** $W$ và **bias** $b$ ngẫu nhiên (thường từ phân phối Gaussian).  
                """, unsafe_allow_html=True)
                try:
                    st.image(os.path.join("plnw", "step1_init.png"), caption="Minh họa Bước 1: Khởi tạo mô hình", width=600)
                except FileNotFoundError:
                    st.error("Không tìm thấy ảnh minh họa cho Bước 1.")

                st.markdown("""
                2. **Lan truyền thuận (Feedforward)**:  
                   - Tính giá trị dự đoán $\\hat{Y}$ từ dữ liệu đầu vào $X$:  
                     - **Lớp đầu vào**: $A^{(0)} = X$ (ma trận $N \\times 784$, $N$ là số mẫu).  
                     - **Cho mỗi lớp $l$**:  
                       - Tổng tuyến tính:  
                         $$ Z^{(l)} = A^{(l-1)} \\cdot W^{(l)} + b^{(l)} $$  
                       - Áp dụng hàm kích hoạt:  
                         $$ A^{(l)} = \\sigma(Z^{(l)}) $$  
                     - **Lớp đầu ra**: $\\hat{Y} = A^{(L)}$ (ma trận $N \\times 10$).  
                   - Ví dụ hàm kích hoạt **sigmoid**:  
                     $$ \\sigma(z) = \\frac{1}{1 + e^{-z}} $$
                """, unsafe_allow_html=True)
                try:
                    st.image(os.path.join("plnw", "step2_feedforward.png"), caption="Minh họa Bước 2: Lan truyền thuận", width=600)
                except FileNotFoundError:
                    st.error("Không tìm thấy ảnh minh họa cho Bước 2.")

                st.markdown("""
                3. **Tính hàm mất mát (Loss Function)**:  
                   - Đo độ sai lệch giữa $\\hat{Y}$ và $Y$ (giá trị thực). Với MNIST, dùng **Cross-Entropy**:  
                     $$ L = -\\frac{1}{N} \\sum_{i=1}^{N} \\sum_{j=0}^{9} y_{ij} \\cdot \\log(\\hat{y}_{ij}) $$  
                   - Trong đó:  
                     - $y_{ij}$: Nhãn thực (dạng one-hot encoded).  
                     - $\\hat{y}_{ij}$: Xác suất dự đoán cho lớp $j$.  
                """, unsafe_allow_html=True)
                try:
                    st.image(os.path.join("plnw", "step3_loss.png"), caption="Minh họa Bước 3: Tính hàm mất mát", width=600)
                except FileNotFoundError:
                    st.error("Không tìm thấy ảnh minh họa cho Bước 3.")

                st.markdown("""
                4. **Lan truyền ngược (Backpropagation)**:  
                   - Tính đạo hàm của $L$ theo $W^{(l)}$ và $b^{(l)}$ để cập nhật tham số:  
                     - Tại **Lớp đầu ra**:  
                       $$ \\delta^{(L)} = \\hat{Y} - Y $$  
                     - Tại **Lớp ẩn**:  
                       $$ \\delta^{(l)} = (\\delta^{(l+1)} \\cdot (W^{(l+1)})^T) \\odot \\sigma'(Z^{(l)}) $$  
                     - $\\sigma'(z)$: Đạo hàm hàm kích hoạt (với sigmoid: $\\sigma'(z) = \\sigma(z) \\cdot (1 - \\sigma(z))$).  
                     - Đạo hàm theo trọng số và bias:  
                       $$ \\frac{\\partial L}{\\partial W^{(l)}} = (A^{(l-1)})^T \\cdot \\delta^{(l)} $$  
                       $$ \\frac{\\partial L}{\\partial b^{(l)}} = \\sum_{i=1}^{N} \\delta^{(l)}_i $$
                """, unsafe_allow_html=True)
                try:
                    st.image(os.path.join("plnw", "step4_backprop.png"), caption="Minh họa Bước 4: Lan truyền ngược", width=600)
                except FileNotFoundError:
                    st.error("Không tìm thấy ảnh minh họa cho Bước 4.")

                st.markdown("""
                5. **Cập nhật tham số (Gradient Descent)**:  
                   - Điều chỉnh $W$ và $b$ để giảm mất mát:  
                     $$ W^{(l)} = W^{(l)} - \\eta \\cdot \\frac{\\partial L}{\\partial W^{(l)}} $$  
                     $$ b^{(l)} = b^{(l)} - \\eta \\cdot \\frac{\\partial L}{\\partial b^{(l)}} $$  
                   - Trong đó: $\\eta$ là **tốc độ học (learning rate)**.  
                """, unsafe_allow_html=True)
                try:
                    st.image(os.path.join("plnw", "step5_gradient.png"), caption="Minh họa Bước 5: Cập nhật tham số", width=600)
                except FileNotFoundError:
                    st.error("Không tìm thấy ảnh minh họa cho Bước 5.")

                st.markdown("""
                6. **Lặp lại**:  
                   - Quay lại bước $2$ qua nhiều **epoch** cho đến khi $L$ hội tụ.  
                """, unsafe_allow_html=True)
                try:
                    st.image(os.path.join("plnw", "step6_repeat_improved.png"), caption="Minh họa Bước 6: Lặp lại", width=600)
                except FileNotFoundError:
                    st.error("Không tìm thấy ảnh minh họa cho Bước 6.")

                st.subheader("⚙️ Các tham số cơ bản và công dụng")
                st.markdown("""
                Dưới đây là các tham số cơ bản bạn sẽ sử dụng để điều chỉnh mô hình trong ứng dụng này:  

                - **Số lớp ẩn (Number of Hidden Layers)**:  
                  - **Ý nghĩa**: Quyết định độ sâu của mạng (từ $1$ đến $3$ lớp).  
                  - **Công dụng**: Nhiều lớp ẩn giúp học đặc trưng phức tạp hơn, nhưng tăng thời gian tính toán.  
                  - **Ví dụ**: $1$ lớp ẩn cho bài toán đơn giản, $2$-$3$ lớp cho độ chính xác cao hơn.  

                - **Số nơ-ron mỗi lớp (Neurons per Layer)**:  
                  - **Ý nghĩa**: Số đơn vị xử lý trong mỗi lớp ẩn (từ $10$ đến $500$).  
                  - **Công dụng**: Nhiều nơ-ron tăng khả năng học, nhưng có thể gây quá tải.  
                  - **Công thức liên quan**: Đầu ra mỗi lớp:  
                    $$ A^{(l)} = \\sigma(W^{(l)} \\cdot A^{(l-1)} + b^{(l)}) $$  

                - **Tốc độ học (Learning Rate, $\\eta$)**:  
                  - **Ý nghĩa**: Tốc độ cập nhật trọng số (ví dụ: $0.01$, $0.001$, $0.0005$, $0.0001$).  
                  - **Công dụng**: Giá trị nhỏ học chậm nhưng ổn định, giá trị lớn học nhanh nhưng có thể không hội tụ.  
                  - **Công thức**: Cập nhật trọng số:  
                    $$ W^{(l)} = W^{(l)} - \\eta \\cdot \\frac{\\partial L}{\\partial W^{(l)}} $$  

                - **Số lần lặp tối đa (Max Iterations)**:  
                  - **Ý nghĩa**: Số epoch tối đa để huấn luyện (từ $50$ đến $500$).  
                  - **Công dụng**: Giới hạn số lần mạng học qua dữ liệu. Nhiều lần lặp tăng độ chính xác nhưng tốn thời gian.  

                - **Hàm kích hoạt (Activation Function, $\\sigma$)**:  
                  - **Ý nghĩa**: Quyết định cách nơ-ron xử lý đầu vào (ReLU, Sigmoid, Tanh).  
                  - **Công dụng**: Giúp mạng học đặc trưng phi tuyến.  
                  - **Công thức**:  
                    - ReLU: $$ \\sigma(z) = \\max(0, z) $$  
                    - Sigmoid: $$ \\sigma(z) = \\frac{1}{1 + e^{-z}} $$  
                    - Tanh: $$ \\sigma(z) = \\tanh(z) $$  

                - **Optimizer (Solver)**:  
                  - **Ý nghĩa**: Phương pháp tối ưu hóa trọng số (LBFGS, SGD, Adam).  
                  - **Công dụng**: Điều chỉnh cách mạng cập nhật tham số để giảm mất mát.  
                  - **Ví dụ**:  
                    - **SGD**: Gradient Descent ngẫu nhiên, đơn giản nhưng chậm.  
                    - **Adam**: Nhanh và hiệu quả với dữ liệu lớn.  
                """, unsafe_allow_html=True)

                st.subheader("🟪 Ưu điểm và nhược điểm")
                st.markdown("""
                ##### ✅ **Ưu điểm**:  
                - Học được các đặc trưng phức tạp từ dữ liệu hình ảnh như MNIST.  
                - Linh hoạt với nhiều tham số để tối ưu hóa.  

                ##### ❌ **Nhược điểm**:  
                - Tốn thời gian huấn luyện nếu số mẫu lớn hoặc cấu trúc mạng phức tạp.  
                - Cần điều chỉnh tham số cẩn thận để đạt hiệu quả tốt nhất.  
                """, unsafe_allow_html=True)
                status_text.text("Đã tải 100%")
                time.sleep(0.5)
                status_text.empty()
                progress_bar.empty()

        elif info_option == "Công thức đánh giá độ chính xác (Accuracy)":
            with st.spinner("Đang tải thông tin..."):
                progress_bar = st.progress(0)
                status_text = st.empty()
                for i in [20, 30, 40, 50, 60, 70, 80, 90, 100]:
                    progress_bar.progress(i)
                    status_text.text(f"Đang tải {i}%")
                    time.sleep(0.05)
                st.subheader("📘 4. Công thức đánh giá độ chính xác (Accuracy)")
                st.markdown("""
                Độ chính xác (**Accuracy**) đo tỷ lệ dự đoán đúng:  
                $$ \\text{Accuracy} = \\frac{\\text{Số mẫu dự đoán đúng}}{\\text{Tổng số mẫu}} $$  
                - **Ví dụ**: Dự đoán đúng $92/100$ ảnh → $\\text{Accuracy} = 92\\%$.  
                - **Ý nghĩa**: Với Neural Network, Accuracy đo khả năng mô hình phân loại đúng các chữ số dựa trên đặc trưng pixel học được.  
                """, unsafe_allow_html=True)
                status_text.text("Đã tải 100%")
                time.sleep(0.5)
                status_text.empty()
                progress_bar.empty()

    with tab_load:
        st.header("Tải Dữ liệu")

        if st.button("Tải dữ liệu MNIST từ OpenML"):
            with st.spinner("Đang tải dữ liệu..."):
                progress_bar = st.progress(0)
                status_text = st.empty()
                for i in [20, 40, 60, 80, 100]:
                    progress_bar.progress(i)
                    status_text.text(f"Đang tải {i}%")
                    time.sleep(0.05)
                try:
                    X, y = fetch_mnist_data()
                    st.session_state['full_data'] = (X, y)
                    with mlflow.start_run(experiment_id=EXPERIMENT_ID, run_name="Data_Load"):
                        mlflow.log_param("total_samples", X.shape[0])
                    st.success("Tải dữ liệu thành công!")
                    st.write("Kích thước dữ liệu gốc:", X.shape)
                    status_text.empty()
                    progress_bar.empty()
                except Exception as e:
                    st.error(f"Không thể tải dữ liệu: {e}")
                    status_text.empty()
                    progress_bar.empty()

        if 'full_data' in st.session_state:
            X_full, y_full = st.session_state['full_data']
            
            st.subheader("Chọn số lượng mẫu dữ liệu")
            st.markdown("""
            Dựa trên bài toán phân loại MNIST với Neural Network, đây là các gợi ý:
            - **100 mẫu**: Dành cho thử nghiệm nhanh, thời gian huấn luyện rất ngắn (~vài giây), nhưng độ chính xác thấp.
            - **1,000 mẫu**: Phù hợp để kiểm tra mô hình cơ bản, thời gian huấn luyện ngắn (~10-20 giây), độ chính xác trung bình.
            - **10,000 mẫu**: Cân bằng giữa tốc độ và hiệu suất, thời gian huấn luyện vừa phải (~1-2 phút), độ chính xác khá tốt.
            - **50,000 mẫu**: Dành cho huấn luyện chuyên sâu, thời gian lâu hơn (~5-10 phút), độ chính xác cao.
            """)
            
            sample_options = {
                "100 mẫu (Thử nghiệm nhanh)": 100,
                "1,000 mẫu (Kiểm tra cơ bản)": 1000,
                "10,000 mẫu (Cân bằng hiệu suất)": 10000,
                "50,000 mẫu (Huấn luyện chuyên sâu)": 50000
            }
            selected_option = st.selectbox("Chọn số lượng mẫu:", list(sample_options.keys()))
            num_samples = sample_options[selected_option]

            if st.button("Chốt số lượng mẫu"):
                with st.spinner(f"Đang lấy {num_samples} mẫu..."):
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    for i in [20, 40, 60, 80, 100]:
                        progress_bar.progress(i)
                        status_text.text(f"Đang xử lý {i}%")
                        time.sleep(0.05)
                    indices = np.random.choice(len(X_full), size=num_samples, replace=False)
                    X_sampled = X_full.iloc[indices]
                    y_sampled = y_full.iloc[indices]
                    st.session_state['data'] = (X_sampled, y_sampled)
                    with mlflow.start_run(experiment_id=EXPERIMENT_ID, run_name="Data_Sample"):
                        mlflow.log_param("num_samples", num_samples)
                    st.success(f"Đã chốt {num_samples} mẫu!")
                    status_text.empty()
                    progress_bar.empty()

    with tab_preprocess:
        st.header("Xử lý Dữ liệu")

        if 'data' not in st.session_state:
            st.info("Vui lòng tải và chốt số lượng mẫu trước.")
        else:
            X, y = st.session_state['data']
            if "data_original" not in st.session_state:
                st.session_state["data_original"] = (X.copy(), y.copy())

            st.subheader("Dữ liệu Gốc")
            fig, axes = plt.subplots(2, 5, figsize=(10, 4))
            for i, ax in enumerate(axes.flat):
                ax.imshow(X.iloc[i].values.reshape(28, 28), cmap='gray')
                ax.set_title(f"Label: {y.iloc[i]}")
                ax.axis("off")
            st.pyplot(fig)

            col1, col2 = st.columns([3, 1])
            with col1:
                if st.button("Normalization", key="normalize_btn"):
                    with st.spinner("Đang chuẩn hóa dữ liệu về [0, 1]..."):
                        progress_bar = st.progress(0)
                        status_text = st.empty()
                        for i in [20, 40, 60, 80, 100]:
                            progress_bar.progress(i)
                            status_text.text(f"Đang xử lý {i}%")
                            time.sleep(0.05)
                        X_norm = X / 255.0
                        st.session_state["data_processed"] = (X_norm, y)
                        st.success("Đã chuẩn hóa dữ liệu về [0, 1]!")
                        status_text.empty()
                        progress_bar.empty()
                        st.rerun()
            with col2:
                st.markdown("""
                    <div class="tooltip">? (Norm)
                        <span class="tooltiptext">
                            Đưa dữ liệu về [0, 1] bằng cách chia cho 255.<br>
                            Công dụng: Đảm bảo thang đo đồng nhất cho Neural Network.
                        </span>
                    </div>
                """, unsafe_allow_html=True)

            if "data_processed" in st.session_state:
                X_processed, y_processed = st.session_state["data_processed"]
                st.subheader("Dữ liệu đã xử lý")
                fig, axes = plt.subplots(2, 5, figsize=(10, 4))
                for i, ax in enumerate(axes.flat):
                    ax.imshow(X_processed.iloc[i].values.reshape(28, 28), cmap='gray')
                    ax.set_title(f"Label: {y_processed.iloc[i]}")
                    ax.axis("off")
                st.pyplot(fig)

    with tab_split:
        st.header("Chia Tập Dữ liệu")

        if 'data' not in st.session_state:
            st.info("Vui lòng tải và xử lý dữ liệu trước.")
        else:
            data_source = st.session_state.get('data_processed', st.session_state['data'])
            X, y = data_source
            total_samples = len(X)
            st.write(f"Tổng số mẫu: {total_samples}")

            test_pct = st.slider("Tỷ lệ Test (%)", 0, 50, 20)
            valid_pct = st.slider("Tỷ lệ Validation (%)", 0, 50, 20)

            test_size = test_pct / 100
            X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
            valid_size = (valid_pct / 100) / (1 - test_size) if test_size < 1 else 0
            X_train, X_valid, y_train, y_valid = train_test_split(X_temp, y_temp, test_size=valid_size, random_state=42)

            st.write(f"Train: {len(X_train)}, Validation: {len(X_valid)}, Test: {len(X_test)}")
            if st.button("Xác nhận", key="confirm_split_button"):
                with st.spinner("Đang chia dữ liệu..."):
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    for i in [20, 40, 60, 80, 100]:
                        progress_bar.progress(i)
                        status_text.text(f"Đang xử lý {i}%")
                        time.sleep(0.05)
                    st.session_state['split_data'] = {
                        "X_train": X_train, "y_train": y_train,
                        "X_valid": X_valid, "y_valid": y_valid,
                        "X_test": X_test, "y_test": y_test
                    }
                    st.success("Đã chia dữ liệu!")
                    status_text.empty()
                    progress_bar.empty()

    with tab_train_eval:
        st.header("Huấn luyện và Đánh giá Mô hình")

        if 'split_data' not in st.session_state:
            st.info("Vui lòng chia dữ liệu trước.")
        else:
            X_train = st.session_state['split_data']["X_train"]
            y_train = st.session_state['split_data']["y_train"]
            X_valid = st.session_state['split_data']["X_valid"]
            y_valid = st.session_state['split_data']["y_valid"]
            X_test = st.session_state['split_data']["X_test"]
            y_test = st.session_state['split_data']["y_test"]

            num_samples = len(X_train)
            st.write(f"**Số mẫu huấn luyện**: {num_samples}")

            def get_optimal_params(num_samples):
                if num_samples <= 100:
                    return {"hidden_layer_sizes": (16,), "learning_rate_init": 0.01, "max_iter": 20, 
                            "activation": "relu", "solver": "adam", "batch_size": 32}
                elif num_samples <= 1000:
                    return {"hidden_layer_sizes": (32,), "learning_rate_init": 0.005, "max_iter": 50, 
                            "activation": "relu", "solver": "adam", "batch_size": 64}
                elif num_samples <= 10000:
                    return {"hidden_layer_sizes": (64, 32), "learning_rate_init": 0.001, "max_iter": 75, 
                            "activation": "relu", "solver": "adam", "batch_size": 128}
                else:
                    return {"hidden_layer_sizes": (128, 64), "learning_rate_init": 0.0005, "max_iter": 100, 
                            "activation": "relu", "solver": "adam", "batch_size": 256}

            if "optimal_params" not in st.session_state:
                st.session_state["optimal_params"] = get_optimal_params(num_samples)
            params = st.session_state.get("training_params", st.session_state["optimal_params"].copy())

            st.subheader("⚙️ Cấu hình tham số mô hình")
            st.markdown("""
            | Số mẫu       | Số lớp ẩn | Kích thước lớp ẩn | Tốc độ học | Số lần lặp | Hàm kích hoạt | Trình tối ưu | Kích thước batch |
            |--------------|-----------|-------------------|------------|------------|---------------|--------------|------------------|
            | ≤100         | 1         | 16                | 0.01       | 20         | ReLU          | adam         | 32               |
            | ≤1,000       | 1         | 32                | 0.005      | 50         | ReLU          | adam         | 64               |
            | ≤10,000      | 2         | (64, 32)          | 0.001      | 75         | ReLU          | adam         | 128              |
            | >10,000      | 2         | (128, 64)         | 0.0005     | 100        | ReLU          | adam         | 256              |
            """, unsafe_allow_html=True)

            st.info(f"Tham số tối ưu cho {num_samples} mẫu: {st.session_state['optimal_params']}")

            col_param1, col_param2 = st.columns(2)
            with col_param1:
                with st.expander("Cấu trúc mạng"):
                    num_hidden_layers = st.number_input("Số lớp ẩn", min_value=1, max_value=2, value=len(params["hidden_layer_sizes"]))
                    hidden_size = st.number_input("Số nơ-ron mỗi lớp", min_value=16, max_value=128, value=params["hidden_layer_sizes"][0])
                    params["hidden_layer_sizes"] = tuple([hidden_size] * num_hidden_layers)
                    params["activation"] = st.selectbox("Hàm kích hoạt", ["relu", "sigmoid", "tanh"], 
                                                        index=["relu", "sigmoid", "tanh"].index(params["activation"]))
            with col_param2:
                with st.expander("Tối ưu hóa"):
                    params["learning_rate_init"] = st.selectbox("Tốc độ học", [0.01, 0.005, 0.001, 0.0005], 
                                                                index=[0.01, 0.005, 0.001, 0.0005].index(params["learning_rate_init"]))
                    params["max_iter"] = st.number_input("Số lần lặp", min_value=10, max_value=100, value=params["max_iter"])
                    params["batch_size"] = st.number_input("Kích thước batch", min_value=32, max_value=256, value=params["batch_size"])
                    params["solver"] = st.selectbox("Trình tối ưu", ["adam", "sgd", "lbfgs"], 
                                                    index=["adam", "sgd", "lbfgs"].index(params["solver"]))

            if st.button("🔄 Khôi phục tham số tối ưu"):
                st.session_state["training_params"] = st.session_state["optimal_params"].copy()
                st.success("Đã khôi phục tham số tối ưu!")
                st.rerun()

            st.session_state["training_params"] = params

            if st.button("🚀 Bắt đầu Huấn luyện", type="primary"):
                try:
                    with st.spinner("Đang huấn luyện mô hình..."):
                        progress_bar = st.progress(0)
                        status_text = st.empty()
                        start_time = time.time()

                        status_text.text("Đang chuẩn bị dữ liệu...")
                        progress_bar.progress(20)

                        model = MLPClassifier(**params, verbose=True)
                        status_text.text("Đang huấn luyện mô hình...")
                        for i in [40, 60, 80]:
                            progress_bar.progress(i)
                            status_text.text(f"Đang huấn luyện {i}%")
                            time.sleep(0.05)
                        model.fit(X_train, y_train)

                        status_text.text("Đang đánh giá mô hình...")
                        progress_bar.progress(90)
                        y_valid_pred = model.predict(X_valid)
                        y_test_pred = model.predict(X_test)
                        acc_valid = accuracy_score(y_valid, y_valid_pred)
                        acc_test = accuracy_score(y_test, y_test_pred)
                        cm_valid = confusion_matrix(y_valid, y_valid_pred)
                        cm_test = confusion_matrix(y_test, y_test_pred)

                        status_text.text("Đang lưu kết quả...")
                        progress_bar.progress(100)
                        run_name = f"NeuralNetwork_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                        with mlflow.start_run(experiment_id=EXPERIMENT_ID, run_name=run_name) as run:
                            mlflow.log_params(params)
                            mlflow.log_metric("accuracy_val", acc_valid)
                            mlflow.log_metric("accuracy_test", acc_test)
                            mlflow.log_metric("training_time", time.time() - start_time)

                            st.session_state['model'] = model
                            st.session_state['training_results'] = {
                                'accuracy_val': acc_valid, 'accuracy_test': acc_test,
                                'cm_valid': cm_valid, 'cm_test': cm_test,
                                'run_name': run_name, 'run_id': run.info.run_id,
                                'params': params, 'training_time': time.time() - start_time
                            }

                        st.success(f"Đã huấn luyện xong! Thời gian: {time.time() - start_time:.2f} giây")
                        status_text.empty()
                        progress_bar.empty()
                        st.rerun()

                except Exception as e:
                    st.error(f"Lỗi trong quá trình huấn luyện: {e}")
                    status_text.empty()
                    progress_bar.empty()

            if 'training_results' in st.session_state:
                results = st.session_state['training_results']
                st.subheader("📊 Kết quả Huấn luyện")
                col_result1, col_result2, col_result3 = st.columns(3)
                with col_result1:
                    st.metric("Thời gian huấn luyện", f"{results['training_time']:.2f} giây")
                with col_result2:
                    st.metric("Độ chính xác Validation", f"{results['accuracy_val']*100:.2f}%")
                with col_result3:
                    st.metric("Độ chính xác Test", f"{results['accuracy_test']*100:.2f}%")

                st.subheader("📈 Ma trận Nhầm lẫn")
                col_cm1, col_cm2 = st.columns(2)
                with col_cm1:
                    fig, ax = plt.subplots(figsize=(6, 5))
                    sns.heatmap(results['cm_valid'], annot=True, fmt="d", cmap="Blues", ax=ax)
                    ax.set_title("Validation")
                    st.pyplot(fig)
                with col_cm2:
                    fig, ax = plt.subplots(figsize=(6, 5))
                    sns.heatmap(results['cm_test'], annot=True, fmt="d", cmap="Blues", ax=ax)
                    ax.set_title("Test")
                    st.pyplot(fig)
                
                st.subheader("ℹ️ Thông tin Chi tiết")
                with st.expander("Xem chi tiết", expanded=False):
                    st.markdown("**Thông tin lần chạy:**")
                    st.write(f"- Tên: {results['run_name']}")
                    st.write(f"- ID: {results['run_id']}")
                    st.write(f"- Thời gian huấn luyện: {results['training_time']:.2f} giây")
                    st.write(f"- Độ chính xác Validation: {results['accuracy_val']*100:.2f}%")
                    st.write(f"- Độ chính xác Test: {results['accuracy_test']*100:.2f}%")
                    st.markdown("**Tham số đã chọn:**")
                    st.json({
                        "Số lớp ẩn": len(results['params']['hidden_layer_sizes']),
                        "Số nơ-ron mỗi lớp": results['params']['hidden_layer_sizes'],
                        "Tốc độ học": results['params']['learning_rate_init'],
                        "Số lần lặp": results['params']['max_iter'],
                        "Kích thước batch": results['params']['batch_size'],
                        "Hàm kích hoạt": results['params']['activation'],
                        "Trình tối ưu": results['params']['solver']
                    })

    with tab_demo:
        st.header("🖼️ Demo Dự đoán Chữ số")

        # Kiểm tra điều kiện tiên quyết
        if 'split_data' not in st.session_state or 'model' not in st.session_state:
            st.info("Vui lòng huấn luyện mô hình trước khi sử dụng Demo.")
        else:
            # CSS tùy chỉnh để cải thiện giao diện (bỏ khung trắng)
            st.markdown("""
                <style>
                    .prediction-box {
                        margin-top: 10px;
                    }
                    .mode-title {
                        font-size: 1.2em;
                        font-weight: bold;
                        color: #1f77b4;
                        margin-bottom: 10px;
                    }
                </style>
            """, unsafe_allow_html=True)

            # Thanh chọn chế độ
            mode = st.selectbox("Chọn phương thức dự đoán:", 
                               ["Dữ liệu Test", "Upload ảnh", "Vẽ số"], 
                               help="Chọn cách bạn muốn thử nghiệm dự đoán: từ dữ liệu có sẵn, ảnh tải lên, hoặc tự vẽ.")

            # Hàm tiền xử lý dữ liệu đầu vào
            def preprocess_input(data, is_normalized):
                data, fixed = validate_and_fix_pixels(data, "dữ liệu đầu vào")
                if fixed:
                    st.success("Đã chuẩn hóa dữ liệu về [0, 255]!")
                if not is_normalized:
                    data = data / 255.0
                return data

            # Lấy trạng thái chuẩn hóa và mô hình
            is_normalized = 'data_processed' in st.session_state
            model = st.session_state['model']

            # Chế độ 1: Dữ liệu Test
            if mode == "Dữ liệu Test":
                st.markdown('<p class="mode-title">Dự đoán từ Dữ liệu Test</p>', unsafe_allow_html=True)
                X_test = st.session_state['split_data']["X_test"]
                y_test = st.session_state['split_data']["y_test"]

                col_select, col_display = st.columns([3, 2])
                with col_select:
                    idx = st.slider("Chọn mẫu Test", 0, len(X_test) - 1, 0, 
                                   help=f"Chọn một mẫu từ {len(X_test)} mẫu trong tập Test.")
                with col_display:
                    st.write("**Ảnh mẫu Test:**")
                    fig, ax = plt.subplots(figsize=(2, 2))
                    ax.imshow(X_test.iloc[idx].values.reshape(28, 28), cmap='gray')
                    ax.axis('off')
                    st.pyplot(fig)
                    st.write(f"**Nhãn thực tế:** {y_test.iloc[idx]}")

                if st.button("🔍 Dự đoán", key="predict_test"):
                    with st.spinner("Đang dự đoán..."):
                        sample = X_test.iloc[idx].values.reshape(1, -1)
                        sample_processed = preprocess_input(sample, is_normalized)
                        prediction = model.predict(sample_processed)[0]
                        proba = model.predict_proba(sample_processed)[0]
                        max_proba = np.max(proba) * 100

                        # Hiển thị kết quả (không có khung trắng)
                        st.markdown(f"""
                            <div class="prediction-box">
                                <strong>Dự đoán:</strong> {prediction}<br>
                                <strong>Xác suất:</strong> {max_proba:.2f}%<br>
                                <strong>Nhãn thực tế:</strong> {y_test.iloc[idx]}
                            </div>
                        """, unsafe_allow_html=True)

                        # Biểu đồ xác suất
                        fig, ax = plt.subplots(figsize=(6, 3))
                        sns.barplot(x=np.arange(10), y=proba, palette="Blues_d", ax=ax)
                        ax.set_title("Xác suất dự đoán cho từng lớp")
                        ax.set_xlabel("Chữ số (0-9)")
                        ax.set_ylabel("Xác suất")
                        st.pyplot(fig)

            # Chế độ 2: Upload ảnh
            elif mode == "Upload ảnh":
                st.markdown('<p class="mode-title">Dự đoán từ Ảnh Tải lên</p>', unsafe_allow_html=True)
                st.markdown("**Hướng dẫn:** Tải lên ảnh chữ số (28x28, thang độ xám) để dự đoán.")

                uploaded_images = st.file_uploader("Chọn ảnh (PNG/JPG)", type=["png", "jpg"], 
                                                  accept_multiple_files=True, 
                                                  help="Tải lên nhiều ảnh nếu muốn.")

                if uploaded_images:
                    for i, uploaded_image in enumerate(uploaded_images):
                        try:
                            # Xử lý và hiển thị ảnh
                            img = Image.open(uploaded_image).convert('L').resize((28, 28))
                            img_array = np.array(img).flatten().reshape(1, -1)

                            col_img, col_btn = st.columns([1, 2])
                            with col_img:
                                st.image(img, caption=f"Ảnh {i+1}", width=150)
                            with col_btn:
                                if st.button(f"Dự đoán ảnh {i+1}", key=f"predict_upload_{i}"):
                                    with st.spinner(f"Đang xử lý ảnh {i+1}..."):
                                        img_processed = preprocess_input(img_array, is_normalized)
                                        prediction = model.predict(img_processed)[0]
                                        proba = model.predict_proba(img_processed)[0]
                                        max_proba = np.max(proba) * 100

                                        # Hiển thị kết quả (không có khung trắng)
                                        st.markdown(f"""
                                            <div class="prediction-box">
                                                <strong>Dự đoán:</strong> {prediction}<br>
                                                <strong>Xác suất:</strong> {max_proba:.2f}%
                                            </div>
                                        """, unsafe_allow_html=True)

                                        # Biểu đồ xác suất
                                        fig, ax = plt.subplots(figsize=(6, 3))
                                        sns.barplot(x=np.arange(10), y=proba, palette="Blues_d", ax=ax)
                                        ax.set_title(f"Xác suất dự đoán cho ảnh {i+1}")
                                        ax.set_xlabel("Chữ số (0-9)")
                                        ax.set_ylabel("Xác suất")
                                        st.pyplot(fig)
                        except Exception as e:
                            st.error(f"Lỗi khi xử lý ảnh {i+1}: {e}")

            # Chế độ 3: Vẽ số (sử dụng lại cách vẽ từ mã cũ)
            elif mode == "Vẽ số":
                st.markdown('<p class="mode-title">Dự đoán từ Hình vẽ</p>', unsafe_allow_html=True)
                st.write("Vẽ chữ số từ 0-9:")

                # Canvas vẽ (từ mã cũ)
                canvas_result = st_canvas(fill_color="black", stroke_width=20, stroke_color="white", 
                                          background_color="black", width=280, height=280, drawing_mode="freedraw", key="canvas")
                col1, col2 = st.columns([1, 1])
                with col1:
                    if st.button("Dự đoán số đã vẽ"):
                        if canvas_result.image_data is not None and np.any(canvas_result.image_data):
                            with st.spinner("Đang xử lý..."):
                                for i in [20, 40, 60, 80, 100]:
                                    time.sleep(0.05)
                                img = Image.fromarray((canvas_result.image_data * 255).astype(np.uint8)).convert('L').resize((28, 28))
                                img_array = np.array(img).flatten().reshape(1, -1)
                                img_array, fixed = validate_and_fix_pixels(img_array, "hình vẽ")
                                if fixed:
                                    st.success("Đã chuẩn hóa hình vẽ về [0, 255]!")
                                if not is_normalized:
                                    img_array = preprocess_input(img_array, is_normalized)
                                prediction = model.predict(img_array)[0]
                                proba = model.predict_proba(img_array)[0]
                                max_proba = np.max(proba) * 100
                                
                                # Hiển thị kết quả (không có khung trắng)
                                st.markdown(f"""
                                    <div class="prediction-box">
                                        <strong>Dự đoán:</strong> {prediction}<br>
                                        <strong>Xác suất:</strong> {max_proba:.2f}%
                                    </div>
                                """, unsafe_allow_html=True)
                                
                                # Hiển thị hình vẽ đã xử lý
                                st.image(img, caption="Hình vẽ của bạn")

                                # Biểu đồ xác suất (giữ lại từ mã mới để trực quan)
                                fig, ax = plt.subplots(figsize=(6, 3))
                                sns.barplot(x=np.arange(10), y=proba, palette="Blues_d", ax=ax)
                                ax.set_title("Xác suất dự đoán")
                                ax.set_xlabel("Chữ số (0-9)")
                                ax.set_ylabel("Xác suất")
                                st.pyplot(fig)
                        else:
                            st.warning("Vui lòng vẽ trước!")
                with col2:
                    if st.button("Xóa Canvas"):
                        st.session_state['canvas_key'] = st.session_state.get('canvas_key', 0) + 1
                        st.rerun()

    with tab_log_info:
        st.header("Theo dõi Kết quả")
        try:
            with st.spinner("Đang tải thông tin huấn luyện..."):
                progress_bar = st.progress(0)
                status_text = st.empty()
                for i in [20, 40, 60, 80, 100]:
                    progress_bar.progress(i)
                    status_text.text(f"Đang tải {i}%")
                    time.sleep(0.05)
                client = MlflowClient()
                runs = client.search_runs(experiment_ids=[EXPERIMENT_ID], order_by=["attributes.start_time DESC"])

                if not runs:
                    st.info(f"Chưa có lần chạy nào trong Experiment ID {EXPERIMENT_ID}.")
                else:
                    run_options = {run.info.run_id: run.data.tags.get('mlflow.runName', f"Run_{run.info.run_id}") for run in runs}
                    selected_run_name = st.selectbox("Chọn run:", list(run_options.values()))
                    selected_run_id = [k for k, v in run_options.items() if v == selected_run_name][0]
                    selected_run = client.get_run(selected_run_id)

                    st.subheader("Đổi tên Run")
                    new_run_name = st.text_input("Nhập tên mới:", value=selected_run_name)
                    if st.button("Cập nhật tên"):
                        client.set_tag(selected_run_id, "mlflow.runName", new_run_name.strip())
                        st.success(f"Đã đổi tên thành: {new_run_name.strip()}")
                        st.rerun()

                    st.subheader("Xóa Run")
                    if st.button("Xóa lần chạy"):
                        client.delete_run(selected_run_id)
                        st.success(f"Đã xóa: {selected_run_name}")
                        st.rerun()

                    st.subheader("Thông tin chi tiết")
                    st.write(f"**Tên:** {selected_run_name}")
                    st.write(f"**ID:** {selected_run_id}")
                    st.write(f"**Thời gian bắt đầu:** {datetime.fromtimestamp(selected_run.info.start_time / 1000)}")
                    st.json(selected_run.data.params, expanded=True)
                    st.json(selected_run.data.metrics, expanded=True)

                status_text.empty()
                progress_bar.empty()
        except Exception as e:
            st.error(f"Lỗi kết nối MLflow: {e}")

if __name__ == "__main__":
    run_mnist_neural_network_app()