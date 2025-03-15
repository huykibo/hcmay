import os
import mlflow
import streamlit as st
import openml
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.neural_network import MLPClassifier
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from mlflow.tracking import MlflowClient
from streamlit_drawable_canvas import st_canvas
from datetime import datetime
import time
import requests

def run_mnist_neural_network_app():
    # Thiết lập MLflow
    mlflow_tracking_uri = "https://dagshub.com/huykibo/streamlit_mlflow.mlflow"
    try:
        os.environ["MLFLOW_TRACKING_USERNAME"] = st.secrets["mlflow"]["MLFLOW_TRACKING_USERNAME"]
        os.environ["MLFLOW_TRACKING_PASSWORD"] = st.secrets["mlflow"]["MLFLOW_TRACKING_PASSWORD"]
        mlflow.set_tracking_uri(mlflow_tracking_uri)
    except KeyError as e:
        st.error(f"Lỗi: Không tìm thấy khóa {e} trong st.secrets. Vui lòng kiểm hình secrets.toml hoặc môi trường triển khai.")
        st.stop()

    # Kiểm tra kết nối đến MLflow server
    try:
        response = requests.get(mlflow_tracking_uri, timeout=5)
        if response.status_code != 200:
            st.error(f"Kết nối đến MLflow server thất bại. Mã trạng thái: {response.status_code}. Vui lòng kiểm tra MLFLOW_TRACKING_URI: {mlflow_tracking_uri}")
            st.stop()
    except requests.exceptions.RequestException as e:
        st.error(f"Không thể kết nối đến MLflow server tại {mlflow_tracking_uri}. Lỗi: {e}. Vui lòng kiểm tra kết nối mạng hoặc URI.")
        st.stop()

    # Định nghĩa Experiment ID
    EXPERIMENT_ID = "5"

    # Kiểm tra xem Experiment ID có tồn tại không
    try:
        client = MlflowClient()
        experiment = client.get_experiment(EXPERIMENT_ID)
        if experiment is None:
            st.error(f"Experiment ID {EXPERIMENT_ID} không tồn tại. Vui lòng kiểm tra Experiment ID trên MLflow UI.")
            st.stop()
        else:
            st.write(f"Kết nối thành công với Experiment ID: {EXPERIMENT_ID} - Tên: {experiment.name}")
    except Exception as e:
        st.error(f"Lỗi khi truy xuất Experiment ID {EXPERIMENT_ID}: {e}. Vui lòng kiểm tra MLflow server hoặc thông tin xác thực.")
        st.stop()

    st.title("Phân loại Chữ số MNIST với Neural Network")

    # CSS cho tooltip và MathJax
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

    # Các tab
    tabs = st.tabs(["Thông tin", "Tải dữ liệu", "Xử lý dữ liệu", "Chia dữ liệu", "Huấn luyện/Đánh giá", "Demo dự đoán", "Thông tin huấn luyện"])
    tab_info, tab_load, tab_preprocess, tab_split, tab_train_eval, tab_demo, tab_log_info = tabs

    # Tab 1: Thông tin
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
                for i in range(0, 91, 10):
                    progress_bar.progress(i)
                    status_text.text(f"Đang tải {i}%...")
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
                progress_bar.progress(100)
                status_text.text("Đã tải 100%!")
                time.sleep(0.5)
                status_text.empty()
                progress_bar.empty()

        elif info_option == "Tập dữ liệu MNIST: Đặc điểm và ý nghĩa":
            with st.spinner("Đang tải thông tin..."):
                progress_bar = st.progress(0)
                status_text = st.empty()
                for i in range(0, 91, 10):
                    progress_bar.progress(i)
                    status_text.text(f"Đang tải {i}%...")
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
                progress_bar.progress(100)
                status_text.text("Đã tải 100%!")
                time.sleep(0.5)
                status_text.empty()
                progress_bar.empty()

        elif info_option == "Neural Network – Mạng nơ-ron nhân tạo":
            with st.spinner("Đang tải thông tin..."):
                progress_bar = st.progress(0)
                status_text = st.empty()
                for i in range(0, 91, 10):
                    progress_bar.progress(i)
                    status_text.text(f"Đang tải {i}%...")
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
                Dưới đây là các tham số bạn sẽ sử dụng để điều chỉnh mô hình trong ứng dụng này, kèm công thức liên quan:  
                
                - **hidden_layer_sizes**:  
                  - **Ý nghĩa**: Số lớp ẩn và số nơ-ron trong mỗi lớp ẩn (ví dụ: $(128, 64)$ cho 2 lớp ẩn với 128 và 64 nơ-ron).  
                  - **Công dụng**: Quyết định cấu trúc mạng, ảnh hưởng đến khả năng học đặc trưng phức tạp. Nhiều lớp và nơ-ron tăng sức mạnh nhưng cũng tăng thời gian tính toán.  
                  - **Công thức liên quan**: Đầu ra mỗi lớp:  
                    $$ A^{(l)} = \\sigma(W^{(l)} \\cdot A^{(l-1)} + b^{(l)}) $$  

                - **learning_rate_init ($\\eta$)**:  
                  - **Ý nghĩa**: Tốc độ học ban đầu (ví dụ: $0.001$).  
                  - **Công dụng**: Điều chỉnh mức độ cập nhật trọng số; nhỏ hơn thì học chậm nhưng ổn định, lớn hơn thì nhanh nhưng có thể không hội tụ.  
                  - **Công thức**: Cập nhật trọng số:  
                    $$ W^{(l)} = W^{(l)} - \\eta \\cdot \\frac{\\partial L}{\\partial W^{(l)}} $$  

                - **max_iter**:  
                  - **Ý nghĩa**: Số lần lặp tối đa (epoch, ví dụ: $200$).  
                  - **Công dụng**: Giới hạn số lần mạng học qua dữ liệu để đạt độ chính xác mong muốn.  

                - **activation ($\\sigma$)**:  
                  - **Ý nghĩa**: Hàm kích hoạt áp dụng cho nơ-ron (ví dụ: ReLU, Sigmoid, Tanh).  
                  - **Công dụng**: Quyết định cách nơ-ron xử lý đầu vào, ảnh hưởng đến khả năng học đặc trưng phi tuyến.  
                  - **Công thức**:  
                    - ReLU: $$ \\sigma(z) = \\max(0, z) $$  
                    - Sigmoid: $$ \\sigma(z) = \\frac{1}{1 + e^{-z}} $$  
                    - Tanh: $$ \\sigma(z) = \\tanh(z) $$  

                - **solver**:  
                  - **Ý nghĩa**: Phương pháp tối ưu hóa (ví dụ: 'lbfgs', 'sgd', 'adam').  
                  - **Công dụng**: Quyết định cách cập nhật trọng số để giảm mất mát.  
                    - **SGD**: Gradient Descent ngẫu nhiên:  
                      $$ W^{(l)} = W^{(l)} - \\eta \\cdot \\frac{\\partial L}{\\partial W^{(l)}} $$  
                    - **Adam**: Kết hợp momentum và RMSProp, thích nghi tốc độ học.  

                - **batch_size**:  
                  - **Ý nghĩa**: Số mẫu trong mỗi lần cập nhật trọng số (ví dụ: $32$).  
                  - **Công dụng**: Ảnh hưởng đến tốc độ và độ ổn định của huấn luyện; batch nhỏ nhanh hơn nhưng nhiễu hơn.  
                  - **Công thức**: Gradient trung bình trên batch:  
                    $$ \\frac{\\partial L}{\\partial W^{(l)}} = \\frac{1}{B} \\sum_{i=1}^{B} (A^{(l-1)}_i)^T \\cdot \\delta^{(l)}_i $$  
                    ($B$ là batch size).  

                - **alpha**:  
                  - **Ý nghĩa**: Hệ số điều chuẩn L2 (ví dụ: $0.0001$).  
                  - **Công dụng**: Giảm overfitting bằng cách phạt các trọng số lớn.  
                  - **Công thức**: Hàm mất mát có điều chuẩn:  
                    $$ L = L_{\\text{data}} + \\frac{\\alpha}{2} \\sum_{l} ||W^{(l)}||^2 $$  
                """, unsafe_allow_html=True)

                st.subheader("📋 Bảng tham số tối ưu dựa trên số mẫu")
                st.markdown("""
                Các tham số tối ưu được tự động chọn dựa trên số lượng mẫu huấn luyện để cân bằng giữa hiệu suất và thời gian tính toán:  
                | Số mẫu       | Hidden Layer Sizes | Learning Rate | Max Iter | Activation | Solver | Batch Size | Alpha   |
                |--------------|--------------------|---------------|----------|------------|--------|------------|---------|
                | <1000        | 50                 | 0.01          | 100      | ReLU       | lbfgs  | auto       | 0.0001  |
                | 1000-5000    | 100                | 0.001         | 200      | ReLU       | adam   | 32         | 0.0001  |
                | 5000-20000   | 200                | 0.0005        | 300      | ReLU       | adam   | 64         | 0.0001  |
                | >20000       | 300                | 0.0001        | 400      | ReLU       | adam   | 128        | 0.0001  |
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
                progress_bar.progress(100)
                status_text.text("Đã tải 100%!")
                time.sleep(0.5)
                status_text.empty()
                progress_bar.empty()

        elif info_option == "Công thức đánh giá độ chính xác (Accuracy)":
            with st.spinner("Đang tải thông tin..."):
                progress_bar = st.progress(0)
                status_text = st.empty()
                for i in range(0, 91, 10):
                    progress_bar.progress(i)
                    status_text.text(f"Đang tải {i}%...")
                    time.sleep(0.05)
                st.subheader("📘 4. Công thức đánh giá độ chính xác (Accuracy)")
                st.markdown("""
                Độ chính xác (**Accuracy**) đo tỷ lệ dự đoán đúng:  
                $$ \\text{Accuracy} = \\frac{\\text{Số mẫu dự đoán đúng}}{\\text{Tổng số mẫu}} $$  
                - **Ví dụ**: Dự đoán đúng $92/100$ ảnh → $\\text{Accuracy} = 92\\%$.  
                - **Ý nghĩa**: Với Neural Network, Accuracy đo khả năng mô hình phân loại đúng các chữ số dựa trên đặc trưng pixel học được.  
                """, unsafe_allow_html=True)
                progress_bar.progress(100)
                status_text.text("Đã tải 100%!")
                time.sleep(0.5)
                status_text.empty()
                progress_bar.empty()

    # Tab 2: Tải dữ liệu
    with tab_load:
        st.header("Tải Dữ liệu")

        if st.button("Tải dữ liệu MNIST từ OpenML"):
            with st.spinner("Đang tải dữ liệu từ OpenML..."):
                progress_bar = st.progress(0)
                status_text = st.empty()
                for i in range(0, 91, 10):
                    progress_bar.progress(i)
                    status_text.text(f"Đang tải {i}%...")
                    time.sleep(0.1)
                try:
                    mnist = openml.datasets.get_dataset(554)
                    X, y, _, _ = mnist.get_data(target=mnist.default_target_attribute)
                    st.session_state['full_data'] = (X, y)
                    with mlflow.start_run(experiment_id=EXPERIMENT_ID, run_name="Data_Load"):
                        mlflow.log_param("total_samples", X.shape[0])
                    st.success("Tải dữ liệu thành công!")
                    st.write("Kích thước dữ liệu gốc:", X.shape)
                    progress_bar.progress(100)
                    status_text.text("Đã tải 100%!")
                    time.sleep(0.5)
                    status_text.empty()
                    progress_bar.empty()
                except Exception as e:
                    st.error(f"Không thể tải dữ liệu: {e}")
                    progress_bar.progress(0)
                    status_text.empty()

        if 'full_data' in st.session_state:
            X_full, y_full = st.session_state['full_data']
            num_samples = st.slider("Chọn số lượng mẫu:", 
                                    min_value=10, max_value=70000, value=min(1000, len(X_full)), step=1)
            if st.button("Chốt số lượng mẫu"):
                with st.spinner(f"Đang lấy {num_samples} mẫu..."):
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    for i in range(0, 91, 10):
                        progress_bar.progress(i)
                        status_text.text(f"Đang xử lý {i}%...")
                        time.sleep(0.05)
                    indices = np.random.choice(len(X_full), size=num_samples, replace=False)
                    X_sampled = X_full.iloc[indices]
                    y_sampled = y_full.iloc[indices]
                    st.session_state['data'] = (X_sampled, y_sampled)
                    with mlflow.start_run(experiment_id=EXPERIMENT_ID, run_name="Data_Sample"):
                        mlflow.log_param("num_samples", num_samples)
                    st.success(f"Đã chốt {num_samples} mẫu!")
                    progress_bar.progress(100)
                    status_text.text("Đã xử lý 100%!")
                    time.sleep(0.5)
                    status_text.empty()
                    progress_bar.empty()

    # Tab 3: Xử lý dữ liệu
    with tab_preprocess:
        st.header("Xử lí Dữ liệu")

        if 'data' not in st.session_state:
            st.info("Vui lòng tải và chốt số lượng mẫu trước.")
        else:
            X, y = st.session_state['data']
            if "data_original" not in st.session_state:
                st.session_state["data_original"] = (X.copy(), y.copy())

            if "data_processed" in st.session_state:
                data_processed = st.session_state["data_processed"]
                if not (isinstance(data_processed, tuple) and len(data_processed) == 2):
                    st.session_state.pop("data_processed", None)

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
                    with st.spinner("Đang chuẩn hóa dữ liệu..."):
                        progress_bar = st.progress(0)
                        status_text = st.empty()
                        for i in range(0, 91, 10):
                            progress_bar.progress(i)
                            status_text.text(f"Đang xử lý {i}%...")
                            time.sleep(0.05)
                        X_norm = X / 255.0
                        st.session_state["data_processed"] = (X_norm, y)
                        st.success("Đã chuẩn hoá dữ liệu!")
                        progress_bar.progress(100)
                        status_text.text("Đã xử lý 100%!")
                        time.sleep(0.5)
                        status_text.empty()
                        progress_bar.empty()
                        st.rerun()
            with col2:
                st.markdown("""
                    <div class="tooltip">
                        ?
                        <span class="tooltiptext">
                            Đưa dữ liệu về khoảng [0, 1] bằng cách chia cho 255.<br>
                            Công dụng: Đảm bảo thang đo đồng nhất, hữu ích cho Neural Network.
                        </span>
                    </div>
                """, unsafe_allow_html=True)

            if "data_processed" in st.session_state:
                data_processed = st.session_state["data_processed"]
                if isinstance(data_processed, tuple) and len(data_processed) == 2:
                    try:
                        X_processed, y_processed = data_processed
                        st.subheader("Dữ liệu đã xử lý")
                        fig, axes = plt.subplots(2, 5, figsize=(10, 4))
                        for i, ax in enumerate(axes.flat):
                            ax.imshow(X_processed.iloc[i].values.reshape(28, 28), cmap='gray')
                            ax.set_title(f"Label: {y_processed.iloc[i]}")
                            ax.axis("off")
                        st.pyplot(fig)
                    except (ValueError, TypeError, AttributeError) as e:
                        st.error(f"Lỗi khi hiển thị dữ liệu đã xử lý: {e}. Vui lòng thử chuẩn hóa lại dữ liệu.")
                        st.session_state.pop("data_processed", None)
                else:
                    st.error("Dữ liệu đã xử lý không đúng định dạng. Vui lòng thử chuẩn hóa lại dữ liệu.")
                    st.session_state.pop("data_processed", None)
            else:
                st.info("Dữ liệu chưa được xử lý. Vui lòng nhấn 'Normalization' để xử lý.")

    # Tab 4: Chia dữ liệu
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
                    for i in range(0, 91, 10):
                        progress_bar.progress(i)
                        status_text.text(f"Đang xử lý {i}%...")
                        time.sleep(0.05)
                    st.session_state['split_data'] = {
                        "X_train": X_train, "y_train": y_train,
                        "X_valid": X_valid, "y_valid": y_valid,
                        "X_test": X_test, "y_test": y_test
                    }
                    st.success("Đã chia dữ liệu!")
                    progress_bar.progress(100)
                    status_text.text("Đã xử lý 100%!")
                    time.sleep(0.5)
                    status_text.empty()
                    progress_bar.empty()

    # Tab 5: Huấn luyện/Đánh giá
    with tab_train_eval:
        st.header("Huấn luyện và Đánh giá Mô hình")

        if 'split_data' not in st.session_state:
            st.info("Vui lòng chia dữ liệu trước khi huấn luyện mô hình.")
        else:
            X_train = st.session_state['split_data']["X_train"]
            num_samples = len(X_train)
            st.write(f"**Số mẫu huấn luyện**: {num_samples}")

            # Bảng tham số tối ưu
            st.subheader("⚙️ Cấu hình tham số mô hình")
            st.markdown("""
            Các tham số tối ưu được tự động chọn dựa trên số mẫu để đảm bảo hiệu suất tốt nhất:
            | Số mẫu       | Hidden Layer Sizes | Learning Rate | Max Iter | Activation | Solver | Batch Size | Alpha   |
            |--------------|--------------------|---------------|----------|------------|--------|------------|---------|
            | <1000        | 50                 | 0.01          | 100      | ReLU       | lbfgs  | auto       | 0.0001  |
            | 1000-5000    | 100                | 0.001         | 200      | ReLU       | adam   | 32         | 0.0001  |
            | 5000-20000   | 200                | 0.0005        | 300      | ReLU       | adam   | 64         | 0.0001  |
            | >20000       | 300                | 0.0001        | 400      | ReLU       | adam   | 128        | 0.0001  |
            """, unsafe_allow_html=True)

            # Hàm chọn tham số tối ưu
            def get_optimal_params(num_samples):
                if num_samples < 1000:
                    return {"hidden_size": 50, "learning_rate": 0.01, "max_iter": 100, "activation": "relu", "solver": "lbfgs", "batch_size": "auto", "alpha": 0.0001}
                elif 1000 <= num_samples <= 5000:
                    return {"hidden_size": 100, "learning_rate": 0.001, "max_iter": 200, "activation": "relu", "solver": "adam", "batch_size": 32, "alpha": 0.0001}
                elif 5000 < num_samples <= 20000:
                    return {"hidden_size": 200, "learning_rate": 0.0005, "max_iter": 300, "activation": "relu", "solver": "adam", "batch_size": 64, "alpha": 0.0001}
                else:
                    return {"hidden_size": 300, "learning_rate": 0.0001, "max_iter": 400, "activation": "relu", "solver": "adam", "batch_size": 128, "alpha": 0.0001}

            # Lưu tham số tối ưu vào session_state để khôi phục
            if "optimal_params" not in st.session_state:
                st.session_state["optimal_params"] = get_optimal_params(num_samples)

            # Khởi tạo tham số từ session_state hoặc tối ưu
            params = st.session_state.get("training_params", st.session_state["optimal_params"].copy())

            # Hiển thị tham số tối ưu được chọn
            st.info(f"**Tham số tối ưu tự động**: Hidden Size = {params['hidden_size']}, Learning Rate = {params['learning_rate']}, "
                    f"Max Iter = {params['max_iter']}, Activation = {params['activation']}, Solver = {params['solver']}, "
                    f"Batch Size = {params['batch_size']}, Alpha = {params['alpha']}")

            # Phân chia giao diện thành các cột
            col1, col2 = st.columns(2)

            with col1:
                with st.expander("🧠 Cấu trúc mạng", expanded=True):
                    st.markdown("##### Số lớp ẩn")
                    num_hidden_layers = st.number_input("Số lớp ẩn", min_value=1, max_value=3, value=1,
                                                       help="Số lớp ẩn quyết định độ sâu của mạng.")
                    with st.expander("ℹ️ Chi tiết về số lớp ẩn"):
                        st.markdown("""
                        - **Ý nghĩa**: Quyết định độ sâu của mạng nơ-ron, tức là số lượng lớp ẩn giữa lớp đầu vào và lớp đầu ra.  
                        - **Công dụng**:  
                          - Tăng số lớp ẩn giúp mô hình học được các đặc trưng phức tạp hơn từ dữ liệu (ví dụ: nhận diện các nét chữ phức tạp trong MNIST).  
                          - Tuy nhiên, quá nhiều lớp có thể dẫn đến hiện tượng quá khớp (overfitting) và tăng thời gian huấn luyện.  
                        - **Lưu ý**: Với MNIST, 1-2 lớp ẩn thường là đủ để đạt hiệu suất tốt.
                        """)

                    st.markdown("##### Số nơ-ron mỗi lớp")
                    params["hidden_size"] = st.number_input("Số nơ-ron mỗi lớp", min_value=10, max_value=500, value=params["hidden_size"],
                                                           help="Số nơ-ron ảnh hưởng đến độ phức tạp của mô hình.")
                    with st.expander("ℹ️ Chi tiết về số nơ-ron mỗi lớp"):
                        st.markdown("""
                        - **Ý nghĩa**: Số lượng nơ-ron (đơn vị tính toán) trong mỗi lớp ẩn.  
                        - **Công dụng**:  
                          - Số nơ-ron lớn giúp mô hình học được nhiều đặc trưng hơn, nhưng cũng làm tăng độ phức tạp và thời gian tính toán.  
                          - Số nơ-ron nhỏ có thể dẫn đến underfitting (mô hình không đủ mạnh để học tốt).  
                        - **Công thức liên quan**: Đầu ra của mỗi lớp:  
                          $$ A^{(l)} = \\sigma(W^{(l)} \\cdot A^{(l-1)} + b^{(l)}) $$  
                          Trong đó:  
                          - $W^{(l)}$: Ma trận trọng số của lớp $l$, kích thước phụ thuộc vào số nơ-ron.  
                          - $A^{(l-1)}$: Đầu ra của lớp trước.  
                          - $b^{(l)}$: Vector bias.  
                          - $\\sigma$: Hàm kích hoạt.  
                        - **Lưu ý**: Với MNIST, 50-300 nơ-ron mỗi lớp thường là lựa chọn hợp lý.
                        """)

                    hidden_sizes = tuple([params["hidden_size"]] * num_hidden_layers)

                    st.markdown("##### Hàm kích hoạt")
                    params["activation"] = st.selectbox("Hàm kích hoạt", ["relu", "sigmoid", "tanh"],
                                                       index=["relu", "sigmoid", "tanh"].index(params["activation"]),
                                                       help="Hàm kích hoạt giúp học đặc trưng phi tuyến.")
                    with st.expander("ℹ️ Chi tiết về hàm kích hoạt"):
                        st.markdown("""
                        - **Ý nghĩa**: Hàm áp dụng lên đầu ra của mỗi nơ-ron để tạo tính phi tuyến tính.  
                        - **Công dụng**:  
                          - Giúp mạng nơ-ron học được các mẫu phức tạp, không chỉ các mối quan hệ tuyến tính.  
                          - Lựa chọn hàm kích hoạt ảnh hưởng đến tốc độ học và khả năng hội tụ.  
                        - **Công thức**:  
                          - **ReLU** (Rectified Linear Unit):  
                            $$ \\sigma(z) = \\max(0, z) $$  
                            Phổ biến vì tính đơn giản và hiệu quả trong việc tránh vanishing gradient.  
                          - **Sigmoid**:  
                            $$ \\sigma(z) = \\frac{1}{1 + e^{-z}} $$  
                            Đưa đầu ra về khoảng [0, 1], phù hợp cho bài toán nhị phân, nhưng dễ gặp vấn đề vanishing gradient.  
                          - **Tanh**:  
                            $$ \\sigma(z) = \\tanh(z) $$  
                            Đưa đầu ra về khoảng [-1, 1], thường tốt hơn Sigmoid cho dữ liệu chuẩn hóa.  
                        - **Lưu ý**: Với MNIST, ReLU thường được ưu tiên vì hiệu suất tốt và tính toán nhanh.
                        """)

            with col2:
                with st.expander("📉 Tối ưu hóa", expanded=True):
                    st.markdown("##### Tốc độ học")
                    params["learning_rate"] = st.selectbox("Tốc độ học", [0.01, 0.001, 0.0005, 0.0001],
                                                          index=[0.01, 0.001, 0.0005, 0.0001].index(params["learning_rate"]),
                                                          help="Điều chỉnh tốc độ cập nhật trọng số.")
                    with st.expander("ℹ️ Chi tiết về tốc độ học"):
                        st.markdown("""
                        - **Ý nghĩa**: Quyết định mức độ thay đổi của trọng số trong mỗi bước huấn luyện. Ký hiệu: $\\eta$.  
                        - **Công dụng**:  
                          - Tốc độ học cao (ví dụ: 0.01) giúp mô hình học nhanh hơn, nhưng có thể vượt qua điểm tối ưu (không hội tụ).  
                          - Tốc độ học thấp (ví dụ: 0.0001) làm mô hình học chậm nhưng ổn định hơn, ít bỏ sót điểm tối ưu.  
                        - **Công thức**: Cập nhật trọng số:  
                          $$ W^{(l)} = W^{(l)} - \\eta \\cdot \\frac{\\partial L}{\\partial W^{(l)}} $$  
                          Trong đó:  
                          - $W^{(l)}$: Trọng số của lớp $l$.  
                          - $\\eta$: Tốc độ học.  
                          - $\\frac{\\partial L}{\\partial W^{(l)}}$: Đạo hàm của hàm mất mát theo $W^{(l)}$.  
                        - **Lưu ý**: Với MNIST, tốc độ học từ 0.0001 đến 0.01 thường phù hợp, tùy vào số mẫu và solver.
                        """)

                    st.markdown("##### Số lần lặp tối đa")
                    params["max_iter"] = st.number_input("Số lần lặp tối đa", min_value=50, max_value=500, value=params["max_iter"],
                                                        help="Số epoch tối đa để huấn luyện.")
                    with st.expander("ℹ️ Chi tiết về số lần lặp tối đa"):
                        st.markdown("""
                        - **Ý nghĩa**: Số lần mô hình lặp lại toàn bộ dữ liệu huấn luyện (epoch).  
                        - **Công dụng**:  
                          - Quyết định thời gian huấn luyện: số lần lặp lớn giúp mô hình học kỹ hơn, nhưng có thể dẫn đến overfitting.  
                          - Số lần lặp nhỏ có thể dẫn đến underfitting (mô hình chưa học đủ).  
                        - **Lưu ý**: Với MNIST, 100-400 lần lặp thường đủ để đạt độ chính xác tốt, tùy thuộc vào số mẫu.
                        """)

                    st.markdown("##### Optimizer")
                    params["solver"] = st.selectbox("Optimizer", ["lbfgs", "sgd", "adam"],
                                                   index=["lbfgs", "sgd", "adam"].index(params["solver"]),
                                                   help="Phương pháp tối ưu hóa trọng số.")
                    with st.expander("ℹ️ Chi tiết về optimizer"):
                        st.markdown("""
                        - **Ý nghĩa**: Phương pháp tối ưu hóa để cập nhật trọng số, giảm hàm mất mát.  
                        - **Công dụng**:  
                          - **lbfgs**: Phương pháp tối ưu dựa trên đạo hàm bậc hai, phù hợp với tập dữ liệu nhỏ, hội tụ nhanh nhưng tốn bộ nhớ.  
                          - **sgd** (Stochastic Gradient Descent): Cập nhật trọng số dựa trên gradient ngẫu nhiên:  
                            $$ W^{(l)} = W^{(l)} - \\eta \\cdot \\frac{\\partial L}{\\partial W^{(l)}} $$  
                            Đơn giản, hiệu quả với tập dữ liệu lớn, nhưng có thể dao động nhiều.  
                          - **adam** (Adaptive Moment Estimation): Kết hợp momentum và RMSProp, điều chỉnh tốc độ học tự động, thường cho kết quả tốt nhất.  
                        - **Lưu ý**: Với MNIST, **adam** thường được ưu tiên vì hiệu suất cao và ổn định.
                        """)

            # Điều chuẩn và batch size
            with st.expander("🔧 Điều chỉnh nâng cao", expanded=False):
                col3, col4 = st.columns(2)
                with col3:
                    st.markdown("##### Kích thước batch")
                    if params["solver"] in ["sgd", "adam"]:
                        params["batch_size"] = st.number_input("Kích thước batch", min_value=1, max_value=512, value=params["batch_size"],
                                                              help="Số mẫu mỗi lần cập nhật (chỉ áp dụng với SGD/Adam).")
                        with st.expander("ℹ️ Chi tiết về kích thước batch"):
                            st.markdown("""
                            - **Ý nghĩa**: Số mẫu dữ liệu được sử dụng để cập nhật trọng số trong mỗi lần lặp.  
                            - **Công dụng**:  
                              - Batch size nhỏ (ví dụ: 32) giúp cập nhật nhanh, nhưng có thể dao động (nhiễu) trong gradient.  
                              - Batch size lớn (ví dụ: 128) làm gradient ổn định hơn, nhưng tốn thời gian và bộ nhớ.  
                            - **Công thức**: Gradient trung bình trên batch:  
                              $$ \\frac{\\partial L}{\\partial W^{(l)}} = \\frac{1}{B} \\sum_{i=1}^{B} (A^{(l-1)}_i)^T \\cdot \\delta^{(l)}_i $$  
                              ($B$ là batch size).  
                            - **Lưu ý**: Với MNIST, batch size từ 32-128 thường là lựa chọn tốt.
                            """)
                    else:
                        params["batch_size"] = "auto"
                        st.write("Kích thước batch: **auto** (dành cho lbfgs)")
                        with st.expander("ℹ️ Chi tiết về kích thước batch"):
                            st.markdown("""
                            - **Ý nghĩa**: Với solver **lbfgs**, kích thước batch được đặt là **auto**, nghĩa là toàn bộ dữ liệu sẽ được sử dụng trong mỗi lần cập nhật.  
                            - **Công dụng**: Phù hợp với tập dữ liệu nhỏ, nhưng không hiệu quả với tập dữ liệu lớn do yêu cầu bộ nhớ cao.
                            """)

                with col4:
                    st.markdown("##### Tham số điều chuẩn (alpha)")
                    params["alpha"] = st.number_input("Tham số điều chuẩn (alpha)", min_value=0.0, max_value=1.0, value=params["alpha"], step=0.0001,
                                                     help="Hệ số L2 để giảm overfitting.")
                    with st.expander("ℹ️ Chi tiết về tham số điều chuẩn (alpha)"):
                        st.markdown("""
                        - **Ý nghĩa**: Hệ số điều chuẩn L2, thêm vào hàm mất mát để phạt các trọng số lớn.  
                        - **Công dụng**:  
                          - Giảm hiện tượng quá khớp (overfitting) bằng cách giới hạn độ lớn của trọng số.  
                          - Giúp mô hình tổng quát hóa tốt hơn trên dữ liệu mới.  
                        - **Công thức**: Hàm mất mát có điều chuẩn:  
                          $$ L = L_{\\text{data}} + \\frac{\\alpha}{2} \\sum_{l} ||W^{(l)}||^2 $$  
                          Trong đó:  
                          - $L_{\\text{data}}$: Hàm mất mát gốc (cross-entropy).  
                          - $\\alpha$: Hệ số điều chuẩn.  
                          - $||W^{(l)}||^2$: Tổng bình phương trọng số.  
                        - **Lưu ý**: Với MNIST, alpha từ 0.0001 đến 0.01 thường là đủ để kiểm soát overfitting.
                        """)

            # Nút khôi phục tham số tối ưu
            if st.button("🔄 Khôi phục tham số tối ưu", help="Quay lại tham số tối ưu dựa trên số mẫu"):
                st.session_state["training_params"] = st.session_state["optimal_params"].copy()
                st.success("Đã khôi phục tham số tối ưu!")
                st.rerun()

            # Lưu tham số vào session_state
            st.session_state["training_params"] = params

            # Nút huấn luyện
            if st.button("🚀 Thực hiện Huấn luyện", key="train_button", type="primary"):
                with st.spinner("Đang huấn luyện mô hình..."):
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    start_time = time.time()
                    for i in range(0, 91, 10):
                        progress_bar.progress(i)
                        status_text.text(f"Tiến độ: {i}%")
                        time.sleep(0.1)

                    X_train = st.session_state['split_data']["X_train"]
                    y_train = st.session_state['split_data']["y_train"]
                    X_valid = st.session_state['split_data']["X_valid"]
                    y_valid = st.session_state['split_data']["y_valid"]
                    X_test = st.session_state['split_data']["X_test"]
                    y_test = st.session_state['split_data']["y_test"]

                    pipeline = Pipeline([
                        ('pca', PCA(n_components=50)),
                        ('classifier', MLPClassifier(
                            hidden_layer_sizes=hidden_sizes,
                            max_iter=params["max_iter"],
                            learning_rate_init=params["learning_rate"],
                            activation=params["activation"],
                            solver=params["solver"],
                            batch_size=params["batch_size"] if params["solver"] in ["sgd", "adam"] else "auto",
                            alpha=params["alpha"]
                        ))
                    ])
                    pipeline.fit(X_train, y_train)

                    run_name = f"NeuralNetwork_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                    with mlflow.start_run(experiment_id=EXPERIMENT_ID, run_name=run_name) as run:
                        mlflow.log_param("hidden_layer_sizes", hidden_sizes)
                        mlflow.log_param("learning_rate_init", params["learning_rate"])
                        mlflow.log_param("max_iter", params["max_iter"])
                        mlflow.log_param("activation", params["activation"])
                        mlflow.log_param("solver", params["solver"])
                        if params["solver"] in ["sgd", "adam"]:
                            mlflow.log_param("batch_size", params["batch_size"])
                        mlflow.log_param("alpha", params["alpha"])

                        y_valid_pred = pipeline.predict(X_valid)
                        y_test_pred = pipeline.predict(X_test)
                        acc_valid = accuracy_score(y_valid, y_valid_pred)
                        acc_test = accuracy_score(y_test, y_test_pred)
                        cm_valid = confusion_matrix(y_valid, y_valid_pred)
                        cm_test = confusion_matrix(y_test, y_test_pred)

                        mlflow.log_metric("accuracy_val", acc_valid)
                        mlflow.log_metric("accuracy_test", acc_test)

                        st.session_state['model'] = pipeline
                        st.session_state['training_results'] = {
                            'accuracy_val': acc_valid,
                            'accuracy_test': acc_test,
                            'cm_valid': cm_valid,
                            'cm_test': cm_test,
                            'run_name': run_name,
                            'run_id': run.info.run_id,
                            'params': params
                        }

                    progress_bar.progress(100)
                    status_text.text("Hoàn tất: 100%")
                    st.success(f"Đã huấn luyện xong! Thời gian: {time.time() - start_time:.2f} giây")

                    # Hiển thị kết quả
                    st.subheader("📊 Kết quả huấn luyện")
                    col_result1, col_result2 = st.columns(2)
                    with col_result1:
                        st.metric("Độ chính xác Validation", f"{acc_valid*100:.2f}%")
                    with col_result2:
                        st.metric("Độ chính xác Test", f"{acc_test*100:.2f}%")

                    st.subheader("📈 Ma trận nhầm lẫn")
                    col_cm1, col_cm2 = st.columns(2)
                    with col_cm1:
                        fig, ax = plt.subplots(figsize=(5, 4))
                        sns.heatmap(cm_valid, annot=True, fmt="d", cmap="Blues", ax=ax)
                        ax.set_title("Validation")
                        st.pyplot(fig)
                    with col_cm2:
                        fig, ax = plt.subplots(figsize=(5, 4))
                        sns.heatmap(cm_test, annot=True, fmt="d", cmap="Blues", ax=ax)
                        ax.set_title("Test")
                        st.pyplot(fig)

                    st.subheader("ℹ️ Chi tiết")
                    with st.expander("Xem chi tiết", expanded=False):
                        st.markdown("**Thông tin lần chạy**:")
                        st.write(f"- Tên: {run_name}")
                        st.write(f"- ID: {run.info.run_id}")
                        st.markdown("**Tham số đã chọn**:")
                        st.write(f"- Số lớp ẩn: {num_hidden_layers}")
                        st.write(f"- Số nơ-ron mỗi lớp: {params['hidden_size']}")
                        st.write(f"- Tốc độ học: {params['learning_rate']}")
                        st.write(f"- Số lần lặp tối đa: {params['max_iter']}")
                        st.write(f"- Hàm kích hoạt: {params['activation']}")
                        st.write(f"- Optimizer: {params['solver']}")
                        if params["solver"] in ["sgd", "adam"]:
                            st.write(f"- Kích thước batch: {params['batch_size']}")
                        st.write(f"- Alpha: {params['alpha']}")

                    status_text.empty()
                    progress_bar.empty()

    # Tab 6: Demo dự đoán
    with tab_demo:
        st.header("Demo Dự đoán")

        if 'split_data' not in st.session_state or 'model' not in st.session_state:
            st.info("Vui lòng huấn luyện mô hình trước.")
        else:
            mode = st.radio("Chọn phương thức:", ["Dữ liệu Test", "Upload ảnh", "Vẽ số"])

            def preprocess_input(data):
                return data / 255.0

            is_normalized = 'data_processed' in st.session_state

            if mode == "Dữ liệu Test":
                X_test = st.session_state['split_data']["X_test"]
                y_test = st.session_state['split_data']["y_test"]
                idx = st.slider("Chọn mẫu Test", 0, len(X_test)-1, 0)
                if st.button("Dự đoán", key="predict_test_button"):
                    with st.spinner("Đang dự đoán..."):
                        progress_bar = st.progress(0)
                        status_text = st.empty()
                        for i in range(0, 91, 10):
                            progress_bar.progress(i)
                            status_text.text(f"Đang xử lý {i}%...")
                            time.sleep(0.05)
                        sample = X_test.iloc[idx].values.reshape(1, -1)
                        if not is_normalized:
                            sample = preprocess_input(sample)
                        pred = st.session_state['model'].predict(sample)[0]
                        true_label = y_test.iloc[idx]
                        st.success(f"Dự đoán: {pred} | Thực tế: {true_label}")
                        fig, ax = plt.subplots()
                        ax.imshow(sample.reshape(28, 28), cmap='gray')
                        ax.axis('off')
                        st.pyplot(fig)
                        progress_bar.progress(100)
                        status_text.text("Đã xử lý 100%!")
                        time.sleep(0.5)
                        status_text.empty()
                        progress_bar.empty()

            elif mode == "Upload ảnh":
                uploaded_images = st.file_uploader("Upload ảnh (28x28, grayscale)", type=["png", "jpg"], accept_multiple_files=True)
                if uploaded_images:
                    for i, img_file in enumerate(uploaded_images):
                        with st.spinner(f"Đang xử lý ảnh {i+1}..."):
                            progress_bar = st.progress(0)
                            status_text = st.empty()
                            for j in range(0, 91, 10):
                                progress_bar.progress(j)
                                status_text.text(f"Đang xử lý {j}%...")
                                time.sleep(0.05)
                            img = Image.open(img_file).convert('L').resize((28, 28))
                            img_array = np.array(img).flatten().reshape(1, -1)
                            if not is_normalized:
                                img_array = preprocess_input(img_array)
                            pred = st.session_state['model'].predict(img_array)[0]
                            st.success(f"Dự đoán ảnh {i+1}: {pred}")
                            st.image(img, caption=f"Ảnh {i+1}", use_container_width=True)
                            progress_bar.progress(100)
                            status_text.text("Đã xử lý 100%!")
                            time.sleep(0.5)
                            status_text.empty()
                            progress_bar.empty()

            elif mode == "Vẽ số":
                st.write("Vẽ số từ 0-9 (28x28 pixel):")
                canvas_result = st_canvas(
                    fill_color="black", stroke_width=20, stroke_color="white",
                    background_color="black", width=280, height=280, drawing_mode="freedraw", key="canvas"
                )
                if st.button("Dự đoán", key="predict_draw_button"):
                    if canvas_result.image_data is not None:
                        with st.spinner("Đang xử lý..."):
                            progress_bar = st.progress(0)
                            status_text = st.empty()
                            for i in range(0, 91, 10):
                                progress_bar.progress(i)
                                status_text.text(f"Đang xử lý {i}%...")
                                time.sleep(0.05)
                            img = Image.fromarray((canvas_result.image_data * 255).astype(np.uint8)).convert('L').resize((28, 28))
                            img_array = np.array(img).flatten().reshape(1, -1)
                            if not is_normalized:
                                img_array = preprocess_input(img_array)
                            pred = st.session_state['model'].predict(img_array)[0]
                            st.success(f"Dự đoán: {pred}")
                            progress_bar.progress(100)
                            status_text.text("Đã xử lý 100%!")
                            time.sleep(0.5)
                            status_text.empty()
                            progress_bar.empty()
                    else:
                        st.warning("Vui lòng vẽ trước!")

    # Tab 7: Thông tin huấn luyện
    with tab_log_info:
        st.header("Theo dõi Kết quả")

        st.markdown(f"""
        Tab này cho phép bạn xem danh sách các lần huấn luyện đã thực hiện từ Experiment ID {EXPERIMENT_ID}. Chọn một lần chạy để xem chi tiết, đổi tên hoặc xóa.
        """, unsafe_allow_html=True)

        try:
            with st.spinner("Đang tải thông tin huấn luyện..."):
                progress_bar = st.progress(0)
                status_text = st.empty()
                for i in range(0, 91, 10):
                    progress_bar.progress(i)
                    status_text.text(f"Đang tải {i}%...")
                    time.sleep(0.05)
                client = MlflowClient()
                runs = client.search_runs(
                    experiment_ids=[EXPERIMENT_ID],
                    order_by=["attributes.start_time DESC"]
                )

                if not runs:
                    st.info(f"Chưa có lần chạy nào được ghi nhận trong Experiment ID {EXPERIMENT_ID}.")
                    progress_bar.progress(100)
                    status_text.text("Đã tải 100%!")
                    time.sleep(0.5)
                    status_text.empty()
                    progress_bar.empty()
                else:
                    run_options = {run.info.run_id: run.data.tags.get('mlflow.runName', f"Run_{run.info.run_id}") for run in runs}
                    run_names = list(run_options.values())
                    run_ids = list(run_options.keys())

                    default_index = 0

                    st.subheader("Danh sách Run")
                    selected_run_name = st.selectbox(
                        "Chọn run:",
                        options=run_names,
                        index=default_index,
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
                        with st.spinner("Đang cập nhật tên..."):
                            progress_bar = st.progress(0)
                            status_text = st.empty()
                            for i in range(0, 91, 10):
                                progress_bar.progress(i)
                                status_text.text(f"Đang xử lý {i}%...")
                                time.sleep(0.05)
                            if new_run_name.strip() and new_run_name.strip() != selected_run_name:
                                client.set_tag(selected_run_id, "mlflow.runName", new_run_name.strip())
                                if 'training_results' in st.session_state and st.session_state['training_results']['run_id'] == selected_run_id:
                                    st.session_state['training_results']['run_name'] = new_run_name.strip()
                                st.success(f"Đã đổi tên thành: {new_run_name.strip()}")
                                progress_bar.progress(100)
                                status_text.text("Đã xử lý 100%!")
                                time.sleep(0.5)
                                status_text.empty()
                                progress_bar.empty()
                                st.rerun()
                            elif not new_run_name.strip():
                                st.warning("Vui lòng nhập tên hợp lệ.")
                                progress_bar.progress(0)
                                status_text.empty()
                            else:
                                st.info("Tên mới trùng với tên hiện tại.")
                                progress_bar.progress(0)
                                status_text.empty()

                    st.subheader("Xóa Run")
                    if st.button("Xóa lần chạy", key="delete_button"):
                        with st.spinner("Đang xóa lần chạy..."):
                            progress_bar = st.progress(0)
                            status_text = st.empty()
                            for i in range(0, 91, 10):
                                progress_bar.progress(i)
                                status_text.text(f"Đang xử lý {i}%...")
                                time.sleep(0.05)
                            client.delete_run(selected_run_id)
                            if 'training_results' in st.session_state and st.session_state['training_results']['run_id'] == selected_run_id:
                                del st.session_state['training_results']
                            st.success(f"Đã xóa: {selected_run_name}")
                            progress_bar.progress(100)
                            status_text.text("Đã xử lý 100%!")
                            time.sleep(0.5)
                            status_text.empty()
                            progress_bar.empty()
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
                        metrics_display = {}
                        training_time = selected_run.data.metrics.get("training_time_seconds", "N/A")
                        metrics_display["Thời gian thực hiện (giây)"] = f"{float(training_time):.2f}" if training_time != "N/A" else "N/A"
                        accuracy_val = selected_run.data.metrics.get("accuracy_val", "N/A")
                        metrics_display["Độ chính xác Validation"] = f"{float(accuracy_val)*100:.2f}%" if accuracy_val != "N/A" else "N/A"
                        accuracy_test = selected_run.data.metrics.get("accuracy_test", "N/A")
                        metrics_display["Độ chính xác Test"] = f"{float(accuracy_test)*100:.2f}%" if accuracy_test != "N/A" else "N/A"
                        st.json(metrics_display, expanded=True)
                    else:
                        st.write("Không có kết quả được ghi nhận.")

                    st.subheader("Truy cập MLflow UI")
                    mlflow_url = "https://dagshub.com/huykibo/streamlit_mlflow.mlflow"
                    if st.button("Mở MLflow UI trên Dagshub"):
                        st.markdown(f'[Click để mở MLflow UI]({mlflow_url})', unsafe_allow_html=True)

                    progress_bar.progress(100)
                    status_text.text("Đã tải 100%!")
                    time.sleep(0.5)
                    status_text.empty()
                    progress_bar.empty()

        except Exception as e:
            st.error(f"Lỗi kết nối MLflow hoặc không tìm thấy Experiment ID {EXPERIMENT_ID}: {e}. Vui lòng kiểm tra MLFLOW_TRACKING_URI và thông tin xác thực.")
            progress_bar.progress(0)
            status_text.empty()

if __name__ == "__main__":
    run_mnist_neural_network_app()