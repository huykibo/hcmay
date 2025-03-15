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

def run_mnist_neural_network_app():
    # Thiết lập MLflow
    try:
        os.environ["MLFLOW_TRACKING_USERNAME"] = st.secrets["mlflow"]["MLFLOW_TRACKING_USERNAME"]
        os.environ["MLFLOW_TRACKING_PASSWORD"] = st.secrets["mlflow"]["MLFLOW_TRACKING_PASSWORD"]
        mlflow.set_tracking_uri(st.secrets["mlflow"]["MLFLOW_TRACKING_URI"])
    except KeyError as e:
        st.error(f"Lỗi: Không tìm thấy khóa {e} trong st.secrets. Vui lòng cấu hình secrets.")
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

        elif info_option == "Tập dữ liệu MNIST: Đặc điểm và ý nghĩa":
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
            with st.spinner("Đang tải ảnh minh họa..."):
                try:
                    mnist_image = Image.open("mnist.png")
                    st.image(mnist_image, caption="Ảnh minh họa $10$ chữ số từ $0$ đến $9$ trong MNIST", width=800)
                except FileNotFoundError:
                    st.error("Không tìm thấy file `mnist.png`. Vui lòng kiểm tra đường dẫn.")
                except Exception as e:
                    st.error(f"Lỗi khi tải ảnh: {e}")

        elif info_option == "Neural Network – Mạng nơ-ron nhân tạo":
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
               - Khởi tạo **trọng số** $W$ và **bias** $b$ ngẫu nhiên hoặc bằng $0$.  
            """, unsafe_allow_html=True)
            try:
                st.image(os.path.join("plnw", "step1_init.png"), caption="Minh họa Bước 1: Khởi tạo mô hình", width=600)
            except FileNotFoundError:
                st.error("Không tìm thấy ảnh minh họa cho Bước 1. Vui lòng chạy mã tạo ảnh trước.")

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
                st.error("Không tìm thấy ảnh minh họa cho Bước 2. Vui lòng chạy mã tạo ảnh trước.")

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
                st.error("Không tìm thấy ảnh minh họa cho Bước 3. Vui lòng chạy mã tạo ảnh trước.")

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
                st.error("Không tìm thấy ảnh minh họa cho Bước 4. Vui lòng chạy mã tạo ảnh trước.")

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
                st.error("Không tìm thấy ảnh minh họa cho Bước 5. Vui lòng chạy mã tạo ảnh trước.")

            st.markdown("""
            6. **Lặp lại**:  
               - Quay lại bước $2$ qua nhiều **epoch** cho đến khi $L$ hội tụ.  
            """, unsafe_allow_html=True)
            try:
                st.image(os.path.join("plnw", "step6_repeat_improved.png"), caption="Minh họa Bước 6: Lặp lại", width=600)
            except FileNotFoundError:
                st.error("Không tìm thấy ảnh minh họa cho Bước 6. Vui lòng chạy mã tạo ảnh trước.")

            st.subheader("⚙️ Các tham số cơ bản và công dụng")
            st.markdown("""
            Dưới đây là các tham số bạn sẽ sử dụng để điều chỉnh mô hình trong ứng dụng này:  
            - **hidden_layer_sizes**:  
              - **Ý nghĩa**: Số nơ-ron trong lớp ẩn (ví dụ: $128$).  
              - **Công dụng**: Quyết định sức mạnh của mô hình; nhiều nơ-ron hơn thì học được đặc trưng phức tạp hơn nhưng tốn thời gian hơn.  
            - **learning_rate_init**:  
              - **Ý nghĩa**: Tốc độ học ban đầu (ví dụ: $0.001$).  
              - **Công dụng**: Điều chỉnh tốc độ cập nhật trọng số; nhỏ hơn thì học chậm nhưng ổn định hơn.  
            - **max_iter**:  
              - **Ý nghĩa**: Số lần huấn luyện tối đa (ví dụ: $200$).  
              - **Công dụng**: Giới hạn số lần mô hình học qua dữ liệu để đạt độ chính xác mong muốn.  
            """, unsafe_allow_html=True)

            st.subheader("🟪 Ưu điểm và nhược điểm")
            st.markdown("""
            ##### ✅ **Ưu điểm**:  
            - Học được các đặc trưng phức tạp từ dữ liệu hình ảnh như MNIST.  
            - Dễ sử dụng với các tham số cơ bản được tối ưu sẵn.  

            ##### ❌ **Nhược điểm**:  
            - Tốn thời gian huấn luyện nếu số mẫu lớn hoặc số nơ-ron nhiều.  
            - Cần dữ liệu được chuẩn hóa để đạt hiệu quả tốt nhất.  
            """, unsafe_allow_html=True)

        elif info_option == "Công thức đánh giá độ chính xác (Accuracy)":
            st.subheader("📘 4. Công thức đánh giá độ chính xác (Accuracy)")
            st.markdown("""
            Độ chính xác (**Accuracy**) đo tỷ lệ dự đoán đúng:  
            $$ \\text{Accuracy} = \\frac{\\text{Số mẫu dự đoán đúng}}{\\text{Tổng số mẫu}} $$  
            - **Ví dụ**: Dự đoán đúng $92/100$ ảnh → $\\text{Accuracy} = 92\\%$.  
            - **Ý nghĩa**: Với Neural Network, Accuracy đo khả năng mô hình phân loại đúng các chữ số dựa trên đặc trưng pixel học được.  
            """, unsafe_allow_html=True)

    # Tab 2: Tải dữ liệu
    with tab_load:
        st.header("Tải Dữ liệu")
        if st.button("Tải dữ liệu MNIST từ OpenML"):
            with st.spinner("Đang tải dữ liệu từ OpenML..."):
                try:
                    mnist = openml.datasets.get_dataset(554)
                    X, y, _, _ = mnist.get_data(target=mnist.default_target_attribute)
                    st.session_state['full_data'] = (X, y)
                    with mlflow.start_run(run_name="Data_Load"):
                        mlflow.log_param("total_samples", X.shape[0])
                    st.success("Tải dữ liệu thành công!")
                    st.write("Kích thước dữ liệu gốc:", X.shape)
                except Exception as e:
                    st.error(f"Không thể tải dữ liệu: {e}")

        if 'full_data' in st.session_state:
            X_full, y_full = st.session_state['full_data']
            num_samples = st.slider("Chọn số lượng mẫu:", 
                                    min_value=10, max_value=len(X_full), value=min(1000, len(X_full)), step=1)
            if st.button("Chốt số lượng mẫu"):
                with st.spinner(f"Đang lấy {num_samples} mẫu..."):
                    indices = np.random.choice(len(X_full), size=num_samples, replace=False)
                    X_sampled = X_full.iloc[indices]
                    y_sampled = y_full.iloc[indices]
                    st.session_state['data'] = (X_sampled, y_sampled)
                    with mlflow.start_run(run_name="Data_Sample"):
                        mlflow.log_param("num_samples", num_samples)
                    st.success(f"Đã chốt {num_samples} mẫu!")

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
                    X_norm = X / 255.0
                    st.session_state["data_processed"] = (X_norm, y)
                    st.success("Đã chuẩn hoá dữ liệu!")
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
                st.session_state['split_data'] = {
                    "X_train": X_train, "y_train": y_train,
                    "X_valid": X_valid, "y_valid": y_valid,
                    "X_test": X_test, "y_test": y_test
                }
                st.success("Đã chia dữ liệu!")

    # Tab 5: Huấn luyện/Đánh giá
    with tab_train_eval:
        st.header("Huấn luyện và Đánh giá")
        if 'split_data' not in st.session_state:
            st.info("Vui lòng chia dữ liệu trước.")
        else:
            X_train = st.session_state['split_data']["X_train"]
            num_samples = len(X_train)
            st.write(f"Số mẫu huấn luyện: {num_samples}")

            st.subheader("⚙️ Cài đặt tham số mô hình")
            st.markdown("""
            Dựa trên số lượng mẫu, đây là gợi ý tham số tối ưu:
            | Số mẫu       | Hidden Layer Sizes | Learning Rate | Max Iter |
            |--------------|--------------------|---------------|----------|
            | <1000        | 50                | 0.01          | 100      |
            | 1000-5000    | 100               | 0.001         | 200      |
            | >5000        | 200               | 0.0001        | 300      |
            """, unsafe_allow_html=True)

            params = {}
            if num_samples < 1000:
                params["hidden_size"] = 50
                params["learning_rate"] = 0.01
                params["max_iter"] = 100
            elif 1000 <= num_samples <= 5000:
                params["hidden_size"] = 100
                params["learning_rate"] = 0.001
                params["max_iter"] = 200
            else:
                params["hidden_size"] = 200
                params["learning_rate"] = 0.0001
                params["max_iter"] = 300

            params["hidden_size"] = st.number_input("Số nơ-ron lớp ẩn", 10, 500, params["hidden_size"])
            params["learning_rate"] = st.selectbox("Tốc độ học", [0.01, 0.001, 0.0001], index=[0.01, 0.001, 0.0001].index(params["learning_rate"]))
            params["max_iter"] = st.number_input("Số lần lặp tối đa", 50, 500, params["max_iter"])

            if st.button("Thực hiện Huấn luyện", key="train_button"):
                with st.spinner("Đang huấn luyện mô hình..."):
                    start_time = time.time()

                    X_train = st.session_state['split_data']["X_train"]
                    y_train = st.session_state['split_data']["y_train"]
                    X_valid = st.session_state['split_data']["X_valid"]
                    y_valid = st.session_state['split_data']["y_valid"]
                    X_test = st.session_state['split_data']["X_test"]
                    y_test = st.session_state['split_data']["y_test"]

                    pipeline = Pipeline([
                        ('pca', PCA(n_components=50)),
                        ('classifier', MLPClassifier(hidden_layer_sizes=(params["hidden_size"],), 
                                                     max_iter=params["max_iter"], 
                                                     learning_rate_init=params["learning_rate"],
                                                     solver='lbfgs'))
                    ])
                    pipeline.fit(X_train, y_train)

                    run_name = f"NeuralNetwork_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                    with mlflow.start_run(run_name=run_name) as run:
                        mlflow.log_param("hidden_size", params["hidden_size"])
                        mlflow.log_param("learning_rate", params["learning_rate"])
                        mlflow.log_param("max_iter", params["max_iter"])

                        y_valid_pred = pipeline.predict(X_valid)
                        y_test_pred = pipeline.predict(X_test)
                        acc_valid = accuracy_score(y_valid, y_valid_pred)
                        acc_test = accuracy_score(y_test, y_test_pred)
                        cm_valid = confusion_matrix(y_valid, y_valid_pred)
                        cm_test = confusion_matrix(y_test, y_test_pred)

                        training_time = time.time() - start_time
                        mlflow.log_metric("accuracy_val", acc_valid)
                        mlflow.log_metric("accuracy_test", acc_test)
                        mlflow.log_metric("training_time_seconds", training_time)
                        mlflow.sklearn.log_model(pipeline, "model")

                        st.session_state['model'] = pipeline
                        st.session_state['training_results'] = {
                            'training_time': training_time,
                            'accuracy_val': acc_valid,
                            'accuracy_test': acc_test,
                            'cm_valid': cm_valid,
                            'cm_test': cm_test,
                            'run_name': run_name,
                            'run_id': run.info.run_id,
                            'params': params
                        }

                    st.success(f"Huấn luyện hoàn tất! Thời gian: {training_time:.2f} giây")
                    st.write(f"Độ chính xác Validation: {acc_valid:.4f}")
                    st.write(f"Độ chính xác Test: {acc_test:.4f}")

                    st.subheader("📈 Ma trận nhầm lẫn")
                    fig, ax = plt.subplots()
                    sns.heatmap(cm_valid, annot=True, fmt="d", cmap="Blues", ax=ax)
                    ax.set_title("Confusion Matrix - Validation")
                    st.pyplot(fig)

                    fig, ax = plt.subplots()
                    sns.heatmap(cm_test, annot=True, fmt="d", cmap="Blues", ax=ax)
                    ax.set_title("Confusion Matrix - Test")
                    st.pyplot(fig)

                    st.subheader("ℹ️ Chi tiết kết quả")
                    with st.expander("Xem chi tiết", expanded=True):
                        st.markdown("#### Thông tin lần chạy:", unsafe_allow_html=True)
                        st.write(f"- **Tên lần chạy**: {run_name}")
                        st.write(f"- **ID lần chạy**: {run.info.run_id}")

                        st.markdown("#### Tham số đã chọn:", unsafe_allow_html=True)
                        st.write(f"- **Số nơ-ron lớp ẩn**: {params['hidden_size']}")
                        st.write(f"- **Tốc độ học**: {params['learning_rate']}")
                        st.write(f"- **Số lần lặp tối đa**: {params['max_iter']}")

                        st.markdown("#### Kết quả đạt được:", unsafe_allow_html=True)
                        st.write(f"- **Độ chính xác Validation**: {acc_valid*100:.2f}%")
                        st.write(f"- **Độ chính xác Test**: {acc_test*100:.2f}%")

    # Tab 6: Demo dự đoán
    with tab_demo:
        st.header("Demo Dự đoán")
        if 'split_data' not in st.session_state or 'model' not in st.session_state:
            st.info("Vui lòng huấn luyện mô hình trước.")
        else:
            mode = st.radio("Chọn phương thức:", ["Dữ liệu Test", "Upload ảnh", "Vẽ số"])
            progress_bar = st.progress(0)
            status_text = st.empty()

            def preprocess_input(data):
                return data / 255.0

            is_normalized = 'data_processed' in st.session_state

            if mode == "Dữ liệu Test":
                X_test = st.session_state['split_data']["X_test"]
                y_test = st.session_state['split_data']["y_test"]
                idx = st.slider("Chọn mẫu Test", 0, len(X_test)-1, 0)
                if st.button("Dự đoán", key="predict_test_button"):
                    with st.spinner("Đang dự đoán..."):
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

            elif mode == "Upload ảnh":
                uploaded_images = st.file_uploader("Upload ảnh (28x28, grayscale)", type=["png", "jpg"], accept_multiple_files=True)
                if uploaded_images:
                    for i, img_file in enumerate(uploaded_images):
                        with st.spinner(f"Đang xử lý ảnh {i+1}..."):
                            img = Image.open(img_file).convert('L').resize((28, 28))
                            img_array = np.array(img).flatten().reshape(1, -1)
                            if not is_normalized:
                                img_array = preprocess_input(img_array)
                            pred = st.session_state['model'].predict(img_array)[0]
                            st.success(f"Dự đoán ảnh {i+1}: {pred}")
                            st.image(img, caption=f"Ảnh {i+1}", use_container_width=True)

            elif mode == "Vẽ số":
                st.write("Vẽ số từ 0-9 (28x28 pixel):")
                canvas_result = st_canvas(
                    fill_color="black", stroke_width=20, stroke_color="white",
                    background_color="black", width=280, height=280, drawing_mode="freedraw", key="canvas"
                )
                if st.button("Dự đoán", key="predict_draw_button"):
                    if canvas_result.image_data is not None:
                        with st.spinner("Đang xử lý..."):
                            img = Image.fromarray((canvas_result.image_data * 255).astype(np.uint8)).convert('L').resize((28, 28))
                            img_array = np.array(img).flatten().reshape(1, -1)
                            if not is_normalized:
                                img_array = preprocess_input(img_array)
                            pred = st.session_state['model'].predict(img_array)[0]
                            st.success(f"Dự đoán: {pred}")
                    else:
                        st.warning("Vui lòng vẽ trước!")

    # Tab 7: Thông tin huấn luyện
    with tab_log_info:
        st.header("Theo dõi Kết quả")
        st.markdown("""
        Tab này cho phép bạn xem danh sách các lần huấn luyện đã thực hiện từ Experiment ID 5. Chọn một lần chạy để xem chi tiết, đổi tên hoặc xóa.
        """, unsafe_allow_html=True)

        try:
            client = MlflowClient()
            experiment_id = "5"
            runs = client.search_runs(
                experiment_ids=[experiment_id],
                order_by=["attributes.start_time DESC"]
            )

            if not runs:
                st.info("Chưa có lần chạy nào được ghi nhận trong Experiment ID 5.")
            else:
                run_options = {run.info.run_id: run.data.tags.get('mlflow.runName', f"Run_{run.info.run_id}") for run in runs}
                run_names = list(run_options.values())

                default_run_name = st.session_state.get('training_results', {}).get('run_name', run_names[0]) if 'training_results' in st.session_state else run_names[0]

                st.subheader("Danh sách Run")
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
                            if 'training_results' in st.session_state and st.session_state['training_results']['run_id'] == selected_run_id:
                                st.session_state['training_results']['run_name'] = new_run_name.strip()
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
                        if 'training_results' in st.session_state and st.session_state['training_results']['run_id'] == selected_run_id:
                            del st.session_state['training_results']
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

        except Exception as e:
            st.error(f"Lỗi kết nối MLflow hoặc không tìm thấy Experiment ID 5: {e}. Vui lòng kiểm tra MLFLOW_TRACKING_URI và thông tin xác thực.")

if __name__ == "__main__":
    run_mnist_neural_network_app()