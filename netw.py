import os
import mlflow
import streamlit as st
import openml
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.neural_network import MLPClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from mlflow.tracking import MlflowClient
from streamlit_drawable_canvas import st_canvas
from datetime import datetime
import time

def    run_mnist_neural_network_app():
    # Thiết lập MLflow
    try:
        os.environ["MLFLOW_TRACKING_USERNAME"] = st.secrets["mlflow"]["MLFLOW_TRACKING_USERNAME"]
        os.environ["MLFLOW_TRACKING_PASSWORD"] = st.secrets["mlflow"]["MLFLOW_TRACKING_PASSWORD"]
        mlflow.set_tracking_uri(st.secrets["mlflow"]["MLFLOW_TRACKING_URI"])
        mlflow.set_experiment("Neural Network ")
    except KeyError as e:
        st.error(f"Lỗi: Không tìm thấy khóa {e} trong st.secrets. Vui lòng cấu hình secrets trong Streamlit.")
        st.stop()

    st.title("Ứng dụng Phân loại Chữ số MNIST bằng Neural Network")

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
    tabs = st.tabs(["Thông tin", "Tải dữ liệu", "Xử lí dữ liệu", "Chia dữ liệu", "Huấn luyện/Đánh Giá", "Demo dự đoán", "Thông tin huấn luyện"])
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

            - **MNIST**:  
              - Tập dữ liệu gồm **70,000 ảnh** chữ số từ **0 đến 9**.  
              - Mỗi ảnh có kích thước **28 × 28 pixel**, tương đương **784 đặc trưng** (giá trị pixel từ 0 đến 255).  

            - **Mục tiêu**:  
              - Xây dựng và huấn luyện một **Neural Network** để nhận diện chính xác các chữ số viết tay.  
              - Cung cấp một công cụ trực quan giúp người dùng học tập và đánh giá hiệu quả của thuật toán phân loại.

            **Thông tin cơ bản**:  
            - **784 đặc trưng**: Mỗi ảnh được biểu diễn dưới dạng vector 784 chiều, với mỗi chiều là giá trị độ sáng của một pixel (từ 0 đến 255).  
            - **70,000 mẫu**: Tổng số ảnh, bao gồm tập huấn luyện và kiểm tra.  
            - **Nhiệm vụ**: Dự đoán nhãn (từ 0 đến 9) của mỗi ảnh dựa trên các đặc trưng pixel.
            """, unsafe_allow_html=True)

        elif info_option == "Tập dữ liệu MNIST: Đặc điểm và ý nghĩa":
            st.subheader("📘 2. Tập dữ liệu MNIST: Đặc điểm và ý nghĩa")
            st.markdown("""
            **MNIST** là một tập dữ liệu chuẩn trong lĩnh vực học máy, được phát triển bởi Yann LeCun và các cộng sự.

            - **Đặc điểm**:  
              - Bao gồm các ảnh chữ số viết tay từ học sinh trung học và nhân viên điều tra dân số Mỹ.  
              - Được chuẩn hóa thành kích thước **28 × 28 pixel**, sử dụng thang độ xám (giá trị từ 0 đến 255).  

            - **Ý nghĩa**:  
              - Là bài toán cơ bản để kiểm tra hiệu quả của các thuật toán phân loại trong học máy.  
              - Dữ liệu đơn giản nhưng đủ phức tạp để đánh giá khả năng phân biệt giữa các lớp tương tự (ví dụ: "4" và "9").  
              - Phù hợp cho cả người mới bắt đầu và các nhà nghiên cứu muốn thử nghiệm mô hình phức tạp.
            """, unsafe_allow_html=True)

            st.subheader("📷 Minh họa dữ liệu MNIST")
            st.markdown("""
            Dưới đây là ảnh minh họa 10 chữ số từ 0 đến 9 từ tập dữ liệu MNIST. Mỗi chữ số được biểu diễn dưới dạng ma trận **28 × 28 pixel**.
            """, unsafe_allow_html=True)
            with st.spinner("Đang tải ảnh minh họa..."):
                try:
                    mnist_image = Image.open("mnist.png")
                    st.image(mnist_image, caption="Ảnh minh họa 10 chữ số từ 0 đến 9 trong MNIST", width=800)
                except FileNotFoundError:
                    st.error("Không tìm thấy file `mnist.png`. Vui lòng kiểm tra đường dẫn.")
                except Exception as e:
                    st.error(f"Lỗi khi tải ảnh: {e}")

        elif info_option == "Neural Network – Mạng nơ-ron nhân tạo":
            st.subheader("📊 3. Neural Network – Mạng nơ-ron nhân tạo")
            st.markdown("""
            **Neural Network (Mạng nơ-ron nhân tạo)** là một mô hình học máy mô phỏng cách hoạt động của mạng nơ-ron sinh học trong não người.

            - **Cấu trúc**:  
              - Gồm các **nơ-ron nhân tạo (nodes)** được tổ chức thành nhiều **lớp (layers)**:  
                - **Lớp đầu vào (Input Layer)**: Nhận dữ liệu thô (784 pixel từ ảnh MNIST).  
                - **Lớp ẩn (Hidden Layers)**: Xử lý thông tin bằng cách kết hợp tuyến tính và áp dụng hàm kích hoạt phi tuyến.  
                - **Lớp đầu ra (Output Layer)**: Đưa ra dự đoán (nhãn từ 0 đến 9).  

            - **Ưu điểm với MNIST**: Neural Network đặc biệt hiệu quả nhờ khả năng học các đặc trưng phức tạp từ dữ liệu hình ảnh.
            """, unsafe_allow_html=True)

            st.subheader("🛠️ Các bước thực hiện trong Neural Network")
            st.markdown("""
            Dưới đây là các bước chi tiết hoạt động của Neural Network trong bài toán MNIST:

            1. **Khởi tạo mô hình**:  
               - Xác định cấu trúc mạng: số lớp ẩn và số nơ-ron mỗi lớp.  
               - Khởi tạo **trọng số (W)** và **bias (b)** ngẫu nhiên hoặc bằng 0.  
            """, unsafe_allow_html=True)
            try:
                st.image(os.path.join("plnw", "step1_init.png"), caption="Bước 1: Khởi tạo mô hình", width=600)
            except FileNotFoundError:
                st.error("Không tìm thấy ảnh minh họa cho Bước 1.")

            st.markdown("""
            2. **Lan truyền thuận (Feedforward)**:  
               - Tính giá trị dự đoán **Ŷ** từ dữ liệu đầu vào **X**:  
                 - **Lớp đầu vào**: \( A^{(0)} = X \) (ma trận \( N \times 784 \), \( N \) là số mẫu).  
                 - **Cho mỗi lớp \( l \)**:  
                   - Tổng tuyến tính:  
                     $$ Z^{(l)} = A^{(l-1)} \cdot W^{(l)} + b^{(l)} $$  
                   - Áp dụng hàm kích hoạt:  
                     $$ A^{(l)} = \sigma(Z^{(l)}) $$  
                 - **Lớp đầu ra**: \( \hat{Y} = A^{(L)} \) (ma trận \( N \times 10 \)).  
               - Ví dụ hàm kích hoạt **sigmoid**:  
                 $$ \sigma(z) = \frac{1}{1 + e^{-z}} $$
            """, unsafe_allow_html=True)
            try:
                st.image(os.path.join("plnw", "step2_feedforward.png"), caption="Bước 2: Lan truyền thuận", width=600)
            except FileNotFoundError:
                st.error("Không tìm thấy ảnh minh họa cho Bước 2.")

            st.markdown("""
            3. **Tính hàm mất mát (Loss Function)**:  
               - Đo độ sai lệch giữa dự đoán \( \hat{Y} \) và nhãn thực \( Y \). Với MNIST, dùng **Cross-Entropy**:  
                 $$ L = -\frac{1}{N} \sum_{i=1}^{N} \sum_{j=0}^{9} y_{ij} \cdot \log(\hat{y}_{ij}) $$  
               - Trong đó:  
                 - \( y_{ij} \): Nhãn thực (dạng one-hot encoded).  
                 - \( \hat{y}_{ij} \): Xác suất dự đoán cho lớp \( j \).
            """, unsafe_allow_html=True)
            try:
                st.image(os.path.join("plnw", "step3_loss.png"), caption="Bước 3: Tính hàm mất mát", width=600)
            except FileNotFoundError:
                st.error("Không tìm thấy ảnh minh họa cho Bước 3.")

            st.markdown("""
            4. **Lan truyền ngược (Backpropagation)**:  
               - Tính đạo hàm của \( L \) theo \( W^{(l)} \) và \( b^{(l)} \) để cập nhật tham số:  
                 - Tại **Lớp đầu ra**:  
                   $$ \delta^{(L)} = \hat{Y} - Y $$  
                 - Tại **Lớp ẩn**:  
                   $$ \delta^{(l)} = (\delta^{(l+1)} \cdot (W^{(l+1)})^T) \odot \sigma'(Z^{(l)}) $$  
                 - \( \sigma'(z) \): Đạo hàm hàm kích hoạt (với sigmoid: \( \sigma'(z) = \sigma(z) \cdot (1 - \sigma(z)) \)).  
                 - Đạo hàm theo trọng số và bias:  
                   $$ \frac{\partial L}{\partial W^{(l)}} = (A^{(l-1)})^T \cdot \delta^{(l)} $$  
                   $$ \frac{\partial L}{\partial b^{(l)}} = \sum_{i=1}^{N} \delta^{(l)}_i $$
            """, unsafe_allow_html=True)
            try:
                st.image(os.path.join("plnw", "step4_backprop.png"), caption="Bước 4: Lan truyền ngược", width=600)
            except FileNotFoundError:
                st.error("Không tìm thấy ảnh minh họa cho Bước 4.")

            st.markdown("""
            5. **Cập nhật tham số (Gradient Descent)**:  
               - Điều chỉnh \( W \) và \( b \) để giảm mất mát:  
                 $$ W^{(l)} = W^{(l)} - \eta \cdot \frac{\partial L}{\partial W^{(l)}} $$  
                 $$ b^{(l)} = b^{(l)} - \eta \cdot \frac{\partial L}{\partial b^{(l)}} $$  
               - Trong đó: \( \eta \) là **tốc độ học (learning rate)**.
            """, unsafe_allow_html=True)
            try:
                st.image(os.path.join("plnw", "step5_gradient.png"), caption="Bước 5: Cập nhật tham số", width=600)
            except FileNotFoundError:
                st.error("Không tìm thấy ảnh minh họa cho Bước 5.")

            st.markdown("""
            6. **Lặp lại**:  
               - Quay lại bước 2 qua nhiều **epoch** cho đến khi \( L \) hội tụ.
            """, unsafe_allow_html=True)
            try:
                st.image(os.path.join("plnw", "step6_repeat_improved.png"), caption="Bước 6: Lặp lại", width=600)
            except FileNotFoundError:
                st.error("Không tìm thấy ảnh minh họa cho Bước 6.")

            st.subheader("⚙️ Các tham số cơ bản và công dụng")
            st.markdown("""
            Dưới đây là các tham số chính bạn có thể điều chỉnh trong ứng dụng:

            - **hidden_layer_sizes**:  
              - **Ý nghĩa**: Số nơ-ron trong các lớp ẩn (ví dụ: 128).  
              - **Công dụng**: Quyết định sức mạnh tính toán của mô hình; nhiều nơ-ron hơn giúp học đặc trưng phức tạp nhưng tốn tài nguyên hơn.  
            - **learning_rate_init**:  
              - **Ý nghĩa**: Tốc độ học ban đầu (ví dụ: 0.001).  
              - **Công dụng**: Điều chỉnh tốc độ cập nhật trọng số; giá trị nhỏ giúp học ổn định nhưng chậm.  
            - **max_iter**:  
              - **Ý nghĩa**: Số lần huấn luyện tối đa (ví dụ: 200).  
              - **Công dụng**: Giới hạn số lần lặp để đạt độ chính xác mong muốn.
            """, unsafe_allow_html=True)

            st.subheader("🟪 Ưu điểm và nhược điểm")
            st.markdown("""
            - **✅ Ưu điểm**:  
              - Học được các đặc trưng phức tạp từ dữ liệu hình ảnh như MNIST.  
              - Dễ sử dụng với các tham số cơ bản được tối ưu sẵn.  

            - **❌ Nhược điểm**:  
              - Tốn thời gian huấn luyện nếu số mẫu lớn hoặc số nơ-ron nhiều.  
              - Cần dữ liệu được chuẩn hóa để đạt hiệu quả tối ưu.
            """, unsafe_allow_html=True)

        elif info_option == "Công thức đánh giá độ chính xác (Accuracy)":
            st.subheader("📘 4. Công thức đánh giá độ chính xác (Accuracy)")
            st.markdown("""
            Độ chính xác (**Accuracy**) đo tỷ lệ dự đoán đúng của mô hình:  
            $$ \text{Accuracy} = \frac{\text{Số mẫu dự đoán đúng}}{\text{Tổng số mẫu}} $$

            - **Ví dụ**: Nếu mô hình dự đoán đúng 92/100 ảnh, thì:  
              $$ \text{Accuracy} = \frac{92}{100} = 92\% $$  

            - **Ý nghĩa**: Với Neural Network, Accuracy thể hiện khả năng mô hình phân loại đúng các chữ số dựa trên các đặc trưng pixel mà nó đã học được từ dữ liệu MNIST.
            """, unsafe_allow_html=True)

    # Tab 2: Tải dữ liệu
    with tab_load:
        st.header("Tải Dữ liệu")
        if st.button("Tải dữ liệu MNIST từ OpenML"):
            with st.spinner("Đang tải dữ liệu từ OpenML..."):
                progress_bar = st.progress(0)
                status_text = st.empty()
                try:
                    mnist = openml.datasets.get_dataset(554)
                    progress_bar.progress(20)
                    status_text.text("Đã tải 20% - Đang lấy dữ liệu...")

                    X, y, _, _ = mnist.get_data(target=mnist.default_target_attribute)
                    progress_bar.progress(50)
                    status_text.text("Đã tải 50% - Đang xử lý dữ liệu...")

                    st.session_state['full_data'] = (X, y)
                    progress_bar.progress(90)
                    status_text.text(f"Đã tải 90% - Hoàn tất {X.shape[0]} mẫu...")

                    with mlflow.start_run(run_name="Data_Load"):
                        mlflow.log_param("total_samples", X.shape[0])

                    progress_bar.progress(100)
                    status_text.text("Đã tải 100% - Hoàn tất!")
                    time.sleep(1)
                    status_text.empty()
                    progress_bar.empty()
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
                    progress_bar = st.progress(0)
                    status_text = st.empty()

                    df = pd.concat([X_full, y_full.rename("label")], axis=1)
                    progress_bar.progress(30)
                    status_text.text("Đã xử lý 30% - Đang nối dữ liệu...")

                    sampled_df = df.sample(n=num_samples, random_state=42)
                    progress_bar.progress(70)
                    status_text.text("Đã xử lý 70% - Đang lấy mẫu...")

                    X_sampled = sampled_df.drop(columns=["label"])
                    y_sampled = sampled_df["label"]
                    st.session_state['data'] = (X_sampled, y_sampled)
                    progress_bar.progress(90)
                    status_text.text("Đã xử lý 90% - Đang lưu dữ liệu...")

                    with mlflow.start_run(run_name="Data_Sample"):
                        mlflow.log_param("num_samples", num_samples)

                    progress_bar.progress(100)
                    status_text.text("Đã xử lý 100% - Hoàn tất!")
                    time.sleep(1)
                    status_text.empty()
                    progress_bar.empty()
                    st.success(f"Đã chốt {num_samples} mẫu!")

    # Tab 3: Xử lý dữ liệu
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
                    X_norm = X / 255.0
                    st.session_state["data_processed"] = (X_norm, y)
                    st.success("Đã chuẩn hóa dữ liệu!")
                    st.rerun()
            with col2:
                st.markdown("""
                    <div class="tooltip">
                        ?
                        <span class="tooltiptext">
                            Đưa dữ liệu về khoảng [0, 1] bằng cách chia cho 255.<br>
                            Công dụng: Đảm bảo thang đo đồng nhất, cần thiết cho Neural Network.
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

    # Tab 4: Chia dữ liệu
    with tab_split:
        st.header("Chia Tập Dữ Liệu")
        if 'data' not in st.session_state:
            st.info("Vui lòng tải và chốt số lượng mẫu trước.")
        else:
            data_source = st.session_state.get("data_processed", st.session_state['data'])
            X, y = data_source
            total_samples = len(X)
            st.write(f"Tổng số mẫu: {total_samples}")

            test_pct = st.slider("Tỷ lệ tập Test (%)", 0, 100, 20)
            valid_pct = st.slider("Tỷ lệ tập Validation (%) từ phần còn lại", 0, 100, 20)
            
            if test_pct + valid_pct > 100:
                st.warning("Tổng tỷ lệ Test và Validation vượt quá 100%!")
            
            test_size = int(total_samples * test_pct / 100)
            if test_size > 0:
                X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=test_size / total_samples, random_state=42)
            else:
                X_temp, y_temp = X, y
                X_test, y_test = pd.DataFrame(), pd.Series()

            valid_size = int(len(X_temp) * valid_pct / 100)
            if valid_size > 0 and len(X_temp) > valid_size:
                X_train, X_valid, y_train, y_valid = train_test_split(X_temp, y_temp, test_size=valid_size / len(X_temp), random_state=42)
            else:
                X_train, y_train = X_temp, y_temp
                X_valid, y_valid = pd.DataFrame(), pd.Series()

            st.write(f"Train: {len(X_train)} mẫu, Validation: {len(X_valid)} mẫu, Test: {len(X_test)} mẫu")
            if st.button("Xác nhận chia dữ liệu"):
                st.session_state['split_data'] = {
                    "X_train": X_train, "y_train": y_train,
                    "X_valid": X_valid, "y_valid": y_valid,
                    "X_test": X_test, "y_test": y_test
                }
                st.success("Dữ liệu đã được chia!")

    # Tab 5: Huấn luyện/Đánh Giá
    with tab_train_eval:
        st.header("Huấn luyện và Đánh Giá")
        if 'split_data' not in st.session_state:
            st.info("Vui lòng chia dữ liệu trước.")
        else:
            st.subheader("Huấn luyện Neural Network")
            X_train = st.session_state['split_data']["X_train"]
            num_samples = len(X_train)
            st.write(f"Số lượng mẫu huấn luyện: {num_samples}")

            st.subheader("Bảng gợi ý tham số tối ưu")
            st.markdown("""
            | Số lượng mẫu | Hidden Layer Sizes | Learning Rate Init | Max Iter |
            |--------------|--------------------|--------------------|----------|
            | <1000        | (64,)             | 0.01              | 100      |
            | 1000-5000    | (128,)            | 0.001             | 200      |
            | 5000-50000   | (256, 128)        | 0.001             | 300      |
            | >50000       | (512, 256)        | 0.0001            | 500      |
            """)
            st.markdown("""
            - **hidden_layer_sizes**: Tuple định nghĩa số nơ-ron trong các lớp ẩn.  
            - **learning_rate_init**: Tốc độ học ban đầu.  
            - **max_iter**: Số lần lặp tối đa.
            """)

            params = {}
            if num_samples < 1000:
                params["hidden_layer_sizes"] = (64,)
                params["learning_rate_init"] = 0.01
                params["max_iter"] = 100
            elif 1000 <= num_samples <= 5000:
                params["hidden_layer_sizes"] = (128,)
                params["learning_rate_init"] = 0.001
                params["max_iter"] = 200
            elif 5000 < num_samples <= 50000:
                params["hidden_layer_sizes"] = (256, 128)
                params["learning_rate_init"] = 0.001
                params["max_iter"] = 300
            else:
                params["hidden_layer_sizes"] = (512, 256)
                params["learning_rate_init"] = 0.0001
                params["max_iter"] = 500

            st.markdown("#### Tham số mô hình (có thể điều chỉnh)")
            hidden_layers_input = st.text_input("Hidden Layer Sizes (dạng tuple, ví dụ: 128, 64)", value=str(params["hidden_layer_sizes"]).strip("()"))
            params["hidden_layer_sizes"] = tuple(int(x.strip()) for x in hidden_layers_input.split(",") if x.strip())
            params["learning_rate_init"] = st.number_input("Learning Rate Init", min_value=0.0001, max_value=0.1, value=params["learning_rate_init"])
            params["max_iter"] = st.number_input("Max Iter", min_value=10, max_value=1000, value=params["max_iter"])

            if st.button("Thực hiện Huấn luyện"):
                with st.spinner("Đang huấn luyện mô hình..."):
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    start_time = time.time()

                    X_train = st.session_state['split_data']["X_train"]
                    y_train = st.session_state['split_data']["y_train"]
                    X_valid = st.session_state['split_data']["X_valid"]
                    y_valid = st.session_state['split_data']["y_valid"]
                    X_test = st.session_state['split_data']["X_test"]
                    y_test = st.session_state['split_data']["y_test"]

                    run_name = f"NeuralNetwork_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                    with mlflow.start_run(run_name=run_name) as run:
                        pipeline = Pipeline([
                            ('imputer', SimpleImputer(strategy='mean')),
                            ('classifier', MLPClassifier(
                                hidden_layer_sizes=params["hidden_layer_sizes"],
                                learning_rate_init=params["learning_rate_init"],
                                max_iter=params["max_iter"],
                                activation='relu',
                                solver='adam',
                                random_state=42,
                                verbose=False
                            ))
                        ])
                        pipeline.fit(X_train, y_train)
                        model = pipeline

                        for i in range(0, 51, 5):
                            progress_bar.progress(i)
                            status_text.text(f"Đang huấn luyện {i}%...")

                        mlflow.log_params(params)
                        y_valid_pred = model.predict(X_valid)
                        accuracy_val = accuracy_score(y_valid, y_valid_pred)
                        mlflow.log_metric("accuracy_val", accuracy_val)
                        cm_valid = confusion_matrix(y_valid, y_valid_pred)

                        y_test_pred = model.predict(X_test)
                        accuracy_test = accuracy_score(y_test, y_test_pred)
                        mlflow.log_metric("accuracy_test", accuracy_test)
                        cm_test = confusion_matrix(y_test, y_test_pred)

                        training_time = time.time() - start_time
                        mlflow.log_metric("training_time_seconds", training_time)
                        mlflow.sklearn.log_model(model, "model")

                        for i in range(50, 101, 5):
                            progress_bar.progress(i)
                            status_text.text(f"Hoàn tất {i}%...")

                        run_id = run.info.run_id
                        st.session_state['model'] = model
                        st.session_state['training_results'] = {
                            'training_time': training_time,
                            'accuracy_val': accuracy_val,
                            'accuracy_test': accuracy_test,
                            'cm_valid': cm_valid,
                            'cm_test': cm_test,
                            'model_choice': 'Neural Network',
                            'params': params,
                            'num_samples': num_samples,
                            'run_name': run_name,
                            'run_id': run_id
                        }

                        status_text.empty()
                        progress_bar.empty()

            if 'training_results' in st.session_state:
                st.success(f"Huấn luyện hoàn tất. Thời gian: {st.session_state['training_results']['training_time']:.2f} giây.")
                st.write(f"Accuracy Validation: {st.session_state['training_results']['accuracy_val']:.4f}")
                st.write(f"Accuracy Test: {st.session_state['training_results']['accuracy_test']:.4f}")

                fig, ax = plt.subplots()
                sns.heatmap(st.session_state['training_results']['cm_valid'], annot=True, fmt="d", cmap="Blues", ax=ax)
                ax.set_title("Confusion Matrix - Validation")
                st.pyplot(fig)

                fig, ax = plt.subplots()
                sns.heatmap(st.session_state['training_results']['cm_test'], annot=True, fmt="d", cmap="Blues", ax=ax)
                ax.set_title("Confusion Matrix - Test")
                st.pyplot(fig)

    # Tab 6: Demo dự đoán
    with tab_demo:
        st.header("Demo Dự đoán")
        if 'split_data' not in st.session_state or 'model' not in st.session_state:
            st.info("Vui lòng huấn luyện mô hình trước.")
        else:
            mode = st.radio("Chọn phương thức dự đoán:", ["Dữ liệu từ Test", "Upload ảnh mới", "Vẽ số"])
            
            progress_bar = st.progress(0)
            status_text = st.empty()

            def preprocess_input(data):
                return data / 255.0

            is_normalized = "data_processed" in st.session_state

            if mode == "Dữ liệu từ Test":
                X_test = st.session_state['split_data']["X_test"]
                y_test = st.session_state['split_data']["y_test"]
                idx = st.slider("Chọn mẫu từ Test", 0, len(X_test)-1, 0)
                if st.button("Dự đoán"):
                    with st.spinner("Đang dự đoán..."):
                        for i in range(0, 51, 5):
                            progress_bar.progress(i)
                            status_text.text(f"Đang xử lý {i}%...")
                            time.sleep(0.1)
                        
                        sample = X_test.iloc[idx].values.reshape(1, -1)
                        if not is_normalized:
                            sample = preprocess_input(sample)
                        
                        prediction = st.session_state['model'].predict(sample)[0]
                        proba = st.session_state['model'].predict_proba(sample)[0]
                        confidence = max(proba) * 100
                        y_true = y_test.iloc[idx]
                        
                        for i in range(50, 101, 5):
                            progress_bar.progress(i)
                            status_text.text(f"Đang dự đoán {i}%...")
                            time.sleep(0.1)
                        
                        st.success(f"Dự đoán: **{prediction}** | Confidence: **{confidence:.2f}%** | Giá trị thực: **{y_true}**")
                        fig, ax = plt.subplots()
                        ax.imshow(X_test.iloc[idx].values.reshape(28, 28), cmap='gray')
                        ax.axis("off")
                        st.pyplot(fig)
                        
                        time.sleep(1)
                        progress_bar.empty()
                        status_text.empty()

            elif mode == "Upload ảnh mới":
                uploaded_images = st.file_uploader("Upload ảnh (28x28, grayscale)", type=["png", "jpg"], accept_multiple_files=True)
                if uploaded_images:
                    for i, uploaded_image in enumerate(uploaded_images):
                        with st.spinner(f"Đang xử lý ảnh {i+1}/{len(uploaded_images)}..."):
                            for j in range(0, 51, 5):
                                progress_bar.progress(j)
                                status_text.text(f"Đang tải ảnh {i+1} - {j}%...")
                                time.sleep(0.1)
                            
                            img = Image.open(uploaded_image).convert('L').resize((28, 28))
                            img_array = np.array(img).flatten().reshape(1, -1)
                            if not is_normalized:
                                img_array = preprocess_input(img_array)
                            
                            prediction = st.session_state['model'].predict(img_array)[0]
                            proba = st.session_state['model'].predict_proba(img_array)[0]
                            confidence = max(proba) * 100
                            
                            for j in range(50, 101, 5):
                                progress_bar.progress(j)
                                status_text.text(f"Đang dự đoán ảnh {i+1} - {j}%...")
                                time.sleep(0.1)
                            
                            st.success(f"Dự đoán: **{prediction}** | Độ tin cậy: **{confidence:.2f}%**")
                            st.image(img, caption=f"Ảnh {i+1} được upload", use_container_width=True)
                            
                            time.sleep(1)
                            progress_bar.empty()
                            status_text.empty()

            elif mode == "Vẽ số":
                st.write("Vẽ một chữ số từ 0-9 trên canvas bên dưới (28x28 pixel):")
                canvas_result = st_canvas(
                    fill_color="black",
                    stroke_width=20,
                    stroke_color="white",
                    background_color="black",
                    width=280,
                    height=280,
                    drawing_mode="freedraw",
                    key="canvas"
                )
                if st.button("Dự đoán số đã vẽ"):
                    if canvas_result.image_data is not None:
                        with st.spinner("Đang xử lý vẽ..."):
                            for i in range(0, 51, 5):
                                progress_bar.progress(i)
                                status_text.text(f"Đang xử lý {i}%...")
                                time.sleep(0.1)
                            
                            img = Image.fromarray((canvas_result.image_data * 255).astype(np.uint8)).convert('L').resize((28, 28))
                            img_array = np.array(img).flatten().reshape(1, -1)
                            if not is_normalized:
                                img_array = preprocess_input(img_array)
                            
                            prediction = st.session_state['model'].predict(img_array)[0]
                            proba = st.session_state['model'].predict_proba(img_array)[0]
                            confidence = max(proba) * 100
                            
                            for i in range(50, 101, 5):
                                progress_bar.progress(i)
                                status_text.text(f"Đang dự đoán {i}%...")
                                time.sleep(0.1)
                            
                            st.success(f"Dự đoán: **{prediction}** | Độ tin cậy: **{confidence:.2f}%**")
                            
                            time.sleep(1)
                            progress_bar.empty()
                            status_text.empty()
                    else:
                        st.warning("Vui lòng vẽ một chữ số trước khi dự đoán!")

    # Tab 7: Thông tin huấn luyện
    with tab_log_info:
        st.header("Theo dõi kết quả")
        st.markdown("""
        Tab này cho phép bạn xem danh sách các lần huấn luyện đã thực hiện. Chọn một lần chạy để xem chi tiết, đổi tên hoặc xóa.
        """, unsafe_allow_html=True)
        
        try:
            client = MlflowClient()
            experiment = client.get_experiment_by_name("Neural Network ")
            if not experiment:
                st.error("Không tìm thấy experiment 'Neural Network '. Vui lòng kiểm tra lại MLflow tracking URI.")
            else:
                experiment_id = experiment.experiment_id
                runs = client.search_runs(experiment_ids=[experiment_id], order_by=["attributes.start_time DESC"])
                
                if not runs:
                    st.info("Chưa có lần chạy nào được ghi nhận.")
                else:
                    run_options = {run.info.run_id: run.data.tags.get('mlflow.runName', f"Run_{run.info.run_id}") for run in runs}
                    run_names = list(run_options.values())
                    selected_run_name = st.selectbox("Chọn run:", run_names)
                    selected_run_id = [k for k, v in run_options.items() if v == selected_run_name][0]
                    selected_run = client.get_run(selected_run_id)

                    st.subheader("Đổi tên Run")
                    new_run_name = st.text_input("Nhập tên mới:", value=selected_run_name)
                    if st.button("Cập nhật tên"):
                        client.set_tag(selected_run_id, "mlflow.runName", new_run_name)
                        st.success(f"Đã đổi tên thành: {new_run_name}")
                        st.rerun()

                    st.subheader("Xóa Run")
                    if st.button("Xóa lần chạy"):
                        client.delete_run(selected_run_id)
                        st.success(f"Đã xóa: {selected_run_name}")
                        st.rerun()

                    st.subheader("Thông tin chi tiết của Run")
                    st.write(f"**Tên lần chạy:** {selected_run_name}")
                    st.write(f"**ID lần chạy:** {selected_run_id}")
                    st.write(f"**Tham số:** {selected_run.data.params}")
                    st.write(f"**Kết quả:** {selected_run.data.metrics}")
        except Exception as e:
            st.error(f"Lỗi kết nối MLflow: {e}")

if __name__ == "__main__":
       run_mnist_neural_network_app()