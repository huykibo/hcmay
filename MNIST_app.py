import os
import mlflow
import streamlit as st
import openml
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.neural_network import MLPClassifier  # Neural Network
from sklearn.impute import SimpleImputer
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
        mlflow.set_experiment("MNIST_NeuralNetwork")
    except KeyError as e:
        st.error(f"Lỗi: Không tìm thấy khóa {e} trong st.secrets. Vui lòng cấu hình secrets trong Streamlit.")
        st.stop()

    st.title("Ứng dụng Phân loại Chữ số MNIST với Neural Network")

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
        info_option = st.selectbox(
            "Chọn thông tin để xem:",
            [
                "Ứng dụng này là gì và mục tiêu của nó?",
                "Tập dữ liệu MNIST: Đặc điểm và ý nghĩa",
                "Neural Network – Mạng nơ-ron nhân tạo",
                "Công thức đánh giá độ chính xác (Accuracy)"
            ],
            index=0,
            key="info_selectbox"
        )

        if info_option == "Ứng dụng này là gì và mục tiêu của nó?":
            st.subheader("1. Ứng dụng này là gì và mục tiêu của nó?")
            st.markdown("""
            Đây là một ứng dụng phân loại chữ số viết tay dựa trên tập dữ liệu **MNIST**, sử dụng **Mạng nơ-ron nhân tạo (Neural Network)**.  
            - **MNIST** bao gồm 70,000 ảnh chữ số từ 0 đến 9, mỗi ảnh kích thước \(28 \times 28\) pixel (784 đặc trưng).  
            - **Mục tiêu**: Huấn luyện một mạng nơ-ron để nhận diện chính xác các chữ số, cung cấp công cụ trực quan cho việc học tập và đánh giá thuật toán.

            **Thông tin cơ bản**:  
            - **784 đặc trưng**: Mỗi ảnh là vector 784 chiều (giá trị pixel từ 0-255).  
            - **70,000 mẫu**: Tổng số ảnh, chia thành tập huấn luyện và kiểm tra.  
            - **Nhiệm vụ**: Dự đoán nhãn (0-9) dựa trên đặc trưng pixel.
            """)

        elif info_option == "Tập dữ liệu MNIST: Đặc điểm và ý nghĩa":
            st.subheader("2. Tập dữ liệu MNIST: Đặc điểm và ý nghĩa")
            st.markdown("""
            **MNIST** là tập dữ liệu chuẩn trong học máy, được tạo bởi Yann LeCun và cộng sự.  
            - **Đặc điểm**: Ảnh chữ số viết tay từ học sinh và nhân viên điều tra dân số Mỹ, chuẩn hóa thành \(28 \times 28\) pixel, thang độ xám.

            **Ý nghĩa**:  
            - Bài toán cơ bản để kiểm tra khả năng phân loại.  
            - Đơn giản nhưng đủ phức tạp để phân biệt các lớp tương tự (ví dụ: "4" và "9").  
            - Phù hợp cho người mới bắt đầu và nghiên cứu mô hình phức tạp.
            """)
            st.subheader("Minh họa dữ liệu MNIST")
            with st.spinner("Đang tải ảnh minh họa..."):
                try:
                    mnist_image = Image.open("mnist.png")
                    st.image(mnist_image, caption="Ảnh minh họa 10 chữ số từ 0 đến 9 trong MNIST", width=800)
                except FileNotFoundError:
                    st.error("Không tìm thấy file `mnist.png`. Vui lòng kiểm tra đường dẫn.")
                except Exception as e:
                    st.error(f"Lỗi khi tải ảnh: {e}")

        elif info_option == "Neural Network – Mạng nơ-ron nhân tạo":
            st.subheader("3. Neural Network – Mạng nơ-ron nhân tạo")
            st.markdown("""
            **Neural Network (Mạng nơ-ron nhân tạo)** mô phỏng cách hoạt động của não người, sử dụng các **nơ-ron nhân tạo** tổ chức thành **lớp (layers)**:  
            - **Input Layer**: Nhận 784 pixel từ ảnh MNIST.  
            - **Hidden Layers**: Xử lý thông tin qua kết hợp tuyến tính và phi tuyến.  
            - **Output Layer**: Dự đoán nhãn (0-9).
            """)

            st.subheader("Cách hoạt động chi tiết:")
            st.markdown("""
            1. **Khởi tạo mô hình**:  
               - Xác định số lớp ẩn và nơ-ron mỗi lớp (ví dụ: \( (128, 64) \)).  
               - Khởi tạo **trọng số** \( W \) và **bias** \( b \) ngẫu nhiên.

            2. **Lan truyền thuận (Feedforward)**:  
               - Tính giá trị dự đoán \( \hat{Y} \) từ dữ liệu đầu vào \( X \):  
               - **Lớp đầu vào**:  
                 $$ A^{(0)} = X $$  
                 (Ma trận \( N \times 784 \), \( N \) là số mẫu).  
               - **Cho mỗi lớp \( l \)**:  
                 - Tính tổng tuyến tính:  
                   $$ Z^{(l)} = A^{(l-1)} \cdot W^{(l)} + b^{(l)} $$  
                 - Áp dụng hàm kích hoạt:  
                   $$ A^{(l)} = \sigma(Z^{(l)}) $$  
               - **Lớp đầu ra**:  
                 $$ \hat{Y} = A^{(L)} $$  
                 (Ma trận \( N \times 10 \), 10 lớp từ 0-9).  
               - Ví dụ hàm kích hoạt **sigmoid**:  
                 $$ \sigma(z) = \frac{1}{1 + e^{-z}} $$

            3. **Tính hàm mất mát (Loss Function)**:  
               - Đo độ sai lệch giữa \( \hat{Y} \) và \( Y \) (giá trị thực). Với MNIST, dùng **Cross-Entropy**:  
                 $$ L = -\frac{1}{N} \sum_{i=1}^{N} \sum_{j=0}^{9} y_{ij} \cdot \log(\hat{y}_{ij}) $$  
               - Trong đó:  
                 - \( y_{ij} \): Nhãn thực (dạng one-hot encoded, ví dụ: nhãn 3 → \( [0, 0, 0, 1, 0, 0, 0, 0, 0, 0] \)).  
                 - \( \hat{y}_{ij} \): Xác suất dự đoán cho lớp \( j \) (từ lớp đầu ra \( \hat{Y} \)).

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

            5. **Cập nhật tham số**:  
               - Sử dụng Gradient Descent:  
                 $$ W^{(l)} = W^{(l)} - \eta \cdot \frac{\partial L}{\partial W^{(l)}} $$  
                 $$ b^{(l)} = b^{(l)} - \eta \cdot \frac{\partial L}{\partial b^{(l)}} $$  
               - \( \eta \): Tốc độ học (learning rate).

            6. **Lặp lại**:  
               - Qua nhiều epoch đến khi \( L \) hội tụ.
            """)

            st.subheader("Tham số chính:")
            st.markdown("""
            | **Tham số**           | **Ý nghĩa**                              | **Công dụng**                              |
            |-----------------------|------------------------------------------|--------------------------------------------|
            | **hidden_layer_sizes**| Số lớp ẩn và nơ-ron (ví dụ: \( (128, 64) \)) | Điều chỉnh độ phức tạp của mạng           |
            | **activation**        | Hàm kích hoạt (relu, sigmoid)           | Thêm tính phi tuyến                       |
            | **learning_rate_init**| Tốc độ học ban đầu (ví dụ: \( 0.001 \))  | Điều chỉnh bước cập nhật tham số          |
            | **max_iter**          | Số epoch tối đa (ví dụ: 200)            | Giới hạn thời gian huấn luyện             |
            | **solver**            | Thuật toán tối ưu (adam, sgd)           | Quyết định cách cập nhật trọng số         |
            """)

            st.subheader("Áp dụng với MNIST:")
            st.markdown("""
            - **Đầu vào**: \( 784 \) pixel (\( 28 \times 28 \)).  
            - **Lớp ẩn**: Ví dụ \( (128, 64) \).  
            - **Đầu ra**: \( 10 \) nơ-ron (0-9).
            """)

            st.subheader("Ưu điểm:")
            st.markdown("""
            - Học được đặc trưng phi tuyến phức tạp.  
            - Hiệu quả với dữ liệu hình ảnh như MNIST.
            """)

            st.subheader("Nhược điểm:")
            st.markdown("""
            - Tốn thời gian huấn luyện với dữ liệu lớn.  
            - Cần điều chỉnh tham số cẩn thận để tránh overfitting.
            """)

        elif info_option == "Công thức đánh giá độ chính xác (Accuracy)":
            st.subheader("4. Công thức đánh giá độ chính xác (Accuracy)")
            st.markdown("""
            Độ chính xác (**Accuracy**) đo tỷ lệ dự đoán đúng:  
            $$ Accuracy = \frac{\text{Số mẫu dự đoán đúng}}{\text{Tổng số mẫu}} $$  
            - **Ví dụ**: Dự đoán đúng 92/100 ảnh → \( Accuracy = 92\% \).  
            - **Ý nghĩa**: Đo khả năng Neural Network phân loại đúng các chữ số dựa trên đặc trưng pixel.
            """)

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

    # Tab 3: Xử lí dữ liệu
    with tab_preprocess:
        st.header("Xử lí Dữ liệu")
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
                            Công dụng: Đảm bảo thang đo đồng nhất, giúp Neural Network học tốt hơn.
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
            X_train = st.session_state['split_data']["X_train"]
            num_samples = len(X_train)
            st.write(f"Số lượng mẫu huấn luyện: {num_samples}")

            st.subheader("Bảng gợi ý tham số tối ưu dựa trên số lượng mẫu")
            st.markdown("""
            | Số lượng mẫu | Hidden Layers      | Activation | Learning Rate | Max Iter |
            |--------------|--------------------|------------|---------------|----------|
            | <1000        | (64,)             | relu       | 0.01          | 100      |
            | 1000-5000    | (128, 64)         | relu       | 0.001         | 200      |
            | 5000-50000   | (256, 128)        | relu       | 0.001         | 300      |
            | >50000       | (512, 256, 128)   | relu       | 0.0001        | 500      |
            """)

            params = {}
            if num_samples < 1000:
                params["hidden_layer_sizes"] = (64,)
                params["activation"] = "relu"
                params["learning_rate_init"] = 0.01
                params["max_iter"] = 100
                params["solver"] = "adam"
            elif 1000 <= num_samples <= 5000:
                params["hidden_layer_sizes"] = (128, 64)
                params["activation"] = "relu"
                params["learning_rate_init"] = 0.001
                params["max_iter"] = 200
                params["solver"] = "adam"
            elif 5000 < num_samples <= 50000:
                params["hidden_layer_sizes"] = (256, 128)
                params["activation"] = "relu"
                params["learning_rate_init"] = 0.001
                params["max_iter"] = 300
                params["solver"] = "adam"
            else:
                params["hidden_layer_sizes"] = (512, 256, 128)
                params["activation"] = "relu"
                params["learning_rate_init"] = ?

                params["max_iter"] = 500
                params["solver"] = "adam"

            st.markdown("#### Tham số mô hình (đã đặt tự động, có thể điều chỉnh)")
            hidden_layers_input = st.text_input("Hidden Layer Sizes (cách nhau bằng dấu phẩy)", 
                                                value=", ".join(map(str, params["hidden_layer_sizes"])))
            params["hidden_layer_sizes"] = tuple(map(int, hidden_layers_input.split(", ")))
            params["activation"] = st.selectbox("Activation", ["relu", "sigmoid", "tanh"], 
                                                index=["relu", "sigmoid", "tanh"].index(params["activation"]))
            params["learning_rate_init"] = st.number_input("Learning Rate", min_value=0.0001, max_value=0.1, 
                                                           value=params["learning_rate_init"], step=0.0001)
            params["max_iter"] = st.number_input("Max Iterations", min_value=10, max_value=1000, 
                                                 value=params["max_iter"], step=10)
            params["solver"] = st.selectbox("Solver", ["adam", "sgd"], 
                                            index=["adam", "sgd"].index(params["solver"]))

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
                            ('classifier', MLPClassifier(**params))
                        ])
                        pipeline.fit(X_train, y_train)
                        model = pipeline

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

                        run_id = run.info.run_id
                        st.session_state['model'] = model
                        st.session_state['training_results'] = {
                            'training_time': training_time,
                            'accuracy_val': accuracy_val,
                            'accuracy_test': accuracy_test,
                            'cm_valid': cm_valid,
                            'cm_test': cm_test,
                            'params': params,
                            'num_samples': len(X_train),
                            'run_name': run_name,
                            'run_id': run_id
                        }

                        progress_bar.progress(100)
                        status_text.text("Hoàn tất!")
                        time.sleep(1)
                        status_text.empty()
                        progress_bar.empty()

            if 'training_results' in st.session_state:
                st.success(f"Huấn luyện hoàn tất. Thời gian thực hiện: {st.session_state['training_results']['training_time']:.2f} giây.")
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
            
            def preprocess_input(data):
                return data / 255.0

            is_normalized = "data_processed" in st.session_state

            if mode == "Dữ liệu từ Test":
                X_test = st.session_state['split_data']["X_test"]
                y_test = st.session_state['split_data']["y_test"]
                idx = st.slider("Chọn mẫu từ Test", 0, len(X_test)-1, 0)
                if st.button("Dự đoán"):
                    with st.spinner("Đang dự đoán..."):
                        sample = X_test.iloc[idx].values.reshape(1, -1)
                        if not is_normalized:
                            sample = preprocess_input(sample)
                        
                        prediction = st.session_state['model'].predict(sample)[0]
                        proba = st.session_state['model'].predict_proba(sample)[0]
                        confidence = max(proba) * 100
                        y_true = y_test.iloc[idx]
                        
                        st.success(f"Dự đoán: **{prediction}** | Confidence: **{confidence:.2f}%** | Giá trị thực: **{y_true}**")
                        fig, ax = plt.subplots()
                        ax.imshow(X_test.iloc[idx].values.reshape(28, 28), cmap='gray')
                        ax.axis("off")
                        st.pyplot(fig)

            elif mode == "Upload ảnh mới":
                uploaded_images = st.file_uploader("Upload ảnh (28x28, grayscale)", type=["png", "jpg"], accept_multiple_files=True)
                if uploaded_images:
                    for i, uploaded_image in enumerate(uploaded_images):
                        with st.spinner(f"Đang xử lý ảnh {i+1}/{len(uploaded_images)}..."):
                            img = Image.open(uploaded_image).convert('L').resize((28, 28))
                            img_array = np.array(img).flatten().reshape(1, -1)
                            if not is_normalized:
                                img_array = preprocess_input(img_array)
                            
                            prediction = st.session_state['model'].predict(img_array)[0]
                            proba = st.session_state['model'].predict_proba(img_array)[0]
                            confidence = max(proba) * 100
                            
                            st.success(f"Dự đoán: **{prediction}** | Độ tin cậy: **{confidence:.2f}%**")
                            st.image(img, caption=f"Ảnh {i+1} được upload", use_container_width=True)

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
                            img = Image.fromarray((canvas_result.image_data * 255).astype(np.uint8)).convert('L').resize((28, 28))
                            img_array = np.array(img).flatten().reshape(1, -1)
                            if not is_normalized:
                                img_array = preprocess_input(img_array)
                            
                            prediction = st.session_state['model'].predict(img_array)[0]
                            proba = st.session_state['model'].predict_proba(img_array)[0]
                            confidence = max(proba) * 100
                            
                            st.success(f"Dự đoán: **{prediction}** | Độ tin cậy: **{confidence:.2f}%**")
                    else:
                        st.warning("Vui lòng vẽ một chữ số trước khi dự đoán!")

    # Tab 7: Thông tin huấn luyện
    with tab_log_info:
        st.header("Theo dõi kết quả")
        try:
            client = MlflowClient()
            experiment = client.get_experiment_by_name("MNIST_NeuralNetwork")
            if not experiment:
                st.error("Không tìm thấy experiment 'MNIST_NeuralNetwork'.")
            else:
                runs = client.search_runs(experiment_ids=[experiment.experiment_id], order_by=["attributes.start_time DESC"])
                if not runs:
                    st.info("Chưa có lần chạy nào được ghi nhận.")
                else:
                    run_options = {run.info.run_id: run.data.tags.get('mlflow.runName', f"Run_{run.info.run_id}") for run in runs}
                    selected_run_name = st.selectbox("Chọn run:", list(run_options.values()))
                    selected_run_id = [k for k, v in run_options.items() if v == selected_run_name][0]
                    selected_run = client.get_run(selected_run_id)

                    st.write(f"**Tên lần chạy:** {selected_run_name}")
                    st.write(f"**ID lần chạy:** {selected_run_id}")
                    st.write(f"**Thời gian bắt đầu:** {datetime.fromtimestamp(selected_run.info.start_time / 1000)}")
                    st.json(selected_run.data.params, expanded=True)
                    st.json(selected_run.data.metrics, expanded=True)

        except Exception as e:
            st.error(f"Lỗi kết nối MLflow: {e}")

if __name__ == "__main__":
    run_mnist_neural_network_app()