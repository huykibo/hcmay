import os
import mlflow
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from mlflow.tracking import MlflowClient
from streamlit_drawable_canvas import st_canvas
from datetime import datetime
import time
import requests
import io
import sys
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
import gc

# Hàm chọn tham số tối ưu dựa trên số mẫu
def get_optimal_params(num_samples):
    """Xác định tham số tối ưu cho mô hình dựa trên số lượng mẫu."""
    if num_samples <= 1000:
        return {
            "hidden_layer_sizes": (32,),
            "learning_rate": 0.001,
            "epochs": 30,
            "activation": "relu",
            "solver": "adam",
            "batch_size": 32,
            "threshold": 0.95,
            "max_iterations": 5
        }
    elif num_samples <= 10000:
        return {
            "hidden_layer_sizes": (64, 32),
            "learning_rate": 0.0005,
            "epochs": 50,
            "activation": "relu",
            "solver": "adam",
            "batch_size": 64,
            "threshold": 0.95,
            "max_iterations": 10
        }
    elif num_samples <= 50000:
        return {
            "hidden_layer_sizes": (128, 64),
            "learning_rate": 0.0003,
            "epochs": 70,
            "activation": "relu",
            "solver": "adam",
            "batch_size": 128,
            "threshold": 0.95,
            "max_iterations": 15
        }
    else:  # > 50,000
        return {
            "hidden_layer_sizes": (128, 64, 32),
            "learning_rate": 0.0001,
            "epochs": 100,
            "activation": "relu",
            "solver": "adam",
            "batch_size": 256,
            "threshold": 0.95,
            "max_iterations": 20
        }

# Hàm xây dựng mô hình Neural Network
def build_model(params):
    """Xây dựng mô hình Neural Network dựa trên tham số."""
    model = models.Sequential()
    model.add(layers.InputLayer(input_shape=(784,)))
    for units in params["hidden_layer_sizes"]:
        model.add(layers.Dense(units, activation=params["activation"]))
    model.add(layers.Dense(10, activation='softmax'))
    optimizer = tf.keras.optimizers.Adam(learning_rate=params["learning_rate"]) if params["solver"] == "adam" else tf.keras.optimizers.SGD(learning_rate=params["learning_rate"])
    model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

# Ứng dụng chính
def run_mnist_pseudo_labeling_app():
    """Chạy ứng dụng Streamlit để phân loại chữ số MNIST với Neural Network và Pseudo-Labeling."""

    ### Thiết lập MLflow
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

    EXPERIMENT_ID = "6"
    try:
        client = MlflowClient()
        experiment = client.get_experiment(EXPERIMENT_ID)
        if experiment is None:
            st.error(f"Experiment ID {EXPERIMENT_ID} không tồn tại.")
            st.stop()
    except Exception as e:
        st.error(f"Lỗi truy xuất Experiment ID {EXPERIMENT_ID}: {e}.")
        st.stop()

    ### Tải dữ liệu MNIST
    if 'full_data' not in st.session_state:
        with st.spinner("Đang tải dữ liệu MNIST..."):
            (X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()
            X_full = np.concatenate([X_train, X_test], axis=0)
            y_full = np.concatenate([y_train, y_test], axis=0)
            X_full = X_full.reshape(-1, 784).astype(np.float32)
            y_full = y_full.astype(np.int32)
            st.session_state['full_data'] = (X_full, y_full)

    st.title("Phân loại Chữ số MNIST với Neural Network và Pseudo-Labeling")

    ### CSS tùy chỉnh
    st.markdown("""
        <style>
            .section-title {
                font-size: 1.5em;
                font-weight: bold;
                color: #2c3e50;
                margin-bottom: 10px;
            }
            .stCanvas {
                border: 1px solid #ddd;
                border-radius: 5px;
            }
            .tooltip {
                position: relative;
                display: inline-block;
                cursor: pointer;
                color: #3498db;
                font-weight: bold;
            }
            .tooltip .tooltiptext {
                visibility: hidden;
                width: 220px;
                background-color: #555;
                color: #fff;
                text-align: center;
                border-radius: 6px;
                padding: 5px;
                position: absolute;
                z-index: 1;
                bottom: 125%;
                left: 50%;
                margin-left: -110px;
                opacity: 0;
                transition: opacity 0.3s;
            }
            .tooltip:hover .tooltiptext {
                visibility: visible;
                opacity: 1;
            }
        </style>
    """, unsafe_allow_html=True)

    ### Tạo các tab
    tab_names = ["Thông tin", "Chọn số lượng dữ liệu", "Xử lý dữ liệu", "Chia dữ liệu", 
                 "Huấn luyện/Đánh giá", "Demo dự đoán", "Thông tin huấn luyện"]
    tab_info, tab_load, tab_preprocess, tab_split, tab_train_eval, tab_demo, tab_log_info = st.tabs(tab_names)

    #### Tab 1: Thông tin
    with tab_info:
        st.header("Giới thiệu Ứng dụng Phân loại Chữ số MNIST với Neural Network và Pseudo-Labeling")
        st.markdown("""
        Chào mừng bạn đến với ứng dụng phân loại chữ số viết tay từ tập dữ liệu **MNIST** sử dụng **Mạng nơ-ron nhân tạo (Neural Network)** kết hợp với kỹ thuật **Pseudo-Labeling**. Ứng dụng này được thiết kế để cung cấp trải nghiệm trực quan, hỗ trợ học tập và nghiên cứu về các thuật toán học máy hiện đại.
        """, unsafe_allow_html=True)

        st.subheader("Chọn nội dung để khám phá")
        info_option = st.selectbox(
            "",
            [
                "Tổng quan về ứng dụng và mục tiêu",
                "Tập dữ liệu MNIST: Đặc điểm và ý nghĩa",
                "Neural Network – Mạng nơ-ron nhân tạo",
                "Pseudo-Labeling – Kỹ thuật học bán giám sát"
            ],
            label_visibility="collapsed",
            help="Khám phá chi tiết về ứng dụng, dữ liệu, mô hình và kỹ thuật Pseudo-Labeling."
        )

        content_placeholder = st.empty()
        with content_placeholder.container():
            if info_option == "Tổng quan về ứng dụng và mục tiêu":
                with st.spinner("Đang tải thông tin..."):
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    for i in range(0, 101, 10):
                        progress_bar.progress(i)
                        status_text.text(f"Đang tải nội dung... {i}%")
                        time.sleep(0.05)
                    st.subheader("📌 Tổng quan về ứng dụng và mục tiêu")
                    st.markdown("""
                    Ứng dụng này tập trung vào việc phân loại chữ số viết tay dựa trên tập dữ liệu **MNIST**, một bộ dữ liệu tiêu chuẩn trong lĩnh vực học máy. Kết hợp **Neural Network** và **Pseudo-Labeling**, ứng dụng không chỉ tối ưu hóa hiệu suất mô hình mà còn tận dụng dữ liệu không có nhãn để nâng cao khả năng học tập.

                    **Mục tiêu chính:**
                    - Phát triển một mô hình Neural Network có khả năng nhận diện chính xác các chữ số từ 0 đến 9.
                    - Áp dụng kỹ thuật Pseudo-Labeling để khai thác dữ liệu không có nhãn, mô phỏng các tình huống thực tế khi dữ liệu có nhãn hạn chế.
                    - Cung cấp giao diện trực quan để người dùng thực hành, đánh giá và tùy chỉnh mô hình.

                    **Thông tin cơ bản về dữ liệu:**
                    - **Quy mô:** 70,000 ảnh, mỗi ảnh kích thước 28x28 pixel (tổng cộng 784 đặc trưng).
                    - **Đặc trưng:** Giá trị pixel từ 0 đến 255, biểu diễn dưới dạng vector 784 chiều.
                    - **Nhiệm vụ:** Dự đoán nhãn tương ứng với từng chữ số từ 0 đến 9.
                    """, unsafe_allow_html=True)
                    status_text.empty()
                    progress_bar.empty()

            elif info_option == "Tập dữ liệu MNIST: Đặc điểm và ý nghĩa":
                with st.spinner("Đang tải thông tin..."):
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    for i in range(0, 101, 10):
                        progress_bar.progress(i)
                        status_text.text(f"Đang tải nội dung... {i}%")
                        time.sleep(0.05)
                    st.subheader("📌 Tập dữ liệu MNIST: Đặc điểm và ý nghĩa")
                    st.markdown("""
                    **MNIST** là một tập dữ liệu tiêu chuẩn trong học máy, được phát triển bởi Yann LeCun và các cộng sự, thường được sử dụng để đánh giá hiệu suất của các mô hình phân loại.

                    **Đặc điểm nổi bật:**
                    - **Nguồn gốc:** Bao gồm ảnh chữ số viết tay từ học sinh trung học và nhân viên điều tra dân số Hoa Kỳ.
                    - **Kích thước:** Mỗi ảnh có độ phân giải 28x28 pixel, thang độ xám với giá trị từ 0 đến 255.
                    - **Quy mô:** Tổng cộng 70,000 ảnh, chia thành tập huấn luyện (60,000 ảnh) và tập kiểm tra (10,000 ảnh).

                    **Ý nghĩa:**
                    - Là nền tảng lý tưởng để thử nghiệm các thuật toán học máy, từ cơ bản đến nâng cao.
                    - Giúp đánh giá khả năng phân biệt các lớp tương tự (ví dụ: 4 và 9) trong các mô hình Neural Network.
                    - Hỗ trợ nghiên cứu và đào tạo cho cả người mới bắt đầu lẫn các chuyên gia trong lĩnh vực học sâu.
                    """, unsafe_allow_html=True)
                    try:
                        st.image(os.path.join("mnist.png"), caption="Tổng quan về tập dữ liệu MNIST", width=800)
                    except FileNotFoundError:
                        st.warning("Không tìm thấy ảnh minh họa 'mnist.png'. Vui lòng kiểm tra đường dẫn.")
                    status_text.empty()
                    progress_bar.empty()

            elif info_option == "Neural Network – Mạng nơ-ron nhân tạo":
                with st.spinner("Đang tải thông tin..."):
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    for i in range(0, 101, 10):
                        progress_bar.progress(i)
                        status_text.text(f"Đang tải nội dung... {i}%")
                        time.sleep(0.05)
                    st.subheader("📌 Neural Network – Mạng nơ-ron nhân tạo")
                    st.markdown("""
                    **Neural Network (Mạng nơ-ron nhân tạo)** là một mô hình học máy mô phỏng cách hoạt động của mạng nơ-ron sinh học trong não người. Nó được thiết kế để học các đặc trưng phức tạp từ dữ liệu, đặc biệt hiệu quả với bài toán nhận diện hình ảnh như MNIST.

                    **Cấu trúc cơ bản:**
                    - **Lớp đầu vào:** Nhận dữ liệu thô (784 pixel từ ảnh MNIST).
                    - **Lớp ẩn:** Xử lý thông tin qua các phép tính tuyến tính và phi tuyến.
                    - **Lớp đầu ra:** Đưa ra dự đoán (10 lớp từ 0-9).

                    **Quy trình hoạt động:**
                    1. **Khởi tạo:** Xác định cấu trúc mạng và khởi tạo trọng số/bias.
                    2. **Lan truyền thuận:** Tính toán đầu ra từ đầu vào qua các lớp.
                    3. **Tính mất mát:** Đo độ sai lệch giữa dự đoán và nhãn thực.
                    4. **Lan truyền ngược:** Cập nhật trọng số để giảm sai số.
                    5. **Lặp lại:** Tinh chỉnh qua nhiều epoch để tối ưu hóa.

                    **Ưu điểm:**
                    - Học được các mối quan hệ phi tuyến tính phức tạp.
                    - Tự động trích xuất đặc trưng từ dữ liệu thô.

                    **Nhược điểm:**
                    - Đòi hỏi nhiều dữ liệu và tài nguyên tính toán.
                    - Khó giải thích kết quả dự đoán.
                    """, unsafe_allow_html=True)
                    status_text.empty()
                    progress_bar.empty()

            elif info_option == "Pseudo-Labeling – Kỹ thuật học bán giám sát":
                with st.spinner("Đang tải thông tin..."):
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    for i in range(0, 101, 10):
                        progress_bar.progress(i)
                        status_text.text(f"Đang tải nội dung... {i}%")
                        time.sleep(0.05)
                    st.subheader("📌 Pseudo-Labeling – Kỹ thuật học bán giám sát")
                    st.markdown("""
                    **Pseudo-Labeling** là một phương pháp học bán giám sát giúp tận dụng dữ liệu không có nhãn bằng cách sử dụng mô hình đã huấn luyện để dự đoán nhãn giả (pseudo-labels), sau đó kết hợp vào quá trình huấn luyện.

                    **Các bước:**
                    1. Huấn luyện mô hình ban đầu trên tập dữ liệu có nhãn nhỏ.
                    2. Dự đoán nhãn cho dữ liệu không có nhãn.
                    3. Gán nhãn giả cho các mẫu có độ tin cậy cao (threshold).
                    4. Huấn luyện lại mô hình với tập dữ liệu mở rộng.
                    5. Lặp lại cho đến khi đạt điều kiện dừng.

                    **Lợi ích:**
                    - Tăng hiệu suất khi dữ liệu có nhãn hạn chế.
                    - Giảm chi phí gắn nhãn thủ công.

                    **Thách thức:**
                    - Nhãn giả có thể gây nhiễu nếu mô hình ban đầu yếu.
                    - Cần điều chỉnh ngưỡng tin cậy phù hợp.
                    """, unsafe_allow_html=True)
                    status_text.empty()
                    progress_bar.empty()

    #### Tab 2: Chọn số lượng dữ liệu
    with tab_load:
        st.markdown('<div class="section-title">Chọn Số lượng Dữ liệu</div>', unsafe_allow_html=True)
        X_full, y_full = st.session_state['full_data']
        st.subheader("Chọn số lượng mẫu")
        sample_options = {
            "1000 mẫu": 1000,
            "10,000 mẫu": 10000,
            "50,000 mẫu": 50000,
            "70,000 mẫu": 70000,
            "Tùy chỉnh": "custom"
        }
        selected_option = st.selectbox("Chọn số lượng mẫu:", list(sample_options.keys()))
        if selected_option == "Tùy chỉnh":
            num_samples = st.number_input("Nhập số lượng mẫu:", min_value=1, max_value=len(X_full), value=1000)
        else:
            num_samples = sample_options[selected_option]

        if st.button("Xác nhận số lượng", type="primary"):
            with st.spinner(f"Đang lấy {num_samples} mẫu..."):
                indices = np.random.choice(len(X_full), size=num_samples, replace=False)
                X_sampled = X_full[indices]
                y_sampled = y_full[indices]
                st.session_state['data'] = (X_sampled.copy(), y_sampled.copy())
                st.session_state['optimal_params'] = get_optimal_params(num_samples)
                with mlflow.start_run(experiment_id=EXPERIMENT_ID, run_name="Data_Sample"):
                    mlflow.log_param("num_samples", num_samples)
                st.success(f"Đã chọn {num_samples} mẫu!")
                del X_sampled, y_sampled
                gc.collect()

    #### Tab 3: Xử lý dữ liệu
    with tab_preprocess:
        st.markdown('<div class="section-title">Xử lý Dữ liệu</div>', unsafe_allow_html=True)
        if 'data' not in st.session_state:
            st.info("Vui lòng chọn số lượng mẫu trước.")
        else:
            X, y = st.session_state['data']
            if "data_original" not in st.session_state:
                st.session_state["data_original"] = (X.copy(), y.copy())

            st.subheader("Dữ liệu Gốc")
            fig, axes = plt.subplots(2, 5, figsize=(10, 4))
            for i, ax in enumerate(axes.flat):
                ax.imshow(X[i].reshape(28, 28), cmap='gray')
                ax.set_title(f"Label: {y[i]}")
                ax.axis("off")
            st.pyplot(fig)
            plt.close(fig)

            col1, col2 = st.columns([3, 1])
            with col1:
                if st.button("Chuẩn hóa dữ liệu (Normalization)", type="primary"):
                    with st.spinner("Đang chuẩn hóa dữ liệu về [0, 1]..."):
                        X_norm = X / 255.0
                        st.session_state["data_processed"] = (X_norm.copy(), y.copy())
                        st.success("Đã xử lý dữ liệu!")
                        del X_norm
                        gc.collect()
                        st.rerun()
            with col2:
                st.markdown("""
                    <div class="tooltip">? (Norm)
                        <span class="tooltiptext">
                            Đưa dữ liệu về $[0, 1]$ bằng cách chia cho $255$.<br>
                            Công dụng: Đảm bảo thang đo đồng nhất cho Neural Network.
                        </span>
                    </div>
                """, unsafe_allow_html=True)

            if "data_processed" in st.session_state:
                X_processed, y_processed = st.session_state["data_processed"]
                st.subheader("Dữ liệu sau khi xử lý")
                fig, axes = plt.subplots(2, 5, figsize=(10, 4))
                for i, ax in enumerate(axes.flat):
                    ax.imshow(X_processed[i].reshape(28, 28), cmap='gray')
                    ax.set_title(f"Label: {y_processed[i]}")
                    ax.axis("off")
                st.pyplot(fig)
                plt.close(fig)

    #### Tab 4: Chia dữ liệu
    with tab_split:
        st.markdown('<div class="section-title">Chia Tập Dữ liệu</div>', unsafe_allow_html=True)
        if 'data' not in st.session_state:
            st.info("Vui lòng chọn và xử lý dữ liệu trước.")
        else:
            data_source = st.session_state.get('data_processed', st.session_state['data'])
            X, y = data_source
            total_samples = len(X)
            st.write(f"Tổng số mẫu: {total_samples}")

            test_pct = st.slider("Tỷ lệ Test (%)", 0, 50, 20)
            test_size = test_pct / 100
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

            st.write(f"**Phân bổ dữ liệu**: Train: {len(X_train)}, Test: {len(X_test)}")
            if st.button("Xác nhận phân chia", type="primary"):
                with st.spinner("Đang chia dữ liệu..."):
                    st.session_state['split_data'] = {
                        "X_train": X_train.copy(), "y_train": y_train.copy(),
                        "X_test": X_test.copy(), "y_test": y_test.copy()
                    }
                    st.success("Đã chia dữ liệu thành công!")
                    del X_train, X_test, y_train, y_test
                    gc.collect()

    #### Tab 5: Huấn luyện/Đánh giá
    with tab_train_eval:
        st.markdown('<div class="section-title">Huấn luyện và Đánh giá</div>', unsafe_allow_html=True)
        if 'split_data' not in st.session_state:
            st.info("Vui lòng chia dữ liệu trước.")
        else:
            split_data = st.session_state['split_data']
            X_train, y_train = split_data["X_train"], split_data["y_train"]
            X_test, y_test = split_data["X_test"], split_data["y_test"]

            num_samples = len(X_train)
            st.write(f"**Số mẫu huấn luyện**: {num_samples}")

            if "optimal_params" not in st.session_state:
                st.session_state["optimal_params"] = get_optimal_params(num_samples)
            params = st.session_state.get("training_params", st.session_state["optimal_params"].copy())

            with st.expander("🔧 Tham số tối ưu đề xuất", expanded=False):
                optimal_table = pd.DataFrame({
                    "Số mẫu": ["≤ 1,000", "≤ 10,000", "≤ 50,000", "> 50,000"],
                    "Số lớp ẩn": [1, 2, 2, 3],
                    "Kích thước lớp ẩn": ["(32,)", "(64, 32)", "(128, 64)", "(128, 64, 32)"],
                    "Tốc độ học": [0.001, 0.0005, 0.0003, 0.0001],
                    "Số lần lặp": [30, 50, 70, 100],
                    "Kích thước batch": [32, 64, 128, 256]
                })
                st.table(optimal_table)
                if st.button("Sử dụng tham số đề xuất"):
                    st.session_state["training_params"] = st.session_state["optimal_params"].copy()
                    st.rerun()

            st.subheader("📊 Tỷ lệ mẫu ban đầu")
            labeled_pct = st.number_input("Tỷ lệ dữ liệu có nhãn ban đầu mỗi lớp (%)", min_value=0.1, max_value=100.0, value=1.0)
            num_labeled = int(num_samples * (labeled_pct / 100))
            st.write(f"**Số mẫu có nhãn ban đầu**: {num_labeled}")
            st.write(f"**Số mẫu không có nhãn**: {num_samples - num_labeled}")

            st.subheader("⚙️ Cấu hình Mô hình")
            col1, col2 = st.columns(2)
            with col1:
                num_hidden_layers = st.number_input("Số lớp ẩn", min_value=1, value=len(params["hidden_layer_sizes"]))
                hidden_sizes = []
                for i in range(num_hidden_layers):
                    default_value = params["hidden_layer_sizes"][i] if i < len(params["hidden_layer_sizes"]) else 32
                    hidden_size = st.number_input(f"Số nơ-ron lớp ẩn {i+1}", min_value=1, value=default_value)
                    hidden_sizes.append(hidden_size)
                params["hidden_layer_sizes"] = tuple(hidden_sizes)
                params["activation"] = st.selectbox("Hàm kích hoạt", ["relu", "tanh"], index=["relu", "tanh"].index(params["activation"]))

            with col2:
                params["learning_rate"] = st.number_input("Tốc độ học", min_value=0.0, step=0.0001, value=params["learning_rate"], format="%.4f")
                params["epochs"] = st.number_input("Số epoch", min_value=1, value=params["epochs"])
                params["batch_size"] = st.number_input("Kích thước batch", min_value=1, value=params["batch_size"])
                params["solver"] = st.selectbox("Trình tối ưu", ["adam", "sgd"], index=["adam", "sgd"].index(params["solver"]))

            st.subheader("🔄 Cấu hình Pseudo-Labeling")
            threshold = st.number_input("Ngưỡng tin cậy", min_value=0.0, max_value=1.0, value=params["threshold"])
            max_iterations = st.number_input("Số vòng lặp tối đa", min_value=1, value=params["max_iterations"])

            st.subheader("Đặt tên cho mô hình")
            if 'model_name' not in st.session_state:
                st.session_state['model_name'] = f"Model_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            model_name = st.text_input("Nhập tên mô hình:", value=st.session_state['model_name'])
            st.session_state['model_name'] = model_name

            if st.button("Bắt đầu Huấn luyện", type="primary"):
                if not model_name.strip():
                    st.error("Tên mô hình không được để trống!")
                else:
                    with mlflow.start_run(experiment_id=EXPERIMENT_ID, run_name=model_name.strip()) as run:
                        mlflow.log_params({**params, "labeled_pct": labeled_pct, "threshold": threshold, "max_iterations": max_iterations})

                        with st.spinner("Đang huấn luyện với Pseudo-Labeling..."):
                            start_time = time.time()
                            progress_bar = st.progress(0)
                            status_text = st.empty()
                            epoch_text = st.empty()
                            loss_text = st.empty()
                            acc_text = st.empty()

                            labeled_indices = []
                            for digit in range(10):
                                digit_indices = np.where(y_train == digit)[0]
                                train_size = max(1, int(len(digit_indices) * (labeled_pct / 100)))
                                labeled_digit = np.random.choice(digit_indices, size=min(train_size, len(digit_indices)), replace=False)
                                labeled_indices.extend(labeled_digit)
                            labeled_indices = np.array(labeled_indices)
                            unlabeled_indices = np.setdiff1d(np.arange(len(X_train)), labeled_indices)

                            X_labeled = X_train[labeled_indices]
                            y_labeled = y_train[labeled_indices]
                            X_unlabeled = X_train[unlabeled_indices]

                            loss_history = []
                            accuracy_history = []
                            iteration = 0

                            class CustomCallback(tf.keras.callbacks.Callback):
                                def on_epoch_end(self, epoch, logs=None):
                                    epoch_text.write(f"Epoch {epoch + 1}/{params['epochs']}")
                                    loss_text.write(f"Loss: {logs['loss']:.4f}")
                                    acc_text.write(f"Accuracy: {logs['accuracy']:.4f}")

                            while iteration < max_iterations and len(unlabeled_indices) > 0:
                                iteration += 1
                                status_text.write(f"Vòng {iteration}/{max_iterations}")

                                model = build_model(params)
                                model.fit(X_labeled, y_labeled, epochs=params["epochs"], batch_size=params["batch_size"], 
                                          verbose=0, callbacks=[CustomCallback()])
                                loss_history.append(model.history.history['loss'][-1])
                                accuracy_history.append(model.history.history['accuracy'][-1])

                                predictions = model.predict(X_unlabeled, verbose=0)
                                max_probs = np.max(predictions, axis=1)
                                pseudo_labels = np.argmax(predictions, axis=1)
                                high_confidence_mask = max_probs >= threshold

                                if not np.any(high_confidence_mask):
                                    break

                                X_labeled = np.vstack((X_labeled, X_unlabeled[high_confidence_mask]))
                                y_labeled = np.hstack((y_labeled, pseudo_labels[high_confidence_mask]))
                                unlabeled_indices = unlabeled_indices[~high_confidence_mask]
                                X_unlabeled = X_unlabeled[~high_confidence_mask]

                                progress_bar.progress(min(iteration / max_iterations, 1.0))

                            model = build_model(params)
                            model.fit(X_labeled, y_labeled, epochs=params["epochs"], batch_size=params["batch_size"], 
                                      verbose=0, callbacks=[CustomCallback()])
                            y_test_pred = np.argmax(model.predict(X_test, verbose=0), axis=1)
                            acc_test = accuracy_score(y_test, y_test_pred)
                            cm_test = confusion_matrix(y_test, y_test_pred)

                            mlflow.log_metric("accuracy_test", acc_test)
                            mlflow.log_metric("training_time", time.time() - start_time)
                            mlflow.keras.log_model(model, "model")

                            st.session_state['training_results'] = {
                                'accuracy_test': acc_test,
                                'cm_test': cm_test,
                                'loss_history': loss_history,
                                'accuracy_history': accuracy_history,
                                'training_time': time.time() - start_time
                            }
                            st.success(f"Đã huấn luyện xong! Thời gian: {time.time() - start_time:.2f} giây")

            if 'training_results' in st.session_state:
                results = st.session_state['training_results']
                st.subheader("📊 Kết quả Huấn luyện")
                col1, col2 = st.columns(2)
                col1.metric("Thời gian huấn luyện", f"{results['training_time']:.2f} giây")
                col2.metric("Độ chính xác Test", f"{results['accuracy_test']*100:.2f}%")

                st.subheader("Ma trận Nhầm lẫn")
                fig, ax = plt.subplots()
                sns.heatmap(results['cm_test'], annot=True, fmt="d", cmap="Blues", ax=ax)
                st.pyplot(fig)
                plt.close(fig)

                st.subheader("Biểu đồ Loss và Accuracy")
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
                ax1.plot(range(1, len(results['loss_history']) + 1), results['loss_history'], 'b-')
                ax1.set_title("Loss qua các vòng")
                ax2.plot(range(1, len(results['accuracy_history']) + 1), results['accuracy_history'], 'g-')
                ax2.set_title("Accuracy qua các vòng")
                st.pyplot(fig)
                plt.close(fig)

    #### Tab 6: Demo dự đoán
    with tab_demo:
        st.markdown('<div class="section-title">Demo Dự đoán</div>', unsafe_allow_html=True)
        if 'split_data' not in st.session_state:
            st.warning("Vui lòng chia dữ liệu trước!")
        else:
            runs = client.search_runs(experiment_ids=[EXPERIMENT_ID], order_by=["attributes.start_time DESC"])
            model_options = {run.info.run_id: run.data.tags.get('mlflow.runName', run.info.run_id) for run in runs}
            if not model_options:
                st.info("Chưa có mô hình nào được huấn luyện.")
            else:
                selected_run_id = st.selectbox("Chọn mô hình:", list(model_options.keys()), 
                                               format_func=lambda x: model_options[x])
                if st.button("Sử dụng mô hình này"):
                    with st.spinner("Đang tải mô hình..."):
                        model = mlflow.keras.load_model(f"runs:/{selected_run_id}/model")
                        st.session_state['selected_model'] = model
                        st.success("Đã tải mô hình!")

                if 'selected_model' in st.session_state:
                    model = st.session_state['selected_model']
                    input_method = st.selectbox("Phương thức nhập liệu", ["Tải ảnh lên", "Dữ liệu Test", "Vẽ trực tiếp"])

                    if input_method == "Tải ảnh lên":
                        uploaded_file = st.file_uploader("Tải lên hình ảnh", type=["png", "jpg"])
                        if uploaded_file:
                            image = Image.open(uploaded_file).convert('L').resize((28, 28))
                            st.image(image, caption="Hình ảnh tải lên", width=100)
                            image_array = np.array(image).reshape(1, 784) / 255.0
                            if st.button("Dự đoán"):
                                pred = model.predict(image_array, verbose=0)
                                st.write(f"Dự đoán: {np.argmax(pred)} (Độ tin cậy: {np.max(pred)*100:.2f}%)")

                    elif input_method == "Dữ liệu Test":
                        X_test = st.session_state['split_data']['X_test']
                        y_test = st.session_state['split_data']['y_test']
                        idx = st.slider("Chọn mẫu", 0, len(X_test)-1, 0)
                        st.image(X_test[idx].reshape(28, 28), caption=f"Nhãn thực tế: {y_test[idx]}", width=100)
                        if st.button("Dự đoán"):
                            pred = model.predict(X_test[idx:idx+1], verbose=0)
                            st.write(f"Dự đoán: {np.argmax(pred)} (Độ tin cậy: {np.max(pred)*100:.2f}%)")

                    elif input_method == "Vẽ trực tiếp":
                        if 'canvas_key' not in st.session_state:
                            st.session_state['canvas_key'] = 0
                        if st.button("Xóa Canvas"):
                            st.session_state['canvas_key'] += 1
                        canvas_result = st_canvas(
                            stroke_width=20,
                            stroke_color="#FFFFFF",
                            background_color="#000000",
                            height=280,
                            width=280,
                            drawing_mode="freedraw",
                            key=f"canvas_{st.session_state['canvas_key']}"
                        )
                        if canvas_result.image_data is not None:
                            image = Image.fromarray(canvas_result.image_data).convert('L').resize((28, 28))
                            st.image(image, caption="Hình ảnh vẽ tay", width=100)
                            image_array = np.array(image).reshape(1, 784) / 255.0
                            if st.button("Dự đoán"):
                                pred = model.predict(image_array, verbose=0)
                                st.write(f"Dự đoán: {np.argmax(pred)} (Độ tin cậy: {np.max(pred)*100:.2f}%)")

    #### Tab 7: Thông tin huấn luyện
    with tab_log_info:
        st.markdown('<div class="section-title">Theo dõi Kết quả</div>', unsafe_allow_html=True)
        with st.spinner("Đang tải thông tin huấn luyện..."):
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
                st.markdown("**Tham số:**")
                st.json(selected_run.data.params)
                st.markdown("**Số liệu:**")
                st.json(selected_run.data.metrics)

                st.markdown(f"📊 **Xem chi tiết trên MLflow UI**: [{mlflow_tracking_uri}/#/experiments/{EXPERIMENT_ID}]({mlflow_tracking_uri}/#/experiments/{EXPERIMENT_ID})", unsafe_allow_html=True)

if __name__ == "__main__":
    run_mnist_pseudo_labeling_app()