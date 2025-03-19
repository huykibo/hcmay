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
            "threshold": 0.95,  # Pseudo-Labeling
            "max_iterations": 5  # Pseudo-Labeling
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

    # Tab 1: Thông tin
    with tab_info:
        # Tạo container riêng để chứa nội dung của tab này
        info_container = st.container()
        with info_container:
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

            # Tạo placeholder để chứa nội dung động
            content_placeholder = st.empty()

            # Hiển thị nội dung mới dựa trên lựa chọn
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
                        time.sleep(0.5)
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
                        except Exception as e:
                            st.error(f"Lỗi khi tải ảnh: {e}")
                        status_text.text("Đã tải xong! 100%")
                        time.sleep(0.5)
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
                        - **Lớp đầu vào (Input Layer)**: Nhận dữ liệu thô (ví dụ: 784 pixel từ ảnh MNIST).
                        - **Lớp ẩn (Hidden Layers)**: Xử lý thông tin thông qua các phép tính tuyến tính và phi tuyến.
                        - **Lớp đầu ra (Output Layer)**: Đưa ra dự đoán (10 lớp, từ 0-9).

                        **Ưu điểm:**
                        - Có khả năng học các mối quan hệ phi tuyến tính phức tạp.
                        - Tự động trích xuất đặc trưng mà không cần xử lý thủ công.

                        **Nhược điểm:**
                        - Yêu cầu nhiều dữ liệu và tài nguyên tính toán.
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
                        **Pseudo-Labeling** là một kỹ thuật học bán giám sát nhằm tận dụng dữ liệu không có nhãn để cải thiện hiệu suất mô hình khi dữ liệu có nhãn khan hiếm.

                        **Quy trình:**
                        1. Huấn luyện mô hình ban đầu trên tập dữ liệu có nhãn nhỏ.
                        2. Dự đoán nhãn cho dữ liệu không có nhãn.
                        3. Gán nhãn giả (pseudo-labels) cho các mẫu có độ tin cậy cao.
                        4. Kết hợp dữ liệu có nhãn và nhãn giả để huấn luyện lại mô hình.
                        5. Lặp lại cho đến khi đạt điều kiện dừng.

                        **Lợi ích:**
                        - Tăng cường hiệu suất mà không cần thêm dữ liệu có nhãn.
                        - Phù hợp với thực tế khi việc gắn nhãn thủ công tốn kém.

                        **Thách thức:**
                        - Nhãn giả có thể chứa nhiễu nếu mô hình ban đầu không đủ tốt.
                        - Cần điều chỉnh ngưỡng tin cậy để cân bằng giữa số lượng và chất lượng nhãn giả.
                        """, unsafe_allow_html=True)
                        status_text.empty()
                        progress_bar.empty()

    # Tab 2: Chọn số lượng dữ liệu
    with tab_load:
        load_container = st.container()
        with load_container:
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
                    del X_full, y_full, X_sampled, y_sampled
                    gc.collect()

    # Tab 3: Xử lý dữ liệu
    with tab_preprocess:
        preprocess_container = st.container()
        with preprocess_container:
            st.markdown('<div class="section-title">Xử lý Dữ liệu</div>', unsafe_allow_html=True)
            if 'data' not in st.session_state:
                st.info("Vui lòng chọn số lượng mẫu trước.")
            else:
                X, y = st.session_state['data']
                if "data_original" not in st.session_state:
                    st.session_state["data_original"] = (X.copy(), y.copy())

                st.subheader("Dữ liệu Gốc")
                fig, axes = plt.subplots(2, 5, figsize=(12, 5))
                for i, ax in enumerate(axes.flat):
                    ax.imshow(X[i].reshape(28, 28), cmap='gray')
                    ax.set_title(f"Label: {y[i]}")
                    ax.axis("off")
                plt.tight_layout()
                st.pyplot(fig)
                plt.close(fig)

                col1, col2 = st.columns([3, 1])
                with col1:
                    if st.button("Chuẩn hóa dữ liệu (Normalization)", type="primary", help="Chuẩn hóa dữ liệu về thang [0, 1]"):
                        with st.spinner("Đang chuẩn hóa dữ liệu về [0, 1]..."):
                            X_norm = X / 255.0
                            st.session_state["data_processed"] = (X_norm.copy(), y.copy())
                            st.success("Đã xử lý dữ liệu!")
                            del X, y, X_norm
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
                    fig, axes = plt.subplots(2, 5, figsize=(12, 5))
                    for i, ax in enumerate(axes.flat):
                        ax.imshow(X_processed[i].reshape(28, 28), cmap='gray')
                        ax.set_title(f"Label: {y_processed[i]}")
                        ax.axis("off")
                    plt.tight_layout()
                    st.pyplot(fig)
                    plt.close(fig)

    # Tab 4: Chia dữ liệu
    with tab_split:
        split_container = st.container()
        with split_container:
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
                        del X, y, X_train, X_test, y_train, y_test
                        gc.collect()

    # Tab 5: Huấn luyện/Đánh giá
    with tab_train_eval:
        train_eval_container = st.container()
        with train_eval_container:
            st.markdown('<div class="section-title">Huấn luyện và Đánh giá</div>', unsafe_allow_html=True)
            if 'split_data' not in st.session_state:
                st.info("Vui lòng chia dữ liệu trước.")
            else:
                split_data = st.session_state['split_data'].copy()
                X_train = split_data["X_train"]
                y_train = split_data["y_train"]
                X_test = split_data["X_test"]
                y_test = split_data["y_test"]

                X_train = np.array(X_train, dtype=np.float32)
                y_train = np.array(y_train, dtype=np.int32)
                X_test = np.array(X_test, dtype=np.float32)
                y_test = np.array(y_test, dtype=np.int32)

                num_samples = len(X_train)
                st.write(f"**Số mẫu huấn luyện**: {num_samples}")

                # Tự động chọn tham số tối ưu
                if "optimal_params" not in st.session_state:
                    st.session_state["optimal_params"] = get_optimal_params(num_samples)
                params = st.session_state.get("training_params", st.session_state["optimal_params"].copy())

                # Thêm bảng tham số tối ưu (ẩn ban đầu, dùng expander)
                with st.expander("🔧 Tham số tối ưu đề xuất", expanded=False):
                    optimal_table = pd.DataFrame({
                        "Số mẫu": ["≤ 1,000", "≤ 10,000", "≤ 50,000", "> 50,000"],
                        "Số lớp ẩn": [1, 2, 2, 3],
                        "Kích thước lớp ẩn": ["(32,)", "(64, 32)", "(128, 64)", "(128, 64, 32)"],
                        "Tốc độ học": [0.001, 0.0005, 0.0003, 0.0001],
                        "Số lần lặp": [30, 50, 70, 100],
                        "Hàm kích hoạt": ["ReLU", "ReLU", "ReLU", "ReLU"],
                        "Trình tối ưu": ["Adam", "Adam", "Adam", "Adam"],
                        "Kích thước batch": [32, 64, 128, 256],
                        "Ngưỡng tin cậy": [0.95, 0.95, 0.95, 0.95],
                        "Số vòng lặp tối đa": [5, 10, 15, 20]
                    })
                    st.table(optimal_table)
                    if st.button("Sử dụng tham số đề xuất"):
                        st.session_state["training_params"] = st.session_state["optimal_params"].copy()
                        st.rerun()

                # Hiển thị tỷ lệ mẫu ban đầu và số lượng mẫu
                st.subheader("📊 Tỷ lệ mẫu ban đầu")
                labeled_pct = st.number_input("Tỷ lệ dữ liệu có nhãn ban đầu mỗi lớp (%)", min_value=0.1, max_value=100.0, value=1.0)
                num_labeled = int(num_samples * (labeled_pct / 100))
                num_unlabeled = num_samples - num_labeled
                st.write(f"**Số mẫu có nhãn ban đầu**: {num_labeled}")
                st.write(f"**Số mẫu không có nhãn**: {num_unlabeled}")

                # Cấu hình mô hình
                st.subheader("⚙️ Cấu hình Mô hình")
                col_param1, col_param2 = st.columns(2)
                with col_param1:
                    num_hidden_layers = st.number_input("Số lớp ẩn", min_value=1, value=len(params["hidden_layer_sizes"]))
                    hidden_sizes = []
                    for i in range(num_hidden_layers):
                        default_value = params["hidden_layer_sizes"][i] if i < len(params["hidden_layer_sizes"]) else 32
                        hidden_size = st.number_input(f"Số nơ-ron lớp ẩn {i+1}", min_value=1, value=default_value)
                        hidden_sizes.append(hidden_size)
                    params["hidden_layer_sizes"] = tuple(hidden_sizes)
                    params["activation"] = st.selectbox("Hàm kích hoạt", ["relu", "tanh", "softmax"], 
                                                        index=["relu", "tanh", "softmax"].index(params["activation"]))

                with col_param2:
                    params["learning_rate"] = st.number_input("Tốc độ học", min_value=0.0, step=0.0001, 
                                                              value=params["learning_rate"], format="%.4f")
                    params["epochs"] = st.number_input("Số epoch", min_value=1, value=params["epochs"])
                    params["batch_size"] = st.number_input("Kích thước batch", min_value=1, value=params["batch_size"])
                    params["solver"] = st.selectbox("Trình tối ưu", ["adam", "sgd"], 
                                                    index=["adam", "sgd"].index(params["solver"]))

                # Cấu hình Pseudo-Labeling
                st.subheader("🔄 Cấu hình Pseudo-Labeling")
                threshold = st.number_input("Ngưỡng tin cậy", min_value=0.0, max_value=1.0, value=params["threshold"])
                max_iterations = st.number_input("Số vòng lặp tối đa", min_value=1, value=params["max_iterations"])

                # Đặt tên cho mô hình
                st.subheader("Đặt tên cho mô hình")
                if 'model_name' not in st.session_state:
                    st.session_state['model_name'] = f"Model_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                model_name = st.text_input("Nhập tên mô hình:", value=st.session_state['model_name'])
                st.session_state['model_name'] = model_name

                if st.button("Bắt đầu Huấn luyện", type="primary"):
                    # Kiểm tra tên mô hình không trống
                    if not model_name.strip():
                        st.error("Tên mô hình không được để trống! Vui lòng nhập tên mô hình.")
                    else:
                        with mlflow.start_run(experiment_id=EXPERIMENT_ID, run_name=model_name.strip()) as run:
                            mlflow.log_params({**params, "labeled_pct": labeled_pct, "threshold": threshold, "max_iterations": max_iterations})
                            run_id = run.info.run_id

                            with st.spinner("Đang huấn luyện với Pseudo-Labeling..."):
                                start_time = time.time()
                                progress_bar = st.progress(0)
                                status_text = st.empty()
                                epoch_text = st.empty()
                                loss_text = st.empty()
                                acc_text = st.empty()

                                # Tạo tập dữ liệu có nhãn ban đầu (dựa trên labeled_pct mỗi lớp)
                                labeled_indices = []
                                for digit in range(10):
                                    digit_indices = np.where(y_train == digit)[0]
                                    if len(digit_indices) > 0:
                                        train_size = min(int(len(digit_indices) * (labeled_pct / 100)), len(digit_indices))
                                        if train_size < 1 and len(digit_indices) > 0:
                                            train_size = 1
                                        if train_size > 0:
                                            labeled_digit, _ = train_test_split(digit_indices, train_size=train_size, random_state=42)
                                            labeled_indices.extend(labeled_digit)
                                labeled_indices = np.array(labeled_indices)
                                unlabeled_indices = np.setdiff1d(np.arange(len(X_train)), labeled_indices)

                                X_labeled = X_train[labeled_indices]
                                y_labeled = y_train[labeled_indices]
                                X_unlabeled = X_train[unlabeled_indices]

                                loss_history = []
                                accuracy_history = []
                                pseudo_samples_history = []  # Lưu trữ mẫu Pseudo-Labeling
                                iteration = 0

                                # Callback để cập nhật thông tin trong quá trình huấn luyện
                                class CustomCallback(tf.keras.callbacks.Callback):
                                    def __init__(self, iteration, max_iterations):
                                        super().__init__()
                                        self.iteration = iteration
                                        self.max_iterations = max_iterations

                                    def on_epoch_end(self, epoch, logs=None):
                                        epoch_text.write(f"Epoch {epoch + 1}/{params['epochs']}")
                                        loss_text.write(f"Loss: {logs['loss']:.4f}")
                                        acc_text.write(f"Accuracy: {logs['accuracy']:.4f}")

                                # Quá trình huấn luyện với Pseudo-Labeling
                                while iteration < max_iterations and len(unlabeled_indices) > 0:
                                    iteration += 1
                                    status_text.write(f"Vòng {iteration}/{max_iterations}")

                                    # Huấn luyện mô hình trên tập dữ liệu có nhãn hiện tại
                                    model = build_model(params)
                                    history = model.fit(
                                        X_labeled, y_labeled,
                                        epochs=params["epochs"],
                                        batch_size=params["batch_size"],
                                        verbose=0,
                                        callbacks=[CustomCallback(iteration, max_iterations)]
                                    )
                                    loss_history.append(history.history['loss'][-1])
                                    accuracy_history.append(history.history['accuracy'][-1])

                                    # Dự đoán nhãn cho tập dữ liệu không có nhãn
                                    predictions = model.predict(X_unlabeled, verbose=0)
                                    max_probs = np.max(predictions, axis=1)
                                    pseudo_labels = np.argmax(predictions, axis=1)

                                    # Lọc các mẫu có độ tin cậy cao
                                    high_confidence_mask = max_probs >= threshold
                                    if not np.any(high_confidence_mask):
                                        break

                                    # Gán nhãn giả và thêm vào tập dữ liệu có nhãn
                                    pseudo_indices = unlabeled_indices[high_confidence_mask]
                                    pseudo_y = pseudo_labels[high_confidence_mask]
                                    pseudo_probs = max_probs[high_confidence_mask]
                                    X_pseudo = X_unlabeled[high_confidence_mask]

                                    # Lưu trữ một số mẫu Pseudo-Labeling để minh họa (tối đa 5 mẫu mỗi vòng)
                                    num_samples_to_show = min(5, len(X_pseudo))
                                    pseudo_samples_history.append({
                                        'iteration': iteration,
                                        'X_pseudo': X_pseudo[:num_samples_to_show],
                                        'pseudo_labels': pseudo_y[:num_samples_to_show],
                                        'confidence': pseudo_probs[:num_samples_to_show]
                                    })

                                    X_labeled = np.vstack((X_labeled, X_unlabeled[high_confidence_mask]))
                                    y_labeled = np.hstack((y_labeled, pseudo_y))

                                    # Cập nhật tập dữ liệu không có nhãn
                                    unlabeled_indices = unlabeled_indices[~high_confidence_mask]
                                    X_unlabeled = X_unlabeled[~high_confidence_mask]

                                    # Cập nhật progress bar
                                    progress = iteration / max_iterations
                                    progress_bar.progress(min(progress, 1.0))

                                # Huấn luyện lần cuối trên toàn bộ dữ liệu đã gắn nhãn
                                model = build_model(params)
                                history = model.fit(
                                    X_labeled, y_labeled,
                                    epochs=params["epochs"],
                                    batch_size=params["batch_size"],
                                    verbose=0,
                                    callbacks=[CustomCallback(iteration, max_iterations)]
                                )
                                loss_history.append(history.history['loss'][-1])
                                accuracy_history.append(history.history['accuracy'][-1])

                                # Đánh giá trên tập test
                                y_test_pred = np.argmax(model.predict(X_test, verbose=0), axis=1)
                                acc_test = accuracy_score(y_test, y_test_pred)
                                cm_test = confusion_matrix(y_test, y_test_pred)

                                # Lưu kết quả vào MLflow
                                mlflow.log_metric("accuracy_test", acc_test)
                                mlflow.log_metric("training_time", time.time() - start_time)
                                mlflow.keras.log_model(model, "model")

                                # Lưu kết quả vào session_state
                                results = {
                                    'accuracy_test': acc_test,
                                    'cm_test': cm_test,
                                    'loss_history': loss_history,
                                    'accuracy_history': accuracy_history,
                                    'pseudo_samples_history': pseudo_samples_history,
                                    'iterations': iteration,
                                    'training_time': time.time() - start_time,
                                    'run_id': run.info.run_id,
                                    'run_name': model_name.strip(),
                                    'params': params,
                                    'n_iter_actual': iteration
                                }
                                st.session_state['training_results'] = results
                                st.success(f"Đã huấn luyện xong sau {iteration} vòng! Thời gian: {results['training_time']:.2f} giây")

                # Hiển thị kết quả
                if 'training_results' in st.session_state:
                    results = st.session_state['training_results']
                    st.subheader("📊 Kết quả Huấn luyện")

                    # Thông tin cơ bản
                    col1, col2 = st.columns(2)
                    col1.metric("Thời gian huấn luyện", f"{results['training_time']:.2f} giây")
                    col2.metric("Độ chính xác Test", f"{results['accuracy_test']*100:.2f}%")

                    # Tham số huấn luyện
                    st.subheader("⚙️ Tham số Huấn luyện")
                    st.json({
                        "Số lớp ẩn": len(results['params']['hidden_layer_sizes']),
                        "Số nơ-ron mỗi lớp": results['params']['hidden_layer_sizes'],
                        "Tốc độ học": results['params']['learning_rate'],
                        "Số lần lặp": results['params']['epochs'],
                        "Kích thước batch": results['params']['batch_size'],
                        "Hàm kích hoạt": results['params']['activation'],
                        "Trình tối ưu": results['params']['solver'],
                        "Ngưỡng tin cậy": threshold,
                        "Số vòng lặp tối đa": max_iterations
                    })

                    # Biểu đồ Loss và Accuracy
                    st.subheader("📈 Biểu đồ Loss và Accuracy")
                    col_loss, col_acc = st.columns(2)
                    with col_loss:
                        fig, ax = plt.subplots(figsize=(6, 4))
                        ax.plot(range(1, len(results['loss_history']) + 1), results['loss_history'], 
                                color='blue', label='Loss', linewidth=2)
                        ax.set_title("Loss qua các vòng", fontsize=12)
                        ax.set_xlabel("Vòng", fontsize=10)
                        ax.set_ylabel("Loss", fontsize=10)
                        ax.grid(True, linestyle='--', alpha=0.7)
                        ax.legend()
                        plt.tight_layout()
                        st.pyplot(fig)
                        plt.close(fig)
                    with col_acc:
                        fig, ax = plt.subplots(figsize=(6, 4))
                        ax.plot(range(1, len(results['accuracy_history']) + 1), results['accuracy_history'], 
                                color='green', label='Accuracy', linewidth=2)
                        ax.set_title("Accuracy qua các vòng", fontsize=12)
                        ax.set_xlabel("Vòng", fontsize=10)
                        ax.set_ylabel("Accuracy", fontsize=10)
                        ax.grid(True, linestyle='--', alpha=0.7)
                        ax.legend()
                        plt.tight_layout()
                        st.pyplot(fig)
                        plt.close(fig)

                    # Ma trận Nhầm lẫn
                    st.subheader("🔢 Ma trận Nhầm lẫn")
                    fig, ax = plt.subplots(figsize=(8, 6))
                    sns.heatmap(results['cm_test'], annot=True, fmt="d", cmap="Blues", ax=ax)
                    ax.set_title("Ma trận Nhầm lẫn trên tập Test", fontsize=14)
                    ax.set_xlabel("Dự đoán", fontsize=12)
                    ax.set_ylabel("Thực tế", fontsize=12)
                    plt.tight_layout()
                    st.pyplot(fig)
                    plt.close(fig)

                    # Minh họa các mẫu Pseudo-Labeling
                    st.subheader("🔄 Minh họa Pseudo-Labeling")
                    if results['pseudo_samples_history']:
                        for pseudo_data in results['pseudo_samples_history']:
                            st.markdown(f"**Vòng {pseudo_data['iteration']}**")
                            fig, axes = plt.subplots(1, len(pseudo_data['X_pseudo']), figsize=(len(pseudo_data['X_pseudo']) * 2, 2.5))
                            if len(pseudo_data['X_pseudo']) == 1:
                                axes = [axes]  # Đảm bảo axes là danh sách ngay cả khi chỉ có 1 mẫu
                            for i, ax in enumerate(axes):
                                ax.imshow(pseudo_data['X_pseudo'][i].reshape(28, 28), cmap='gray')
                                ax.set_title(f"Label: {pseudo_data['pseudo_labels'][i]}\nConf: {pseudo_data['confidence'][i]:.2f}", fontsize=10)
                                ax.axis("off")
                            plt.tight_layout()
                            st.pyplot(fig)
                            plt.close(fig)
                    else:
                        st.info("Không có mẫu Pseudo-Labeling nào được gán.")

                    # Tóm tắt kết quả huấn luyện
                    st.subheader("📋 Tóm tắt Kết quả")
                    full_data = {
                        "Vòng": list(range(1, len(results['loss_history']) + 1)),
                        "Loss": results['loss_history'],
                        "Accuracy": results['accuracy_history'],
                    }
                    df_full = pd.DataFrame(full_data)
                    st.table(df_full)

                    # Chi tiết lần chạy
                    with st.expander("Xem chi tiết lần chạy", expanded=False):
                        st.markdown("**Thông tin lần chạy:**")
                        st.write(f"- Tên: {results['run_name']}")
                        st.write(f"- ID: {results['run_id']}")
                        st.write(f"- Thời gian huấn luyện: {results['training_time']:.2f} giây")
                        st.write(f"- Số lần lặp thực tế: {results['n_iter_actual']}")

    # Tab 6: Demo dự đoán
    with tab_demo:
        demo_container = st.container()
        with demo_container:
            st.markdown('<div class="section-title">Demo Dự đoán</div>', unsafe_allow_html=True)
            if 'split_data' not in st.session_state:
                st.warning("Vui lòng chia dữ liệu trước!")
            else:
                if st.button("Làm mới danh sách mô hình"):
                    st.rerun()

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
                            canvas_result = st_canvas(stroke_width=20, stroke_color="#FFFFFF", background_color="#000000", 
                                                      height=280, width=280, drawing_mode="freedraw")
                            if canvas_result.image_data is not None:
                                image = Image.fromarray(canvas_result.image_data).convert('L').resize((28, 28))
                                st.image(image, caption="Hình ảnh vẽ tay", width=100)
                                image_array = np.array(image).reshape(1, 784) / 255.0
                                if st.button("Dự đoán"):
                                    pred = model.predict(image_array, verbose=0)
                                    st.write(f"Dự đoán: {np.argmax(pred)} (Độ tin cậy: {np.max(pred)*100:.2f}%)")

    # Tab 7: Thông tin huấn luyện
    with tab_log_info:
        log_info_container = st.container()
        with log_info_container:
            st.markdown('<div class="section-title">Theo dõi Kết quả</div>', unsafe_allow_html=True)
            try:
                with st.spinner("Đang tải thông tin huấn luyện..."):
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
                        
                        st.markdown("**Tham số huấn luyện:**")
                        st.json(selected_run.data.params, expanded=True)
                        
                        st.markdown("**Số liệu huấn luyện:**")
                        st.json(selected_run.data.metrics, expanded=True)

                        st.subheader("📈 Lịch sử Huấn luyện")
                        col_loss, col_acc = st.columns(2)
                        with col_loss:
                            if 'training_results' in st.session_state and selected_run_id == st.session_state['training_results']['run_id']:
                                results = st.session_state['training_results']
                                if 'loss_history' in results:
                                    fig, ax = plt.subplots(figsize=(6, 4))
                                    ax.plot(range(1, len(results['loss_history']) + 1), results['loss_history'], 
                                            label='Training Loss', color='blue', linewidth=2)
                                    ax.set_xlabel("Vòng", fontsize=10)
                                    ax.set_ylabel("Loss", fontsize=10)
                                    ax.set_title("Lịch sử Mất mát", fontsize=12)
                                    ax.legend()
                                    ax.grid(True, linestyle='--', alpha=0.7)
                                    plt.tight_layout()
                                    st.pyplot(fig)
                                    plt.close(fig)
                        with col_acc:
                            if 'training_results' in st.session_state and selected_run_id == st.session_state['training_results']['run_id']:
                                results = st.session_state['training_results']
                                if 'accuracy_history' in results:
                                    fig, ax = plt.subplots(figsize=(6, 4))
                                    ax.plot(range(1, len(results['accuracy_history']) + 1), results['accuracy_history'], 
                                            label='Training Accuracy', color='green', linewidth=2)
                                    ax.set_xlabel("Vòng", fontsize=10)
                                    ax.set_ylabel("Accuracy", fontsize=10)
                                    ax.set_title("Lịch sử Độ chính xác", fontsize=12)
                                    ax.legend()
                                    ax.grid(True, linestyle='--', alpha=0.7)
                                    plt.tight_layout()
                                    st.pyplot(fig)
                                    plt.close(fig)

                        st.subheader("So sánh các Run")
                        selected_runs = st.multiselect("Chọn các run để so sánh:", list(run_options.values()), default=[selected_run_name])
                        if selected_runs:
                            selected_run_ids = [k for k, v in run_options.items() if v in selected_runs]
                            comparison_data = []
                            for run_id in selected_run_ids:
                                run = client.get_run(run_id)
                                run_data = {
                                    "Tên": run.data.tags.get('mlflow.runName', run_id),
                                    "Accuracy Test": run.data.metrics.get('accuracy_test', 'N/A'),
                                    "Thời gian": run.data.metrics.get('training_time', 'N/A'),
                                    "Số lớp ẩn": run.data.params.get('hidden_layer_sizes', 'N/A'),
                                    "Learning Rate": run.data.params.get('learning_rate', 'N/A'),
                                    "Epochs": run.data.params.get('epochs', 'N/A')
                                }
                                comparison_data.append(run_data)
                            st.table(pd.DataFrame(comparison_data))

            except Exception as e:
                st.error(f"Lỗi khi tải thông tin huấn luyện: {e}. Vui lòng kiểm tra kết nối MLflow hoặc thông tin Experiment ID.")
            mlflow_ui_link = f"{mlflow_tracking_uri}/#/experiments/{EXPERIMENT_ID}"
            st.markdown("---")
            st.markdown(f"📊 **Xem chi tiết trên MLflow UI**: [Nhấn vào đây]({mlflow_ui_link})", unsafe_allow_html=True)

if __name__ == "__main__":
    run_mnist_pseudo_labeling_app()