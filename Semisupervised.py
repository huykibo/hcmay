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
    if num_samples <= 1000:
        return {
            "hidden_layer_sizes": (32,),
            "learning_rate": 0.001,
            "epochs": 30,
            "activation": "relu",
            "solver": "adam",
            "batch_size": 32,
            "threshold": 0.9,
            "max_iterations": 3
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
            "max_iterations": 5
        }
    elif num_samples <= 50000:
        return {
            "hidden_layer_sizes": (128, 64),
            "learning_rate": 0.0003,
            "epochs": 70,
            "activation": "relu",
            "solver": "adam",
            "batch_size": 128,
            "threshold": 0.97,
            "max_iterations": 7
        }
    else:
        return {
            "hidden_layer_sizes": (128, 64, 32),
            "learning_rate": 0.0001,
            "epochs": 100,
            "activation": "relu",
            "solver": "adam",
            "batch_size": 256,
            "threshold": 0.98,
            "max_iterations": 10
        }

# Hàm chính chạy ứng dụng
def run_mnist_pseudo_labeling_app():
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

    # Tải dữ liệu MNIST mặc định (giả lập nếu không có file cục bộ)
    if 'full_data' not in st.session_state:
        (X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()
        X_full = np.concatenate([X_train, X_test], axis=0).reshape(-1, 784)
        y_full = np.concatenate([y_train, y_test], axis=0)
        st.session_state['full_data'] = (X_full, y_full)

    st.title("Phân loại Chữ số MNIST với Neural Network và Pseudo-Labeling")

    # CSS tùy chỉnh
    st.markdown("""
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
            .section-title {
                font-size: 1.5em;
                font-weight: bold;
                color: #2c3e50;
                margin-bottom: 10px;
            }
            .info-box {
                background-color: #f8f9fa;
                padding: 10px;
                border-left: 4px solid #3498db;
                margin-bottom: 15px;
            }
            .prediction-box {
                margin-top: 10px;
                padding: 10px;
                border: 1px solid #ccc;
                border-radius: 5px;
                background-color: #f9f9f9;
            }
            .mode-title {
                font-size: 1.2em;
                font-weight: bold;
                color: #2c3e50;
                margin-bottom: 10px;
            }
            .stCanvas {
                border: 1px solid #ddd;
                border-radius: 5px;
            }
        </style>
    """, unsafe_allow_html=True)

    # Tạo các tab
    tabs = st.tabs(["Thông tin", "Tải dữ liệu", "Xử lý dữ liệu", "Chia dữ liệu", "Huấn luyện/Đánh giá", "Demo dự đoán", "Thông tin huấn luyện"])
    tab_info, tab_load, tab_preprocess, tab_split, tab_train_eval, tab_demo, tab_log_info = tabs

    ### Tab 1: Thông tin
    with tab_info:
        st.header("Giới thiệu Ứng dụng")
        st.markdown("""
        Ứng dụng này phân loại chữ số viết tay từ tập dữ liệu **MNIST** bằng **Neural Network** và **Pseudo-Labeling**, cung cấp trải nghiệm trực quan cho học tập và nghiên cứu học máy.
        """)
        st.subheader("Chọn nội dung")
        info_option = st.selectbox("", ["Tổng quan", "MNIST", "Neural Network", "Pseudo-Labeling"], label_visibility="collapsed")
        if info_option == "Tổng quan":
            st.markdown("""
            **Mục tiêu:**
            - Nhận diện chữ số 0-9 bằng Neural Network.
            - Sử dụng Pseudo-Labeling để khai thác dữ liệu không nhãn.
            - Giao diện trực quan để thực hành và đánh giá.
            **Dữ liệu:** 70,000 ảnh 28x28 pixel, 784 đặc trưng, giá trị 0-255.
            """)
        elif info_option == "MNIST":
            st.markdown("""
            **MNIST:** Bộ dữ liệu chuẩn với 70,000 ảnh (60,000 train, 10,000 test), 28x28 pixel, thang độ xám.
            **Ý nghĩa:** Nền tảng thử nghiệm học máy, đánh giá khả năng phân biệt.
            """)
        elif info_option == "Neural Network":
            st.markdown("""
            **Neural Network:** Mô hình học máy mô phỏng mạng nơ-ron sinh học, gồm lớp đầu vào (784 pixel), lớp ẩn, lớp đầu ra (10 chữ số).
            **Quy trình:** Khởi tạo, lan truyền thuận, tính mất mát, lan truyền ngược, cập nhật tham số, lặp lại.
            """)
        elif info_option == "Pseudo-Labeling":
            st.markdown("""
            **Pseudo-Labeling:** Học bán giám sát, dự đoán nhãn giả cho dữ liệu không nhãn, kết hợp huấn luyện.
            **Các bước:** Chuẩn bị dữ liệu, lấy 1% nhãn, huấn luyện, dự đoán, gán nhãn giả (ngưỡng 0.95), huấn luyện lại, lặp lại.
            """)

    ### Tab 2: Tải dữ liệu
    with tab_load:
        st.markdown('<div class="section-title">Chọn Số lượng Dữ liệu</div>', unsafe_allow_html=True)
        X_full, y_full = st.session_state['full_data']
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
                st.session_state['data'] = (X_full[indices].copy(), y_full[indices].copy())
                st.session_state['optimal_params'] = get_optimal_params(num_samples)
                with mlflow.start_run(experiment_id=EXPERIMENT_ID, run_name="Data_Sample"):
                    mlflow.log_param("num_samples", num_samples)
                st.success(f"Đã chọn {num_samples} mẫu!")
                gc.collect()

    ### Tab 3: Xử lý dữ liệu
    with tab_preprocess:
        st.markdown('<div class="section-title">Xử lý Dữ liệu</div>', unsafe_allow_html=True)
        if 'data' not in st.session_state:
            st.info("Vui lòng chọn số lượng mẫu trước.")
        else:
            X, y = st.session_state['data']
            st.session_state["data_original"] = (X.copy(), y.copy())
            st.subheader("Dữ liệu Gốc")
            fig, axes = plt.subplots(2, 5, figsize=(10, 4))
            for i, ax in enumerate(axes.flat):
                ax.imshow(X[i].reshape(28, 28), cmap='gray')
                ax.set_title(f"Label: {y[i]}")
                ax.axis("off")
            st.pyplot(fig)
            plt.close(fig)

            if st.button("Chuẩn hóa dữ liệu", type="primary"):
                with st.spinner("Đang chuẩn hóa dữ liệu..."):
                    X_norm = X / 255.0
                    st.session_state["data_processed"] = (X_norm.copy(), y.copy())
                    st.success("Đã xử lý dữ liệu!")
                    gc.collect()
                    st.rerun()

            if "data_processed" in st.session_state:
                st.success("Dữ liệu đã được chuẩn hóa!")

    ### Tab 4: Chia dữ liệu
    with tab_split:
        st.markdown('<div class="section-title">Chia Tập Dữ liệu</div>', unsafe_allow_html=True)
        if 'data' not in st.session_state:
            st.info("Vui lòng chọn và xử lý dữ liệu trước.")
        else:
            data_source = st.session_state.get('data_processed', st.session_state['data'])
            X, y = data_source
            test_pct = st.slider("Tỷ lệ Test (%)", 0, 50, 20)
            valid_pct = st.slider("Tỷ lệ Validation (%)", 0, 50, 20)
            test_size = test_pct / 100
            X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
            valid_size = (valid_pct / 100) / (1 - test_size) if test_size < 1 else 0
            X_train, X_valid, y_train, y_valid = train_test_split(X_temp, y_temp, test_size=valid_size, random_state=42)

            st.write(f"Train: {len(X_train)}, Validation: {len(X_valid)}, Test: {len(X_test)}")
            if st.button("Xác nhận phân chia", type="primary"):
                with st.spinner("Đang chia dữ liệu..."):
                    st.session_state['split_data'] = {
                        "X_train": X_train.copy(), "y_train": y_train.copy(),
                        "X_valid": X_valid.copy(), "y_valid": y_valid.copy(),
                        "X_test": X_test.copy(), "y_test": y_test.copy()
                    }
                    st.success("Đã chia dữ liệu thành công!")
                    gc.collect()

     # Tab 5: Huấn luyện/Đánh giá
    with tab_train_eval:
        st.markdown('<div class="section-title">Huấn luyện và Đánh giá Mô hình với Pseudo-Labeling</div>', unsafe_allow_html=True)

        if 'split_data' not in st.session_state:
            st.info("Vui lòng chia dữ liệu trước.")
        else:
            split_data = st.session_state['split_data'].copy()
            X_train_full = split_data["X_train"]
            y_train_full = split_data["y_train"]
            X_test = split_data["X_test"]
            y_test = split_data["y_test"]

            X_train_full = np.array(X_train_full, dtype=np.float32)
            y_train_full = np.array(y_train_full, dtype=np.int32)
            X_test = np.array(X_test, dtype=np.float32)
            y_test = np.array(y_test, dtype=np.int32)

            if np.any(np.isnan(X_train_full)) or np.any(np.isnan(y_train_full)):
                st.error("Dữ liệu huấn luyện chứa giá trị NaN. Đang xử lý...")
                X_train_full = np.nan_to_num(X_train_full, nan=0.0)
                y_train_full = np.nan_to_num(y_train_full, nan=0.0)
                st.success("Đã thay thế NaN bằng 0 trong dữ liệu huấn luyện!")

            num_samples = len(X_train_full)
            st.write(f"**Tổng số mẫu huấn luyện ban đầu**: {num_samples}")

            # Bước 1: Lấy 1% số lượng ảnh cho mỗi class (0-9) để làm tập train ban đầu
            st.subheader("Tạo tập dữ liệu ban đầu (1% mỗi lớp)")
            classes = np.unique(y_train_full)
            X_train_initial = []
            y_train_initial = []
            X_unlabeled = []
            y_unlabeled_indices = []

            for cls in classes:
                cls_indices = np.where(y_train_full == cls)[0]
                num_cls_samples = len(cls_indices)
                num_initial = max(1, int(0.01 * num_cls_samples))  # Lấy 1% mỗi lớp
                initial_indices = np.random.choice(cls_indices, num_initial, replace=False)
                unlabeled_indices = np.setdiff1d(cls_indices, initial_indices)

                X_train_initial.append(X_train_full[initial_indices])
                y_train_initial.append(y_train_full[initial_indices])
                X_unlabeled.append(X_train_full[unlabeled_indices])
                y_unlabeled_indices.extend(unlabeled_indices)

            X_train_initial = np.concatenate(X_train_initial, axis=0)
            y_train_initial = np.concatenate(y_train_initial, axis=0)
            X_unlabeled = np.concatenate(X_unlabeled, axis=0)

            st.write(f"**Tập dữ liệu huấn luyện ban đầu lấy (1%)**: {len(X_train_initial)} mẫu")
            st.write(f"**Tập dữ liệu huấn luyện chưa gắn nhãn (99%)**: {len(X_unlabeled)} mẫu")

            # Lưu trữ tập dữ liệu ban đầu
            st.session_state['pseudo_data'] = {
                'X_train_initial': X_train_initial.copy(),
                'y_train_initial': y_train_initial.copy(),
                'X_unlabeled': X_unlabeled.copy(),
                'y_unlabeled_indices': y_unlabeled_indices,
                'X_test': X_test.copy(),
                'y_test': y_test.copy()
            }

            if "optimal_params" not in st.session_state:
                st.session_state["optimal_params"] = get_optimal_params(num_samples)
            
            params = st.session_state.get("training_params", st.session_state["optimal_params"].copy())

            st.subheader("⚙️ Cấu hình tham khảo Tham số Mô hình")
            st.markdown(f"""
            Dựa trên số mẫu huấn luyện ban đầu ({num_samples} mẫu), bảng dưới đây gợi ý các tham số tối ưu cho bài toán **Pseudo-Labeling với Neural Network**:

            | Số mẫu       | Số lớp ẩn | Kích thước lớp ẩn | Tốc độ học | Số lần lặp | Hàm kích hoạt | Trình tối ưu | Kích thước batch | Ngưỡng tin cậy | Số vòng lặp tối đa |
            |--------------|-----------|-------------------|------------|------------|---------------|--------------|------------------|----------------|-------------------|
            | ≤ 1,000      | 1         | 32                | 0.001      | 30         | ReLU          | Adam         | 32               | 0.9            | 3                 |
            | ≤ 10,000     | 2         | (64, 32)          | 0.0005     | 50         | ReLU          | Adam         | 64               | 0.95           | 5                 |
            | ≤ 50,000     | 2         | (128, 64)         | 0.0003     | 70         | ReLU          | Adam         | 128              | 0.97           | 7                 |
            | > 50,000     | 3         | (128, 64, 32)     | 0.0001     | 100        | ReLU          | Adam         | 256              | 0.98           | 10                |
            """, unsafe_allow_html=True)
            st.info(f"Tham số tối ưu gợi ý cho {num_samples} mẫu: {st.session_state['optimal_params']}")

            col_param1, col_param2 = st.columns(2)
            with col_param1:
                with st.expander("🧠 Cấu trúc Mạng", expanded=True):
                    st.markdown("**Tùy chỉnh số lớp ẩn và nơ-ron**", unsafe_allow_html=True)
                    num_hidden_layers = st.number_input("Số lớp ẩn", min_value=1, max_value=3, value=len(params["hidden_layer_sizes"]), 
                                                       help="Chọn 1, 2 hoặc 3 lớp ẩn để điều chỉnh độ phức tạp của mô hình.")
                    hidden_sizes = list(params["hidden_layer_sizes"])
                    
                    if num_hidden_layers == 1:
                        hidden_size_1 = st.number_input("Số nơ-ron lớp ẩn 1", min_value=16, max_value=128, 
                                                        value=hidden_sizes[0] if len(hidden_sizes) > 0 else 32, 
                                                        help="Số nơ-ron cho lớp ẩn duy nhất (16-128).")
                        hidden_sizes = [hidden_size_1]
                    elif num_hidden_layers == 2:
                        hidden_size_1 = st.number_input("Số nơ-ron lớp ẩn 1", min_value=16, max_value=128, 
                                                        value=hidden_sizes[0] if len(hidden_sizes) > 0 else 64, 
                                                        help="Số nơ-ron cho lớp ẩn đầu tiên (16-128).")
                        hidden_size_2 = st.number_input("Số nơ-ron lớp ẩn 2", min_value=16, max_value=128, 
                                                        value=hidden_sizes[1] if len(hidden_sizes) > 1 else 32, 
                                                        help="Số nơ-ron cho lớp ẩn thứ hai (16-128).")
                        hidden_sizes = [hidden_size_1, hidden_size_2]
                    elif num_hidden_layers == 3:
                        hidden_size_1 = st.number_input("Số nơ-ron lớp ẩn 1", min_value=16, max_value=128, 
                                                        value=hidden_sizes[0] if len(hidden_sizes) > 0 else 128, 
                                                        help="Số nơ-ron cho lớp ẩn đầu tiên (16-128).")
                        hidden_size_2 = st.number_input("Số nơ-ron lớp ẩn 2", min_value=16, max_value=128, 
                                                        value=hidden_sizes[1] if len(hidden_sizes) > 1 else 64, 
                                                        help="Số nơ-ron cho lớp ẩn thứ hai (16-128).")
                        hidden_size_3 = st.number_input("Số nơ-ron lớp ẩn 3", min_value=16, max_value=128, 
                                                        value=hidden_sizes[2] if len(hidden_sizes) > 2 else 32, 
                                                        help="Số nơ-ron cho lớp ẩn thứ ba (16-128).")
                        hidden_sizes = [hidden_size_1, hidden_size_2, hidden_size_3]
                    
                    params["hidden_layer_sizes"] = tuple(hidden_sizes)
                    params["activation"] = st.selectbox("Hàm kích hoạt", ["relu", "sigmoid", "tanh"], 
                                                        index=["relu", "sigmoid", "tanh"].index(params["activation"]),
                                                        help="Chọn hàm kích hoạt: ReLU (nhanh), Sigmoid (xác suất), Tanh (cân bằng).")
            
            with col_param2:
                with st.expander("🔧 Tối ưu hóa", expanded=True):
                    st.markdown("**Cấu hình huấn luyện**", unsafe_allow_html=True)
                    params["learning_rate"] = st.selectbox("Tốc độ học", [0.01, 0.005, 0.001, 0.0005, 0.0003, 0.0001], 
                                                           index=[0.01, 0.005, 0.001, 0.0005, 0.0003, 0.0001].index(params["learning_rate"]),
                                                           help="Tốc độ học càng nhỏ càng ổn định nhưng chậm.")
                    params["epochs"] = st.number_input("Số lần lặp (Epochs)", min_value=10, max_value=100, value=params["epochs"], 
                                                       help="Số lần lặp qua toàn bộ dữ liệu (10-100).")
                    params["batch_size"] = st.number_input("Kích thước batch", min_value=32, max_value=256, value=params["batch_size"], 
                                                           help="Số mẫu mỗi lần cập nhật trọng số (32-256).")
                    params["solver"] = st.selectbox("Trình tối ưu", ["adam", "sgd"], 
                                                    index=["adam", "sgd"].index(params["solver"]),
                                                    help="Adam (nhanh, hiệu quả), SGD (đơn giản, chậm hơn).")
                    threshold_default = st.session_state.get("optimal_params", {}).get("threshold", 0.95)
                    threshold = st.slider("Ngưỡng tin cậy Pseudo-Label", 0.5, 1.0, 
                                          threshold_default, 
                                          help="Ngưỡng để gán nhãn giả cho dữ liệu không có nhãn.")
                    max_iterations = st.number_input("Số vòng lặp tối đa", min_value=1, max_value=10, 
                                                     value=st.session_state["optimal_params"]["max_iterations"], 
                                                     help="Số lần lặp tối đa cho quá trình Pseudo-Labeling.")

            col_reset, col_train = st.columns([1, 3])
            with col_reset:
                if st.button("🔄 Khôi phục tham số tối ưu", key="reset_params"):
                    st.session_state["training_params"] = st.session_state["optimal_params"].copy()
                    st.success("Đã khôi phục tham số tối ưu!")
                    st.rerun()

            st.session_state["training_params"] = params

            with col_train:
                if st.button("🚀 Bắt đầu Huấn luyện với Pseudo-Labeling", type="primary", key="start_training"):
                    try:
                        with st.spinner("Đang thực hiện quy trình Pseudo-Labeling..."):
                            start_time = time.time()

                            # Khởi tạo tập dữ liệu huấn luyện
                            X_train = st.session_state['pseudo_data']['X_train_initial'].copy()
                            y_train = st.session_state['pseudo_data']['y_train_initial'].copy()
                            X_unlabeled = st.session_state['pseudo_data']['X_unlabeled'].copy()
                            y_unlabeled_indices = st.session_state['pseudo_data']['y_unlabeled_indices'].copy()

                            iteration = 0
                            pseudo_labeled_history = []
                            accuracy_test_history = []

                            # Tạo các container cố định để hiển thị thông tin vòng lặp
                            iteration_container = st.empty()  # Container cho tiêu đề vòng lặp
                            progress_bar_container = st.empty()  # Container cho progress bar
                            status_container = st.empty()  # Container cho thông tin epoch
                            pseudo_container = st.empty()  # Container cho số mẫu gán nhãn giả
                            accuracy_container = st.empty()  # Container cho độ chính xác

                            while iteration < max_iterations and len(X_unlabeled) > 0:
                                iteration += 1
                                # Hiển thị tiêu đề vòng lặp trong container cố định
                                iteration_container.markdown(f"**Vòng lặp {iteration}/{max_iterations}**")

                                # Bước 2: Huấn luyện mô hình trên tập dữ liệu hiện tại
                                model = models.Sequential()
                                model.add(layers.Input(shape=(784,)))
                                for neurons in params["hidden_layer_sizes"]:
                                    model.add(layers.Dense(neurons, activation=params["activation"]))
                                model.add(layers.Dense(10, activation='softmax'))

                                optimizer = tf.keras.optimizers.Adam(learning_rate=params["learning_rate"]) if params["solver"] == "adam" else tf.keras.optimizers.SGD(learning_rate=params["learning_rate"])

                                model.compile(optimizer=optimizer,
                                              loss='sparse_categorical_crossentropy',
                                              metrics=['accuracy'])

                                # Tạo progress bar trong container cố định
                                with progress_bar_container:
                                    progress_bar = st.progress(0)

                                class ProgressCallback(callbacks.Callback):
                                    def on_epoch_end(self, epoch, logs=None):
                                        progress = (epoch + 1) / params["epochs"] * 100
                                        progress_bar.progress(int(progress))
                                        # Cập nhật thông tin epoch trong container cố định
                                        status_container.markdown(
                                            f"**Vòng lặp {iteration}/{max_iterations} - Epoch {epoch+1}/{params['epochs']}**: "
                                            f"Loss: {logs['loss']:.4f}, Accuracy: {logs['accuracy']:.4f}"
                                        )

                                history = model.fit(X_train, y_train, epochs=params["epochs"], batch_size=params["batch_size"],
                                                    callbacks=[ProgressCallback()], verbose=0)

                                # Bước 3: Dự đoán nhãn cho tập dữ liệu chưa gắn nhãn
                                predictions = model.predict(X_unlabeled, verbose=0)
                                predicted_labels = np.argmax(predictions, axis=1)
                                confidences = np.max(predictions, axis=1)

                                # Bước 4: Gán nhãn giả với ngưỡng tin cậy
                                pseudo_mask = confidences >= threshold
                                X_pseudo = X_unlabeled[pseudo_mask]
                                y_pseudo = predicted_labels[pseudo_mask]

                                # Hiển thị số mẫu được gán nhãn giả trong container cố định
                                pseudo_container.markdown(f"**Số mẫu được gán nhãn giả trong vòng {iteration}**: {len(X_pseudo)}")

                                # Cập nhật tập dữ liệu huấn luyện
                                X_train = np.concatenate([X_train, X_pseudo], axis=0)
                                y_train = np.concatenate([y_train, y_pseudo], axis=0)

                                # Loại bỏ các mẫu đã được gán nhãn giả khỏi tập unlabeled
                                remaining_mask = ~pseudo_mask
                                X_unlabeled = X_unlabeled[remaining_mask]

                                # Ghi lại số lượng mẫu được gán nhãn giả
                                pseudo_labeled_history.append(len(X_pseudo))

                                # Đánh giá trên tập test
                                y_test_pred = np.argmax(model.predict(X_test, verbose=0), axis=1)
                                acc_test = accuracy_score(y_test, y_test_pred)
                                accuracy_test_history.append(acc_test)
                                # Hiển thị độ chính xác trong container cố định
                                accuracy_container.markdown(f"**Độ chính xác trên tập Test sau vòng {iteration}**: {acc_test*100:.2f}%")

                                tf.keras.backend.clear_session()
                                del model, predictions, predicted_labels, confidences, pseudo_mask, X_pseudo, y_pseudo, remaining_mask
                                gc.collect()

                            # Huấn luyện lần cuối trên toàn bộ tập dữ liệu đã gắn nhãn
                            iteration_container.markdown("**Huấn luyện lần cuối trên toàn bộ tập dữ liệu đã gắn nhãn**")
                            model = models.Sequential()
                            model.add(layers.Input(shape=(784,)))
                            for neurons in params["hidden_layer_sizes"]:
                                model.add(layers.Dense(neurons, activation=params["activation"]))
                            model.add(layers.Dense(10, activation='softmax'))

                            optimizer = tf.keras.optimizers.Adam(learning_rate=params["learning_rate"]) if params["solver"] == "adam" else tf.keras.optimizers.SGD(learning_rate=params["learning_rate"])

                            model.compile(optimizer=optimizer,
                                          loss='sparse_categorical_crossentropy',
                                          metrics=['accuracy'])

                            with progress_bar_container:
                                progress_bar_final = st.progress(0)
                            status_container_final = st.empty()

                            class ProgressCallback(callbacks.Callback):
                                def on_epoch_end(self, epoch, logs=None):
                                    progress = (epoch + 1) / params["epochs"] * 100
                                    progress_bar_final.progress(int(progress))
                                    status_container_final.markdown(
                                        f"**Huấn luyện cuối - Epoch {epoch+1}/{params['epochs']}**: "
                                        f"Loss: {logs['loss']:.4f}, Accuracy: {logs['accuracy']:.4f}"
                                    )

                            history = model.fit(X_train, y_train, epochs=params["epochs"], batch_size=params["batch_size"],
                                                callbacks=[ProgressCallback()], verbose=0)

                            # Đánh giá trên tập test
                            y_test_pred = np.argmax(model.predict(X_test, verbose=0), axis=1)
                            acc_test = accuracy_score(y_test, y_test_pred)
                            cm_test = confusion_matrix(y_test, y_test_pred)

                            run_name = f"PseudoLabeling_NN_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                            with mlflow.start_run(experiment_id=EXPERIMENT_ID, run_name=run_name) as run:
                                mlflow.log_params({
                                    'hidden_layer_sizes': params["hidden_layer_sizes"],
                                    'learning_rate': params["learning_rate"],
                                    'epochs': params["epochs"],
                                    'batch_size': params["batch_size"],
                                    'activation': params["activation"],
                                    'solver': params["solver"],
                                    'threshold': threshold,
                                    'max_iterations': max_iterations
                                })
                                mlflow.log_metric("accuracy_test", acc_test)
                                mlflow.log_metric("training_time", time.time() - start_time)
                                mlflow.log_metric("total_iterations", iteration)

                            st.session_state['model'] = model
                            st.session_state['training_results'] = {
                                'accuracy_test': acc_test,
                                'cm_test': cm_test,
                                'run_name': run_name,
                                'run_id': run.info.run_id,
                                'params': params,
                                'training_time': time.time() - start_time,
                                'loss_history': history.history['loss'][-10:],
                                'accuracy_history': history.history['accuracy'][-10:],
                                'pseudo_labeled_history': pseudo_labeled_history,
                                'accuracy_test_history': accuracy_test_history,
                                'total_iterations': iteration
                            }

                            # Hiển thị thông tin vòng lặp cuối cùng
                            final_iteration = iteration
                            iteration_container.markdown(f"**Vòng lặp {final_iteration}/{max_iterations}**")
                            pseudo_container.markdown(f"**Số mẫu được gán nhãn giả trong vòng {final_iteration}**: {pseudo_labeled_history[-1] if pseudo_labeled_history else 0}")
                            accuracy_container.markdown(f"**Độ chính xác trên tập Test sau vòng {final_iteration}**: {acc_test*100:.2f}%")

                            st.success(f"Đã hoàn thành Pseudo-Labeling! Thời gian: {time.time() - start_time:.2f} giây, Tổng số vòng lặp: {final_iteration}")
                            tf.keras.backend.clear_session()
                            del X_train, y_train, X_unlabeled, X_test, y_test, split_data, history
                            gc.collect()
                            st.rerun()

                    except Exception as e:
                        st.error(f"Lỗi trong quá trình huấn luyện với Pseudo-Labeling: {e}")

            if 'training_results' in st.session_state:
                results = st.session_state['training_results']
                st.subheader("📊 Kết quả Huấn luyện với Pseudo-Labeling")
                col_result1, col_result2, col_result3 = st.columns(3)
                with col_result1:
                    st.metric("Thời gian huấn luyện", f"{results['training_time']:.2f} giây")
                with col_result2:
                    st.metric("Độ chính xác Test", f"{results['accuracy_test']*100:.2f}%")
                with col_result3:
                    st.metric("Tổng số vòng lặp", f"{results['total_iterations']}")

                st.subheader("📈 Ma trận Nhầm lẫn trên tập Test")
                fig, ax = plt.subplots(figsize=(6, 5))
                sns.heatmap(results['cm_test'], annot=True, fmt="d", cmap="Blues", ax=ax)
                ax.set_title("Test")
                st.pyplot(fig)
                plt.close(fig)

                st.subheader("📉 Biểu đồ Kết quả Huấn luyện")
                col_chart1, col_chart2 = st.columns(2)
                with col_chart1:
                    if results['loss_history']:
                        fig, ax = plt.subplots(figsize=(6, 3))
                        ax.plot(range(1, len(results['loss_history']) + 1), results['loss_history'], 
                                label='Loss', linestyle='-', color='blue', linewidth=2)
                        ax.set_xlabel("Epochs")
                        ax.set_ylabel("Loss")
                        ax.set_title("Training Loss (Final)")
                        ax.legend()
                        ax.grid(True)
                        st.pyplot(fig)
                        plt.close(fig)
                    st.markdown("**Giải thích:** Biểu đồ này thể hiện mức độ mất mát (loss) của mô hình trong 10 epoch cuối cùng của lần huấn luyện cuối, cho thấy khả năng hội tụ.")

                    if results['pseudo_labeled_history']:
                        fig, ax = plt.subplots(figsize=(6, 3))
                        ax.plot(range(1, len(results['pseudo_labeled_history']) + 1), results['pseudo_labeled_history'], 
                                label='Số mẫu', linestyle='-', color='purple', linewidth=2)
                        ax.set_xlabel("Vòng lặp")
                        ax.set_ylabel("Số mẫu")
                        ax.set_title("Số mẫu Pseudo-Label")
                        ax.legend()
                        ax.grid(True)
                        st.pyplot(fig)
                        plt.close(fig)
                    st.markdown("**Giải thích:** Hiển thị số lượng mẫu được gán nhãn giả qua từng vòng lặp Pseudo-Labeling, phản ánh mức độ mở rộng tập dữ liệu.")

                with col_chart2:
                    if results['accuracy_history']:
                        fig, ax = plt.subplots(figsize=(6, 3))
                        ax.plot(range(1, len(results['accuracy_history']) + 1), results['accuracy_history'], 
                                label='Accuracy', linestyle='-', color='green', linewidth=2)
                        ax.set_xlabel("Epochs")
                        ax.set_ylabel("Accuracy")
                        ax.set_title("Training Accuracy (Final)")
                        ax.legend()
                        ax.grid(True)
                        st.pyplot(fig)
                        plt.close(fig)
                    st.markdown("**Giải thích:** Biểu đồ này cho thấy độ chính xác huấn luyện trong 10 epoch cuối cùng của lần huấn luyện cuối, đánh giá hiệu suất mô hình.")

                    if results['accuracy_test_history']:
                        fig, ax = plt.subplots(figsize=(6, 3))
                        ax.plot(range(1, len(results['accuracy_test_history']) + 1), results['accuracy_test_history'], 
                                label='Test Accuracy', linestyle='-', color='red', linewidth=2)
                        ax.set_xlabel("Vòng lặp")
                        ax.set_ylabel("Accuracy")
                        ax.set_title("Test Accuracy qua vòng lặp")
                        ax.legend()
                        ax.grid(True)
                        st.pyplot(fig)
                        plt.close(fig)
                    st.markdown("**Giải thích:** Thể hiện độ chính xác trên tập kiểm tra qua các vòng lặp, đánh giá sự cải thiện nhờ Pseudo-Labeling.")

                with st.expander("Xem chi tiết", expanded=False):
                    st.markdown("**Thông tin lần chạy:**")
                    st.write(f"- Tên: {results['run_name']}")
                    st.write(f"- ID: {results['run_id']}")
                    st.write(f"- Thời gian huấn luyện: {results['training_time']:.2f} giây")
                    st.write(f"- Tổng số vòng lặp: {results['total_iterations']}")
                    st.write(f"- Độ chính xác Test: {results['accuracy_test']*100:.2f}%")
                    st.markdown("**Tham số đã chọn:**")
                    st.json({
                        "Số lớp ẩn": len(results['params']['hidden_layer_sizes']),
                        "Số nơ-ron mỗi lớp": results['params']['hidden_layer_sizes'],
                        "Tốc độ học": results['params']['learning_rate'],
                        "Số lần lặp mỗi vòng (Epochs)": results['params']['epochs'],
                        "Kích thước batch": results['params']['batch_size'],
                        "Hàm kích hoạt": results['params']['activation'],
                        "Trình tối ưu": results['params']['solver'],
                        "Ngưỡng tin cậy": threshold,
                        "Số vòng lặp tối đa": max_iterations
                    })
     # Tab 6: Demo dự đoán
    with tab_demo:
        st.markdown('<div class="section-title">Demo Dự đoán Chữ số</div>', unsafe_allow_html=True)
        st.header("Dự đoán số viết tay")
        st.write("Chọn cách nhập liệu: tải lên hình ảnh, sử dụng dữ liệu Test hoặc vẽ trực tiếp.")

        if 'split_data' not in st.session_state:
            st.warning("⚠️ Vui lòng chia dữ liệu trước trong tab 'Chia dữ liệu'!")
        else:
            # Khởi tạo client MLflow chỉ một lần
            if 'mlflow_client' not in st.session_state:
                st.session_state['mlflow_client'] = MlflowClient()

            # Lấy danh sách runs một lần và lưu vào session_state
            if 'model_options' not in st.session_state or st.button("Làm mới danh sách mô hình"):
                with st.spinner("Đang tải danh sách mô hình..."):
                    runs = st.session_state['mlflow_client'].search_runs(
                        experiment_ids=[EXPERIMENT_ID], 
                        order_by=["attributes.start_time DESC"]
                    )
                    st.session_state['model_options'] = {
                        run.info.run_id: run.data.tags.get('mlflow.runName', f"Run_{run.info.run_id}") 
                        for run in runs if 'mlflow.runName' in run.data.tags
                    }

            model_options = st.session_state['model_options']

            if model_options:
                # Tự động chọn model mới nhất sau khi huấn luyện
                if 'latest_run_id' in st.session_state:
                    default_run_id = st.session_state['latest_run_id']
                else:
                    default_run_id = list(model_options.keys())[0]
                
                default_model_name = model_options.get(default_run_id, list(model_options.values())[0])
                
                # Chọn mô hình
                selected_model_name = st.selectbox(
                    "Chọn mô hình:", 
                    list(model_options.values()), 
                    index=list(model_options.values()).index(default_model_name),
                    key="model_select"
                )
                selected_run_id = [k for k, v in model_options.items() if v == selected_model_name][0]

                # Tải mô hình một lần và lưu vào session_state
                if 'selected_model' not in st.session_state or st.session_state['selected_run_id'] != selected_run_id:
                    with st.spinner("Đang tải mô hình..."):
                        model_uri = f"runs:/{selected_run_id}/model"
                        try:
                            model = mlflow.keras.load_model(model_uri)
                            st.session_state['selected_model'] = model
                            st.session_state['selected_run_id'] = selected_run_id
                        except Exception as e:
                            st.error(f"Không thể tải mô hình từ MLflow: {e}")
                            model = None
                else:
                    model = st.session_state['selected_model']

                if model is not None:
                    st.write(f"**Mô hình hiện tại**: {selected_model_name}")

                    input_method = st.selectbox(
                        "Chọn phương thức nhập liệu", 
                        ["Tải ảnh lên", "Dữ liệu Test", "Vẽ trực tiếp"],
                        key="input_method"
                    )
                    is_normalized = 'data_processed' in st.session_state

                    def preprocess_input(data, is_normalized):
                        if not is_normalized:
                            data = data / 255.0
                        return data

                    if input_method == "Tải ảnh lên":
                        st.markdown('<p class="mode-title">Dự đoán từ Ảnh Tải lên</p>', unsafe_allow_html=True)
                        uploaded_file = st.file_uploader("Tải lên hình ảnh", type=["png", "jpg", "jpeg"])
                        if uploaded_file is not None:
                            image = Image.open(uploaded_file).convert('L')
                            image = image.resize((28, 28))
                            st.image(image, caption="Hình ảnh đầu vào", width=100)

                            if st.button("Dự đoán", key="predict_upload_button"):
                                with st.spinner("Đang xử lý ảnh..."):
                                    image_array = np.array(image, dtype=np.float32)
                                    image_array = image_array.reshape(1, 784)
                                    image_processed = preprocess_input(image_array, is_normalized)
                                    prediction = model.predict(image_processed, verbose=0)[0]
                                    predicted_class = np.argmax(prediction)
                                    confidence = prediction[predicted_class] * 100
                                    st.markdown(f"""
                                        <div>
                                            <strong>Dự đoán:</strong> {predicted_class}<br>
                                            <strong>Độ tin cậy:</strong> {confidence:.2f}%
                                        </div>
                                    """, unsafe_allow_html=True)
                                    fig, ax = plt.subplots(figsize=(6, 4))
                                    ax.bar(range(10), prediction * 100, color='blue')
                                    ax.set_xlabel("Chữ số")
                                    ax.set_ylabel("Xác suất (%)")
                                    ax.set_title("Phân bố xác suất")
                                    st.pyplot(fig)
                                    plt.close(fig)
                                    st.success("Dự đoán hoàn tất!")
                                    del image, image_array, image_processed, prediction
                                    gc.collect()

                    elif input_method == "Dữ liệu Test":
                        st.markdown('<p class="mode-title">Dự đoán từ Dữ liệu Test</p>', unsafe_allow_html=True)
                        X_test = st.session_state['split_data']["X_test"]
                        y_test = st.session_state['split_data']["y_test"]
                        if len(X_test) == 0:
                            st.warning("Tập Test rỗng. Vui lòng chia lại dữ liệu với tỷ lệ Test > 0%.")
                        else:
                            col_select, col_display = st.columns([3, 2])
                            with col_select:
                                idx = st.slider("Chọn mẫu Test", 0, min(len(X_test) - 1, 100), 0)
                            with col_display:
                                st.write("**Ảnh mẫu Test:**")
                                fig, ax = plt.subplots(figsize=(2, 2))
                                ax.imshow(X_test[idx].reshape(28, 28), cmap='gray')
                                ax.axis('off')
                                st.pyplot(fig)
                                plt.close(fig)
                                st.write(f"**Nhãn thực tế:** {y_test[idx]}")

                            if st.button("🔍 Dự đoán", key="predict_test"):
                                with st.spinner("Đang dự đoán..."):
                                    sample = X_test[idx].reshape(1, -1)
                                    sample_processed = preprocess_input(sample, is_normalized)
                                    prediction = model.predict(sample_processed, verbose=0)[0]
                                    predicted_class = np.argmax(prediction)
                                    confidence = prediction[predicted_class] * 100
                                    st.markdown(f"""
                                        <div class="prediction-box">
                                            <strong>Dự đoán:</strong> {predicted_class}<br>
                                            <strong>Độ tin cậy:</strong> {confidence:.2f}%<br>
                                            <strong>Nhãn thực tế:</strong> {y_test[idx]}
                                        </div>
                                    """, unsafe_allow_html=True)
                                    fig, ax = plt.subplots(figsize=(6, 4))
                                    ax.bar(range(10), prediction * 100, color='blue')
                                    ax.set_xlabel("Chữ số")
                                    ax.set_ylabel("Xác suất (%)")
                                    ax.set_title("Phân bố xác suất")
                                    st.pyplot(fig)
                                    plt.close(fig)
                                    st.success("Dự đoán hoàn tất!")
                                    del sample, sample_processed, prediction
                                    gc.collect()

                    elif input_method == "Vẽ trực tiếp":
                        st.markdown('<p class="mode-title">Vẽ trực tiếp</p>', unsafe_allow_html=True)
                        st.write("Vẽ chữ số từ 0-9 (nét trắng trên nền đen):")

                        # Sử dụng key cố định cho canvas
                        if 'canvas_result' not in st.session_state:
                            st.session_state['canvas_result'] = None

                        canvas_result = st_canvas(
                            fill_color="rgba(255, 165, 0, 0.3)",
                            stroke_width=20,
                            stroke_color="#FFFFFF",
                            background_color="#000000",
                            height=280,
                            width=280,
                            drawing_mode="freedraw",
                            key="canvas_fixed_key",  # Key cố định
                            update_streamlit=False  # Ngăn rerender tự động
                        )

                        # Lưu kết quả canvas vào session_state
                        if canvas_result.image_data is not None:
                            st.session_state['canvas_result'] = canvas_result

                        col_pred, col_clear = st.columns([2, 1])
                        with col_pred:
                            if st.button("Dự đoán", key="predict_button"):
                                if st.session_state['canvas_result'] is not None:
                                    with st.spinner("Đang xử lý hình vẽ..."):
                                        image = Image.fromarray(
                                            st.session_state['canvas_result'].image_data.astype('uint8'), 'RGBA'
                                        ).convert('L')
                                        image_resized = image.resize((28, 28))
                                        image_array = np.array(image_resized, dtype=np.float32).reshape(1, 784)
                                        image_processed = preprocess_input(image_array, is_normalized)
                                        prediction = model.predict(image_processed, verbose=0)[0]
                                        predicted_class = np.argmax(prediction)
                                        confidence = prediction[predicted_class] * 100
                                        st.markdown(f"""
                                            <div>
                                                <strong>Dự đoán:</strong> {predicted_class}<br>
                                                <strong>Độ tin cậy:</strong> {confidence:.2f}%
                                            </div>
                                        """, unsafe_allow_html=True)
                                        fig, ax = plt.subplots(figsize=(6, 4))
                                        ax.bar(range(10), prediction * 100, color='blue')
                                        ax.set_xlabel("Chữ số")
                                        ax.set_ylabel("Xác suất (%)")
                                        ax.set_title("Phân bố xác suất")
                                        st.pyplot(fig)
                                        plt.close(fig)
                                        st.success("Dự đoán hoàn tất!")
                                        del image, image_resized, image_array, image_processed, prediction
                                        gc.collect()
                                else:
                                    st.warning("Vui lòng vẽ trước khi dự đoán!")

            else:
                st.warning("Chưa có mô hình nào được lưu trong MLflow.")
     # Tab 7: Thông tin huấn luyện
    with tab_log_info:
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
                    col_history1, col_history2 = st.columns(2)
                    with col_history1:
                        if 'training_results' in st.session_state and selected_run_id == st.session_state['training_results']['run_id']:
                            results = st.session_state['training_results']
                            if results['loss_history']:
                                fig, ax = plt.subplots(figsize=(6, 3))
                                ax.plot(range(1, len(results['loss_history']) + 1), results['loss_history'], 
                                        label='Loss', linestyle='-', color='blue', linewidth=2)
                                ax.set_xlabel("Epochs")
                                ax.set_ylabel("Loss")
                                ax.set_title("Training Loss (Final)")
                                ax.legend()
                                ax.grid(True)
                                st.pyplot(fig)
                                plt.close(fig)
                            st.markdown("**Giải thích:** Biểu đồ Loss của 10 epoch cuối trong lần huấn luyện cuối.")

                            if results['pseudo_labeled_history']:
                                fig, ax = plt.subplots(figsize=(6, 3))
                                ax.plot(range(1, len(results['pseudo_labeled_history']) + 1), results['pseudo_labeled_history'], 
                                        label='Số mẫu', linestyle='-', color='purple', linewidth=2)
                                ax.set_xlabel("Vòng lặp")
                                ax.set_ylabel("Số mẫu")
                                ax.set_title("Số mẫu Pseudo-Label")
                                ax.legend()
                                ax.grid(True)
                                st.pyplot(fig)
                                plt.close(fig)
                            st.markdown("**Giải thích:** Số mẫu được gán nhãn giả qua các vòng lặp.")

                    with col_history2:
                        if 'training_results' in st.session_state and selected_run_id == st.session_state['training_results']['run_id']:
                            results = st.session_state['training_results']
                            if results['accuracy_history']:
                                fig, ax = plt.subplots(figsize=(6, 3))
                                ax.plot(range(1, len(results['accuracy_history']) + 1), results['accuracy_history'], 
                                        label='Accuracy', linestyle='-', color='green', linewidth=2)
                                ax.set_xlabel("Epochs")
                                ax.set_ylabel("Accuracy")
                                ax.set_title("Training Accuracy (Final)")
                                ax.legend()
                                ax.grid(True)
                                st.pyplot(fig)
                                plt.close(fig)
                            st.markdown("**Giải thích:** Độ chính xác huấn luyện của 10 epoch cuối trong lần huấn luyện cuối.")

                            if results['accuracy_test_history']:
                                fig, ax = plt.subplots(figsize=(6, 3))
                                ax.plot(range(1, len(results['accuracy_test_history']) + 1), results['accuracy_test_history'], 
                                        label='Test Accuracy', linestyle='-', color='red', linewidth=2)
                                ax.set_xlabel("Vòng lặp")
                                ax.set_ylabel("Accuracy")
                                ax.set_title("Test Accuracy qua vòng lặp")
                                ax.legend()
                                ax.grid(True)
                                st.pyplot(fig)
                                plt.close(fig)
                            st.markdown("**Giải thích:** Độ chính xác trên tập test qua các vòng lặp.")

                    mlflow_ui_link = f"{mlflow_tracking_uri}/#/experiments/{EXPERIMENT_ID}"
                    st.markdown("---")
                    st.markdown(f"📊 **Xem chi tiết trên MLflow UI**: [Nhấn vào đây]({mlflow_ui_link})", unsafe_allow_html=True)

        except Exception as e:
            st.error(f"Lỗi khi tải thông tin huấn luyện: {e}")
if __name__ == "__main__":
    run_mnist_pseudo_labeling_app()