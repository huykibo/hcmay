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
            "batch_size": 32
        }
    elif num_samples <= 10000:
        return {
            "hidden_layer_sizes": (64, 32),
            "learning_rate": 0.0005,
            "epochs": 50,
            "activation": "relu",
            "solver": "adam",
            "batch_size": 64
        }
    elif num_samples <= 50000:
        return {
            "hidden_layer_sizes": (128, 64),
            "learning_rate": 0.0003,
            "epochs": 70,
            "activation": "relu",
            "solver": "adam",
            "batch_size": 128
        }
    else:
        return {
            "hidden_layer_sizes": (128, 64, 32),
            "learning_rate": 0.0001,
            "epochs": 100,
            "activation": "relu",
            "solver": "adam",
            "batch_size": 256
        }

def run_mnist_pseudo_labeling_app():
    # Thiết lập MLflow
    mlflow_tracking_uri = "https://dagshub.com/huykibo/MNIST_Pseudo_Labeling.mlflow"
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

    st.title("Phân loại Chữ số MNIST với Pseudo Labeling và Neural Network")

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
            .action-container {
                background-color: #ffffff;
                padding: 15px;
                border-radius: 5px;
                box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
                margin-bottom: 20px;
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

    # Thêm tab "Thông tin huấn luyện"
    tabs = st.tabs(["Thông tin", "Tải và Chia Dữ liệu", "Pseudo Labeling", "Kết quả", "Thông tin huấn luyện"])
    tab_info, tab_load_split, tab_pseudo_labeling, tab_results, tab_log_info = tabs

    # Tab 1: Thông tin (đầy đủ như mã gốc)
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
                "Pseudo Labeling – Gán nhãn giả",
                "Công thức đánh giá độ chính xác (Accuracy)"
            ],
            label_visibility="collapsed",
            help="Chọn để xem chi tiết về ứng dụng, dữ liệu, hoặc mô hình."
        )

        if info_option == "Ứng dụng này là gì và mục tiêu của nó?":
            with st.spinner("Đang tải thông tin..."):
                progress_bar = st.progress(0)
                status_text = st.empty()
                for i in range(0, 101, 10):
                    progress_bar.progress(i)
                    status_text.text(f"Đang tải thông tin... {i}%")
                    time.sleep(0.05)
                st.subheader("📘 1. Ứng dụng này là gì và mục tiêu của nó?")
                st.markdown("""
                Đây là một ứng dụng phân loại chữ số viết tay dựa trên tập dữ liệu **MNIST**, sử dụng **Mạng nơ-ron nhân tạo (Neural Network)**.  
                - **MNIST**: Tập dữ liệu gồm $70,000$ ảnh chữ số từ $0$ đến $9$, mỗi ảnh kích thước $28 \\times 28$ pixel (tổng cộng $784$ đặc trưng).  
                - **Mục tiêu**:  
                  - Xây dựng và huấn luyện một mạng nơ-ron để nhận diện chính xác các chữ số.  
                  - Cung cấp công cụ trực quan để học tập và đánh giá hiệu quả của thuật toán.  
                """, unsafe_allow_html=True)
                status_text.text("Đã tải xong! 100%")
                time.sleep(0.5)
                status_text.empty()
                progress_bar.empty()

        elif info_option == "Tập dữ liệu MNIST: Đặc điểm và ý nghĩa":
            with st.spinner("Đang tải thông tin..."):
                progress_bar = st.progress(0)
                status_text = st.empty()
                for i in range(0, 101, 10):
                    progress_bar.progress(i)
                    status_text.text(f"Đang tải thông tin... {i}%")
                    time.sleep(0.05)
                st.subheader("📘 2. Tập dữ liệu MNIST: Đặc điểm và ý nghĩa")
                st.markdown("""
                **MNIST** là tập dữ liệu chuẩn trong học máy, được tạo bởi Yann LeCun và các cộng sự.  
                - **Đặc điểm**:  
                  - Gồm các ảnh chữ số viết tay từ học sinh trung học và nhân viên điều tra dân số Mỹ.  
                  - Chuẩn hóa thành kích thước $28 \\times 28$ pixel, thang độ xám (giá trị từ $0$ đến $255$).  
                - **Ý nghĩa**:  
                  - Là bài toán cơ bản để kiểm tra khả năng phân loại của các mô hình học máy.  
                  - Đơn giản nhưng đủ phức tạp để đánh giá khả năng phân biệt các lớp tương tự (ví dụ: "$4$" và "$9$").  
                """, unsafe_allow_html=True)
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
                    status_text.text(f"Đang tải thông tin... {i}%")
                    time.sleep(0.05)
                st.subheader("📊 3. Neural Network – Mạng nơ-ron nhân tạo")
                st.markdown("""
                **Neural Network** là một mô hình học máy mô phỏng cách hoạt động của mạng nơ-ron sinh học trong não người.  
                - **Cấu trúc**:  
                  - **Lớp đầu vào**: $784$ pixel từ ảnh MNIST.  
                  - **Lớp ẩn**: Xử lý thông tin qua các phép tính tuyến tính và phi tuyến.  
                  - **Lớp đầu ra**: Dự đoán 10 lớp ($0$-$9$).  
                - **Quy trình**: Lan truyền thuận, tính mất mát, lan truyền ngược, cập nhật trọng số.  
                """, unsafe_allow_html=True)
                status_text.text("Đã tải xong! 100%")
                time.sleep(0.5)
                status_text.empty()
                progress_bar.empty()

        elif info_option == "Pseudo Labeling – Gán nhãn giả":
            with st.spinner("Đang tải thông tin..."):
                progress_bar = st.progress(0)
                status_text = st.empty()
                for i in range(0, 101, 10):
                    progress_bar.progress(i)
                    status_text.text(f"Đang tải thông tin... {i}%")
                    time.sleep(0.05)
                st.subheader("📘 4. Pseudo Labeling – Gán Nhãn Giả")
                st.markdown("""
                **Pseudo Labeling** là kỹ thuật học bán giám sát, tận dụng dữ liệu chưa có nhãn để cải thiện mô hình.  
                - **Cách thực hiện**:  
                  1. Chia tập train/test.  
                  2. Lấy 1% mẫu từ mỗi lớp (0-9) làm tập train ban đầu.  
                  3. Huấn luyện Neural Network trên tập 1%.  
                  4. Dự đoán nhãn cho 99% còn lại.  
                  5. Gán nhãn giả với ngưỡng (ví dụ: 0.95).  
                  6. Lặp lại từ bước 3 với tập dữ liệu mới cho đến khi gán hết hoặc đạt số lần lặp tối đa.  
                - **Ưu điểm**: Tăng cường hiệu suất khi dữ liệu có nhãn hạn chế.  
                - **Nhược điểm**: Có thể lan truyền sai sót nếu nhãn giả không chính xác.  
                """, unsafe_allow_html=True)
                status_text.text("Đã tải xong! 100%")
                time.sleep(0.5)
                status_text.empty()
                progress_bar.empty()

        elif info_option == "Công thức đánh giá độ chính xác (Accuracy)":
            with st.spinner("Đang tải thông tin..."):
                progress_bar = st.progress(0)
                status_text = st.empty()
                for i in range(0, 101, 10):
                    progress_bar.progress(i)
                    status_text.text(f"Đang tải thông tin... {i}%")
                    time.sleep(0.05)
                st.subheader("📘 5. Công thức đánh giá độ chính xác (Accuracy)")
                st.markdown("""
                - **Accuracy** đo tỷ lệ dự đoán đúng:  
                  $$ \\text{Accuracy} = \\frac{\\text{Số mẫu dự đoán đúng}}{\\text{Tổng số mẫu}} $$  
                - **Ví dụ**: Dự đoán đúng $92/100$ ảnh → $92\%$.  
                """, unsafe_allow_html=True)
                status_text.text("Đã tải xong! 100%")
                time.sleep(0.5)
                status_text.empty()
                progress_bar.empty()

    # Tab 2: Tải và Chia Dữ liệu
    with tab_load_split:
        st.markdown('<div class="section-title">Tải và Chia Dữ liệu MNIST</div>', unsafe_allow_html=True)

        if 'data' not in st.session_state:
            if st.button("Tải dữ liệu MNIST", type="primary"):
                with st.spinner("Đang tải dữ liệu..."):
                    (X_train_full, y_train_full), (X_test_full, y_test_full) = tf.keras.datasets.mnist.load_data()
                    X = np.concatenate([X_train_full, X_test_full], axis=0)
                    y = np.concatenate([y_train_full, y_test_full], axis=0)
                    X = X.reshape(-1, 784).astype(np.float32) / 255.0  # Chuẩn hóa
                    y = y.astype(np.int32)
                    st.session_state['data'] = (X, y)
                    st.success(f"Đã tải {X.shape[0]} mẫu!")
                    del X_train_full, X_test_full, y_train_full, y_test_full
                    gc.collect()
                    st.rerun()

        if 'data' in st.session_state and 'split_data' not in st.session_state:
            X, y = st.session_state['data']
            test_size = st.slider("Tỷ lệ Test (%)", 0, 50, 20) / 100
            if st.button("Chia dữ liệu", type="primary"):
                with st.spinner("Đang chia dữ liệu..."):
                    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
                    st.session_state['split_data'] = {
                        "X_train": X_train, "y_train": y_train,
                        "X_test": X_test, "y_test": y_test
                    }
                    st.success(f"Đã chia: Train ({len(X_train)} mẫu), Test ({len(X_test)} mẫu)")
                    del X, y, X_train, X_test, y_train, y_test
                    gc.collect()
                    st.rerun()

        if 'split_data' in st.session_state:
            split_data = st.session_state['split_data']
            st.write(f"Tập train: {len(split_data['X_train'])} mẫu")
            st.write(f"Tập test: {len(split_data['X_test'])} mẫu")

    # Tab 3: Pseudo Labeling
    with tab_pseudo_labeling:
        st.markdown('<div class="section-title">Pseudo Labeling với Neural Network</div>', unsafe_allow_html=True)
        if 'split_data' not in st.session_state:
            st.info("Vui lòng tải và chia dữ liệu trước.")
        else:
            split_data = st.session_state['split_data']
            X_train = split_data["X_train"]
            y_train = split_data["y_train"]
            X_test = split_data["X_test"]
            y_test = split_data["y_test"]

            num_samples = len(X_train)
            st.session_state['optimal_params'] = get_optimal_params(num_samples)
            params = st.session_state.get("training_params", st.session_state["optimal_params"].copy())

            st.subheader("⚙️ Cấu hình Tham số Mô hình")
            col_param1, col_param2 = st.columns(2)
            with col_param1:
                with st.expander("🧠 Cấu trúc Mạng", expanded=True):
                    num_hidden_layers = st.number_input("Số lớp ẩn", min_value=1, max_value=3, value=len(params["hidden_layer_sizes"]))
                    hidden_sizes = list(params["hidden_layer_sizes"])
                    if num_hidden_layers == 1:
                        hidden_size_1 = st.number_input("Số nơ-ron lớp ẩn 1", min_value=16, max_value=128, value=hidden_sizes[0] if hidden_sizes else 32)
                        hidden_sizes = [hidden_size_1]
                    elif num_hidden_layers == 2:
                        hidden_size_1 = st.number_input("Số nơ-ron lớp ẩn 1", min_value=16, max_value=128, value=hidden_sizes[0] if hidden_sizes else 64)
                        hidden_size_2 = st.number_input("Số nơ-ron lớp ẩn 2", min_value=16, max_value=128, value=hidden_sizes[1] if len(hidden_sizes) > 1 else 32)
                        hidden_sizes = [hidden_size_1, hidden_size_2]
                    elif num_hidden_layers == 3:
                        hidden_size_1 = st.number_input("Số nơ-ron lớp ẩn 1", min_value=16, max_value=128, value=hidden_sizes[0] if hidden_sizes else 128)
                        hidden_size_2 = st.number_input("Số nơ-ron lớp ẩn 2", min_value=16, max_value=128, value=hidden_sizes[1] if len(hidden_sizes) > 1 else 64)
                        hidden_size_3 = st.number_input("Số nơ-ron lớp ẩn 3", min_value=16, max_value=128, value=hidden_sizes[2] if len(hidden_sizes) > 2 else 32)
                        hidden_sizes = [hidden_size_1, hidden_size_2, hidden_size_3]
                    params["hidden_layer_sizes"] = tuple(hidden_sizes)
                    params["activation"] = st.selectbox("Hàm kích hoạt", ["relu", "sigmoid", "tanh"], index=["relu", "sigmoid", "tanh"].index(params["activation"]))

            with col_param2:
                with st.expander("🔧 Tối ưu hóa", expanded=True):
                    params["learning_rate"] = st.selectbox("Tốc độ học", [0.01, 0.005, 0.001, 0.0005, 0.0003, 0.0001], index=[0.01, 0.005, 0.001, 0.0005, 0.0003, 0.0001].index(params["learning_rate"]))
                    params["epochs"] = st.number_input("Số lần lặp (Epochs)", min_value=10, max_value=100, value=params["epochs"])
                    params["batch_size"] = st.number_input("Kích thước batch", min_value=32, max_value=256, value=params["batch_size"])
                    params["solver"] = st.selectbox("Trình tối ưu", ["adam", "sgd"], index=["adam", "sgd"].index(params["solver"]))
                    threshold = st.slider("Ngưỡng gán nhãn giả (Threshold)", 0.5, 1.0, 0.95, step=0.01)
                    max_iterations = st.number_input("Số vòng lặp tối đa", min_value=1, max_value=10, value=5)

            if st.button("🚀 Bắt đầu Pseudo Labeling", type="primary"):
                with st.spinner("Đang thực hiện Pseudo Labeling..."):
                    start_time = time.time()

                    # Bước 1: Lấy 1% mẫu từ mỗi class trong tập train
                    labeled_X, labeled_y, unlabeled_X = [], [], []
                    for digit in range(10):
                        digit_indices = np.where(y_train == digit)[0]
                        num_labeled = max(1, int(len(digit_indices) * 0.01))  # 1% hoặc ít nhất 1 mẫu
                        labeled_indices = np.random.choice(digit_indices, num_labeled, replace=False)
                        unlabeled_indices = np.setdiff1d(digit_indices, labeled_indices)
                        labeled_X.append(X_train[labeled_indices])
                        labeled_y.append(y_train[labeled_indices])
                        unlabeled_X.append(X_train[unlabeled_indices])
                    labeled_X = np.concatenate(labeled_X, axis=0)
                    labeled_y = np.concatenate(labeled_y, axis=0)
                    unlabeled_X = np.concatenate(unlabeled_X, axis=0)
                    st.write(f"Tập labeled ban đầu: {len(labeled_X)} mẫu")
                    st.write(f"Tập unlabeled: {len(unlabeled_X)} mẫu")

                    # Khởi tạo mô hình
                    def create_model():
                        model = models.Sequential()
                        model.add(layers.Input(shape=(784,)))
                        for neurons in params["hidden_layer_sizes"]:
                            model.add(layers.Dense(neurons, activation=params["activation"]))
                        model.add(layers.Dense(10, activation="softmax"))
                        optimizer = tf.keras.optimizers.Adam(learning_rate=params["learning_rate"]) if params["solver"] == "adam" else tf.keras.optimizers.SGD(learning_rate=params["learning_rate"])
                        model.compile(optimizer=optimizer, loss="sparse_categorical_crossentropy", metrics=["accuracy"])
                        return model

                    model = create_model()
                    pseudo_labeled_X, pseudo_labeled_y = labeled_X.copy(), labeled_y.copy()
                    iteration = 0
                    total_unlabeled = len(unlabeled_X)

                    # Vòng lặp Pseudo Labeling
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    while iteration < max_iterations and len(unlabeled_X) > 0:
                        iteration += 1
                        st.write(f"**Vòng lặp {iteration}/{max_iterations}**")

                        # Bước 2: Huấn luyện trên tập labeled hiện tại
                        history = model.fit(pseudo_labeled_X, pseudo_labeled_y, epochs=params["epochs"],
                                            batch_size=params["batch_size"], verbose=0)
                        status_text.text(f"Vòng {iteration}: Loss: {history.history['loss'][-1]:.4f}, Accuracy: {history.history['accuracy'][-1]:.4f}")

                        # Bước 3: Dự đoán nhãn cho tập unlabeled
                        pseudo_predictions = model.predict(unlabeled_X, verbose=0)
                        pseudo_confidences = np.max(pseudo_predictions, axis=1)
                        pseudo_labels = np.argmax(pseudo_predictions, axis=1)

                        # Bước 4: Gán nhãn giả với ngưỡng
                        confident_mask = pseudo_confidences >= threshold
                        new_labeled_X = unlabeled_X[confident_mask]
                        new_labeled_y = pseudo_labels[confident_mask]
                        if len(new_labeled_X) > 0:
                            pseudo_labeled_X = np.concatenate([pseudo_labeled_X, new_labeled_X], axis=0)
                            pseudo_labeled_y = np.concatenate([pseudo_labeled_y, new_labeled_y], axis=0)
                            unlabeled_X = unlabeled_X[~confident_mask]
                            st.write(f"Đã gán nhãn giả cho {len(new_labeled_X)} mẫu, còn lại {len(unlabeled_X)} mẫu chưa gán.")
                        else:
                            st.write("Không có mẫu nào đạt ngưỡng trong vòng này.")
                            break

                        progress_bar.progress(int((total_unlabeled - len(unlabeled_X)) / total_unlabeled * 100))

                    # Đánh giá trên tập test
                    y_test_pred = np.argmax(model.predict(X_test, verbose=0), axis=1)
                    acc_test = accuracy_score(y_test, y_test_pred)
                    cm_test = confusion_matrix(y_test, y_test_pred)

                    # Ghi log MLflow
                    run_name = f"PseudoLabeling_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                    with mlflow.start_run(experiment_id=EXPERIMENT_ID, run_name=run_name) as run:
                        mlflow.log_params({
                            "hidden_layer_sizes": params["hidden_layer_sizes"],
                            "learning_rate": params["learning_rate"],
                            "epochs": params["epochs"],
                            "batch_size": params["batch_size"],
                            "activation": params["activation"],
                            "solver": params["solver"],
                            "threshold": threshold,
                            "max_iterations": max_iterations
                        })
                        mlflow.log_metric("accuracy_test", acc_test)
                        mlflow.log_metric("training_time", time.time() - start_time)
                        mlflow.log_metric("n_iter_actual", iteration)
                        mlflow.log_metric("pseudo_labeled_samples", len(pseudo_labeled_X))

                        # Ghi log lịch sử loss và accuracy
                        for epoch, (loss, acc) in enumerate(zip(history.history['loss'], history.history['accuracy']), 1):
                            mlflow.log_metric("loss", loss, step=epoch)
                            mlflow.log_metric("accuracy", acc, step=epoch)

                    st.session_state['results'] = {
                        'model': model,
                        'accuracy_test': acc_test,
                        'cm_test': cm_test,
                        'training_time': time.time() - start_time,
                        'n_iter_actual': iteration,
                        'pseudo_labeled_samples': len(pseudo_labeled_X),
                        'loss_history': history.history['loss'],
                        'accuracy_history': history.history['accuracy'],
                        'run_id': run.info.run_id
                    }
                    st.success(f"Đã hoàn thành Pseudo Labeling! Thời gian: {time.time() - start_time:.2f} giây")
                    tf.keras.backend.clear_session()
                    del X_train, y_train, X_test, y_test, split_data, history
                    gc.collect()
                    st.rerun()

    # Tab 4: Kết quả
    with tab_results:
        st.markdown('<div class="section-title">Kết quả Pseudo Labeling</div>', unsafe_allow_html=True)
        if 'results' not in st.session_state:
            st.info("Vui lòng thực hiện Pseudo Labeling trước.")
        else:
            results = st.session_state['results']
            st.subheader("Số liệu")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Thời gian huấn luyện", f"{results['training_time']:.2f} giây")
            with col2:
                st.metric("Độ chính xác Test", f"{results['accuracy_test']*100:.2f}%")
            with col3:
                st.metric("Số mẫu được gán nhãn", f"{results['pseudo_labeled_samples']}")

            st.subheader("Ma trận Nhầm lẫn (Test)")
            fig, ax = plt.subplots(figsize=(6, 5))
            sns.heatmap(results['cm_test'], annot=True, fmt="d", cmap="Blues", ax=ax)
            st.pyplot(fig)
            plt.close(fig)

            st.subheader("Biểu đồ Kết quả Huấn luyện")
            if results['loss_history']:
                fig, ax = plt.subplots(figsize=(8, 4))
                ax.plot(range(1, len(results['loss_history']) + 1), results['loss_history'], label='Training Loss', color='blue')
                ax.set_xlabel("Epochs")
                ax.set_ylabel("Loss")
                ax.set_title("Training Loss")
                ax.legend()
                ax.grid(True)
                st.pyplot(fig)
                plt.close(fig)

            if results['accuracy_history']:
                fig, ax = plt.subplots(figsize=(8, 4))
                ax.plot(range(1, len(results['accuracy_history']) + 1), results['accuracy_history'], label='Training Accuracy', color='green')
                ax.set_xlabel("Epochs")
                ax.set_ylabel("Accuracy")
                ax.set_title("Training Accuracy")
                ax.legend()
                ax.grid(True)
                st.pyplot(fig)
                plt.close(fig)

    # Tab 5: Thông tin huấn luyện
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
                    # Biểu đồ Loss
                    history_metrics = client.get_metric_history(selected_run_id, "loss")
                    if history_metrics:
                        epochs = range(1, len(history_metrics) + 1)
                        loss_values = [metric.value for metric in history_metrics]
                        fig, ax = plt.subplots(figsize=(10, 5))
                        ax.plot(epochs, loss_values, label='Training Loss', linestyle='-', color='blue', linewidth=2)
                        ax.set_xlabel("Epochs")
                        ax.set_ylabel("Loss")
                        ax.set_title("Lịch sử Mất mát")
                        ax.legend()
                        ax.grid(True)
                        st.pyplot(fig)
                        plt.close(fig)
                    else:
                        if 'results' in st.session_state and selected_run_id == st.session_state['results']['run_id']:
                            results = st.session_state['results']
                            if results['loss_history']:
                                fig, ax = plt.subplots(figsize=(10, 5))
                                ax.plot(range(1, len(results['loss_history']) + 1), results['loss_history'], 
                                        label='Training Loss', linestyle='-', color='blue', linewidth=2)
                                ax.set_xlabel("Epochs")
                                ax.set_ylabel("Loss")
                                ax.set_title("Lịch sử Mất mát")
                                ax.legend()
                                ax.grid(True)
                                st.pyplot(fig)
                                plt.close(fig)
                        else:
                            st.info("Không có dữ liệu lịch sử Loss để hiển thị.")

                    # Biểu đồ Accuracy
                    history_accuracy = client.get_metric_history(selected_run_id, "accuracy")
                    if history_accuracy:
                        epochs = range(1, len(history_accuracy) + 1)
                        accuracy_values = [metric.value for metric in history_accuracy]
                        fig, ax = plt.subplots(figsize=(10, 5))
                        ax.plot(epochs, accuracy_values, label='Training Accuracy', linestyle='-', color='green', linewidth=2)
                        ax.set_xlabel("Epochs")
                        ax.set_ylabel("Accuracy")
                        ax.set_title("Lịch sử Độ chính xác")
                        ax.legend()
                        ax.grid(True)
                        st.pyplot(fig)
                        plt.close(fig)
                    else:
                        if 'results' in st.session_state and selected_run_id == st.session_state['results']['run_id']:
                            results = st.session_state['results']
                            if results['accuracy_history']:
                                fig, ax = plt.subplots(figsize=(10, 5))
                                ax.plot(range(1, len(results['accuracy_history']) + 1), results['accuracy_history'], 
                                        label='Training Accuracy', linestyle='-', color='green', linewidth=2)
                                ax.set_xlabel("Epochs")
                                ax.set_ylabel("Accuracy")
                                ax.set_title("Lịch sử Độ chính xác")
                                ax.legend()
                                ax.grid(True)
                                st.pyplot(fig)
                                plt.close(fig)
                        else:
                            st.info("Không có dữ liệu lịch sử Accuracy để hiển thị.")

                    mlflow_ui_link = f"{mlflow_tracking_uri}/#/experiments/{EXPERIMENT_ID}"
                    st.markdown("---")
                    st.markdown(f"📊 **Xem chi tiết trên MLflow UI**: [Nhấn vào đây]({mlflow_ui_link})", unsafe_allow_html=True)

        except Exception as e:
            st.error(f"Lỗi khi tải thông tin huấn luyện: {e}")

if __name__ == "__main__":
    run_mnist_pseudo_labeling_app()