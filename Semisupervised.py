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

    ### Tab 5: Huấn luyện/Đánh giá
    with tab_train_eval:
        st.markdown('<div class="section-title">Huấn luyện và Đánh giá</div>', unsafe_allow_html=True)
        if 'split_data' not in st.session_state:
            st.info("Vui lòng chia dữ liệu trước.")
        else:
            split_data = st.session_state['split_data'].copy()
            X_train_full, y_train_full = split_data["X_train"], split_data["y_train"]
            X_test, y_test = split_data["X_test"], split_data["y_test"]

            # Chuẩn bị tập dữ liệu ban đầu (1% mỗi lớp)
            classes = np.unique(y_train_full)
            X_train_initial, y_train_initial, X_unlabeled = [], [], []
            for cls in classes:
                cls_indices = np.where(y_train_full == cls)[0]
                num_initial = max(1, int(0.01 * len(cls_indices)))
                initial_indices = np.random.choice(cls_indices, num_initial, replace=False)
                X_train_initial.append(X_train_full[initial_indices])
                y_train_initial.append(y_train_full[initial_indices])
                X_unlabeled.append(X_train_full[np.setdiff1d(cls_indices, initial_indices)])
            X_train_initial = np.concatenate(X_train_initial)
            y_train_initial = np.concatenate(y_train_initial)
            X_unlabeled = np.concatenate(X_unlabeled)

            st.session_state['pseudo_data'] = {
                'X_train_initial': X_train_initial.copy(),
                'y_train_initial': y_train_initial.copy(),
                'X_unlabeled': X_unlabeled.copy(),
                'X_test': X_test.copy(),
                'y_test': y_test.copy()
            }

            params = st.session_state.get("training_params", get_optimal_params(len(X_train_full)))
            st.subheader("Cấu hình Tham số")
            num_hidden_layers = st.number_input("Số lớp ẩn", 1, 3, len(params["hidden_layer_sizes"]))
            hidden_sizes = list(params["hidden_layer_sizes"])
            if num_hidden_layers == 1:
                hidden_sizes = [st.number_input("Lớp ẩn 1", 16, 128, hidden_sizes[0] if hidden_sizes else 32)]
            elif num_hidden_layers == 2:
                hidden_sizes = [
                    st.number_input("Lớp ẩn 1", 16, 128, hidden_sizes[0] if hidden_sizes else 64),
                    st.number_input("Lớp ẩn 2", 16, 128, hidden_sizes[1] if len(hidden_sizes) > 1 else 32)
                ]
            else:
                hidden_sizes = [
                    st.number_input("Lớp ẩn 1", 16, 128, hidden_sizes[0] if hidden_sizes else 128),
                    st.number_input("Lớp ẩn 2", 16, 128, hidden_sizes[1] if len(hidden_sizes) > 1 else 64),
                    st.number_input("Lớp ẩn 3", 16, 128, hidden_sizes[2] if len(hidden_sizes) > 2 else 32)
                ]
            params["hidden_layer_sizes"] = tuple(hidden_sizes)
            params["learning_rate"] = st.selectbox("Tốc độ học", [0.01, 0.001, 0.0005, 0.0001], index=2)
            params["epochs"] = st.number_input("Epochs", 10, 100, params["epochs"])
            params["batch_size"] = st.number_input("Batch size", 32, 256, params["batch_size"])
            params["activation"] = st.selectbox("Hàm kích hoạt", ["relu", "sigmoid"], index=0)
            params["solver"] = st.selectbox("Trình tối ưu", ["adam", "sgd"], index=0)
            threshold = st.slider("Ngưỡng tin cậy", 0.5, 1.0, params["threshold"])
            max_iterations = st.number_input("Số vòng lặp tối đa", 1, 10, params["max_iterations"])

            st.session_state["training_params"] = params

            if st.button("Bắt đầu Huấn luyện", type="primary"):
                with st.spinner("Đang huấn luyện..."):
                    X_train = X_train_initial.copy()
                    y_train = y_train_initial.copy()
                    X_unlabeled = X_unlabeled.copy()
                    iteration, pseudo_history, acc_history, pseudo_samples_history = 0, [], [], []

                    while iteration < max_iterations and len(X_unlabeled) > 0:
                        iteration += 1
                        st.write(f"**Vòng lặp {iteration}/{max_iterations}**")

                        model = models.Sequential([
                            layers.Input(shape=(784,)),
                            *[layers.Dense(n, activation=params["activation"]) for n in params["hidden_layer_sizes"]],
                            layers.Dense(10, activation='softmax')
                        ])
                        optimizer = tf.keras.optimizers.Adam(params["learning_rate"]) if params["solver"] == "adam" else tf.keras.optimizers.SGD(params["learning_rate"])
                        model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
                        model.fit(X_train, y_train, epochs=params["epochs"], batch_size=params["batch_size"], verbose=0)

                        predictions = model.predict(X_unlabeled, verbose=0)
                        predicted_labels = np.argmax(predictions, axis=1)
                        confidences = np.max(predictions, axis=1)
                        pseudo_mask = confidences >= threshold
                        X_pseudo = X_unlabeled[pseudo_mask]
                        y_pseudo = predicted_labels[pseudo_mask]

                        st.write(f"Số mẫu gán nhãn giả: {len(X_pseudo)}")
                        if len(X_pseudo) > 0:
                            num_display = min(5, len(X_pseudo))
                            fig, axes = plt.subplots(1, num_display, figsize=(num_display * 2, 2))
                            for i in range(num_display):
                                ax = axes[i] if num_display > 1 else axes
                                ax.imshow(X_pseudo[i].reshape(28, 28), cmap='gray')
                                ax.set_title(f"Label: {y_pseudo[i]}")
                                ax.axis('off')
                            st.pyplot(fig)
                            plt.close(fig)
                            pseudo_samples_history.append((X_pseudo[:num_display].copy(), y_pseudo[:num_display].copy()))

                        X_train = np.concatenate([X_train, X_pseudo])
                        y_train = np.concatenate([y_train, y_pseudo])
                        X_unlabeled = X_unlabeled[~pseudo_mask]
                        pseudo_history.append(len(X_pseudo))
                        acc_test = accuracy_score(y_test, np.argmax(model.predict(X_test, verbose=0), axis=1))
                        acc_history.append(acc_test)
                        st.write(f"Độ chính xác Test: {acc_test*100:.2f}%")

                    # Huấn luyện lần cuối
                    model.fit(X_train, y_train, epochs=params["epochs"], batch_size=params["batch_size"], verbose=0)
                    y_test_pred = np.argmax(model.predict(X_test, verbose=0), axis=1)
                    acc_test = accuracy_score(y_test, y_test_pred)
                    cm_test = confusion_matrix(y_test, y_test_pred)

                    run_name = f"PseudoLabeling_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                    with mlflow.start_run(experiment_id=EXPERIMENT_ID, run_name=run_name) as run:
                        mlflow.log_params(params)
                        mlflow.log_metric("accuracy_test", acc_test)
                    st.session_state['model'] = model
                    st.session_state['training_results'] = {
                        'accuracy_test': acc_test,
                        'cm_test': cm_test,
                        'run_id': run.info.run_id,
                        'pseudo_samples_history': pseudo_samples_history,
                        'accuracy_history': acc_history
                    }
                    st.success("Hoàn thành huấn luyện!")
                    gc.collect()

            if 'training_results' in st.session_state:
                results = st.session_state['training_results']
                st.subheader("Kết quả")
                st.write(f"Độ chính xác Test: {results['accuracy_test']*100:.2f}%")
                fig, ax = plt.subplots()
                sns.heatmap(results['cm_test'], annot=True, fmt="d", cmap="Blues", ax=ax)
                st.pyplot(fig)
                plt.close(fig)

                st.subheader("Minh họa mẫu Pseudo-Label")
                for iter_idx, (X_pseudo, y_pseudo) in enumerate(results['pseudo_samples_history'], 1):
                    if len(X_pseudo) > 0:
                        st.write(f"**Vòng lặp {iter_idx}:**")
                        fig, axes = plt.subplots(1, len(X_pseudo), figsize=(len(X_pseudo) * 2, 2))
                        for i, ax in enumerate(axes if len(X_pseudo) > 1 else [axes]):
                            ax.imshow(X_pseudo[i].reshape(28, 28), cmap='gray')
                            ax.set_title(f"Label: {y_pseudo[i]}")
                            ax.axis('off')
                        st.pyplot(fig)
                        plt.close(fig)

    ### Tab 6: Demo dự đoán
    with tab_demo:
        st.markdown('<div class="section-title">Demo Dự đoán</div>', unsafe_allow_html=True)
        if 'split_data' not in st.session_state:
            st.warning("Vui lòng chia dữ liệu trước!")
        else:
            # Huấn luyện với 1% dữ liệu
            st.subheader("Kiểm chứng với 1% dữ liệu")
            if st.button("Huấn luyện với 1% dữ liệu"):
                with st.spinner("Đang huấn luyện..."):
                    X_train_initial = st.session_state['pseudo_data']['X_train_initial'].copy()
                    y_train_initial = st.session_state['pseudo_data']['y_train_initial'].copy()
                    X_test = st.session_state['pseudo_data']['X_test'].copy()
                    y_test = st.session_state['pseudo_data']['y_test'].copy()
                    params = st.session_state.get("training_params", get_optimal_params(len(X_train_full)))

                    model_1percent = models.Sequential([
                        layers.Input(shape=(784,)),
                        *[layers.Dense(n, activation=params["activation"]) for n in params["hidden_layer_sizes"]],
                        layers.Dense(10, activation='softmax')
                    ])
                    optimizer = tf.keras.optimizers.Adam(params["learning_rate"]) if params["solver"] == "adam" else tf.keras.optimizers.SGD(params["learning_rate"])
                    model_1percent.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
                    model_1percent.fit(X_train_initial, y_train_initial, epochs=params["epochs"], batch_size=params["batch_size"], verbose=0)
                    acc_test = accuracy_score(y_test, np.argmax(model_1percent.predict(X_test, verbose=0), axis=1))
                    st.session_state['model_1percent'] = model_1percent
                    st.session_state['model_1percent_accuracy'] = acc_test
                    st.success(f"Độ chính xác: {acc_test*100:.2f}%")
                    gc.collect()

            if 'model_1percent_accuracy' in st.session_state:
                st.write(f"Độ chính xác của mô hình 1%: {st.session_state['model_1percent_accuracy']*100:.2f}%")

            # Chọn mô hình
            model_options = {"Pseudo-Labeling": st.session_state.get('model'), "1% Data": st.session_state.get('model_1percent')}
            selected_model_name = st.selectbox("Chọn mô hình:", [k for k, v in model_options.items() if v is not None])
            model = model_options[selected_model_name]

            if model:
                input_method = st.selectbox("Phương thức nhập liệu", ["Tải ảnh", "Dữ liệu Test", "Vẽ"])
                if input_method == "Tải ảnh":
                    uploaded_file = st.file_uploader("Tải lên hình ảnh", type=["png", "jpg"])
                    if uploaded_file and st.button("Dự đoán"):
                        image = Image.open(uploaded_file).convert('L').resize((28, 28))
                        st.image(image, width=100)
                        image_array = np.array(image, dtype=np.float32).reshape(1, 784) / 255.0
                        prediction = model.predict(image_array, verbose=0)[0]
                        predicted_class = np.argmax(prediction)
                        confidence = prediction[predicted_class] * 100
                        st.write(f"Dự đoán: {predicted_class}, Độ tin cậy: {confidence:.2f}%")
                        fig, ax = plt.subplots()
                        ax.bar(range(10), prediction * 100)
                        st.pyplot(fig)
                        plt.close(fig)

                elif input_method == "Dữ liệu Test":
                    X_test = st.session_state['split_data']["X_test"]
                    y_test = st.session_state['split_data']["y_test"]
                    idx = st.slider("Chọn mẫu", 0, len(X_test) - 1, 0)
                    st.image(X_test[idx].reshape(28, 28), caption=f"Nhãn thực: {y_test[idx]}", width=100)
                    if st.button("Dự đoán"):
                        sample = X_test[idx].reshape(1, -1) / 255.0
                        prediction = model.predict(sample, verbose=0)[0]
                        predicted_class = np.argmax(prediction)
                        confidence = prediction[predicted_class] * 100
                        st.write(f"Dự đoán: {predicted_class}, Độ tin cậy: {confidence:.2f}%")
                        fig, ax = plt.subplots()
                        ax.bar(range(10), prediction * 100)
                        st.pyplot(fig)
                        plt.close(fig)

                elif input_method == "Vẽ":
                    canvas_result = st_canvas(
                        stroke_width=20,
                        stroke_color="#FFFFFF",
                        background_color="#000000",
                        height=280,
                        width=280,
                        drawing_mode="freedraw",
                        key="canvas"
                    )
                    if canvas_result.image_data is not None and st.button("Dự đoán"):
                        image = Image.fromarray(canvas_result.image_data.astype('uint8'), 'RGBA').convert('L').resize((28, 28))
                        image_array = np.array(image, dtype=np.float32).reshape(1, 784) / 255.0
                        prediction = model.predict(image_array, verbose=0)[0]
                        predicted_class = np.argmax(prediction)
                        confidence = prediction[predicted_class] * 100
                        st.write(f"Dự đoán: {predicted_class}, Độ tin cậy: {confidence:.2f}%")
                        fig, ax = plt.subplots()
                        ax.bar(range(10), prediction * 100)
                        st.pyplot(fig)
                        plt.close(fig)

    ### Tab 7: Thông tin huấn luyện
    with tab_log_info:
        st.markdown('<div class="section-title">Theo dõi Kết quả</div>', unsafe_allow_html=True)
        runs = client.search_runs(experiment_ids=[EXPERIMENT_ID], order_by=["attributes.start_time DESC"])
        if not runs:
            st.info("Chưa có lần chạy nào.")
        else:
            run_options = {run.info.run_id: run.data.tags.get('mlflow.runName', f"Run_{run.info.run_id}") for run in runs}
            selected_run_name = st.selectbox("Chọn run:", list(run_options.values()))
            selected_run_id = [k for k, v in run_options.items() if v == selected_run_name][0]
            selected_run = client.get_run(selected_run_id)

            st.write(f"Tên: {selected_run_name}, ID: {selected_run_id}")
            st.json(selected_run.data.params)
            st.json(selected_run.data.metrics)

if __name__ == "__main__":
    run_mnist_pseudo_labeling_app()