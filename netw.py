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
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
import gc
import keras_tuner as kt

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

    EXPERIMENT_ID = "5"
    client = MlflowClient()

    # Tải dữ liệu MNIST
    if 'full_data' not in st.session_state:
        (X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()
        X_full = np.concatenate([X_train, X_test], axis=0)
        y_full = np.concatenate([y_train, y_test], axis=0)
        X_full = X_full.reshape(-1, 784).astype(np.float32)
        y_full = y_full.astype(np.int32)
        st.session_state['full_data'] = (X_full, y_full)

    st.title("Phân loại Chữ số MNIST với Neural Network")

    # CSS tùy chỉnh
    st.markdown("""
        <style>
            .section-title { font-size: 1.5em; font-weight: bold; color: #2c3e50; margin-bottom: 10px; }
            .prediction-box { margin-top: 10px; padding: 10px; border: 1px solid #ccc; border-radius: 5px; background-color: #f9f9f9; }
            .mode-title { font-size: 1.2em; font-weight: bold; color: #2c3e50; margin-bottom: 10px; }
        </style>
    """, unsafe_allow_html=True)

    tabs = st.tabs(["Thông tin", "Chọn dữ liệu", "Xử lý dữ liệu", "Chia dữ liệu", "Huấn luyện/Đánh giá", "Demo dự đoán", "Thông tin huấn luyện"])

    # Tab 1: Thông tin
    with tabs[0]:
        st.header("Giới thiệu")
        st.write("Ứng dụng này phân loại chữ số viết tay từ tập dữ liệu MNIST bằng Neural Network.")

    # Tab 2: Chọn dữ liệu
    with tabs[1]:
        st.markdown('<div class="section-title">Chọn Số lượng Dữ liệu</div>', unsafe_allow_html=True)
        X_full, y_full = st.session_state['full_data']
        num_samples = st.number_input("Số lượng mẫu:", min_value=1, max_value=len(X_full), value=1000)
        if st.button("Xác nhận"):
            indices = np.random.choice(len(X_full), size=num_samples, replace=False)
            st.session_state['data'] = (X_full[indices].copy(), y_full[indices].copy())
            st.session_state['optimal_params'] = get_optimal_params(num_samples)
            st.success(f"Đã chọn {num_samples} mẫu!")
            gc.collect()

    # Tab 3: Xử lý dữ liệu
    with tabs[2]:
        st.markdown('<div class="section-title">Xử lý Dữ liệu</div>', unsafe_allow_html=True)
        if 'data' not in st.session_state:
            st.info("Chọn dữ liệu trước.")
        else:
            X, y = st.session_state['data']
            if st.button("Chuẩn hóa dữ liệu"):
                X_norm = X / 255.0
                st.session_state["data_processed"] = (X_norm.copy(), y.copy())
                st.success("Đã chuẩn hóa dữ liệu!")
                gc.collect()

    # Tab 4: Chia dữ liệu
    with tabs[3]:
        st.markdown('<div class="section-title">Chia Tập Dữ liệu</div>', unsafe_allow_html=True)
        if 'data' not in st.session_state:
            st.info("Chọn và xử lý dữ liệu trước.")
        else:
            data_source = st.session_state.get('data_processed', st.session_state['data'])
            X, y = data_source
            test_pct = st.slider("Tỷ lệ Test (%)", 0, 50, 20)
            valid_pct = st.slider("Tỷ lệ Validation (%)", 0, 50, 20)
            test_size = test_pct / 100
            X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
            valid_size = (valid_pct / 100) / (1 - test_size) if test_size < 1 else 0
            X_train, X_valid, y_train, y_valid = train_test_split(X_temp, y_temp, test_size=valid_size, random_state=42)
            if st.button("Xác nhận phân chia"):
                st.session_state['split_data'] = {
                    "X_train": X_train.copy(), "y_train": y_train.copy(),
                    "X_valid": X_valid.copy(), "y_valid": y_valid.copy(),
                    "X_test": X_test.copy(), "y_test": y_test.copy()
                }
                st.success("Đã chia dữ liệu!")
                gc.collect()

    # Tab 5: Huấn luyện/Đánh giá
    with tabs[4]:
        st.markdown('<div class="section-title">Huấn luyện và Đánh giá</div>', unsafe_allow_html=True)
        if 'split_data' not in st.session_state:
            st.info("Chia dữ liệu trước.")
        else:
            split_data = st.session_state['split_data']
            X_train, y_train = split_data["X_train"], split_data["y_train"]
            X_valid, y_valid = split_data["X_valid"], split_data["y_valid"]
            X_test, y_test = split_data["X_test"], split_data["y_test"]

            num_samples = len(X_train)
            params = st.session_state.get("training_params", get_optimal_params(num_samples))

            st.subheader("Cấu hình Mô hình")
            num_hidden_layers = st.number_input("Số lớp ẩn", min_value=1, value=len(params["hidden_layer_sizes"]))
            hidden_sizes = [st.number_input(f"Số nơ-ron lớp ẩn {i+1}", min_value=1, value=params["hidden_layer_sizes"][i] if i < len(params["hidden_layer_sizes"]) else 32) for i in range(num_hidden_layers)]
            params["hidden_layer_sizes"] = tuple(hidden_sizes)
            params["learning_rate"] = st.number_input("Tốc độ học", min_value=0.00001, max_value=1.0, value=float(params["learning_rate"]))
            params["epochs"] = st.number_input("Số lần lặp", min_value=1, value=params["epochs"])
            params["batch_size"] = st.number_input("Kích thước batch", min_value=1, value=params["batch_size"])
            st.session_state["training_params"] = params

            if st.button("Bắt đầu Huấn luyện"):
                st.write("### Xác nhận Tham số")
                st.write(f"- Số lớp ẩn: {len(params['hidden_layer_sizes'])}")
                st.write(f"- Số nơ-ron: {params['hidden_layer_sizes']}")
                st.write(f"- Tốc độ học: {params['learning_rate']}")
                st.write(f"- Số lần lặp: {params['epochs']}")
                st.write(f"- Kích thước batch: {params['batch_size']}")
                if st.button("Xác nhận và Huấn luyện"):
                    with st.spinner("Đang huấn luyện..."):
                        model = models.Sequential([
                            layers.Input(shape=(784,)),
                            *[layers.Dense(neurons, activation="relu") for neurons in params["hidden_layer_sizes"]],
                            layers.Dense(10, activation='softmax')
                        ])
                        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=params["learning_rate"]),
                                      loss='sparse_categorical_crossentropy', metrics=['accuracy'])
                        history = model.fit(X_train, y_train, epochs=params["epochs"], batch_size=params["batch_size"],
                                            validation_data=(X_valid, y_valid), verbose=0)
                        y_valid_pred = np.argmax(model.predict(X_valid, verbose=0), axis=1)
                        y_test_pred = np.argmax(model.predict(X_test, verbose=0), axis=1)
                        acc_valid = accuracy_score(y_valid, y_valid_pred)
                        acc_test = accuracy_score(y_test, y_test_pred)

                        # Kiểm tra overfitting
                        train_acc = history.history['accuracy'][-1]
                        val_acc = history.history['val_accuracy'][-1]
                        if train_acc - val_acc > 0.1:
                            st.warning("Cảnh báo: Mô hình có dấu hiệu overfitting.")

                        st.session_state['model'] = model
                        st.session_state['training_results'] = {
                            'accuracy_val': acc_valid, 'accuracy_test': acc_test,
                            'loss_history': history.history['loss'],
                            'val_accuracy_history': history.history['val_accuracy']
                        }
                        st.success("Huấn luyện xong!")
                        tf.keras.backend.clear_session()
                        del model, history
                        gc.collect()

            # Huấn luyện AutoML
            if st.button("Huấn luyện AutoML"):
                with st.spinner("Đang tìm tham số tối ưu..."):
                    def build_model(hp):
                        model = models.Sequential([
                            layers.Input(shape=(784,)),
                            *[layers.Dense(hp.Int(f'units_{i}', 32, 128, step=32), activation='relu') for i in range(hp.Int('num_layers', 1, 3))],
                            layers.Dense(10, activation='softmax')
                        ])
                        model.compile(optimizer=tf.keras.optimizers.Adam(hp.Float('learning_rate', 1e-4, 1e-2, sampling='log')),
                                      loss='sparse_categorical_crossentropy', metrics=['accuracy'])
                        return model
                    tuner = kt.Hyperband(build_model, objective='val_accuracy', max_epochs=10)
                    tuner.search(X_train, y_train, epochs=10, validation_data=(X_valid, y_valid))
                    best_model = tuner.get_best_models(num_models=1)[0]
                    st.session_state['model'] = best_model
                    st.success("Huấn luyện AutoML xong!")

    # Tab 6: Demo dự đoán
    with tabs[5]:
        st.markdown('<div class="section-title">Demo Dự đoán</div>', unsafe_allow_html=True)
        if 'model' not in st.session_state:
            st.info("Huấn luyện mô hình trước.")
        else:
            model = st.session_state['model']
            input_method = st.selectbox("Chọn phương thức nhập liệu", ["Tải ảnh lên", "Vẽ trực tiếp"])
            is_normalized = 'data_processed' in st.session_state

            def preprocess_input(data):
                return data / 255.0 if not is_normalized else data

            if input_method == "Tải ảnh lên":
                st.write("**Hướng dẫn**: Tải lên ảnh chữ số (28x28, nền đen, nét trắng).")
                uploaded_file = st.file_uploader("Tải ảnh", type=["png", "jpg"])
                if uploaded_file:
                    image = Image.open(uploaded_file).convert('L').resize((28, 28))
                    st.image(image, caption="Ảnh sau resize (28x28)", width=100)
                    if st.button("Dự đoán"):
                        image_array = np.array(image, dtype=np.float32).reshape(1, 784)
                        if np.sum(image_array > 50) < 50:
                            st.warning("Ảnh không chứa chữ số. Thử lại!")
                        else:
                            prediction = model.predict(preprocess_input(image_array), verbose=0)[0]
                            predicted_class = np.argmax(prediction)
                            st.markdown(f'<div class="prediction-box">Dự đoán: {predicted_class}</div>', unsafe_allow_html=True)
                            gc.collect()

            elif input_method == "Vẽ trực tiếp":
                st.write("**Hướng dẫn**: Vẽ chữ số (nét trắng trên nền đen).")
                canvas_result = st_canvas(
                    stroke_width=20,
                    stroke_color="#FFFFFF",
                    background_color="#000000",
                    height=280,
                    width=280,
                    drawing_mode="freedraw",
                    key="canvas"
                )
                if canvas_result.image_data is not None:
                    image = Image.fromarray(canvas_result.image_data.astype('uint8'), 'RGBA').convert('L').resize((28, 28))
                    st.image(image, caption="Ảnh sau resize (28x28)", width=100)
                    if st.button("Dự đoán"):
                        image_array = np.array(image).reshape(1, 784)
                        if np.sum(image_array > 50) < 50:
                            st.warning("Hình vẽ không chứa chữ số. Thử lại!")
                        else:
                            prediction = model.predict(preprocess_input(image_array), verbose=0)[0]
                            predicted_class = np.argmax(prediction)
                            st.markdown(f'<div class="prediction-box">Dự đoán: {predicted_class}</div>', unsafe_allow_html=True)
                            gc.collect()

    # Tab 7: Thông tin huấn luyện
    with tabs[6]:
        st.markdown('<div class="section-title">Thông tin Huấn luyện</div>', unsafe_allow_html=True)
        runs = client.search_runs(experiment_ids=[EXPERIMENT_ID], order_by=["attributes.start_time DESC"])
        if runs and 'training_results' in st.session_state:
            run_options = {run.info.run_id: run.data.tags.get('mlflow.runName', run.info.run_id) for run in runs}
            selected_runs = st.multiselect("Chọn run để so sánh", list(run_options.values()))
            if selected_runs:
                acc_tests = []
                run_names = []
                for run_id in [k for k, v in run_options.items() if v in selected_runs]:
                    run = client.get_run(run_id)
                    acc_test = run.data.metrics.get('accuracy_test', st.session_state['training_results']['accuracy_test'])
                    acc_tests.append(acc_test * 100)
                    run_names.append(run.data.tags.get('mlflow.runName', run_id))
                fig, ax = plt.subplots()
                ax.bar(run_names, acc_tests)
                ax.set_ylabel("Độ chính xác Test (%)")
                ax.set_title("So sánh các mô hình")
                st.pyplot(fig)
                plt.close(fig)

if __name__ == "__main__":
    run_mnist_neural_network_app()