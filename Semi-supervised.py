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

# Hàm tiền xử lý ảnh: căn giữa và làm rõ nét vẽ
def preprocess_image(image):
    image = np.array(image, dtype=np.float32)
    image = np.where(image > 127, 255, 0)
    return image / 255.0  # Chuẩn hóa ngay trong hàm này

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

    EXPERIMENT_ID = "5"
    client = MlflowClient()
    experiment = client.get_experiment(EXPERIMENT_ID)
    if experiment is None:
        st.error(f"Experiment ID {EXPERIMENT_ID} không tồn tại.")
        st.stop()

    st.title("Pseudo Labeling với Neural Network trên MNIST")

    # CSS tùy chỉnh
    st.markdown("""
        <style>
            .section-title { font-size: 1.5em; font-weight: bold; color: #2c3e50; margin-bottom: 10px; }
            .info-box { background-color: #f8f9fa; padding: 10px; border-left: 4px solid #3498db; margin-bottom: 15px; }
            .prediction-box { margin-top: 10px; padding: 10px; border: 1px solid #ccc; border-radius: 5px; background-color: #f9f9f9; }
            .mode-title { font-size: 1.2em; font-weight: bold; color: #2c3e50; margin-bottom: 10px; }
        </style>
    """, unsafe_allow_html=True)

    tabs = st.tabs(["Thông tin", "Tải dữ liệu", "Chia dữ liệu", "Pseudo Labeling", "Demo dự đoán", "Thông tin huấn luyện"])
    tab_info, tab_load, tab_split, tab_pseudo, tab_demo, tab_log_info = tabs

    # Tab 1: Thông tin
    with tab_info:
        st.header("Giới thiệu về Pseudo Labeling")
        st.markdown("""
        Ứng dụng này triển khai thuật toán **Pseudo Labeling** với Neural Network trên tập dữ liệu **MNIST**.  
        **Pseudo Labeling** là một phương pháp học bán giám sát (semi-supervised learning) nhằm tận dụng dữ liệu chưa gán nhãn bằng cách:
        1. Huấn luyện mô hình trên một tập dữ liệu nhỏ có nhãn (1% mỗi lớp).
        2. Dự đoán nhãn giả (pseudo labels) cho dữ liệu chưa gán nhãn.
        3. Gán nhãn giả cho các mẫu có độ tin cậy cao (threshold ≥ 0.95).
        4. Lặp lại quá trình để mở rộng tập dữ liệu có nhãn.
        """, unsafe_allow_html=True)

        # Thêm hình ảnh minh họa có tên "labelding"
        st.subheader("📊 Quy trình Pseudo Labeling")
        st.markdown("""
        Dưới đây là hình ảnh minh họa quy trình **Pseudo Labeling**:
        """, unsafe_allow_html=True)
        try:
            # Giả sử hình ảnh có tên "labelding.png" (thêm đuôi file nếu cần)
            pseudo_image = Image.open("labelding.png")
            st.image(pseudo_image, caption="Quy trình Pseudo Labeling", use_column_width=True)
        except FileNotFoundError:
            st.error("Không tìm thấy file `labelding.png`. Vui lòng kiểm tra đường dẫn hoặc tên file.")
        except Exception as e:
            st.error(f"Lỗi khi tải ảnh: {e}")

        # Thêm chú thích chi tiết cho từng bước trong hình ảnh
        st.markdown("""
        **Giải thích các bước trong hình ảnh:**
        - **Bước 1: Huấn luyện với dữ liệu có nhãn**  
          Một tập dữ liệu nhỏ (màu xanh) được sử dụng để huấn luyện một **Initial Neural Network**. Đây là tập dữ liệu labeled ban đầu (1% mỗi lớp từ 0-9).  
        - **Bước 2: Dự đoán nhãn giả cho dữ liệu chưa gán nhãn**  
          Mô hình đã huấn luyện được sử dụng để dự đoán nhãn cho tập dữ liệu chưa gán nhãn (màu xám). Các mẫu có độ tin cậy cao (threshold ≥ 0.95) được gán nhãn giả (màu cam).  
        - **Bước 3: Huấn luyện lại với dữ liệu có nhãn và nhãn giả**  
          Tập dữ liệu mới (bao gồm dữ liệu labeled ban đầu màu xanh và dữ liệu vừa được gán nhãn giả màu cam) được sử dụng để huấn luyện một **New Neural Network**. Quá trình này lặp lại cho đến khi gán hết nhãn hoặc đạt số vòng lặp tối đa.
        """, unsafe_allow_html=True)

    # Tab 2: Tải dữ liệu
    with tab_load:
        st.markdown('<div class="section-title">Tải Dữ liệu</div>', unsafe_allow_html=True)
        if 'full_data' not in st.session_state:
            if st.button("Tải dữ liệu MNIST", type="primary"):
                with st.spinner("Đang tải dữ liệu MNIST..."):
                    (X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()
                    X = np.concatenate([X_train, X_test], axis=0)
                    y = np.concatenate([y_train, y_test], axis=0)
                    X = X.reshape(-1, 784).astype(np.float32) / 255.0  # Chuẩn hóa ngay lúc tải
                    y = y.astype(np.int32)
                    st.session_state['full_data'] = (X, y)
                    st.success(f"Đã tải dữ liệu: {X.shape[0]} mẫu, {X.shape[1]} đặc trưng")
                    st.rerun()
        else:
            st.success("Dữ liệu đã được tải!")

    # Tab 3: Chia dữ liệu
    with tab_split:
        st.markdown('<div class="section-title">Chia Tập Dữ liệu</div>', unsafe_allow_html=True)
        if 'full_data' not in st.session_state:
            st.info("Vui lòng tải dữ liệu trước.")
        else:
            X_full, y_full = st.session_state['full_data']
            test_size = st.slider("Tỷ lệ Test (%)", 0, 50, 20, help="Tỷ lệ dữ liệu dùng để kiểm tra")
            if st.button("Chia dữ liệu", type="primary"):
                with st.spinner("Đang chia dữ liệu..."):
                    X_train_full, X_test, y_train_full, y_test = train_test_split(
                        X_full, y_full, test_size=test_size / 100, stratify=y_full, random_state=42
                    )
                    # Lấy 1% dữ liệu có nhãn cho mỗi lớp
                    X_labeled, y_labeled = [], []
                    X_unlabeled, y_unlabeled = [], []
                    for digit in range(10):
                        digit_indices = np.where(y_train_full == digit)[0]
                        num_labeled = max(1, int(0.01 * len(digit_indices)))  # 1% mỗi lớp
                        labeled_indices = np.random.choice(digit_indices, num_labeled, replace=False)
                        unlabeled_indices = np.setdiff1d(digit_indices, labeled_indices)
                        X_labeled.append(X_train_full[labeled_indices])
                        y_labeled.append(y_train_full[labeled_indices])
                        X_unlabeled.append(X_train_full[unlabeled_indices])
                        y_unlabeled.append(y_train_full[unlabeled_indices])
                    X_labeled = np.concatenate(X_labeled)
                    y_labeled = np.concatenate(y_labeled)
                    X_unlabeled = np.concatenate(X_unlabeled)
                    y_unlabeled = np.concatenate(y_unlabeled)

                    st.session_state['split_data'] = {
                        "X_labeled": X_labeled, "y_labeled": y_labeled,
                        "X_unlabeled": X_unlabeled, "y_unlabeled": y_unlabeled,
                        "X_test": X_test, "y_test": y_test
                    }
                    st.success(f"Đã chia dữ liệu: Labeled: {len(X_labeled)}, Unlabeled: {len(X_unlabeled)}, Test: {len(X_test)}")
                    del X_full, y_full, X_train_full, y_train_full
                    gc.collect()

    # Tab 4: Pseudo Labeling
    with tab_pseudo:
        st.markdown('<div class="section-title">Pseudo Labeling</div>', unsafe_allow_html=True)
        if 'split_data' not in st.session_state:
            st.info("Vui lòng chia dữ liệu trước.")
        else:
            split_data = st.session_state['split_data']
            X_labeled = split_data["X_labeled"].copy()
            y_labeled = split_data["y_labeled"].copy()
            X_unlabeled = split_data["X_unlabeled"].copy()
            X_test = split_data["X_test"].copy()
            y_test = split_data["y_test"].copy()

            num_samples = len(X_labeled)
            params = get_optimal_params(num_samples)

            st.subheader("Cấu hình Pseudo Labeling")
            threshold = st.slider("Ngưỡng gán nhãn giả (Threshold)", 0.5, 1.0, 0.95, step=0.01)
            max_iterations = st.number_input("Số lần lặp tối đa", 1, 10, 5)
            col_param1, col_param2 = st.columns(2)
            with col_param1:
                num_hidden_layers = st.number_input("Số lớp ẩn", 1, 3, len(params["hidden_layer_sizes"]))
                hidden_sizes = list(params["hidden_layer_sizes"])
                for i in range(num_hidden_layers):
                    hidden_sizes[i] = st.number_input(f"Số nơ-ron lớp {i+1}", 16, 128, hidden_sizes[i])
                params["hidden_layer_sizes"] = tuple(hidden_sizes[:num_hidden_layers])
            with col_param2:
                params["learning_rate"] = st.selectbox("Tốc độ học", [0.01, 0.001, 0.0001], index=1)
                params["epochs"] = st.number_input("Số epoch", 10, 100, params["epochs"])
                params["batch_size"] = st.number_input("Kích thước batch", 32, 256, params["batch_size"])

            if st.button("Bắt đầu Pseudo Labeling", type="primary"):
                with st.spinner("Đang thực hiện Pseudo Labeling..."):
                    iteration = 0
                    accuracy_history = []  # Lưu lịch sử độ chính xác
                    num_labeled_history = [len(X_labeled)]  # Lưu số lượng mẫu labeled qua từng vòng lặp

                    while len(X_unlabeled) > 0 and iteration < max_iterations:
                        iteration += 1
                        st.write(f"**Vòng lặp {iteration}**: Huấn luyện với {len(X_labeled)} mẫu có nhãn")

                        # Huấn luyện mô hình
                        model = models.Sequential([
                            layers.Input(shape=(784,)),
                            *[layers.Dense(n, activation="relu") for n in params["hidden_layer_sizes"]],
                            layers.Dense(10, activation="softmax")
                        ])
                        model.compile(optimizer=tf.keras.optimizers.Adam(params["learning_rate"]),
                                    loss="sparse_categorical_crossentropy",
                                    metrics=["accuracy"])
                        history = model.fit(X_labeled, y_labeled, epochs=params["epochs"],
                                          batch_size=params["batch_size"], verbose=0)

                        # Dự đoán nhãn giả
                        pseudo_probs = model.predict(X_unlabeled, verbose=0)
                        pseudo_labels = np.argmax(pseudo_probs, axis=1)
                        max_probs = np.max(pseudo_probs, axis=1)

                        # Gán nhãn giả cho các mẫu vượt ngưỡng
                        confident_mask = max_probs >= threshold
                        X_confident = X_unlabeled[confident_mask]
                        y_confident = pseudo_labels[confident_mask]

                        # Cập nhật tập dữ liệu
                        X_labeled = np.concatenate([X_labeled, X_confident])
                        y_labeled = np.concatenate([y_labeled, y_confident])
                        X_unlabeled = X_unlabeled[~confident_mask]

                        # Đánh giá trên tập test
                        y_test_pred = np.argmax(model.predict(X_test, verbose=0), axis=1)
                        acc_test = accuracy_score(y_test, y_test_pred)

                        # Lưu lịch sử
                        accuracy_history.append(acc_test)
                        num_labeled_history.append(len(X_labeled))

                        with mlflow.start_run(experiment_id=EXPERIMENT_ID, run_name=f"Pseudo_Iter_{iteration}"):
                            mlflow.log_params(params)
                            mlflow.log_metric("num_labeled", len(X_labeled))
                            mlflow.log_metric("accuracy_test", acc_test)
                            mlflow.log_metric("iteration", iteration)

                        st.write(f"Đã gán nhãn cho {len(X_confident)} mẫu. Độ chính xác Test: {acc_test:.4f}")

                    # Lưu mô hình cuối cùng
                    st.session_state['model'] = model
                    st.session_state['training_results'] = {
                        "X_labeled": X_labeled, "y_labeled": y_labeled,
                        "accuracy_test": acc_test,
                        "iterations": iteration,
                        "accuracy_history": accuracy_history,
                        "num_labeled_history": num_labeled_history
                    }
                    st.success(f"Hoàn tất sau {iteration} vòng lặp! Số mẫu được gán nhãn: {len(X_labeled)}")

                    # Vẽ biểu đồ độ chính xác qua các vòng lặp
                    st.subheader("📈 Kết quả Pseudo Labeling")
                    fig, ax1 = plt.subplots(figsize=(8, 4))
                    ax1.plot(range(1, len(accuracy_history) + 1), accuracy_history, 
                            label='Test Accuracy', color='blue', marker='o')
                    ax1.set_xlabel("Vòng lặp")
                    ax1.set_ylabel("Độ chính xác Test", color='blue')
                    ax1.tick_params(axis='y', labelcolor='blue')
                    ax1.grid(True)

                    # Thêm trục thứ hai để vẽ số lượng mẫu labeled
                    ax2 = ax1.twinx()
                    ax2.plot(range(0, len(num_labeled_history)), num_labeled_history, 
                            label='Số mẫu labeled', color='orange', marker='x')
                    ax2.set_ylabel("Số mẫu labeled", color='orange')
                    ax2.tick_params(axis='y', labelcolor='orange')

                    fig.tight_layout()
                    fig.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=2)
                    st.pyplot(fig)
                    plt.close(fig)

    # Tab 5: Demo dự đoán
    with tab_demo:
        st.markdown('<div class="section-title">Demo Dự đoán</div>', unsafe_allow_html=True)
        if 'model' not in st.session_state:
            st.warning("Vui lòng thực hiện Pseudo Labeling trước!")
        else:
            model = st.session_state['model']
            input_method = st.selectbox("Chọn phương thức nhập liệu", ["Tải ảnh lên", "Dữ liệu Test", "Vẽ trực tiếp"])

            if input_method == "Tải ảnh lên":
                uploaded_file = st.file_uploader("Tải lên hình ảnh", type=["png", "jpg"])
                if uploaded_file:
                    image = Image.open(uploaded_file).convert('L').resize((28, 28))
                    image_array = preprocess_image(image).reshape(1, 784)
                    if st.button("Dự đoán"):
                        pred = model.predict(image_array, verbose=0)
                        predicted_class = np.argmax(pred[0])
                        confidence = pred[0][predicted_class] * 100
                        st.markdown(f"**Dự đoán:** {predicted_class}, **Độ tin cậy:** {confidence:.2f}%")

            elif input_method == "Dữ liệu Test":
                X_test = st.session_state['split_data']["X_test"]
                y_test = st.session_state['split_data']["y_test"]
                idx = st.slider("Chọn mẫu Test", 0, len(X_test) - 1, 0)
                st.image(X_test[idx].reshape(28, 28), width=100)
                if st.button("Dự đoán"):
                    pred = model.predict(X_test[idx].reshape(1, -1), verbose=0)
                    predicted_class = np.argmax(pred[0])
                    confidence = pred[0][predicted_class] * 100
                    st.markdown(f"**Dự đoán:** {predicted_class}, **Độ tin cậy:** {confidence:.2f}%, **Thực tế:** {y_test[idx]}")

            elif input_method == "Vẽ trực tiếp":
                canvas_result = st_canvas(stroke_width=20, stroke_color="#FFFFFF", background_color="#000000", height=280, width=280)
                if canvas_result.image_data is not None:
                    image = Image.fromarray(canvas_result.image_data.astype('uint8'), 'RGBA').convert('L').resize((28, 28))
                    image_array = preprocess_image(image).reshape(1, 784)
                    st.image(image_array.reshape(28, 28), width=100)
                    if st.button("Dự đoán"):
                        pred = model.predict(image_array, verbose=0)
                        predicted_class = np.argmax(pred[0])
                        confidence = pred[0][predicted_class] * 100
                        st.markdown(f"**Dự đoán:** {predicted_class}, **Độ tin cậy:** {confidence:.2f}%")

    # Tab 6: Thông tin huấn luyện
    with tab_log_info:
        st.markdown('<div class="section-title">Thông tin Huấn luyện</div>', unsafe_allow_html=True)
        runs = client.search_runs(experiment_ids=[EXPERIMENT_ID], order_by=["attributes.start_time DESC"])
        if not runs:
            st.info("Chưa có lần chạy nào.")
        else:
            run_id = st.selectbox("Chọn run:", [run.info.run_id for run in runs])
            run = client.get_run(run_id)
            st.write(f"**Tên:** {run.data.tags.get('mlflow.runName', run_id)}")
            st.write(f"**Tham số:** {run.data.params}")
            st.write(f"**Số liệu:** {run.data.metrics}")

if __name__ == "__main__":
    run_mnist_pseudo_labeling_app()