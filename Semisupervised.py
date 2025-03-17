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
    else:  # > 50,000
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

    tabs = st.tabs(["Thông tin", "Tải dữ liệu", "Xử lý dữ liệu", "Chia dữ liệu", "Huấn luyện/Đánh giá", "Demo dự đoán", "Thông tin huấn luyện"])
    tab_info, tab_load, tab_preprocess, tab_split, tab_train_eval, tab_demo, tab_log_info = tabs

    # Tab 1: Thông tin
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
                st.subheader("📊 Neural Network – Mạng nơ-ron nhân tạo")
                st.markdown("""
                **Neural Network (Mạng nơ-ron nhân tạo)** là một mô hình học máy mô phỏng cách hoạt động của mạng nơ-ron sinh học trong não người. Nó được thiết kế để học các đặc trưng phức tạp từ dữ liệu, đặc biệt hiệu quả với bài toán nhận diện hình ảnh như MNIST.

                **Cấu trúc cơ bản:**
                - **Lớp đầu vào (Input Layer)**: Nhận dữ liệu thô (784 pixel từ ảnh MNIST).
                - **Lớp ẩn (Hidden Layers)**: Xử lý thông tin qua các phép tính tuyến tính và phi tuyến.
                - **Lớp đầu ra (Output Layer)**: Đưa ra dự đoán (10 lớp, từ 0-9).

                **Quy trình hoạt động:**
                - **Lan truyền thuận**: Tính toán dự đoán từ đầu vào qua các lớp.
                - **Tính hàm mất mát**: Đo độ sai lệch giữa dự đoán và nhãn thực tế.
                - **Lan truyền ngược**: Cập nhật trọng số để giảm mất mát.

                **Ưu điểm:**
                - Học được các đặc trưng phức tạp từ dữ liệu hình ảnh.
                - Linh hoạt với nhiều tham số để tối ưu hóa.

                **Nhược điểm:**
                - Tốn thời gian huấn luyện với dữ liệu lớn.
                - Yêu cầu điều chỉnh tham số cẩn thận.
                """, unsafe_allow_html=True)
                status_text.text("Đã tải xong! 100%")
                time.sleep(0.5)
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
                **Pseudo-Labeling** là một phương pháp học bán giám sát giúp tận dụng cả dữ liệu có nhãn và không có nhãn để nâng cao hiệu suất mô hình, đặc biệt khi dữ liệu có nhãn khan hiếm. Kỹ thuật này sử dụng mô hình đã huấn luyện để dự đoán nhãn giả (pseudo-labels) cho dữ liệu không có nhãn, sau đó kết hợp chúng vào quá trình huấn luyện.

                **Các bước thực hiện:**
                1. Lấy 1% dữ liệu có nhãn ban đầu từ mỗi lớp (0-9).
                2. Huấn luyện mô hình Neural Network trên tập dữ liệu ban đầu.
                3. Dự đoán nhãn cho dữ liệu không có nhãn.
                4. Gán nhãn giả cho các dự đoán có độ tin cậy cao (threshold mặc định = 0.95).
                5. Huấn luyện lại mô hình với tập dữ liệu mới (bao gồm nhãn thật và nhãn giả).
                6. Lặp lại các bước 3-5 cho đến khi đạt số vòng lặp tối đa hoặc không còn dữ liệu không nhãn.

                **Lợi ích:**
                - Tối ưu hóa hiệu suất khi dữ liệu có nhãn hạn chế.
                - Giảm chi phí gắn nhãn thủ công.

                **Thách thức:**
                - Nhãn giả có thể chứa nhiễu nếu mô hình ban đầu không chính xác.
                - Yêu cầu điều chỉnh ngưỡng tin cậy hợp lý để cân bằng chất lượng và số lượng nhãn giả.
                """, unsafe_allow_html=True)

                st.subheader("⚙️ Các Tham số Sử dụng trong Huấn luyện với Pseudo-Labeling")
                st.markdown("""
                Trong quá trình huấn luyện với Pseudo-Labeling, các tham số sau được sử dụng để tối ưu hóa mô hình Neural Network. Các tham số này có thể được tùy chỉnh trong tab **Huấn luyện/Đánh giá**:

                - **Số lớp ẩn (Number of Hidden Layers)**: 1-3 (mặc định: dựa vào số mẫu).  
                - **Số nơ-ron mỗi lớp (Hidden Layer Sizes)**: 16-128 (mặc định: dựa vào số mẫu).  
                - **Tốc độ học (Learning Rate)**: 0.01-0.0001 (mặc định: dựa vào số mẫu).  
                - **Số lần lặp mỗi vòng (Epochs)**: 10-100 (mặc định: dựa vào số mẫu).  
                - **Kích thước batch (Batch Size)**: 32-256 (mặc định: dựa vào số mẫu).  
                - **Hàm kích hoạt (Activation Function)**: ReLU, Sigmoid, Tanh (mặc định: ReLU).  
                - **Trình tối ưu (Solver)**: Adam, SGD (mặc định: Adam).  
                - **Ngưỡng tin cậy (Confidence Threshold)**: 0.5-1.0 (mặc định: 0.95).  
                - **Số vòng lặp tối đa (Max Iterations)**: 1-10 (mặc định: 5).  
                """, unsafe_allow_html=True)

                st.subheader("⭐ Gợi ý Tham số Tối ưu Nhất cho Bài toán MNIST")
                st.markdown("""
                Dựa trên đặc điểm của tập dữ liệu MNIST (70,000 mẫu, 10 lớp, ảnh 28x28), tham số tối ưu được đề xuất là cấu hình cho dữ liệu lớn (> 50,000 mẫu) để đạt hiệu suất cao nhất với Pseudo-Labeling:
                - **Số lớp ẩn**: 3  
                - **Số nơ-ron mỗi lớp**: (128, 64, 32)  
                - **Tốc độ học**: 0.0001  
                - **Số lần lặp mỗi vòng**: 100  
                - **Kích thước batch**: 256  
                - **Hàm kích hoạt**: ReLU  
                - **Trình tối ưu**: Adam  
                - **Ngưỡng tin cậy**: 0.95  
                - **Số vòng lặp tối đa**: 5  

                **Lý do chọn:**
                - MNIST có số lượng mẫu lớn và đặc trưng phức tạp, cần mạng sâu (3 lớp ẩn) với số nơ-ron giảm dần để học tốt các đặc trưng.
                - Tốc độ học nhỏ (0.0001) và số lần lặp lớn (100) đảm bảo hội tụ ổn định.
                - Ngưỡng 0.95 đảm bảo nhãn giả có độ tin cậy cao, giảm nhiễu.
                """, unsafe_allow_html=True)

                if 'data' in st.session_state:
                    X, _ = st.session_state['data']
                    num_samples = len(X)
                    optimal_params = get_optimal_params(num_samples)
                    st.markdown(f"""
                    **Tham số tối ưu tự động chọn cho {num_samples} mẫu đã tải:**
                    - **Số lớp ẩn**: {len(optimal_params['hidden_layer_sizes'])}  
                    - **Số nơ-ron mỗi lớp**: {optimal_params['hidden_layer_sizes']}  
                    - **Tốc độ học**: {optimal_params['learning_rate']}  
                    - **Số lần lặp mỗi vòng**: {optimal_params['epochs']}  
                    - **Kích thước batch**: {optimal_params['batch_size']}  
                    - **Hàm kích hoạt**: {optimal_params['activation']}  
                    - **Trình tối ưu**: {optimal_params['solver']}  
                    - **Ngưỡng tin cậy**: 0.95 (khuyến nghị)  
                    - **Số vòng lặp tối đa**: 5 (khuyến nghị)  
                    """, unsafe_allow_html=True)
                else:
                    st.info("Vui lòng chọn số lượng mẫu trong tab 'Tải dữ liệu' để xem tham số tối ưu tự động.")

                status_text.text("Đã tải xong! 100%")
                time.sleep(0.5)
                status_text.empty()
                progress_bar.empty()

    # Tab 2: Tải dữ liệu
    with tab_load:
        st.markdown('<div class="section-title">Tải Dữ liệu</div>', unsafe_allow_html=True)

        if 'full_data' not in st.session_state:
            if st.button("Tải dữ liệu MNIST", type="primary"):
                with st.spinner("Đang tải dữ liệu MNIST..."):
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    try:
                        (X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()
                        for i in range(0, 101, 20):
                            progress_bar.progress(i)
                            status_text.text(f"Đang tải dữ liệu... {i}%")
                            time.sleep(0.1)
                        X = np.concatenate([X_train, X_test], axis=0)
                        y = np.concatenate([y_train, y_test], axis=0)
                        X = X.reshape(-1, 784).astype(np.float64)
                        y = y.astype(np.int32)
                        st.session_state['full_data'] = (X, y)
                        progress_bar.progress(100)
                        status_text.text("Đã tải xong! 100%")
                        st.success("Đã tải dữ liệu thành công!")
                        st.write(f"Kích thước dữ liệu: {X.shape[0]} mẫu, mỗi mẫu {X.shape[1]} đặc trưng")
                        time.sleep(0.5)
                        status_text.empty()
                        progress_bar.empty()
                        st.rerun()
                    except Exception as e:
                        st.error(f"Lỗi khi tải dữ liệu: {e}")
        else:
            X_full, y_full = st.session_state['full_data']
            st.subheader("Chọn số lượng mẫu")
            col1, col_center, col2 = st.columns([2, 1, 2])
            with col1:
                sample_options = {
                    "1000 mẫu (Thử nghiệm nhanh)": 1000,
                    "10,000 mẫu (Kiểm tra cơ bản)": 10000,
                    "50,000 mẫu (Cân bằng hiệu suất)": 50000,
                    "70,000 mẫu (Huấn luyện chuyên sâu)": 70000
                }
                selected_option = st.selectbox("Chọn số lượng mẫu:", list(sample_options.keys()))
                num_samples = min(sample_options[selected_option], len(X_full))

                if st.button("Xác nhận số lượng (tùy chọn có sẵn)", type="primary"):
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

            with col_center:
                st.markdown("<h3 style='text-align: center; margin-top: 30px;'>hoặc</h3>", unsafe_allow_html=True)

            with col2:
                custom_num_samples = st.number_input("Nhập số lượng tùy ý (tối đa 70,000):", min_value=1, max_value=70000, value=1000, step=100)
                if st.button("Xác nhận số lượng (tùy ý)", type="primary"):
                    if custom_num_samples <= len(X_full):
                        with st.spinner(f"Đang lấy {custom_num_samples} mẫu..."):
                            indices = np.random.choice(len(X_full), size=custom_num_samples, replace=False)
                            X_sampled = X_full[indices]
                            y_sampled = y_full[indices]
                            st.session_state['data'] = (X_sampled.copy(), y_sampled.copy())
                            st.session_state['optimal_params'] = get_optimal_params(custom_num_samples)
                            with mlflow.start_run(experiment_id=EXPERIMENT_ID, run_name="Data_Sample_Custom"):
                                mlflow.log_param("num_samples", custom_num_samples)
                            st.success(f"Đã chọn {custom_num_samples} mẫu!")
                            del X_sampled, y_sampled
                            gc.collect()
                    else:
                        st.error("Số lượng mẫu vượt quá dữ liệu hiện có!")

    # Tab 3: Xử lý dữ liệu
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
                st.success("Đã xử lý dữ liệu!")

    # Tab 4: Chia dữ liệu
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

            # Bước 1: Lấy 1% số lượng ảnh cho mỗi class (0-9)
            st.subheader("Bước 1: Tạo tập dữ liệu ban đầu (1% mỗi lớp)")
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

            st.write(f"**Tập dữ liệu ban đầu (1%)**: {len(X_train_initial)} mẫu")
            st.write(f"**Tập dữ liệu chưa gắn nhãn (99%)**: {len(X_unlabeled)} mẫu")

            st.session_state['pseudo_data'] = {
                'X_train_initial': X_train_initial.copy(),
                'y_train_initial': y_train_initial.copy(),
                'X_unlabeled': X_unlabeled.copy(),
                'y_unlabeled_indices': y_unlabeled_indices,
                'X_test': X_test.copy(),
                'y_test': y_test.copy()
            }

            if "optimal_params" not in st.session_state:
                st.session_state["optimal_params"] = get_optimal_params(len(X_train_initial))
            
            params = st.session_state.get("training_params", st.session_state["optimal_params"].copy())

            st.subheader("⚙️ Cấu hình Tham số Mô hình")
            st.info(f"Tham số tối ưu tự động cho {len(X_train_initial)} mẫu: {st.session_state['optimal_params']}")

            col_param1, col_param2 = st.columns(2)
            with col_param1:
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
                params["learning_rate"] = st.selectbox("Tốc độ học", [0.01, 0.005, 0.001, 0.0005, 0.0003, 0.0001], 
                                                       index=[0.01, 0.005, 0.001, 0.0005, 0.0003, 0.0001].index(params["learning_rate"]))
                params["epochs"] = st.number_input("Số lần lặp (Epochs)", min_value=10, max_value=100, value=params["epochs"])
                params["batch_size"] = st.number_input("Kích thước batch", min_value=32, max_value=256, value=params["batch_size"])
                params["solver"] = st.selectbox("Trình tối ưu", ["adam", "sgd"], index=["adam", "sgd"].index(params["solver"]))
                threshold = st.slider("Ngưỡng tin cậy Pseudo-Label", 0.5, 1.0, 0.95)
                max_iterations = st.number_input("Số vòng lặp tối đa", min_value=1, max_value=10, value=5)

            col_reset, col_train = st.columns([1, 3])
            with col_reset:
                if st.button("🔄 Khôi phục tham số tối ưu"):
                    st.session_state["training_params"] = st.session_state["optimal_params"].copy()
                    st.success("Đã khôi phục tham số tối ưu!")
                    st.rerun()

            st.session_state["training_params"] = params

            with col_train:
                if st.button("🚀 Bắt đầu Huấn luyện với Pseudo-Labeling", type="primary"):
                    with st.spinner("Đang thực hiện quy trình Pseudo-Labeling..."):
                        start_time = time.time()

                        X_train = st.session_state['pseudo_data']['X_train_initial'].copy()
                        y_train = st.session_state['pseudo_data']['y_train_initial'].copy()
                        X_unlabeled = st.session_state['pseudo_data']['X_unlabeled'].copy()
                        X_test = st.session_state['pseudo_data']['X_test'].copy()
                        y_test = st.session_state['pseudo_data']['y_test'].copy()

                        iteration = 0
                        pseudo_labeled_history = []
                        accuracy_test_history = []

                        while iteration < max_iterations and len(X_unlabeled) > 0:
                            iteration += 1
                            st.write(f"**Vòng lặp {iteration}/{max_iterations}**")

                            model = models.Sequential()
                            model.add(layers.Input(shape=(784,)))
                            for neurons in params["hidden_layer_sizes"]:
                                model.add(layers.Dense(neurons, activation=params["activation"]))
                            model.add(layers.Dense(10, activation='softmax'))

                            optimizer = tf.keras.optimizers.Adam(learning_rate=params["learning_rate"]) if params["solver"] == "adam" else tf.keras.optimizers.SGD(learning_rate=params["learning_rate"])
                            model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

                            progress_bar = st.progress(0)
                            status_text = st.empty()

                            class ProgressCallback(callbacks.Callback):
                                def on_epoch_end(self, epoch, logs=None):
                                    progress = (epoch + 1) / params["epochs"] * 100
                                    progress_bar.progress(int(progress))
                                    status_text.text(f"Epoch {epoch+1}/{params['epochs']}, Loss: {logs['loss']:.4f}, Accuracy: {logs['accuracy']:.4f}")

                            history = model.fit(X_train, y_train, epochs=params["epochs"], batch_size=params["batch_size"], callbacks=[ProgressCallback()], verbose=0)

                            predictions = model.predict(X_unlabeled, verbose=0)
                            predicted_labels = np.argmax(predictions, axis=1)
                            confidences = np.max(predictions, axis=1)

                            pseudo_mask = confidences >= threshold
                            X_pseudo = X_unlabeled[pseudo_mask]
                            y_pseudo = predicted_labels[pseudo_mask]

                            st.write(f"**Số mẫu được gán nhãn giả trong vòng {iteration}**: {len(X_pseudo)}")

                            X_train = np.concatenate([X_train, X_pseudo], axis=0)
                            y_train = np.concatenate([y_train, y_pseudo], axis=0)
                            X_unlabeled = X_unlabeled[~pseudo_mask]

                            pseudo_labeled_history.append(len(X_pseudo))
                            y_test_pred = np.argmax(model.predict(X_test, verbose=0), axis=1)
                            acc_test = accuracy_score(y_test, y_test_pred)
                            accuracy_test_history.append(acc_test)
                            st.write(f"**Độ chính xác trên tập Test sau vòng {iteration}**: {acc_test*100:.2f}%")

                            tf.keras.backend.clear_session()
                            del model, predictions, predicted_labels, confidences, pseudo_mask, X_pseudo, y_pseudo
                            gc.collect()

                        st.write("**Huấn luyện lần cuối trên toàn bộ tập dữ liệu đã gắn nhãn**")
                        model = models.Sequential()
                        model.add(layers.Input(shape=(784,)))
                        for neurons in params["hidden_layer_sizes"]:
                            model.add(layers.Dense(neurons, activation=params["activation"]))
                        model.add(layers.Dense(10, activation='softmax'))

                        optimizer = tf.keras.optimizers.Adam(learning_rate=params["learning_rate"]) if params["solver"] == "adam" else tf.keras.optimizers.SGD(learning_rate=params["learning_rate"])
                        model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

                        history = model.fit(X_train, y_train, epochs=params["epochs"], batch_size=params["batch_size"], callbacks=[ProgressCallback()], verbose=0)

                        y_test_pred = np.argmax(model.predict(X_test, verbose=0), axis=1)
                        acc_test = accuracy_score(y_test, y_test_pred)
                        cm_test = confusion_matrix(y_test, y_test_pred)

                        run_name = f"PseudoLabeling_NN_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                        with mlflow.start_run(experiment_id=EXPERIMENT_ID, run_name=run_name) as run:
                            mlflow.log_params(params)
                            mlflow.log_param("threshold", threshold)
                            mlflow.log_param("max_iterations", max_iterations)
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

                        st.success(f"Đã hoàn thành Pseudo-Labeling! Thời gian: {time.time() - start_time:.2f} giây")
                        tf.keras.backend.clear_session()
                        del X_train, y_train, X_unlabeled, X_test, y_test, history
                        gc.collect()
                        st.rerun()

            if 'training_results' in st.session_state:
                results = st.session_state['training_results']
                st.subheader("📊 Kết quả Huấn luyện")
                col_result1, col_result2, col_result3 = st.columns(3)
                with col_result1:
                    st.metric("Thời gian huấn luyện", f"{results['training_time']:.2f} giây")
                with col_result2:
                    st.metric("Độ chính xác Test", f"{results['accuracy_test']*100:.2f}%")
                with col_result3:
                    st.metric("Tổng số vòng lặp", f"{results['total_iterations']}")

                st.subheader("📈 Ma trận Nhầm lẫn")
                fig, ax = plt.subplots(figsize=(6, 5))
                sns.heatmap(results['cm_test'], annot=True, fmt="d", cmap="Blues", ax=ax)
                st.pyplot(fig)
                plt.close(fig)

                st.subheader("📉 Biểu đồ Đánh giá")
                col_chart1, col_chart2 = st.columns(2)
                with col_chart1:
                    if results['loss_history']:
                        fig, ax = plt.subplots(figsize=(5, 3))
                        ax.plot(range(1, len(results['loss_history']) + 1), results['loss_history'], label='Loss', color='blue')
                        ax.set_xlabel("Epochs")
                        ax.set_ylabel("Loss")
                        ax.set_title("Loss (10 Epochs Cuối)")
                        ax.grid(True)
                        st.pyplot(fig)
                        plt.close(fig)
                        st.markdown("*Biểu đồ Loss*: Giảm dần cho thấy mô hình học tốt, hội tụ ổn định.")
                    
                    if results['pseudo_labeled_history']:
                        fig, ax = plt.subplots(figsize=(5, 3))
                        ax.plot(range(1, len(results['pseudo_labeled_history']) + 1), results['pseudo_labeled_history'], label='Pseudo-Labeled', color='purple')
                        ax.set_xlabel("Iterations")
                        ax.set_ylabel("Số mẫu")
                        ax.set_title("Số mẫu Pseudo-Labeled")
                        ax.grid(True)
                        st.pyplot(fig)
                        plt.close(fig)
                        st.markdown("*Số mẫu Pseudo-Labeled*: Thể hiện lượng dữ liệu không nhãn được gán qua từng vòng.")

                with col_chart2:
                    if results['accuracy_history']:
                        fig, ax = plt.subplots(figsize=(5, 3))
                        ax.plot(range(1, len(results['accuracy_history']) + 1), results['accuracy_history'], label='Accuracy', color='green')
                        ax.set_xlabel("Epochs")
                        ax.set_ylabel("Accuracy")
                        ax.set_title("Accuracy (10 Epochs Cuối)")
                        ax.grid(True)
                        st.pyplot(fig)
                        plt.close(fig)
                        st.markdown("*Biểu đồ Accuracy*: Tăng dần cho thấy khả năng phân loại cải thiện.")
                    
                    if results['accuracy_test_history']:
                        fig, ax = plt.subplots(figsize=(5, 3))
                        ax.plot(range(1, len(results['accuracy_test_history']) + 1), results['accuracy_test_history'], label='Test Accuracy', color='red')
                        ax.set_xlabel("Iterations")
                        ax.set_ylabel("Accuracy")
                        ax.set_title("Test Accuracy Qua Vòng Lặp")
                        ax.grid(True)
                        st.pyplot(fig)
                        plt.close(fig)
                        st.markdown("*Test Accuracy*: Đánh giá hiệu suất thực tế trên tập kiểm tra.")

    # Tab 6: Demo dự đoán
    with tab_demo:
        st.markdown('<div class="section-title">Demo Dự đoán Chữ số</div>', unsafe_allow_html=True)

        if 'split_data' not in st.session_state or 'model' not in st.session_state:
            st.warning("Vui lòng huấn luyện mô hình trước!")
        else:
            model = st.session_state['model']
            input_method = st.selectbox("Chọn phương thức nhập liệu", ["Tải ảnh lên", "Dữ liệu Test", "Vẽ trực tiếp"])
            is_normalized = 'data_processed' in st.session_state

            def preprocess_input(data, is_normalized):
                if not is_normalized:
                    data = data / 255.0
                return data

            if input_method == "Tải ảnh lên":
                uploaded_file = st.file_uploader("Tải lên hình ảnh", type=["png", "jpg", "jpeg"])
                if uploaded_file is not None:
                    image = Image.open(uploaded_file).convert('L').resize((28, 28))
                    st.image(image, caption="Hình ảnh đầu vào", width=100)
                    if st.button("Dự đoán"):
                        image_array = np.array(image, dtype=np.float32).reshape(1, 784)
                        image_processed = preprocess_input(image_array, is_normalized)
                        prediction = model.predict(image_processed, verbose=0)
                        predicted_class = np.argmax(prediction[0])
                        confidence = prediction[0][predicted_class] * 100
                        st.write(f"Dự đoán: {predicted_class}, Độ tin cậy: {confidence:.2f}%")

            elif input_method == "Dữ liệu Test":
                X_test = st.session_state['split_data']["X_test"]
                y_test = st.session_state['split_data']["y_test"]
                idx = st.slider("Chọn mẫu Test", 0, len(X_test) - 1, 0)
                st.image(X_test[idx].reshape(28, 28), caption=f"Nhãn thực tế: {y_test[idx]}", width=100)
                if st.button("Dự đoán"):
                    sample = X_test[idx].reshape(1, -1)
                    sample_processed = preprocess_input(sample, is_normalized)
                    prediction = model.predict(sample_processed, verbose=0)
                    predicted_class = np.argmax(prediction[0])
                    confidence = prediction[0][predicted_class] * 100
                    st.write(f"Dự đoán: {predicted_class}, Độ tin cậy: {confidence:.2f}%")

            elif input_method == "Vẽ trực tiếp":
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
                    st.image(image, caption="Hình vẽ", width=100)
                    if st.button("Dự đoán"):
                        image_array = np.array(image, dtype=np.float32).reshape(1, 784)
                        image_processed = preprocess_input(image_array, is_normalized)
                        prediction = model.predict(image_processed, verbose=0)
                        predicted_class = np.argmax(prediction[0])
                        confidence = prediction[0][predicted_class] * 100
                        st.write(f"Dự đoán: {predicted_class}, Độ tin cậy: {confidence:.2f}%")

    # Tab 7: Thông tin huấn luyện
    with tab_log_info:
        st.markdown('<div class="section-title">Thông tin Huấn luyện</div>', unsafe_allow_html=True)
        if 'training_results' in st.session_state:
            results = st.session_state['training_results']
            st.write(f"Tên lần chạy: {results['run_name']}")
            st.write(f"ID lần chạy: {results['run_id']}")
            st.write(f"Thời gian huấn luyện: {results['training_time']:.2f} giây")
            st.write(f"Độ chính xác Test: {results['accuracy_test']*100:.2f}%")
            st.write("Tham số:", results['params'])
        else:
            st.info("Chưa có thông tin huấn luyện. Vui lòng huấn luyện mô hình trước.")

if __name__ == "__main__":
    run_mnist_pseudo_labeling_app()