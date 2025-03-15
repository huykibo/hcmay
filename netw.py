import os
import mlflow
import streamlit as st
import openml
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.neural_network import MLPClassifier
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from mlflow.tracking import MlflowClient
from streamlit_drawable_canvas import st_canvas
from datetime import datetime
import time
import requests

# Hàm tải dữ liệu MNIST
def fetch_mnist_data():
    mnist = openml.datasets.get_dataset(554)
    X, y, _, _ = mnist.get_data(target=mnist.default_target_attribute)
    return X, y

# Hàm kiểm tra và chuẩn hóa dữ liệu pixel về [0, 255] (dùng cho các tab khác)
def validate_and_fix_pixels(X, name="dữ liệu"):
    invalid_mask = (X < 0) | (X > 255)
    if np.any(invalid_mask):
        st.warning(f"Phát hiện giá trị pixel không hợp lệ trong {name} (ngoài [0, 255]). Đang chuẩn hóa...")
        X_fixed = np.clip(X, 0, 255)
        return X_fixed, True
    return X, False

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

    try:
        response = requests.get(mlflow_tracking_uri, timeout=5)
        if response.status_code != 200:
            st.error(f"Kết nối MLflow thất bại. Mã trạng thái: {response.status_code}.")
            st.stop()
    except requests.exceptions.RequestException as e:
        st.error(f"Không thể kết nối MLflow: {e}.")
        st.stop()

    EXPERIMENT_ID = "5"

    try:
        client = MlflowClient()
        experiment = client.get_experiment(EXPERIMENT_ID)
        if experiment is None:
            st.error(f"Experiment ID {EXPERIMENT_ID} không tồn tại.")
            st.stop()
    except Exception as e:
        st.error(f"Lỗi truy xuất Experiment ID {EXPERIMENT_ID}: {e}.")
        st.stop()

    st.title("Phân loại Chữ số MNIST với Neural Network")

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

    tabs = st.tabs(["Thông tin", "Tải dữ liệu", "Xử lý dữ liệu", "Chia dữ liệu", "Huấn luyện/Đánh giá", "Demo dự đoán", "Thông tin huấn luyện"])
    tab_info, tab_load, tab_preprocess, tab_split, tab_train_eval, tab_demo, tab_log_info = tabs

    with tab_info:
        st.header("Giới thiệu về Ứng dụng và Mạng Neural Network")
        st.markdown("""
        Đây là ứng dụng phân loại chữ số viết tay từ **MNIST** bằng **Neural Network**.
        """, unsafe_allow_html=True)

        st.subheader("Chọn thông tin để xem")
        info_option = st.selectbox("", ["Ứng dụng này là gì và mục tiêu của nó?", "Tập dữ liệu MNIST: Đặc điểm và ý nghĩa", 
                                        "Neural Network – Mạng nơ-ron nhân tạo", "Công thức đánh giá độ chính xác (Accuracy)"],
                                   label_visibility="collapsed")
        
        if info_option == "Ứng dụng này là gì và mục tiêu của nó?":
            with st.spinner("Đang tải thông tin..."):
                progress_bar = st.progress(0)
                status_text = st.empty()
                for i in [20, 40, 60, 80, 100]:
                    progress_bar.progress(i)
                    status_text.text(f"Đang tải {i}%")
                    time.sleep(0.05)
                st.subheader("📘 1. Ứng dụng này là gì và mục tiêu của nó?")
                st.markdown("""
                Ứng dụng phân loại chữ số viết tay từ **MNIST** bằng **Neural Network**.  
                - **MNIST**: $70,000$ ảnh chữ số (0-9), mỗi ảnh $28 \\times 28$ pixel ($784$ đặc trưng).  
                - **Mục tiêu**: Nhận diện chính xác chữ số và cung cấp công cụ học tập trực quan.  
                """, unsafe_allow_html=True)
                status_text.empty()
                progress_bar.empty()
        
        elif info_option == "Tập dữ liệu MNIST: Đặc điểm và ý nghĩa":
            st.subheader("📊 2. Tập dữ liệu MNIST: Đặc điểm và ý nghĩa")
            st.markdown("""
            - **Nguồn gốc**: MNIST (Modified National Institute of Standards and Technology) là tập dữ liệu chuẩn trong học máy.  
            - **Cấu trúc**: $60,000$ mẫu huấn luyện + $10,000$ mẫu kiểm tra, mỗi mẫu là ảnh thang độ xám $28 \\times 28$.  
            - **Ý nghĩa**: Được sử dụng rộng rãi để kiểm tra hiệu suất các thuật toán phân loại hình ảnh.  
            """, unsafe_allow_html=True)
        
        elif info_option == "Neural Network – Mạng nơ-ron nhân tạo":
            st.subheader("🧠 3. Neural Network – Mạng nơ-ron nhân tạo")
            st.markdown("""
            - **Khái niệm**: Mô hình học máy mô phỏng não người, gồm các lớp nơ-ron (input, hidden, output).  
            - **Ứng dụng trong MNIST**: Nhận diện chữ số qua các lớp ẩn xử lý đặc trưng pixel.  
            - **Tham số chính**: Số lớp ẩn, số nơ-ron, tốc độ học, hàm kích hoạt (ReLU, sigmoid,...).  
            """, unsafe_allow_html=True)
        
        elif info_option == "Công thức đánh giá độ chính xác (Accuracy)":
            st.subheader("📈 4. Công thức đánh giá độ chính xác (Accuracy)")
            st.markdown("""
            Độ chính xác được tính bằng:  
            $$ \\text{Accuracy} = \\frac{\\text{Số mẫu dự đoán đúng}}{\\text{Tổng số mẫu}} \\times 100\\% $$  
            - **Ý nghĩa**: Đo lường tỷ lệ dự đoán chính xác của mô hình trên tập kiểm tra hoặc validation.  
            """, unsafe_allow_html=True)

    with tab_load:
        st.header("Tải Dữ liệu")

        if st.button("Tải dữ liệu MNIST từ OpenML"):
            with st.spinner("Đang tải dữ liệu..."):
                progress_bar = st.progress(0)
                status_text = st.empty()
                for i in [20, 40, 60, 80, 100]:
                    progress_bar.progress(i)
                    status_text.text(f"Đang tải {i}%")
                    time.sleep(0.05)
                try:
                    X, y = fetch_mnist_data()
                    st.session_state['full_data'] = (X, y)
                    with mlflow.start_run(experiment_id=EXPERIMENT_ID, run_name="Data_Load"):
                        mlflow.log_param("total_samples", X.shape[0])
                    st.success("Tải dữ liệu thành công!")
                    st.write("Kích thước dữ liệu gốc:", X.shape)
                    status_text.empty()
                    progress_bar.empty()
                except Exception as e:
                    st.error(f"Không thể tải dữ liệu: {e}")
                    status_text.empty()
                    progress_bar.empty()

        if 'full_data' in st.session_state:
            X_full, y_full = st.session_state['full_data']
            
            st.subheader("Chọn số lượng mẫu dữ liệu")
            st.markdown("""
            Dựa trên bài toán phân loại MNIST với Neural Network, đây là các gợi ý:
            - **100 mẫu**: Dành cho thử nghiệm nhanh, thời gian huấn luyện rất ngắn (~vài giây), nhưng độ chính xác thấp.
            - **1,000 mẫu**: Phù hợp để kiểm tra mô hình cơ bản, thời gian huấn luyện ngắn (~10-20 giây), độ chính xác trung bình.
            - **10,000 mẫu**: Cân bằng giữa tốc độ và hiệu suất, thời gian huấn luyện vừa phải (~1-2 phút), độ chính xác khá tốt.
            - **50,000 mẫu**: Dành cho huấn luyện chuyên sâu, thời gian lâu hơn (~5-10 phút), độ chính xác cao.
            """)
            
            sample_options = {
                "100 mẫu (Thử nghiệm nhanh)": 100,
                "1,000 mẫu (Kiểm tra cơ bản)": 1000,
                "10,000 mẫu (Cân bằng hiệu suất)": 10000,
                "50,000 mẫu (Huấn luyện chuyên sâu)": 50000
            }
            selected_option = st.selectbox("Chọn số lượng mẫu:", list(sample_options.keys()))
            num_samples = sample_options[selected_option]

            if st.button("Chốt số lượng mẫu"):
                with st.spinner(f"Đang lấy {num_samples} mẫu..."):
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    for i in [20, 40, 60, 80, 100]:
                        progress_bar.progress(i)
                        status_text.text(f"Đang xử lý {i}%")
                        time.sleep(0.05)
                    indices = np.random.choice(len(X_full), size=num_samples, replace=False)
                    X_sampled = X_full.iloc[indices]
                    y_sampled = y_full.iloc[indices]
                    st.session_state['data'] = (X_sampled, y_sampled)
                    with mlflow.start_run(experiment_id=EXPERIMENT_ID, run_name="Data_Sample"):
                        mlflow.log_param("num_samples", num_samples)
                    st.success(f"Đã chốt {num_samples} mẫu!")
                    status_text.empty()
                    progress_bar.empty()

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
                    with st.spinner("Đang chuẩn hóa dữ liệu về [0, 1]..."):
                        progress_bar = st.progress(0)
                        status_text = st.empty()
                        for i in [20, 40, 60, 80, 100]:
                            progress_bar.progress(i)
                            status_text.text(f"Đang xử lý {i}%")
                            time.sleep(0.05)
                        X_norm = X / 255.0
                        st.session_state["data_processed"] = (X_norm, y)
                        st.success("Đã chuẩn hóa dữ liệu về [0, 1]!")
                        status_text.empty()
                        progress_bar.empty()
                        st.rerun()
            with col2:
                st.markdown("""
                    <div class="tooltip">? (Norm)
                        <span class="tooltiptext">
                            Đưa dữ liệu về [0, 1] bằng cách chia cho 255.<br>
                            Công dụng: Đảm bảo thang đo đồng nhất cho Neural Network.
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

    with tab_split:
        st.header("Chia Tập Dữ liệu")

        if 'data' not in st.session_state:
            st.info("Vui lòng tải và xử lý dữ liệu trước.")
        else:
            data_source = st.session_state.get('data_processed', st.session_state['data'])
            X, y = data_source
            total_samples = len(X)
            st.write(f"Tổng số mẫu: {total_samples}")

            test_pct = st.slider("Tỷ lệ Test (%)", 0, 50, 20)
            valid_pct = st.slider("Tỷ lệ Validation (%)", 0, 50, 20)

            test_size = test_pct / 100
            X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
            valid_size = (valid_pct / 100) / (1 - test_size) if test_size < 1 else 0
            X_train, X_valid, y_train, y_valid = train_test_split(X_temp, y_temp, test_size=valid_size, random_state=42)

            st.write(f"Train: {len(X_train)}, Validation: {len(X_valid)}, Test: {len(X_test)}")
            if st.button("Xác nhận", key="confirm_split_button"):
                with st.spinner("Đang chia dữ liệu..."):
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    for i in [20, 40, 60, 80, 100]:
                        progress_bar.progress(i)
                        status_text.text(f"Đang xử lý {i}%")
                        time.sleep(0.05)
                    st.session_state['split_data'] = {
                        "X_train": X_train, "y_train": y_train,
                        "X_valid": X_valid, "y_valid": y_valid,
                        "X_test": X_test, "y_test": y_test
                    }
                    st.success("Đã chia dữ liệu!")
                    status_text.empty()
                    progress_bar.empty()

    with tab_train_eval:
        st.header("Huấn luyện và Đánh giá Mô hình")

        if 'split_data' not in st.session_state:
            st.info("Vui lòng chia dữ liệu trước.")
        else:
            X_train = st.session_state['split_data']["X_train"]
            y_train = st.session_state['split_data']["y_train"]
            X_valid = st.session_state['split_data']["X_valid"]
            y_valid = st.session_state['split_data']["y_valid"]
            X_test = st.session_state['split_data']["X_test"]
            y_test = st.session_state['split_data']["y_test"]

            num_samples = len(X_train)
            st.write(f"**Số mẫu huấn luyện**: {num_samples}")

            # Hàm tối ưu tham số
            def get_optimal_params(num_samples):
                if num_samples < 1000:
                    return {"hidden_layer_sizes": (16,), "learning_rate_init": 0.01, "max_iter": 30, 
                            "activation": "relu", "solver": "adam", "batch_size": 64}
                elif 1000 <= num_samples < 5000:
                    return {"hidden_layer_sizes": (32,), "learning_rate_init": 0.005, "max_iter": 50, 
                            "activation": "relu", "solver": "adam", "batch_size": 128}
                elif 5000 <= num_samples <= 20000:
                    return {"hidden_layer_sizes": (64, 32), "learning_rate_init": 0.001, "max_iter": 75, 
                            "activation": "relu", "solver": "adam", "batch_size": 256}
                else:
                    return {"hidden_layer_sizes": (128, 64), "learning_rate_init": 0.0005, "max_iter": 100, 
                            "activation": "relu", "solver": "adam", "batch_size": 512}

            if "optimal_params" not in st.session_state:
                st.session_state["optimal_params"] = get_optimal_params(num_samples)
            params = st.session_state.get("training_params", st.session_state["optimal_params"].copy())

            st.subheader("⚙️ Cấu hình tham số mô hình")
            st.markdown("""
            | Số mẫu       | Số lớp ẩn | Kích thước lớp ẩn | Tốc độ học | Số lần lặp | Hàm kích hoạt | Trình tối ưu | Kích thước batch |
            |--------------|-----------|-------------------|------------|------------|---------------|--------------|------------------|
            | <1000        | 1         | 16                | 0.01       | 30         | ReLU          | adam         | 64               |
            | 1000-5000    | 1         | 32                | 0.005      | 50         | ReLU          | adam         | 128              |
            | 5000-20000   | 2         | (64, 32)          | 0.001      | 75         | ReLU          | adam         | 256              |
            | >20000       | 2         | (128, 64)         | 0.0005     | 100        | ReLU          | adam         | 512              |
            """, unsafe_allow_html=True)

            st.info(f"Tham số tối ưu cho {num_samples} mẫu: {st.session_state['optimal_params']}")

            col_param1, col_param2 = st.columns(2)
            with col_param1:
                with st.expander("Cấu trúc mạng"):
                    num_hidden_layers = st.number_input("Số lớp ẩn", min_value=1, max_value=2, value=len(params["hidden_layer_sizes"]))
                    hidden_size = st.number_input("Số nơ-ron mỗi lớp", min_value=16, max_value=128, value=params["hidden_layer_sizes"][0])
                    params["hidden_layer_sizes"] = tuple([hidden_size] * num_hidden_layers)
                    params["activation"] = st.selectbox("Hàm kích hoạt", ["relu", "sigmoid", "tanh"], 
                                                        index=["relu", "sigmoid", "tanh"].index(params["activation"]))
            with col_param2:
                with st.expander("Tối ưu hóa"):
                    params["learning_rate_init"] = st.selectbox("Tốc độ học", [0.01, 0.005, 0.001, 0.0005], 
                                                                index=[0.01, 0.005, 0.001, 0.0005].index(params["learning_rate_init"]))
                    params["max_iter"] = st.number_input("Số lần lặp", min_value=10, max_value=100, value=params["max_iter"])
                    params["batch_size"] = st.number_input("Kích thước batch", min_value=64, max_value=512, value=params["batch_size"])
                    params["solver"] = st.selectbox("Trình tối ưu", ["adam", "sgd", "lbfgs"], 
                                                    index=["adam", "sgd", "lbfgs"].index(params["solver"]))

            if st.button("🔄 Khôi phục tham số tối ưu"):
                st.session_state["training_params"] = st.session_state["optimal_params"].copy()
                st.success("Đã khôi phục tham số tối ưu!")
                st.rerun()

            st.session_state["training_params"] = params

            if st.button("🚀 Bắt đầu Huấn luyện", type="primary"):
                try:
                    with st.spinner("Đang huấn luyện mô hình..."):
                        progress_bar = st.progress(0)
                        status_text = st.empty()
                        start_time = time.time()

                        status_text.text("Đang chuẩn bị dữ liệu...")
                        progress_bar.progress(20)

                        model = MLPClassifier(**params, verbose=True)
                        status_text.text("Đang huấn luyện mô hình...")
                        for i in [40, 60, 80]:
                            progress_bar.progress(i)
                            status_text.text(f"Đang huấn luyện {i}%")
                            time.sleep(0.05)
                        model.fit(X_train, y_train)

                        status_text.text("Đang đánh giá mô hình...")
                        progress_bar.progress(90)
                        y_valid_pred = model.predict(X_valid)
                        y_test_pred = model.predict(X_test)
                        acc_valid = accuracy_score(y_valid, y_valid_pred)
                        acc_test = accuracy_score(y_test, y_test_pred)
                        cm_valid = confusion_matrix(y_valid, y_valid_pred)
                        cm_test = confusion_matrix(y_test, y_test_pred)

                        status_text.text("Đang lưu kết quả...")
                        progress_bar.progress(100)
                        run_name = f"NeuralNetwork_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                        with mlflow.start_run(experiment_id=EXPERIMENT_ID, run_name=run_name) as run:
                            mlflow.log_params(params)
                            mlflow.log_metric("accuracy_val", acc_valid)
                            mlflow.log_metric("accuracy_test", acc_test)
                            mlflow.log_metric("training_time", time.time() - start_time)

                            st.session_state['model'] = model
                            st.session_state['training_results'] = {
                                'accuracy_val': acc_valid, 'accuracy_test': acc_test,
                                'cm_valid': cm_valid, 'cm_test': cm_test,
                                'run_name': run_name, 'run_id': run.info.run_id,
                                'params': params, 'training_time': time.time() - start_time
                            }

                        st.success(f"Đã huấn luyện xong! Thời gian: {time.time() - start_time:.2f} giây")
                        status_text.empty()
                        progress_bar.empty()
                        st.rerun()

                except Exception as e:
                    st.error(f"Lỗi trong quá trình huấn luyện: {e}")
                    status_text.empty()
                    progress_bar.empty()

            if 'training_results' in st.session_state:
                results = st.session_state['training_results']
                st.subheader("📊 Kết quả Huấn luyện")
                col_result1, col_result2, col_result3 = st.columns(3)
                with col_result1:
                    st.metric("Thời gian huấn luyện", f"{results['training_time']:.2f} giây")
                with col_result2:
                    st.metric("Độ chính xác Validation", f"{results['accuracy_val']*100:.2f}%")
                with col_result3:
                    st.metric("Độ chính xác Test", f"{results['accuracy_test']*100:.2f}%")

                st.subheader("📈 Ma trận Nhầm lẫn")
                col_cm1, col_cm2 = st.columns(2)
                with col_cm1:
                    fig, ax = plt.subplots(figsize=(6, 5))
                    sns.heatmap(results['cm_valid'], annot=True, fmt="d", cmap="Blues", ax=ax)
                    ax.set_title("Validation")
                    st.pyplot(fig)
                with col_cm2:
                    fig, ax = plt.subplots(figsize=(6, 5))
                    sns.heatmap(results['cm_test'], annot=True, fmt="d", cmap="Blues", ax=ax)
                    ax.set_title("Test")
                    st.pyplot(fig)
                
                st.subheader("ℹ️ Thông tin Chi tiết")
                with st.expander("Xem chi tiết", expanded=False):
                    st.markdown("**Thông tin lần chạy:**")
                    st.write(f"- Tên: {results['run_name']}")
                    st.write(f"- ID: {results['run_id']}")
                    st.write(f"- Thời gian huấn luyện: {results['training_time']:.2f} giây")
                    st.write(f"- Độ chính xác Validation: {results['accuracy_val']*100:.2f}%")
                    st.write(f"- Độ chính xác Test: {results['accuracy_test']*100:.2f}%")
                    st.markdown("**Tham số đã chọn:**")
                    st.json({
                        "Số lớp ẩn": len(results['params']['hidden_layer_sizes']),
                        "Số nơ-ron mỗi lớp": results['params']['hidden_layer_sizes'],
                        "Tốc độ học": results['params']['learning_rate_init'],
                        "Số lần lặp": results['params']['max_iter'],
                        "Kích thước batch": results['params']['batch_size'],
                        "Hàm kích hoạt": results['params']['activation'],
                        "Trình tối ưu": results['params']['solver']
                    })

    with tab_demo:
        st.header("Demo Dự đoán")

        if 'split_data' not in st.session_state or 'model' not in st.session_state:
            st.info("Vui lòng huấn luyện mô hình trước.")
        else:
            mode = st.radio("Chọn phương thức:", ["Dữ liệu Test", "Upload ảnh", "Vẽ số"])
            progress_bar = st.progress(0)
            status_text = st.empty()

            def preprocess_input(data):
                return data / 255.0

            is_normalized = 'data_processed' in st.session_state

            if mode == "Dữ liệu Test":
                X_test = st.session_state['split_data']["X_test"]
                y_test = st.session_state['split_data']["y_test"]
                idx = st.slider("Chọn mẫu Test", 0, len(X_test)-1, 0)
                col1, col2 = st.columns([2, 1])
                with col1:
                    st.write("**Ảnh mẫu Test:**")
                    fig, ax = plt.subplots(figsize=(2, 2))
                    ax.imshow(X_test.iloc[idx].values.reshape(28, 28), cmap='gray')
                    ax.axis('off')
                    st.pyplot(fig)
                with col2:
                    st.write(f"**Nhãn thực tế:** {y_test.iloc[idx]}")

                if st.button("Dự đoán"):
                    with st.spinner("Đang dự đoán..."):
                        for i in [20, 40, 60, 80, 100]:
                            progress_bar.progress(i)
                            status_text.text(f"Đang xử lý {i}%")
                            time.sleep(0.05)
                        sample = X_test.iloc[idx].values.reshape(1, -1)
                        if not is_normalized:
                            sample = preprocess_input(sample)
                        model = st.session_state['model']
                        prediction = model.predict(sample)[0]
                        proba = model.predict_proba(sample)[0]
                        max_proba = np.max(proba) * 100
                        st.success(f"Dự đoán: **{prediction}** | Xác suất: **{max_proba:.2f}%** | Nhãn thực tế: **{y_test.iloc[idx]}**")
                        status_text.empty()
                        progress_bar.empty()

            elif mode == "Upload ảnh":
                uploaded_images = st.file_uploader("Upload ảnh (28x28, thang độ xám)", type=["png", "jpg"], accept_multiple_files=True)
                if uploaded_images:
                    for i, uploaded_image in enumerate(uploaded_images):
                        try:
                            img = Image.open(uploaded_image).convert('L').resize((28, 28))
                            st.image(img, caption=f"Ảnh {i+1}", width=280)
                            img_array = np.array(img).flatten().reshape(1, -1)
                            img_array, fixed = validate_and_fix_pixels(img_array, f"ảnh upload {i+1}")
                            if fixed:
                                st.success(f"Đã chuẩn hóa ảnh {i+1} về [0, 255]!")
                            if not is_normalized:
                                img_array = preprocess_input(img_array)
                            
                            if st.button(f"Dự đoán ảnh {i+1}", key=f"predict_upload_{i}"):
                                with st.spinner(f"Đang dự đoán ảnh {i+1}..."):
                                    for j in [20, 40, 60, 80, 100]:
                                        progress_bar.progress(j)
                                        status_text.text(f"Đang xử lý {j}%")
                                        time.sleep(0.05)
                                    model = st.session_state['model']
                                    prediction = model.predict(img_array)[0]
                                    proba = model.predict_proba(img_array)[0]
                                    max_proba = np.max(proba) * 100
                                    st.success(f"Dự đoán: **{prediction}** | Xác suất: **{max_proba:.2f}%**")
                                    status_text.empty()
                                    progress_bar.empty()
                        except Exception as e:
                            st.error(f"Lỗi khi xử lý ảnh {i+1}: {e}")

            elif mode == "Vẽ số":
                st.write("Vẽ chữ số từ 0-9:")
                canvas_result = st_canvas(fill_color="black", stroke_width=20, stroke_color="white", 
                                          background_color="black", width=280, height=280, drawing_mode="freedraw", key="canvas")
                col1, col2 = st.columns([1, 1])
                with col1:
                    if st.button("Dự đoán số đã vẽ"):
                        if canvas_result.image_data is not None and np.any(canvas_result.image_data):
                            with st.spinner("Đang xử lý..."):
                                for i in [20, 40, 60, 80, 100]:
                                    progress_bar.progress(i)
                                    status_text.text(f"Đang xử lý {i}%")
                                    time.sleep(0.05)
                                img = Image.fromarray((canvas_result.image_data * 255).astype(np.uint8)).convert('L').resize((28, 28))
                                img_array = np.array(img).flatten().reshape(1, -1)
                                img_array, fixed = validate_and_fix_pixels(img_array, "hình vẽ")
                                if fixed:
                                    st.success("Đã chuẩn hóa hình vẽ về [0, 255]!")
                                if not is_normalized:
                                    img_array = preprocess_input(img_array)
                                model = st.session_state['model']
                                prediction = model.predict(img_array)[0]
                                proba = model.predict_proba(img_array)[0]
                                max_proba = np.max(proba) * 100
                                st.success(f"Dự đoán: **{prediction}** | Xác suất: **{max_proba:.2f}%**")
                                st.image(img, caption="Hình vẽ của bạn")
                                status_text.empty()
                                progress_bar.empty()
                        else:
                            st.warning("Vui lòng vẽ trước!")
                with col2:
                    if st.button("Xóa Canvas"):
                        st.session_state['canvas_key'] = st.session_state.get('canvas_key', 0) + 1
                        st.rerun()

    with tab_log_info:
        st.header("Theo dõi Kết quả")
        try:
            with st.spinner("Đang tải thông tin huấn luyện..."):
                progress_bar = st.progress(0)
                status_text = st.empty()
                for i in [20, 40, 60, 80, 100]:
                    progress_bar.progress(i)
                    status_text.text(f"Đang tải {i}%")
                    time.sleep(0.05)
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
                    st.json(selected_run.data.params, expanded=True)
                    st.json(selected_run.data.metrics, expanded=True)

                status_text.empty()
                progress_bar.empty()
        except Exception as e:
            st.error(f"Lỗi kết nối MLflow: {e}")

if __name__ == "__main__":
    run_mnist_neural_network_app()