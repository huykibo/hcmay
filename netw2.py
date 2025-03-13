import os
import mlflow
import streamlit as st
import openml
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from mlflow.tracking import MlflowClient
from streamlit_drawable_canvas import st_canvas
from datetime import datetime
import time
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Flatten, Dropout
from keras.utils import to_categorical

def run_mnist_neural_network_app():
    # Thiết lập MLflow
    # try:
    #     os.environ["MLFLOW_TRACKING_USERNAME"] = st.secrets["mlflow"]["MLFLOW_TRACKING_USERNAME"]
    #     os.environ["MLFLOW_TRACKING_PASSWORD"] = st.secrets["mlflow"]["MLFLOW_TRACKING_PASSWORD"]
    #     mlflow.set_tracking_uri(st.secrets["mlflow"]["MLFLOW_TRACKING_URI"])
    #     mlflow.set_experiment("MNIST_Neural_Network")
    # except KeyError as e:
    #     st.error(f"Lỗi: Không tìm thấy khóa {e} trong st.secrets. Vui lòng cấu hình secrets trong Streamlit.")
    #     st.stop()

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
    tabs = st.tabs(["Thông tin", "Tải dữ liệu", "Xử lý dữ liệu", "Chia dữ liệu", "Huấn luyện/Đánh Giá", "Demo dự đoán", "Thông tin huấn luyện"])
    tab_info, tab_load, tab_preprocess, tab_split, tab_train_eval, tab_demo, tab_log_info = tabs

    # Tab 1: Thông tin
    with tab_info:
        st.header("Giới thiệu về Ứng dụng và Neural Network với MNIST")
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

        content_placeholder = st.empty()

        with content_placeholder.container():
            if info_option == "Ứng dụng này là gì và mục tiêu của nó?":
                st.subheader("1. Ứng dụng này là gì và mục tiêu của nó?")
                st.markdown("""
                Đây là một ứng dụng phân loại chữ số viết tay dựa trên tập dữ liệu MNIST, sử dụng **Neural Network (Mạng nơ-ron nhân tạo)** – một thuật toán học sâu mạnh mẽ. MNIST bao gồm 70,000 ảnh chữ số từ 0 đến 9, mỗi ảnh có kích thước 28x28 pixel, tương đương với 784 đặc trưng (pixel). Mục tiêu của ứng dụng là xây dựng và huấn luyện một mô hình mạng nơ-ron để nhận diện chính xác các chữ số này, đồng thời cung cấp một công cụ trực quan để học tập, thử nghiệm và đánh giá hiệu quả của thuật toán học sâu.

                Để dễ hình dung:  
                - **784 đặc trưng**: Mỗi ảnh được biểu diễn dưới dạng một vector 784 chiều, với mỗi chiều là giá trị độ sáng của một pixel (từ 0 đến 255).  
                - **70,000 mẫu**: Tổng số ảnh trong tập dữ liệu, bao gồm cả tập huấn luyện và kiểm tra.  
                - **Nhiệm vụ**: Dự đoán nhãn (từ 0 đến 9) của mỗi ảnh dựa trên các đặc trưng pixel.
                """)

            elif info_option == "Tập dữ liệu MNIST: Đặc điểm và ý nghĩa":
                st.subheader("2. Tập dữ liệu MNIST: Đặc điểm và ý nghĩa")
                st.markdown("""
                MNIST được tạo ra bởi Yann LeCun và các cộng sự, là một tập dữ liệu chuẩn trong nghiên cứu học máy và thị giác máy tính. Các ảnh trong MNIST được thu thập từ chữ số viết tay của học sinh trung học và nhân viên điều tra dân số Mỹ, sau đó được chuẩn hóa thành kích thước 28x28 pixel và chuyển thành thang độ xám (grayscale).  

                **Ý nghĩa của MNIST**:  
                - Là bài toán cơ bản để kiểm tra hiệu quả của các thuật toán phân loại, đặc biệt là mạng nơ-ron.  
                - Dữ liệu đơn giản nhưng đủ phức tạp để đánh giá khả năng phân biệt giữa các lớp tương tự (ví dụ: "4" và "9").  
                - Phù hợp cho cả người mới bắt đầu và các nhà nghiên cứu muốn thử nghiệm các mô hình học sâu phức tạp hơn.
                """)
                st.subheader("Minh họa dữ liệu MNIST")
                with st.spinner("Đang tải ảnh minh họa..."):
                    try:
                        mnist_image = Image.open("mnist.png")
                        st.image(mnist_image, caption="Ảnh minh họa 10 chữ số từ 0 đến 9 trong MNIST", use_container_width=True)
                    except FileNotFoundError:
                        if 'full_data' in st.session_state:
                            X, y = st.session_state['full_data']
                            fig, axes = plt.subplots(1, 10, figsize=(20, 2))
                            for i in range(10):
                                sample = X[y == str(i)].iloc[0].values.reshape(28, 28)
                                axes[i].imshow(sample, cmap='gray')
                                axes[i].set_title(f"{i}")
                                axes[i].axis('off')
                            st.pyplot(fig)
                            st.caption("Ảnh minh họa được tạo từ dữ liệu MNIST do không tìm thấy file `mnist.png`.")
                        else:
                            st.error("Không tìm thấy file `mnist.png`. Vui lòng đặt file vào thư mục hiện tại hoặc tải dữ liệu MNIST trước.")
                    except Exception as e:
                        st.error(f"Lỗi khi tải ảnh: {e}")

            elif info_option == "Neural Network – Mạng nơ-ron nhân tạo":
                st.subheader("3. Neural Network – Mạng nơ-ron nhân tạo")
                st.markdown("""
                **Neural Network (Mạng nơ-ron nhân tạo)** là một mô hình học sâu mô phỏng cách hoạt động của não bộ con người, bao gồm các lớp nơ-ron (neurons) kết nối với nhau để xử lý dữ liệu và đưa ra dự đoán. Với MNIST, mạng nơ-ron nhận đầu vào là 784 đặc trưng (pixel) và trả về xác suất cho 10 nhãn (0-9).

                ### Cách hoạt động chi tiết:
                1. **Lớp đầu vào (Input Layer)**:  
                   - Nhận dữ liệu thô: vector 784 chiều từ ảnh 28x28 pixel.  
                   - Ví dụ: Một ảnh số "3" được biểu diễn bằng các giá trị pixel từ 0 (đen) đến 255 (trắng).  
                """)
                try:
                    nn_step_1 = Image.open("netw/illustrations/nn_step_1.png")
                    st.image(nn_step_1, caption="Bước 1: Lớp đầu vào nhận dữ liệu từ ảnh MNIST", use_container_width=True)
                except FileNotFoundError:
                    st.warning("Không tìm thấy file `netw/illustrations/nn_step_1.png`. Vui lòng chạy file `netw/netw.py` để tạo ảnh minh họa.")

                st.markdown("""
                2. **Lớp ẩn (Hidden Layers)**:  
                   - Các lớp này học cách trích xuất đặc trưng từ dữ liệu đầu vào thông qua trọng số và hàm kích hoạt (activation function).  
                   - Công thức tính tại mỗi nơ-ron:  
                     $$ z = w \\cdot x + b $$  
                     $$ a = \\sigma(z) $$  
                     - $w$: Trọng số (weights).  
                     - $x$: Đầu vào (input).  
                     - $b$: Độ lệch (bias).  
                     - $\\sigma$: Hàm kích hoạt (ví dụ: ReLU hoặc Sigmoid).  
                   - Ví dụ: Một lớp ẩn có thể học cách nhận diện các nét ngang hoặc vòng tròn trong chữ số.  
                """)
                try:
                    nn_step_2 = Image.open("netw/illustrations/nn_step_2.png")
                    st.image(nn_step_2, caption="Bước 2: Lớp ẩn trích xuất đặc trưng từ dữ liệu", use_container_width=True)
                except FileNotFoundError:
                    st.warning("Không tìm thấy file `netw/illustrations/nn_step_2.png`. Vui lòng chạy file `netw/netw.py` để tạo ảnh minh họa.")

                st.markdown("""
                3. **Lớp đầu ra (Output Layer)**:  
                   - Bao gồm 10 nơ-ron, mỗi nơ-ron đại diện cho một chữ số (0-9).  
                   - Sử dụng hàm **Softmax** để chuyển đổi đầu ra thành xác suất:  
                     $$ P(y=i) = \\frac{e^{z_i}}{\\sum_{j=0}^{9} e^{z_j}} $$  
                   - Ví dụ: Đầu ra [0.05, 0.1, ..., 0.7, ...] → Dự đoán là "7" với xác suất 70%.  
                """)
                try:
                    nn_step_3 = Image.open("netw/illustrations/nn_step_3.png")
                    st.image(nn_step_3, caption="Bước 3: Lớp đầu ra dự đoán nhãn với Softmax", use_container_width=True)
                except FileNotFoundError:
                    st.warning("Không tìm thấy file `netw/illustrations/nn_step_3.png`. Vui lòng chạy file `netw/netw.py` để tạo ảnh minh họa.")

                st.markdown("""
                4. **Huấn luyện**:  
                   - Sử dụng hàm mất mát (loss function) như **Cross-Entropy**:  
                     $$ L = -\\frac{1}{N} \\sum_{i=1}^{N} y_i \\log(\\hat{y}_i) $$  
                     - $y_i$: Nhãn thực tế (one-hot encoded).  
                     - $\\hat{y}_i$: Xác suất dự đoán.  
                   - Tối ưu hóa bằng thuật toán **Gradient Descent** hoặc **Adam** để điều chỉnh trọng số.  
                """)
                try:
                    nn_step_4 = Image.open("netw/illustrations/nn_step_4.png")
                    st.image(nn_step_4, caption="Bước 4: Huấn luyện tối ưu hóa trọng số", use_container_width=True)
                except FileNotFoundError:
                    st.warning("Không tìm thấy file `netw/illustrations/nn_step_4.png`. Vui lòng chạy file `netw/netw.py` để tạo ảnh minh họa.")

                st.markdown("""
                ### Áp dụng với MNIST:
                - Mạng nơ-ron học cách ánh xạ từ 784 pixel sang nhãn 0-9 thông qua các lớp ẩn.  

                ### Ưu điểm:
                - Khả năng học các đặc trưng phức tạp, hiệu quả cao với dữ liệu phi tuyến như MNIST.  
                - Linh hoạt với kiến trúc mạng (số lớp, số nơ-ron).  

                ### Nhược điểm:
                - Yêu cầu dữ liệu lớn và thời gian huấn luyện lâu hơn so với các mô hình đơn giản.  
                - Cần chuẩn hóa dữ liệu và điều chỉnh siêu tham số cẩn thận.
                """)

            elif info_option == "Công thức đánh giá độ chính xác (Accuracy)":
                st.subheader("4. Công thức đánh giá độ chính xác (Accuracy)")
                st.markdown("""
                Độ chính xác (Accuracy) đo tỷ lệ dự đoán đúng:  
                $$ Accuracy = \\frac{\\text{Số mẫu dự đoán đúng}}{\\text{Tổng số mẫu}} $$  
                - **Ví dụ**: Dự đoán đúng 92/100 ảnh → Accuracy = 92%.  

                **Ý nghĩa**:  
                - Với Neural Network, Accuracy đo khả năng mạng phân loại chính xác các chữ số dựa trên đặc trưng pixel đã học.
                """)

    # Tab 2: Tải dữ liệu
    with tab_load:
        st.header("Tải Dữ liệu")
        if st.button("Tải dữ liệu MNIST từ OpenML"):
            with st.spinner("Đang tải dữ liệu..."):
                progress_bar = st.progress(0)
                status_text = st.empty()
                try:
                    mnist = openml.datasets.get_dataset(554)
                    for i in range(20, 51, 5):
                        progress_bar.progress(i)
                        status_text.text(f"Đang tải dữ liệu {i}%{i % 4 * '.'}")
                        time.sleep(0.1)

                    X, y, _, _ = mnist.get_data(target=mnist.default_target_attribute)
                    for i in range(50, 91, 5):
                        progress_bar.progress(i)
                        status_text.text(f"Đang xử lý dữ liệu {i}%{i % 4 * '.'}")
                        time.sleep(0.1)

                    st.session_state['full_data'] = (X, y)
                    for i in range(90, 101, 2):
                        progress_bar.progress(i)
                        status_text.text(f"Hoàn tất {i}% - Đã tải {X.shape[0]} mẫu{i % 4 * '.'}")
                        time.sleep(0.1)

                    with mlflow.start_run(run_name="Data_Load"):
                        mlflow.log_param("total_samples", X.shape[0])

                    status_text.text("Đã tải 100% - Hoàn tất!")
                    time.sleep(1)
                    status_text.empty()
                    progress_bar.empty()
                    st.success(f"Tải dữ liệu thành công! Kích thước dữ liệu gốc: {X.shape}")
                except Exception as e:
                    st.error(f"Không thể tải dữ liệu: {e}")

        if 'full_data' in st.session_state:
            X_full, y_full = st.session_state['full_data']
            num_samples = st.slider("Chọn số lượng mẫu:",
                                    min_value=10, max_value=len(X_full), value=min(1000, len(X_full)), step=1)
           
            if st.button("Chốt số lượng mẫu"):
                with st.spinner("Đang xử lý mẫu..."):
                    progress_bar = st.progress(0)
                    status_text = st.empty()

                    df = pd.concat([X_full, y_full.rename("label")], axis=1)
                    for i in range(0, 31, 5):
                        progress_bar.progress(i)
                        status_text.text(f"Đang nối dữ liệu {i}%{i % 4 * '.'}")
                        time.sleep(0.1)

                    sampled_df = df.sample(n=num_samples, random_state=42)
                    for i in range(30, 71, 5):
                        progress_bar.progress(i)
                        status_text.text(f"Đang lấy mẫu {i}%{i % 4 * '.'}")
                        time.sleep(0.1)

                    X_sampled = sampled_df.drop(columns=["label"])
                    y_sampled = sampled_df["label"]
                    st.session_state['data'] = (X_sampled, y_sampled)
                    for i in range(70, 101, 5):
                        progress_bar.progress(i)
                        status_text.text(f"Đang lưu dữ liệu {i}%{i % 4 * '.'}")
                        time.sleep(0.1)

                    with mlflow.start_run(run_name="Data_Sample"):
                        mlflow.log_param("num_samples", num_samples)

                    status_text.text("Đã xử lý 100% - Hoàn tất!")
                    time.sleep(1)
                    status_text.empty()
                    progress_bar.empty()
                    st.success(f"Đã chốt {num_samples} mẫu!")

    # Tab 3: Xử lý dữ liệu
    with tab_preprocess:
        st.header("Xử lý Dữ liệu")
        if 'data' not in st.session_state:
            st.info("Vui lòng tải và chốt số lượng mẫu trước.")
        else:
            if "data_original" not in st.session_state:
                X, y = st.session_state['data']
                st.session_state["data_original"] = (X.copy(), y.copy())
           
            current_data = st.session_state.get("data_processed", st.session_state["data_original"])
            X_current, y_current = current_data

            st.subheader("Dữ liệu Gốc")
            fig, axes = plt.subplots(2, 5, figsize=(10, 4))
            for i, ax in enumerate(axes.flat):
                ax.imshow(st.session_state["data_original"][0].iloc[i].values.reshape(28, 28), cmap='gray')
                ax.set_title(f"Label: {st.session_state['data_original'][1].iloc[i]}")
                ax.axis("off")
            st.pyplot(fig)

            col1, col2 = st.columns([3, 1])
            with col1:
                if st.button("Normalization", key="normalize_btn"):
                    with st.spinner("Đang chuẩn hóa dữ liệu..."):
                        progress_bar = st.progress(0)
                        status_text = st.empty()

                        for i in range(0, 21, 5):
                            progress_bar.progress(i)
                            status_text.text(f"Đang chuẩn bị dữ liệu {i}%{i % 4 * '.'}")
                            time.sleep(0.1)
                        X_norm = X_current / 255.0

                        for i in range(20, 61, 5):
                            progress_bar.progress(i)
                            status_text.text(f"Đang chuẩn hóa {i}%{i % 4 * '.'}")
                            time.sleep(0.1)
                        st.session_state["data_processed"] = (X_norm, y_current)

                        for i in range(60, 101, 5):
                            progress_bar.progress(i)
                            status_text.text(f"Hoàn tất {i}%{i % 4 * '.'}")
                            time.sleep(0.1)

                        status_text.empty()
                        progress_bar.empty()
                        st.success("Đã chuẩn hóa dữ liệu!")
                        st.rerun()
            with col2:
                st.markdown("""
                    <div class="tooltip">
                        ?
                        <span class="tooltiptext">
                            Đưa dữ liệu về khoảng [0, 1] bằng cách chia cho 255.<br>
                            Công dụng: Đảm bảo thang đo đồng nhất, rất quan trọng cho Neural Network.
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
            else:
                st.info("Dữ liệu chưa được xử lý. Vui lòng nhấn 'Normalization' để xử lý.")

    # Tab 4: Chia dữ liệu
    with tab_split:
        st.header("Chia Tập Dữ Liệu")
        if 'data' not in st.session_state:
            st.info("Vui lòng tải và chốt số lượng mẫu trước.")
        else:
            data_source = st.session_state.get("data_processed", st.session_state['data'])
            try:
                X, y = data_source
            except (ValueError, TypeError) as e:
                st.error(f"Lỗi: Dữ liệu không hợp lệ. Vui lòng kiểm tra bước tải hoặc xử lý dữ liệu. Chi tiết lỗi: {e}")
            else:
                total_samples = len(X)
                st.write(f"Tổng số mẫu: {total_samples}")

                test_pct = st.slider("Tỷ lệ tập Test (%)", 0, 100, 20)
                valid_pct = st.slider("Tỷ lệ tập Validation (%) từ phần còn lại", 0, 100, 20)
               
                if test_pct + valid_pct > 100:
                    st.warning("Tổng tỷ lệ Test và Validation vượt quá 100%!")
               
                test_size = test_pct / 100
                if test_size > 0:
                    X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=test_size, random_state=42, stratify=y)
                else:
                    X_temp, y_temp = X, y
                    X_test, y_test = pd.DataFrame(), pd.Series()

                valid_size = valid_pct / 100
                if valid_size > 0 and len(X_temp) > int(len(X_temp) * valid_size):
                    X_train, X_valid, y_train, y_valid = train_test_split(X_temp, y_temp, test_size=valid_size, random_state=42, stratify=y_temp)
                else:
                    X_train, y_train = X_temp, y_temp
                    X_valid, y_valid = pd.DataFrame(), pd.Series()

                st.write(f"Train: {len(X_train)} mẫu ({(len(X_train) / total_samples) * 100:.1f}%)")
                st.write(f"Validation: {len(X_valid)} mẫu ({(len(X_valid) / total_samples) * 100:.1f}%)")
                st.write(f"Test: {len(X_test)} mẫu ({(len(X_test) / total_samples) * 100:.1f}%)")

                if st.button("Xác nhận chia dữ liệu"):
                    with st.spinner("Đang chia dữ liệu..."):
                        progress_bar = st.progress(0)
                        status_text = st.empty()
                        for i in range(0, 101, 5):
                            progress_bar.progress(i)
                            status_text.text(f"Đang chia dữ liệu {i}%{i % 4 * '.'}")
                            time.sleep(0.1)

                        st.session_state['split_data'] = {
                            "X_train": X_train, "y_train": y_train,
                            "X_valid": X_valid, "y_valid": y_valid,
                            "X_test": X_test, "y_test": y_test
                        }
                        status_text.empty()
                        progress_bar.empty()
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
            | Số lượng mẫu | Số lớp ẩn | Nơ-ron mỗi lớp | Epochs | Dropout |
            |--------------|-----------|----------------|--------|---------|
            | <1000        | 1-2       | 64-128         | 5-10   | 0.2     |
            | 1000-5000    | 2-3       | 128-256        | 10-20  | 0.3     |
            | 5000-50000   | 3-4       | 256-512        | 20-30  | 0.4     |
            | >50000       | 4-5       | 512-1024       | 30-50  | 0.5     |
            """)
            st.markdown("""
            - **Số lớp ẩn**: Số lượng hidden layers trong mạng.  
            - **Nơ-ron mỗi lớp**: Số đơn vị tính toán trong mỗi lớp ẩn.  
            - **Epochs**: Số lần lặp qua toàn bộ dữ liệu huấn luyện.  
            - **Dropout**: Tỷ lệ bỏ qua nơ-ron để tránh overfitting.
            """)

            params = {}
            if num_samples < 1000:
                params["hidden_layers"] = 1
                params["neurons"] = 64
                params["epochs"] = 5
                params["dropout"] = 0.2
            elif 1000 <= num_samples <= 5000:
                params["hidden_layers"] = 2
                params["neurons"] = 128
                params["epochs"] = 10
                params["dropout"] = 0.3
            elif 5000 < num_samples <= 50000:
                params["hidden_layers"] = 3
                params["neurons"] = 256
                params["epochs"] = 20
                params["dropout"] = 0.4
            else:
                params["hidden_layers"] = 4
                params["neurons"] = 512
                params["epochs"] = 30
                params["dropout"] = 0.5

            st.markdown("#### Tham số mô hình (đã đặt tự động, có thể điều chỉnh)")
            params["hidden_layers"] = st.number_input("Số lớp ẩn", min_value=1, max_value=5, value=params["hidden_layers"])
            params["neurons"] = st.number_input("Số nơ-ron mỗi lớp", min_value=32, max_value=1024, value=params["neurons"], step=32)
            params["epochs"] = st.number_input("Số epochs", min_value=1, max_value=100, value=params["epochs"])
            params["dropout"] = st.slider("Dropout rate", 0.0, 0.5, value=params["dropout"], step=0.05)

            if st.button("Thực hiện Huấn luyện"):
                with st.spinner("Đang huấn luyện mô hình..."):
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    start_time = time.time()
                   
                    X_train = st.session_state['split_data']["X_train"].values.reshape(-1, 28, 28)
                    y_train = to_categorical(st.session_state['split_data']["y_train"])
                    X_valid = st.session_state['split_data']["X_valid"].values.reshape(-1, 28, 28)
                    y_valid = to_categorical(st.session_state['split_data']["y_valid"])
                    X_test = st.session_state['split_data']["X_test"].values.reshape(-1, 28, 28)
                    y_test = to_categorical(st.session_state['split_data']["y_test"])

                    run_name = f"Neural_Network_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                    with mlflow.start_run(run_name=run_name) as run:
                        for i in range(0, 11, 2):
                            progress_bar.progress(i)
                            status_text.text(f"Đang khởi tạo mô hình {i}%{i % 4 * '.'}")
                            time.sleep(0.1)

                        model = Sequential()
                        model.add(Flatten(input_shape=(28, 28)))
                        for _ in range(params["hidden_layers"]):
                            model.add(Dense(params["neurons"], activation='relu'))
                            model.add(Dropout(params["dropout"]))
                        model.add(Dense(10, activation='softmax'))
                        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

                        for i in range(10, 51, 5):
                            progress_bar.progress(i)
                            status_text.text(f"Đang huấn luyện {i}%{i % 4 * '.'}")
                            time.sleep(0.1)
                        history = model.fit(X_train, y_train, epochs=params["epochs"], batch_size=32, validation_data=(X_valid, y_valid), verbose=0)

                        mlflow.log_params(params)

                        y_valid_pred = np.argmax(model.predict(X_valid, verbose=0), axis=1)
                        y_valid_true = np.argmax(y_valid, axis=1)
                        accuracy_val = accuracy_score(y_valid_true, y_valid_pred)
                        mlflow.log_metric("accuracy_val", accuracy_val)
                        cm_valid = confusion_matrix(y_valid_true, y_valid_pred)

                        for i in range(50, 76, 5):
                            progress_bar.progress(i)
                            status_text.text(f"Đang đánh giá validation {i}%{i % 4 * '.'}")
                            time.sleep(0.1)

                        y_test_pred = np.argmax(model.predict(X_test, verbose=0), axis=1)
                        y_test_true = np.argmax(y_test, axis=1)
                        accuracy_test = accuracy_score(y_test_true, y_test_pred)
                        mlflow.log_metric("accuracy_test", accuracy_test)
                        cm_test = confusion_matrix(y_test_true, y_test_pred)
                        training_time = time.time() - start_time
                        mlflow.log_metric("training_time_seconds", training_time)
                        mlflow.keras.log_model(model, "model")

                        for i in range(75, 101, 5):
                            progress_bar.progress(i)
                            status_text.text(f"Hoàn tất {i}%{i % 4 * '.'}")
                            time.sleep(0.1)

                        run_id = run.info.run_id
                        st.session_state['model'] = model
                        st.session_state['latest_run'] = {
                            'run_name': run_name,
                            'run_id': run_id
                        }

                        st.session_state['training_results'] = {
                            'training_time': training_time,
                            'accuracy_val': accuracy_val,
                            'accuracy_test': accuracy_test,
                            'cm_valid': cm_valid,
                            'cm_test': cm_test,
                            'model_choice': 'Neural Network',
                            'params': params,
                            'num_samples': len(X_train),
                            'run_name': run_name,
                            'run_id': run_id,
                            'history': history.history  # Lưu lịch sử huấn luyện để hiển thị
                        }

                        status_text.empty()
                        progress_bar.empty()

            if 'training_results' in st.session_state:
                st.success(f"Huấn luyện hoàn tất. Thời gian thực hiện: {st.session_state['training_results']['training_time']:.2f} giây.")
                st.write(f"Accuracy Validation: {st.session_state['training_results']['accuracy_val']:.4f}")
                st.write(f"Accuracy Test: {st.session_state['training_results']['accuracy_test']:.4f}")

                st.markdown("### Confusion Matrix")
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
                sns.heatmap(st.session_state['training_results']['cm_valid'], annot=True, fmt="d", cmap="Blues", ax=ax1)
                ax1.set_title("Confusion Matrix - Validation")
                sns.heatmap(st.session_state['training_results']['cm_test'], annot=True, fmt="d", cmap="Blues", ax=ax2)
                ax2.set_title("Confusion Matrix - Test")
                st.pyplot(fig)

                # Thêm biểu đồ Loss và Accuracy
                if 'history' in st.session_state['training_results']:
                    st.markdown("### Biểu đồ Loss và Accuracy trong quá trình huấn luyện")
                    history = st.session_state['training_results']['history']
                    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
                    ax1.plot(history['loss'], label='Train Loss')
                    ax1.plot(history['val_loss'], label='Validation Loss')
                    ax1.set_title("Loss qua các Epoch")
                    ax1.set_xlabel("Epoch")
                    ax1.set_ylabel("Loss")
                    ax1.legend()
                    ax1.grid(True, alpha=0.3)

                    ax2.plot(history['accuracy'], label='Train Accuracy')
                    ax2.plot(history['val_accuracy'], label='Validation Accuracy')
                    ax2.set_title("Accuracy qua các Epoch")
                    ax2.set_xlabel("Epoch")
                    ax2.set_ylabel("Accuracy")
                    ax2.legend()
                    ax2.grid(True, alpha=0.3)
                    st.pyplot(fig)

                st.subheader("Thông tin Kết quả")
                with st.expander("Xem chi tiết kết quả", expanded=True):
                    run_name = st.session_state['training_results']['run_name']
                    run_id = st.session_state['training_results']['run_id']
                    model_choice = st.session_state['training_results']['model_choice']
                    params = st.session_state['training_results']['params']
                    training_time = st.session_state['training_results']['training_time']
                    accuracy_val = st.session_state['training_results']['accuracy_val']
                    accuracy_test = st.session_state['training_results']['accuracy_test']
                    X_train = st.session_state['split_data']["X_train"]

                    st.markdown("#### Thông tin lần chạy:", unsafe_allow_html=True)
                    st.write(f"- **Tên lần chạy (Run Name)**: {run_name}")
                    st.write(f"- **ID lần chạy (Run ID)**: {run_id}")

                    st.markdown("#### Cài đặt bạn đã chọn:", unsafe_allow_html=True)
                    st.write(f"- **Mô hình**: {model_choice}")
                    st.write(f"- **Tham số**:")
                    for key, value in params.items():
                        st.write(f"  - {key}: {value}")
                    st.write(f"- **Thời gian chạy**: {training_time:.2f} giây")
                    st.write(f"- **Số mẫu huấn luyện**: {len(X_train)}")

                    st.markdown("#### Kết quả đạt được:", unsafe_allow_html=True)
                    st.markdown(f"""
                    - **Độ chính xác Validation**: {accuracy_val*100:.2f}%  
                    - **Độ chính xác Test**: {accuracy_test*100:.2f}%  
                    - **Nhận xét**: Mô hình đạt độ chính xác {accuracy_test*100:.2f}% trên tập Test, cho thấy khả năng tổng quát hóa {'tốt' if accuracy_test > 0.9 else 'trung bình' if accuracy_test > 0.7 else 'kém'}.
                    """, unsafe_allow_html=True)

    # Tab 6: Demo dự đoán
    with tab_demo:
        st.header("Demo Dự đoán")
        if 'split_data' not in st.session_state or 'model' not in st.session_state:
            st.info("Vui lòng huấn luyện mô hình trước.")
        else:
            mode = st.radio("Chọn phương thức dự đoán:", ["Dữ liệu từ Test", "Upload ảnh mới", "Vẽ số"])
           
            progress_bar = st.progress(0)
            status_text = st.empty()

            def preprocess_input(data):
                return data / 255.0

            is_normalized = "data_processed" in st.session_state

            if mode == "Dữ liệu từ Test":
                X_test = st.session_state['split_data']["X_test"]
                y_test = st.session_state['split_data']["y_test"]
                if X_test.empty or y_test.empty:
                    st.warning("Tập Test không có dữ liệu. Vui lòng chia dữ liệu với tỷ lệ Test > 0.")
                else:
                    idx = st.slider("Chọn mẫu từ Test", 0, len(X_test)-1, 0)
                    if st.button("Dự đoán"):
                        with st.spinner("Đang dự đoán..."):
                            for i in range(0, 51, 5):
                                progress_bar.progress(i)
                                status_text.text(f"Đang xử lý {i}%{i % 4 * '.'}")
                                time.sleep(0.1)
                       
                            sample = X_test.iloc[idx].values.reshape(1, 28, 28)
                            if not is_normalized:
                                sample = preprocess_input(sample)
                       
                            prediction = np.argmax(st.session_state['model'].predict(sample, verbose=0), axis=1)[0]
                            proba = st.session_state['model'].predict(sample, verbose=0)[0]
                            confidence = max(proba) * 100
                            y_true = y_test.iloc[idx]
                       
                            for i in range(50, 101, 5):
                                progress_bar.progress(i)
                                status_text.text(f"Đang dự đoán {i}%{i % 4 * '.'}")
                                time.sleep(0.1)
                       
                            st.success(f"Dự đoán: **{prediction}** | Confidence: **{confidence:.2f}%** | Giá trị thực: **{y_true}**")
                            fig, ax = plt.subplots()
                            ax.imshow(X_test.iloc[idx].values.reshape(28, 28), cmap='gray')
                            ax.axis("off")
                            st.pyplot(fig)
                       
                            time.sleep(1)
                            progress_bar.empty()
                            status_text.empty()

            elif mode == "Upload ảnh mới":
                uploaded_images = st.file_uploader("Upload ảnh (28x28, grayscale)", type=["png", "jpg"], accept_multiple_files=True)
                if uploaded_images:
                    for i, uploaded_image in enumerate(uploaded_images):
                        with st.spinner(f"Đang xử lý ảnh {i+1}/{len(uploaded_images)}..."):
                            for j in range(0, 51, 5):
                                progress_bar.progress(j)
                                status_text.text(f"Đang tải ảnh {i+1} - {j}%{j % 4 * '.'}")
                                time.sleep(0.1)
                           
                            img = Image.open(uploaded_image).convert('L').resize((28, 28))
                            img_array = np.array(img).reshape(1, 28, 28)
                            if not is_normalized:
                                img_array = preprocess_input(img_array)
                           
                            for j in range(50, 101, 5):
                                progress_bar.progress(j)
                                status_text.text(f"Đang dự đoán ảnh {i+1} - {j}%{j % 4 * '.'}")
                                time.sleep(0.1)
                           
                            prediction = np.argmax(st.session_state['model'].predict(img_array, verbose=0), axis=1)[0]
                            proba = st.session_state['model'].predict(img_array, verbose=0)[0]
                            confidence = max(proba) * 100
                           
                            st.success(f"Dự đoán: **{prediction}** | Confidence: **{confidence:.2f}%**")
                            st.image(img, caption=f"Ảnh {i+1} được upload", use_container_width=True)
                           
                            time.sleep(1)
                            progress_bar.empty()
                            status_text.empty()

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
                            for i in range(0, 51, 5):
                                progress_bar.progress(i)
                                status_text.text(f"Đang xử lý {i}%{i % 4 * '.'}")
                                time.sleep(0.1)
                           
                            image_data = canvas_result.image_data
                            if image_data is None or image_data.size == 0:
                                st.warning("Không có dữ liệu từ canvas. Vui lòng vẽ một số!")
                                progress_bar.empty()
                                status_text.empty()
                                return
                           
                            img = Image.fromarray((image_data[:, :, 0]).astype(np.uint8)).convert('L').resize((28, 28))
                            img_array = np.array(img).reshape(1, 28, 28)
                            if not is_normalized:
                                img_array = preprocess_input(img_array)
                           
                            for i in range(50, 101, 5):
                                progress_bar.progress(i)
                                status_text.text(f"Đang dự đoán {i}%{i % 4 * '.'}")
                                time.sleep(0.1)
                           
                            prediction = np.argmax(st.session_state['model'].predict(img_array, verbose=0), axis=1)[0]
                            proba = st.session_state['model'].predict(img_array, verbose=0)[0]
                            confidence = max(proba) * 100
                           
                            st.success(f"Dự đoán: **{prediction}** | Confidence: **{confidence:.2f}%**")
                            st.image(img, caption="Ảnh đã vẽ", width=150)
                           
                            time.sleep(1)
                            progress_bar.empty()
                            status_text.empty()
                    else:
                        st.warning("Vui lòng vẽ một chữ số trước khi dự đoán!")

    # Tab 7: Thông tin huấn luyện
    with tab_log_info:
        st.header("Theo dõi kết quả")
        st.markdown("""
        Tab này cho phép bạn xem danh sách các lần huấn luyện đã thực hiện. Chọn một lần chạy để xem chi tiết, đổi tên hoặc xóa.
        """, unsafe_allow_html=True)
        
        try:
            client = MlflowClient()
            experiment = client.get_experiment_by_name("MNIST_Neural_Network")
            if not experiment:
                st.error("Không tìm thấy experiment 'MNIST_Neural_Network'. Vui lòng kiểm tra lại MLflow tracking URI.")
            else:
                experiment_id = experiment.experiment_id
                runs = client.search_runs(experiment_ids=[experiment_id], order_by=["attributes.start_time DESC"])
                
                if not runs:
                    st.info("Chưa có lần chạy nào được ghi nhận.")
                else:
                    run_options = {run.info.run_id: run.data.tags.get('mlflow.runName', f"Run_{run.info.run_id}") for run in runs}
                    run_names = list(run_options.values())

                    default_run_name = st.session_state.get('training_results', {}).get('run_name', run_names[0]) if 'training_results' in st.session_state else run_names[0]

                    st.subheader("Danh sách run")
                    selected_run_name = st.selectbox(
                        "Chọn run:",
                        options=run_names,
                        index=run_names.index(default_run_name) if default_run_name in run_names else 0,
                        key="main_select",
                        help="Chọn một lần chạy để xem chi tiết, đổi tên hoặc xóa."
                    )
                    selected_run_id = [k for k, v in run_options.items() if v == selected_run_name][0]
                    selected_run = client.get_run(selected_run_id)

                    st.subheader("Đổi tên Run")
                    new_run_name = st.text_input(
                        "Nhập tên mới:",
                        value=selected_run_name,
                        key="rename_input"
                    )
                    if st.button("Cập nhật tên", key="rename_button"):
                        if new_run_name.strip() and new_run_name.strip() != selected_run_name:
                            with st.spinner("Đang cập nhật tên..."):
                                client.set_tag(selected_run_id, "mlflow.runName", new_run_name.strip())
                                if 'training_results' in st.session_state and st.session_state['training_results']['run_id'] == selected_run_id:
                                    st.session_state['training_results']['run_name'] = new_run_name.strip()
                                st.success(f"Đã đổi tên thành: {new_run_name.strip()}")
                                time.sleep(0.5)
                                st.rerun()
                        elif not new_run_name.strip():
                            st.warning("Vui lòng nhập tên hợp lệ.")
                        else:
                            st.info("Tên mới trùng với tên hiện tại.")

                    st.subheader("Xóa Run")
                    if st.button("Xóa lần chạy", key="delete_button"):
                        with st.spinner("Đang xóa lần chạy..."):
                            client.delete_run(selected_run_id)
                            if 'training_results' in st.session_state and st.session_state['training_results']['run_id'] == selected_run_id:
                                del st.session_state['training_results']
                                del st.session_state['model']
                            st.success(f"Đã xóa: {selected_run_name}")
                            time.sleep(0.5)
                            st.rerun()

                    st.subheader("Thông tin chi tiết của Run")
                    st.write(f"**Tên lần chạy:** {selected_run_name}")
                    st.write(f"**ID lần chạy:** {selected_run_id}")
                    st.write(f"**Thời gian bắt đầu:** {datetime.fromtimestamp(selected_run.info.start_time / 1000)}")

                    st.markdown("**Tham số:**", unsafe_allow_html=True)
                    if selected_run.data.params:
                        st.json(selected_run.data.params, expanded=True)
                    else:
                        st.write("Không có tham số được ghi nhận.")

                    st.markdown("**Kết quả:**", unsafe_allow_html=True)
                    if selected_run.data.metrics:
                        metrics_display = {}
                        training_time = selected_run.data.metrics.get("training_time_seconds", "N/A")
                        metrics_display["Thời gian thực hiện (giây)"] = f"{float(training_time):.2f}" if training_time != "N/A" else "N/A"
                        accuracy_val = selected_run.data.metrics.get("accuracy_val", "N/A")
                        metrics_display["Độ chính xác Validation"] = f"{float(accuracy_val)*100:.2f}%" if accuracy_val != "N/A" else "N/A"
                        accuracy_test = selected_run.data.metrics.get("accuracy_test", "N/A")
                        metrics_display["Độ chính xác Test"] = f"{float(accuracy_test)*100:.2f}%" if accuracy_test != "N/A" else "N/A"
                        st.json(metrics_display, expanded=True)
                    else:
                        st.write("Không có kết quả được ghi nhận.")

                    st.subheader("Truy cập MLflow UI")
                    mlflow_url = "https://dagshub.com/huykibo/streamlit_mlflow.mlflow"
                    if st.button("Mở MLflow UI trên Dagshub"):
                        st.markdown(f'[Click để mở MLflow UI]({mlflow_url})', unsafe_allow_html=True)
        except Exception as e:
            st.error(f"Lỗi kết nối MLflow: {e}. Vui lòng kiểm tra MLFLOW_TRACKING_URI và thông tin xác thực.")

if __name__ == "__main__":
    run_mnist_neural_network_app()