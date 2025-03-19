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

        # Tạo placeholder để chứa nội dung động
        content_placeholder = st.empty()

        # Xóa nội dung cũ trước khi hiển thị nội dung mới
        content_placeholder.empty()

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
                        status_text.text(f"Đang tải thông tin... {i}%")
                        time.sleep(0.05)
                    st.subheader("📊 Neural Network – Mạng nơ-ron nhân tạo")
                    st.markdown("""
                    **Neural Network (Mạng nơ-ron nhân tạo)** là một mô hình học máy mô phỏng cách hoạt động của mạng nơ-ron sinh học trong não người. Nó được thiết kế để học các đặc trưng phức tạp từ dữ liệu, đặc biệt hiệu quả với bài toán nhận diện hình ảnh như MNIST.
                    """, unsafe_allow_html=True)

                    st.subheader("🌐 Cấu trúc cơ bản của Neural Network")
                    st.markdown("""
                    - **Lớp đầu vào (Input Layer)**: Nhận dữ liệu thô (ví dụ: $784$ pixel từ ảnh MNIST $28 \\times 28$).  
                    - **Lớp ẩn (Hidden Layers)**: Xử lý thông tin thông qua các phép tính tuyến tính và phi tuyến.  
                    - **Lớp đầu ra (Output Layer)**: Đưa ra dự đoán (10 lớp, tương ứng với các chữ số $0$-$9$).  
                    """, unsafe_allow_html=True)

                    st.subheader("🔧 Quy trình hoạt động")
                    st.markdown("""
                    Neural Network hoạt động qua các bước sau, được tối ưu hóa dựa trên các tham số bạn có thể điều chỉnh trong tab **Huấn luyện/Đánh giá**:
                    """, unsafe_allow_html=True)

                    st.subheader("1. Khởi tạo mô hình")
                    st.markdown("""
                    - Xác định cấu trúc mạng (số lớp ẩn, số nơ-ron mỗi lớp) và khởi tạo **trọng số** ($W$) và **bias** ($b$) ngẫu nhiên (thường từ phân phối Gaussian).  
                    - **Tham số liên quan**: Số lớp ẩn, số nơ-ron mỗi lớp.  
                    - **Chú thích**:  
                      - $W$: Ma trận trọng số (weights) kết nối các nơ-ron giữa các lớp.  
                      - $b$: Vector bias (độ lệch) giúp điều chỉnh đầu ra của nơ-ron.  
                    - Mục đích: Thiết lập cấu trúc ban đầu để bắt đầu quá trình học.  
                    """, unsafe_allow_html=True)
                    try:
                        st.image(os.path.join("plnw", "step1_init.png"), caption="Minh họa: Khởi tạo mô hình", width=700)
                    except FileNotFoundError:
                        st.error("Không tìm thấy ảnh minh họa 'step1_init.png'.")
                    except Exception as e:
                        st.error(f"Lỗi khi tải ảnh: {e}")

                    st.subheader("2. Lan truyền thuận (Feedforward)")
                    st.markdown("""
                    - Tính toán đầu ra dự đoán ($\\hat{Y}$) từ đầu vào $X$ qua các lớp:  
                      $$ Z^{(l)} = A^{(l-1)} \\cdot W^{(l)} + b^{(l)} $$  
                      $$ A^{(l)} = \\text{hàm kích hoạt}(Z^{(l)}) $$  
                    - **Chú thích**:  
                      - $Z^{(l)}$: Tổng trọng số đầu vào tại lớp $l$ (trước khi áp dụng hàm kích hoạt).  
                      - $A^{(l-1)}$: Đầu ra của lớp trước ($l-1$), là đầu vào của lớp $l$.  
                      - $W^{(l)}$: Ma trận trọng số của lớp $l$.  
                      - $b^{(l)}$: Vector bias của lớp $l$.  
                      - $A^{(l)}$: Đầu ra của lớp $l$ sau khi áp dụng hàm kích hoạt.  
                    - Mục đích: Tạo dự đoán ban đầu từ dữ liệu đầu vào qua các lớp nơ-ron.  
                    """, unsafe_allow_html=True)
                    try:
                        st.image(os.path.join("plnw", "step2_feedforward.png"), caption="Minh họa: Lan truyền thuận", width=700)
                    except FileNotFoundError:
                        st.error("Không tìm thấy ảnh minh họa 'step2_feedforward.png'.")
                    except Exception as e:
                        st.error(f"Lỗi khi tải ảnh: {e}")

                    st.subheader("3. Tính hàm mất mát (Loss Function)")
                    st.markdown("""
                    - Đo độ sai lệch giữa dự đoán ($\\hat{Y}$) và nhãn thực ($Y$) bằng **Cross-Entropy**:  
                      $$ L = -\\frac{1}{N} \\sum_{i=1}^{N} \\sum_{j=0}^{9} y_{ij} \\cdot \\log(\\hat{y}_{ij}) $$  
                    - **Chú thích**:  
                      - $L$: Giá trị mất mát (loss) tổng thể của mô hình.  
                      - $N$: Số lượng mẫu trong tập dữ liệu.  
                      - $y_{ij}$: Giá trị thực tế (1 nếu mẫu $i$ thuộc lớp $j$, 0 nếu không).  
                      - $\\hat{y}_{ij}$: Xác suất dự đoán bởi mô hình cho mẫu $i$ thuộc lớp $j$.  
                    - Mục đích: Định lượng sai lệch để điều chỉnh mô hình trong bước tiếp theo.  
                    """, unsafe_allow_html=True)
                    try:
                        st.image(os.path.join("plnw", "step3_loss.png"), caption="Minh họa: Tính hàm mất mát", width=700)
                    except FileNotFoundError:
                        st.error("Không tìm thấy ảnh minh họa 'step3_loss.png'.")
                    except Exception as e:
                        st.error(f"Lỗi khi tải ảnh: {e}")

                    st.subheader("4. Lan truyền ngược (Backpropagation)")
                    st.markdown("""
                    - Tính đạo hàm của $L$ để cập nhật $W^{(l)}$ và $b^{(l)}$ nhằm giảm sai số dự đoán.  
                    - **Chú thích**:  
                      - $\\frac{\\partial L}{\\partial W^{(l)}}$: Đạo hàm riêng của mất mát $L$ theo trọng số $W^{(l)}$.  
                      - $\\frac{\\partial L}{\\partial b^{(l)}}$: Đạo hàm riêng của mất mát $L$ theo bias $b^{(l)}$.  
                    - Mục đích: Xác định hướng điều chỉnh tham số dựa trên sai số.  
                    """, unsafe_allow_html=True)
                    try:
                        st.image(os.path.join("plnw", "step4_backprop.png"), caption="Minh họa: Lan truyền ngược", width=700)
                    except FileNotFoundError:
                        st.error("Không tìm thấy ảnh minh họa 'step4_backprop.png'.")
                    except Exception as e:
                        st.error(f"Lỗi khi tải ảnh: {e}")

                    st.subheader("5. Cập nhật tham số (Gradient Descent)")
                    st.markdown("""
                    - Điều chỉnh $W^{(l)}$ và $b^{(l)}$ để giảm mất mát:  
                      $$ W^{(l)} = W^{(l)} - \\eta \\cdot \\frac{\\partial L}{\\partial W^{(l)}} $$  
                      $$ b^{(l)} = b^{(l)} - \\eta \\cdot \\frac{\\partial L}{\\partial b^{(l)}} $$  
                    - **Chú thích**:  
                      - $\\eta$: Tốc độ học (learning rate), kiểm soát mức độ thay đổi của $W$ và $b$.  
                      - $\\frac{\\partial L}{\\partial W^{(l)}}$: Gradient của $L$ theo $W^{(l)}$.  
                      - $\\frac{\\partial L}{\\partial b^{(l)}}$: Gradient của $L$ theo $b^{(l)}$.  
                    - Mục đích: Tối ưu hóa tham số để giảm sai số dự đoán.  
                    """, unsafe_allow_html=True)
                    try:
                        st.image(os.path.join("plnw", "step5_gradient.png"), caption="Minh họa: Cập nhật tham số", width=700)
                    except FileNotFoundError:
                        st.error("Không tìm thấy ảnh minh họa 'step5_gradient.png'.")
                    except Exception as e:
                        st.error(f"Lỗi khi tải ảnh: {e}")

                    st.subheader("6. Lặp lại")
                    st.markdown("""
                    - Lặp lại từ bước 2 qua nhiều **epoch** cho đến khi mất mát $L$ hội tụ.  
                    - **Chú thích**:  
                      - **Epoch**: Một lần lặp qua toàn bộ tập dữ liệu huấn luyện.  
                    - Mục đích: Tinh chỉnh mô hình qua nhiều vòng lặp để đạt hiệu suất tối ưu.  
                    """, unsafe_allow_html=True)
                    try:
                        st.image(os.path.join("plnw", "step6_repeat.png"), caption="Minh họa: Lặp lại", width=700)
                    except FileNotFoundError:
                        st.error("Không tìm thấy ảnh minh họa 'step6_repeat.png'.")
                    except Exception as e:
                        st.error(f"Lỗi khi tải ảnh: {e}")

                    st.subheader("🔧 Các tham số huấn luyện.")
                    st.markdown("""
                    Dưới đây là các tham số chính trong quá trình huấn luyện Neural Network, ý nghĩa của chúng, cách hoạt động và công thức (nếu có):

                    1. **Số lớp ẩn (Number of Hidden Layers):**  
                       - **Ý nghĩa**: Quyết định độ sâu của mạng, ảnh hưởng đến khả năng học các đặc trưng phức tạp.  
                       - **Hoạt động**: Tăng số lớp ẩn giúp mạng học được các đặc trưng cấp cao hơn, nhưng quá nhiều lớp có thể gây khó hội tụ hoặc overfitting.  
                       - **Công thức**: Không có công thức cụ thể, thường được chọn dựa trên kinh nghiệm hoặc thử nghiệm (trong ứng dụng này: từ 1 đến 5).  

                    2. **Số nơ-ron mỗi lớp ẩn (Number of Neurons per Layer):**  
                       - **Ý nghĩa**: Quyết định độ rộng của mạng, tức là khả năng biểu diễn thông tin trong mỗi lớp.  
                       - **Hoạt động**: Nhiều nơ-ron hơn giúp mạng học được nhiều đặc trưng hơn, nhưng cũng tăng chi phí tính toán.  
                       - **Công thức**: Không có, thường là lũy thừa của 2 (16, 32, 64, 128, v.v.) để tối ưu hóa phần cứng.  

                    3. **Tốc độ học (Learning Rate - η):**  
                       - **Ý nghĩa**: Điều chỉnh mức độ thay đổi của trọng số trong mỗi lần cập nhật.  
                       - **Hoạt động**: Giá trị nhỏ (ví dụ: 0.0001) làm mô hình học chậm nhưng ổn định; giá trị lớn (ví dụ: 0.01) học nhanh hơn nhưng dễ vượt qua điểm tối ưu.  
                       - **Công thức**:  
                         $$ W_{t+1} = W_t - \\eta \\cdot \\frac{\\partial L}{\\partial W_t} $$  
                         - $W_{t+1}$: Trọng số sau khi cập nhật.  
                         - $W_t$: Trọng số tại bước hiện tại.  
                         - $\\eta$: Tốc độ học.  
                         - $\\frac{\\partial L}{\\partial W_t}$: Gradient của mất mát theo trọng số.  

                    4. **Số lần lặp (Epochs):**  
                       - **Ý nghĩa**: Số lần toàn bộ dữ liệu huấn luyện được đưa qua mạng.  
                       - **Hoạt động**: Tăng số lần lặp giúp mạng học tốt hơn, nhưng quá nhiều có thể dẫn đến overfitting.  
                       - **Công thức**: Không có, là tham số người dùng chọn (trong ứng dụng này: 10-200).  

                    5. **Kích thước batch (Batch Size):**  
                       - **Ý nghĩa**: Số mẫu được xử lý trước khi cập nhật trọng số.  
                       - **Hoạt động**: Batch nhỏ (ví dụ: 16) giúp cập nhật thường xuyên hơn nhưng chậm; batch lớn (ví dụ: 512) nhanh hơn nhưng cần nhiều bộ nhớ.  
                       - **Công thức**: Không có, thường là lũy thừa của 2 để tối ưu hóa tính toán.  

                    6. **Hàm kích hoạt (Activation Function):**  
                       - **Ý nghĩa**: Quyết định cách nơ-ron "kích hoạt" đầu ra dựa trên đầu vào.  
                       - **Hoạt động**: Chuyển đổi đầu ra tuyến tính thành phi tuyến để mạng học được các đặc trưng phức tạp.  
                       - **Chi tiết các hàm kích hoạt phổ biến:**  
                         - **ReLU (Rectified Linear Unit):**  
                           - **Ý nghĩa**: Đơn giản, nhanh, tránh vấn đề biến mất gradient.  
                           - **Hoạt động**: Chỉ cho phép các giá trị dương đi qua, đặt giá trị âm về 0.  
                           - **Công thức**:  
                             $$ f(x) = \\max(0, x) $$  
                             - $x$: Đầu vào của hàm.  
                         - **Tanh (Hyperbolic Tangent):**  
                           - **Ý nghĩa**: Chuẩn hóa đầu ra về khoảng [-1, 1], phù hợp khi cần cân bằng giá trị âm/dương.  
                           - **Hoạt động**: Tạo đầu ra phi tuyến, nhưng dễ gặp vấn đề biến mất gradient với mạng sâu.  
                           - **Công thức**:  
                             $$ f(x) = \\frac{e^x - e^{-x}}{e^x + e^{-x}} $$  
                             - $x$: Đầu vào của hàm.  
                         - **Softmax:**  
                           - **Ý nghĩa**: Dùng ở lớp đầu ra để chuyển đổi thành xác suất cho phân loại đa lớp.  
                           - **Hoạt động**: Chuẩn hóa tổng các đầu ra thành 1, giúp dự đoán lớp có xác suất cao nhất.  
                           - **Công thức**:  
                             $$ f(x_i) = \\frac{e^{x_i}}{\\sum_{j=0}^{k} e^{x_j}} $$  
                             - $x_i$: Đầu vào của nơ-ron thứ $i$.  
                             - $k$: Số lớp (ở đây là 10).  

                    7. **Trình tối ưu (Optimizer):**  
                       - **Ý nghĩa**: Thuật toán điều chỉnh trọng số để giảm hàm mất mát.  
                       - **Hoạt động**: Dùng gradient để cập nhật tham số, với cách tiếp cận khác nhau tùy thuật toán.  
                       - **Ví dụ phổ biến:**  
                         - **SGD (Stochastic Gradient Descent):**  
                           - **Ý nghĩa**: Cập nhật trọng số dựa trên gradient của một mẫu/mini-batch.  
                           - **Công thức**:  
                             $$ W_{t+1} = W_t - \\eta \\cdot \\frac{\\partial L}{\\partial W_t} $$  
                             - $W_t$: Trọng số hiện tại.  
                             - $\\eta$: Tốc độ học.  
                             - $\\frac{\\partial L}{\\partial W_t}$: Gradient.  
                           - **Ưu điểm**: Đơn giản, nhanh với dữ liệu lớn.  
                           - **Nhược điểm**: Dao động, hội tụ chậm.  
                         - **Adam (Adaptive Moment Estimation):**  
                           - **Ý nghĩa**: Kết hợp động lượng và RMSProp, thích nghi tốc độ học cho từng tham số.  
                           - **Công thức**:  
                             1. $m_t = \\beta_1 \\cdot m_{t-1} + (1 - \\beta_1) \\cdot g_t$ (moment bậc 1).  
                             2. $v_t = \\beta_2 \\cdot v_{t-1} + (1 - \\beta_2) \\cdot g_t^2$ (moment bậc 2).  
                             3. $\\hat{m}_t = \\frac{m_t}{1 - \\beta_1^t}, \\hat{v}_t = \\frac{v_t}{1 - \\beta_2^t}$ (hiệu chỉnh).  
                             4. $W_{t+1} = W_t - \\eta \\cdot \\frac{\\hat{m}_t}{\\sqrt{\\hat{v}_t} + \\epsilon}$.  
                             - $g_t$: Gradient.  
                             - $\\beta_1 \\approx 0.9, \\beta_2 \\approx 0.999, \\epsilon \\approx 10^{-8}$.  
                           - **Ưu điểm**: Nhanh, ổn định, hiệu quả.  
                           - **Nhược điểm**: Phức tạp, đôi khi kém trên hàm không lồi.  
                       - **So sánh**: SGD chậm, dao động; Adam nhanh, ổn định.  
                    """, unsafe_allow_html=True)

                    st.subheader("🌟 Ưu điểm và nhược điểm của Neural Network")
                    st.markdown("""
                    #### **Ưu điểm:**  
                    - **Khả năng học phi tuyến tính**: Neural Network có thể học các mối quan hệ phức tạp, phi tuyến tính trong dữ liệu mà các mô hình tuyến tính không làm được.  
                    - **Khả năng mở rộng**: Có thể xử lý dữ liệu lớn và nhiều chiều (như ảnh, âm thanh) khi được huấn luyện đúng cách.  
                    - **Tính linh hoạt**: Có thể áp dụng cho nhiều bài toán khác nhau (phân loại, hồi quy, nhận diện hình ảnh, v.v.).  
                    - **Tự động học đặc trưng**: Không cần trích xuất đặc trưng thủ công, mạng tự động học từ dữ liệu thô.  

                    #### **Nhược điểm:**  
                    - **Đòi hỏi tài nguyên lớn**: Cần nhiều dữ liệu và sức mạnh tính toán (CPU/GPU) để huấn luyện hiệu quả.  
                    - **Khó giải thích**: Mạng hoạt động như "hộp đen", khó hiểu tại sao lại đưa ra dự đoán cụ thể.  
                    - **Dễ bị overfitting**: Nếu không được điều chỉnh tốt (ví dụ: thiếu dữ liệu hoặc không dùng regularization), mô hình có thể học quá mức dữ liệu huấn luyện.  
                    - **Thời gian huấn luyện lâu**: Đặc biệt với mạng sâu hoặc dữ liệu lớn.  
                    """, unsafe_allow_html=True)

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
                    **Pseudo-Labeling** là một phương pháp học bán giám sát (semi-supervised learning) giúp tận dụng cả dữ liệu có nhãn và không có nhãn để nâng cao hiệu suất mô hình, đặc biệt hữu ích khi dữ liệu có nhãn khan hiếm. Kỹ thuật này sử dụng mô hình đã huấn luyện để dự đoán nhãn giả (pseudo-labels) cho dữ liệu không có nhãn, sau đó kết hợp chúng vào quá trình huấn luyện.

                    **Các bước thực hiện Pseudo-Labeling với Neural Network:**
                    1. **Chuẩn bị dữ liệu và chia tập train/test**  
                       - Chuẩn hóa dữ liệu (ví dụ: đưa về thang [0, 1]) và chia thành tập huấn luyện (train) và tập kiểm tra (test).  
                       - Minh họa:  
                       """, unsafe_allow_html=True)
                    try:
                        st.image(os.path.join("mhpersoudo", "pseudo_step1.png"), caption="Chuẩn bị dữ liệu và chia tập", width=600)
                    except FileNotFoundError:
                        st.warning("Không tìm thấy ảnh minh họa 'pseudo_step1.png'. Vui lòng kiểm tra đường dẫn.")
                    except Exception as e:
                        st.error(f"Lỗi khi tải ảnh: {e}")
                    st.markdown("""
                    2. **Lấy 1% số lượng ảnh cho mỗi lớp (0-9) làm tập ban đầu**  
                       - Chọn 1% mẫu từ mỗi lớp trong tập train để tạo tập dữ liệu có nhãn ban đầu, phần còn lại (99%) là dữ liệu không có nhãn.  
                       - Minh họa:  
                       """, unsafe_allow_html=True)
                    try:
                        st.image(os.path.join("mhpersoudo", "pseudo_step2.png"), caption="Lấy 1% dữ liệu có nhãn ban đầu", width=600)
                    except FileNotFoundError:
                        st.warning("Không tìm thấy ảnh minh họa 'pseudo_step2.png'. Vui lòng kiểm tra đường dẫn.")
                    except Exception as e:
                        st.error(f"Lỗi khi tải ảnh: {e}")
                    st.markdown("""
                    3. **Huấn luyện mô hình Neural Network trên tập 1% ban đầu**  
                       - Sử dụng tập dữ liệu có nhãn (1%) để huấn luyện một mô hình Neural Network cơ bản.  
                       - Minh họa:  
                       """, unsafe_allow_html=True)
                    try:
                        st.image(os.path.join("mhpersoudo", "pseudo_step3.png"), caption="Huấn luyện mô hình trên 1% dữ liệu", width=600)
                    except FileNotFoundError:
                        st.warning("Không tìm thấy ảnh minh họa 'pseudo_step3.png'. Vui lòng kiểm tra đường dẫn.")
                    except Exception as e:
                        st.error(f"Lỗi khi tải ảnh: {e}")
                    st.markdown("""
                    4. **Dự đoán nhãn cho dữ liệu không có nhãn (99%)**  
                       - Sử dụng mô hình đã huấn luyện để dự đoán nhãn và độ tin cậy cho tập dữ liệu không có nhãn.  
                       - Minh họa:  
                       """, unsafe_allow_html=True)
                    try:
                        st.image(os.path.join("mhpersoudo", "pseudo_step4.png"), caption="Dự đoán nhãn cho dữ liệu không có nhãn", width=600)
                    except FileNotFoundError:
                        st.warning("Không tìm thấy ảnh minh họa 'pseudo_step4.png'. Vui lòng kiểm tra đường dẫn.")
                    except Exception as e:
                        st.error(f"Lỗi khi tải ảnh: {e}")
                    st.markdown("""
                    5. **Gán nhãn giả với ngưỡng tin cậy (threshold = 0.95)**  
                       - Lọc các dự đoán có độ tin cậy ≥ 0.95 để gán nhãn giả, các mẫu còn lại giữ nguyên là không có nhãn.  
                       - Minh họa:  
                       """, unsafe_allow_html=True)
                    try:
                        st.image(os.path.join("mhpersoudo", "pseudo_step5.png"), caption="Gán nhãn giả với ngưỡng tin cậy", width=600)
                    except FileNotFoundError:
                        st.warning("Không tìm thấy ảnh minh họa 'pseudo_step5.png'. Vui lòng kiểm tra đường dẫn.")
                    except Exception as e:
                        st.error(f"Lỗi khi tải ảnh: {e}")
                    st.markdown("""
                    6. **Huấn luyện lại mô hình với tập dữ liệu mới**  
                       - Kết hợp tập 1% ban đầu với dữ liệu vừa gán nhãn giả để huấn luyện lại mô hình.  
                       - Minh họa:  
                       """, unsafe_allow_html=True)
                    try:
                        st.image(os.path.join("mhpersoudo", "pseudo_step6.png"), caption="Huấn luyện lại với dữ liệu mới", width=600)
                    except FileNotFoundError:
                        st.warning("Không tìm thấy ảnh minh họa 'pseudo_step6.png'. Vui lòng kiểm tra đường dẫn.")
                    except Exception as e:
                        st.error(f"Lỗi khi tải ảnh: {e}")
                    st.markdown("""
                    7. **Lặp lại các bước 4-6 cho đến khi đạt điều kiện dừng**  
                       - Tiếp tục dự đoán, gán nhãn giả và huấn luyện lại cho đến khi không còn dữ liệu không có nhãn hoặc đạt số vòng lặp tối đa (ví dụ: 5 vòng).  
                       - Minh họa:  
                       """, unsafe_allow_html=True)
                    try:
                        st.image(os.path.join("mhpersoudo", "pseudo_step7.png"), caption="Lặp lại quy trình cho đến điều kiện dừng", width=600)
                    except FileNotFoundError:
                        st.warning("Không tìm thấy ảnh minh họa 'pseudo_step7.png'. Vui lòng kiểm tra đường dẫn.")
                    except Exception as e:
                        st.error(f"Lỗi khi tải ảnh: {e}")
                    st.markdown("""
                    8. **Huấn luyện lần cuối và đánh giá**  
                       - Huấn luyện mô hình cuối cùng trên toàn bộ dữ liệu đã gắn nhãn và đánh giá trên tập test.  
                       - Minh họa:  
                       """, unsafe_allow_html=True)
                    try:
                        st.image(os.path.join("mhpersoudo", "pseudo_step8.png"), caption="Huấn luyện lần cuối và đánh giá", width=600)
                    except FileNotFoundError:
                        st.warning("Không tìm thấy ảnh minh họa 'pseudo_step8.png'. Vui lòng kiểm tra đường dẫn.")
                    except Exception as e:
                        st.error(f"Lỗi khi tải ảnh: {e}")

                    st.markdown("""
                    **Lợi ích:**
                    - Tối ưu hóa hiệu suất mô hình bằng cách khai thác dữ liệu không có nhãn.
                    - Giảm chi phí gắn nhãn thủ công trong các dự án thực tế.

                    **Thách thức:**
                    - Nhãn giả có thể chứa nhiễu nếu mô hình ban đầu chưa đủ chính xác.
                    - Yêu cầu điều chỉnh ngưỡng tin cậy để cân bằng giữa chất lượng và số lượng nhãn giả.
                    """, unsafe_allow_html=True)

                    st.subheader("⚙️ Các tham số của Pseudo-Labeling trong Huấn luyện")
                    st.markdown("""
                    Trong quá trình huấn luyện bài toán phân loại MNIST với Pseudo-Labeling, các tham số sau được sử dụng để điều khiển kỹ thuật học bán giám sát này:

                    | **Tham số**            | **Mô tả**                                                                |
                    |------------------------|--------------------------------------------------------------------------|
                    | **Ngưỡng tin cậy**     | Mức độ tin cậy tối thiểu để gán nhãn giả cho dữ liệu không có nhãn.      |
                    | **Số vòng lặp tối đa** | Số lần lặp tối đa của quy trình Pseudo-Labeling để gắn nhãn và huấn luyện.|

                    **Chi tiết:**
                    - **Ngưỡng tin cậy (threshold)**:  
                      - Công thức: Nếu độ tin cậy dự đoán $P(y|x) \geq \text{threshold}$, mẫu sẽ được gán nhãn giả.  
                      - Ví dụ: Với threshold = 0.95, chỉ các dự đoán có độ tin cậy ≥ 95% được chấp nhận.  
                      - Tác động: Giá trị cao đảm bảo chất lượng nhãn giả nhưng giảm số lượng mẫu được gắn nhãn; giá trị thấp tăng số lượng mẫu nhưng có thể gây nhiễu.

                    - **Số vòng lặp tối đa (max_iterations)**:  
                      - Quy định số lần mô hình dự đoán nhãn giả và huấn luyện lại trên dữ liệu mới.  
                      - Điều kiện dừng: Quy trình kết thúc khi hết dữ liệu không có nhãn hoặc đạt số vòng lặp tối đa.  
                      - Tác động: Giá trị lớn tăng cơ hội khai thác dữ liệu không nhãn nhưng kéo dài thời gian huấn luyện.
                    """, unsafe_allow_html=True)

    ### Tab 2: Chọn số lượng dữ liệu
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
                del X_full, y_full, X_sampled, y_sampled
                gc.collect()

    ### Tab 3: Xử lý dữ liệu
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
                fig, axes = plt.subplots(2, 5, figsize=(10, 4))
                for i, ax in enumerate(axes.flat):
                    ax.imshow(X_processed[i].reshape(28, 28), cmap='gray')
                    ax.set_title(f"Label: {y_processed[i]}")
                    ax.axis("off")
                st.pyplot(fig)
                plt.close(fig)

    ### Tab 4: Chia dữ liệu
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
                    del X, y, X_train, X_test, y_train, y_test
                    gc.collect()

    ### Tab 5: Huấn luyện/Đánh giá
    with tab_train_eval:
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
                            status_text = st.empty()  # Placeholder cho vòng hiện tại
                            epoch_text = st.empty()   # Placeholder cho epoch hiện tại
                            loss_text = st.empty()    # Placeholder cho loss
                            acc_text = st.empty()     # Placeholder cho accuracy

                            # Tạo tập dữ liệu có nhãn ban đầu (dựa trên labeled_pct mỗi lớp)
                            labeled_indices = []
                            for digit in range(10):
                                digit_indices = np.where(y_train == digit)[0]
                                if len(digit_indices) > 0:
                                    train_size = min(int(len(digit_indices) * (labeled_pct / 100)), len(digit_indices))
                                    if train_size < 1 and len(digit_indices) > 0:
                                        train_size = 1  # Đảm bảo lấy ít nhất 1 mẫu nếu lớp có dữ liệu
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
                            test_acc_history = []  # Lưu độ chính xác trên tập test sau mỗi vòng
                            pseudo_samples = []    # Lưu thông tin mẫu được gán nhãn giả
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

                                # Đánh giá trên tập test sau mỗi vòng để kiểm chứng hiệu quả
                                test_pred = np.argmax(model.predict(X_test, verbose=0), axis=1)
                                test_acc = accuracy_score(y_test, test_pred)
                                test_acc_history.append(test_acc)

                                # Dự đoán nhãn cho tập dữ liệu không có nhãn
                                predictions = model.predict(X_unlabeled, verbose=0)
                                max_probs = np.max(predictions, axis=1)
                                pseudo_labels = np.argmax(predictions, axis=1)

                                # Lọc các mẫu có độ tin cậy cao
                                high_confidence_mask = max_probs >= threshold
                                if not np.any(high_confidence_mask):
                                    break

                                pseudo_indices = unlabeled_indices[high_confidence_mask]

                                # Thu thập thông tin mẫu được gán nhãn giả để minh họa
                                if len(pseudo_indices) > 0:
                                    selected_indices = np.random.choice(pseudo_indices, size=min(5, len(pseudo_indices)), replace=False)
                                    samples = []
                                    for idx in selected_indices:
                                        i = np.where(unlabeled_indices == idx)[0][0]
                                        image = X_unlabeled[i].copy()
                                        pseudo_label = pseudo_labels[i]
                                        confidence = max_probs[i]
                                        true_label = y_train[idx]
                                        samples.append({
                                            'image': image,
                                            'pseudo_label': pseudo_label,
                                            'confidence': confidence,
                                            'true_label': true_label
                                        })
                                    pseudo_samples.append({
                                        'iteration': iteration,
                                        'samples': samples,
                                        'num_added': len(pseudo_indices),
                                        'total_labeled': len(X_labeled) + len(pseudo_indices)
                                    })

                                # Gán nhãn giả và thêm vào tập dữ liệu có nhãn
                                X_labeled = np.vstack((X_labeled, X_unlabeled[high_confidence_mask]))
                                y_labeled = np.hstack((y_labeled, pseudo_labels[high_confidence_mask]))

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
                                'test_acc_history': test_acc_history,
                                'pseudo_samples': pseudo_samples,
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
                col1, col2 = st.columns(2)
                col1.metric("Thời gian huấn luyện", f"{results['training_time']:.2f} giây")
                col2.metric("Độ chính xác Test", f"{results['accuracy_test']*100:.2f}%")

                st.subheader("Ma trận Nhầm lẫn")
                fig, ax = plt.subplots()
                sns.heatmap(results['cm_test'], annot=True, fmt="d", cmap="Blues", ax=ax)
                ax.set_title("Test")
                st.pyplot(fig)
                plt.close(fig)

                # Biểu đồ Loss và Accuracy theo số vòng
                st.subheader("Biểu đồ Loss và Accuracy theo Vòng")
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
                ax1.plot(range(1, len(results['loss_history']) + 1), results['loss_history'])
                ax1.set_title("Loss qua các vòng")
                ax1.set_xlabel("Vòng")
                ax1.set_ylabel("Loss")
                ax2.plot(range(1, len(results['accuracy_history']) + 1), results['accuracy_history'])
                ax2.set_title("Accuracy qua các vòng")
                ax2.set_xlabel("Vòng")
                ax2.set_ylabel("Accuracy")
                st.pyplot(fig)
                plt.close(fig)

                # Biểu đồ độ chính xác trên Test qua các vòng
                if 'test_acc_history' in results:
                    st.subheader("Biểu đồ Độ chính xác trên Test qua các Vòng")
                    fig, ax = plt.subplots()
                    ax.plot(range(1, len(results['test_acc_history']) + 1), results['test_acc_history'])
                    ax.set_title("Độ chính xác trên Test qua các Vòng")
                    ax.set_xlabel("Vòng")
                    ax.set_ylabel("Độ chính xác")
                    st.pyplot(fig)
                    plt.close(fig)

                # Minh họa các mẫu được gán nhãn Pseudo
                if 'pseudo_samples' in results:
                    st.subheader("Minh họa các mẫu được gán nhãn Pseudo")
                    for iter_data in results['pseudo_samples']:
                        with st.expander(f"Vòng {iter_data['iteration']}"):
                            st.write(f"Số mẫu được thêm vào: {iter_data['num_added']}")
                            st.write(f"Tổng số mẫu có nhãn sau vòng này: {iter_data['total_labeled']}")
                            num_samples = len(iter_data['samples'])
                            if num_samples > 0:
                                fig, axes = plt.subplots(1, num_samples, figsize=(3*num_samples, 3))
                                if num_samples == 1:
                                    axes = [axes]
                                for ax, sample in zip(axes, iter_data['samples']):
                                    ax.imshow(sample['image'].reshape(28, 28), cmap='gray')
                                    ax.set_title(f"Pseudo: {sample['pseudo_label']}\nTrue: {sample['true_label']}\nConf: {sample['confidence']:.2f}")
                                    ax.axis('off')
                                st.pyplot(fig)
                                plt.close(fig)

                # Tóm tắt kết quả huấn luyện
                st.markdown("#### 📋 Tóm tắt Kết quả Huấn luyện")
                full_data = {
                    "Vòng": list(range(1, len(results['loss_history']) + 1)),
                    "Loss": results['loss_history'],
                    "Accuracy": results['accuracy_history'],
                }
                df_full = pd.DataFrame(full_data)

                if 'display_iterations' not in st.session_state:
                    st.session_state['display_iterations'] = 5

                st.table(df_full.head(st.session_state['display_iterations']))

                if len(results['loss_history']) > st.session_state['display_iterations']:
                    if st.button("Xem thêm 5 vòng", key="show_more_iterations"):
                        st.session_state['display_iterations'] += 5
                        st.rerun()

                if st.session_state['display_iterations'] > 5:
                    if st.button("Thu gọn", key="collapse_iterations"):
                        st.session_state['display_iterations'] = 5
                        st.rerun()

                # Thêm phần chi tiết kết quả huấn luyện
                with st.expander("Xem chi tiết", expanded=False):
                    st.markdown("**Thông tin lần chạy:**")
                    st.write(f"- Tên: {results['run_name']}")
                    st.write(f"- ID: {results['run_id']}")
                    st.write(f"- Thời gian huấn luyện: {results['training_time']:.2f} giây")
                    st.write(f"- Số lần lặp thực tế: {results['n_iter_actual']}")
                    st.write(f"- Độ chính xác Test: {results['accuracy_test']*100:.2f}%")
                    st.markdown("**Tham số đã chọn:**")
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

    ### Tab 6: Demo dự đoán
    with tab_demo:
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

    ### Tab 7: Thông tin huấn luyện
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
                    col_loss, col_acc = st.columns(2)
                    with col_loss:
                        if 'training_results' in st.session_state and selected_run_id == st.session_state['training_results']['run_id']:
                            results = st.session_state['training_results']
                            if 'loss_history' in results:
                                fig, ax = plt.subplots(figsize=(6, 4))
                                ax.plot(range(1, len(results['loss_history']) + 1), results['loss_history'], 
                                        label='Training Loss', color='blue', linewidth=2)
                                ax.set_xlabel("Vòng")
                                ax.set_ylabel("Loss")
                                ax.set_title("Lịch sử Mất mát")
                                ax.legend()
                                ax.grid(True)
                                st.pyplot(fig)
                                plt.close(fig)
                    with col_acc:
                        if 'training_results' in st.session_state and selected_run_id == st.session_state['training_results']['run_id']:
                            results = st.session_state['training_results']
                            if 'accuracy_history' in results:
                                fig, ax = plt.subplots(figsize=(6, 4))
                                ax.plot(range(1, len(results['accuracy_history']) + 1), results['accuracy_history'], 
                                        label='Training Accuracy', color='green', linewidth=2)
                                ax.set_xlabel("Vòng")
                                ax.set_ylabel("Accuracy")
                                ax.set_title("Lịch sử Độ chính xác")
                                ax.legend()
                                ax.grid(True)
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