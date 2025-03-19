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

# Hàm xây dựng mô hình Neural Network
def build_model(params):
    """Xây dựng và biên dịch mô hình Neural Network với tham số được cung cấp."""
    model = models.Sequential()
    model.add(layers.Input(shape=(784,)))
    for neurons in params["hidden_layer_sizes"]:
        model.add(layers.Dense(neurons, activation=params["activation"]))
    model.add(layers.Dense(10, activation='softmax'))
    optimizer = (tf.keras.optimizers.Adam(learning_rate=params["learning_rate"])
                 if params["solver"] == "adam" else
                 tf.keras.optimizers.SGD(learning_rate=params["learning_rate"]))
    model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

# Ứng dụng chính
def run_mnist_neural_network_app():
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

    st.title("Phân loại Chữ số MNIST với Neural Network")

    ### CSS tùy chỉnh
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
                    st.warning("Không tìm thấy ảnh minh họa 'mnist_overview.png'. Vui lòng kiểm tra đường dẫn.")
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
                st.subheader("📊 3. Neural Network – Mạng nơ-ron nhân tạo")
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
                    st.error("Không tìm thấy ảnh minh họa cho Bước 1.")
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
                    st.error("Không tìm thấy ảnh minh họa cho Bước 2.")
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
                    st.error("Không tìm thấy ảnh minh họa cho Bước 3.")
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
                    st.error("Không tìm thấy ảnh minh họa cho Bước 4.")
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
                    st.error("Không tìm thấy ảnh minh họa cho Bước 5.")
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
                    st.error("Không tìm thấy ảnh minh họa cho Bước 6.")
                except Exception as e:
                    st.error(f"Lỗi khi tải ảnh: {e}")

                st.subheader("🔧 Các tham số huấn luyện: Ý nghĩa, hoạt động và công thức")
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
                |-----------------------|---------------------------------------------------------------------------|
                | **Ngưỡng tin cậy**    | Mức độ tin cậy tối thiểu để gán nhãn giả cho dữ liệu không có nhãn.       |
                | **Số vòng lặp tối đa**| Số lần lặp tối đa của quy trình Pseudo-Labeling để gắn nhãn và huấn luyện.|

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
            "1000 mẫu (Thử nghiệm nhanh)": 1000,
            "10,000 mẫu (Kiểm tra cơ bản)": 10000,
            "50,000 mẫu (Cân bằng hiệu suất)": 50000,
            "70,000 mẫu (Huấn luyện chuyên sâu)": 70000,
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

            if st.button("Chuẩn hóa dữ liệu (Normalization)", type="primary"):
                with st.spinner("Đang chuẩn hóa dữ liệu về [0, 1]..."):
                    X_norm = X / 255.0
                    st.session_state["data_processed"] = (X_norm.copy(), y.copy())
                    st.success("Đã xử lý dữ liệu!")
                    del X, y, X_norm
                    gc.collect()
                    st.rerun()

            if "data_processed" in st.session_state:
                st.subheader("Dữ liệu sau khi xử lý")
                X_proc, y_proc = st.session_state["data_processed"]
                fig, axes = plt.subplots(2, 5, figsize=(10, 4))
                for i, ax in enumerate(axes.flat):
                    ax.imshow(X_proc[i].reshape(28, 28), cmap='gray')
                    ax.set_title(f"Label: {y_proc[i]}")
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
            valid_pct = st.slider("Tỷ lệ Validation (%)", 0, 50, 20)

            test_size = test_pct / 100
            X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
            valid_size = (valid_pct / 100) / (1 - test_size) if test_size < 1 else 0
            X_train, X_valid, y_train, y_valid = train_test_split(X_temp, y_temp, test_size=valid_size, random_state=42)

            st.write(f"**Phân bổ dữ liệu**: Train: {len(X_train)}, Validation: {len(X_valid)}, Test: {len(X_test)}")
            if st.button("Xác nhận phân chia", type="primary"):
                with st.spinner("Đang chia dữ liệu..."):
                    st.session_state['split_data'] = {
                        "X_train": X_train.copy(), "y_train": y_train.copy(),
                        "X_valid": X_valid.copy(), "y_valid": y_valid.copy(),
                        "X_test": X_test.copy(), "y_test": y_test.copy()
                    }
                    st.success("Đã chia dữ liệu thành công!")
                    del X, y, X_temp, y_temp, X_test, y_test, X_train, X_valid, y_train, y_valid
                    gc.collect()

    ### Tab 5: Huấn luyện/Đánh giá
    with tab_train_eval:
        st.markdown('<div class="section-title">Huấn luyện và Đánh giá Mô hình</div>', unsafe_allow_html=True)
        if 'split_data' not in st.session_state:
            st.info("Vui lòng chia dữ liệu trước.")
        else:
            split_data = st.session_state['split_data'].copy()
            X_train = split_data["X_train"]
            y_train = split_data["y_train"]
            X_valid = split_data["X_valid"]
            y_valid = split_data["y_valid"]
            X_test = split_data["X_test"]
            y_test = split_data["y_test"]

            X_train = np.array(X_train, dtype=np.float32)
            y_train = np.array(y_train, dtype=np.int32)
            X_valid = np.array(X_valid, dtype=np.float32)
            y_valid = np.array(y_valid, dtype=np.int32)
            X_test = np.array(X_test, dtype=np.float32)
            y_test = np.array(y_test, dtype=np.int32)

            if np.any(np.isnan(X_train)) or np.any(np.isnan(y_train)):
                st.error("Dữ liệu huấn luyện chứa NaN. Đang xử lý...")
                X_train = np.nan_to_num(X_train, nan=0.0)
                y_train = np.nan_to_num(y_train, nan=0.0)
                st.success("Đã thay thế NaN bằng 0!")

            num_samples = len(X_train)
            st.write(f"**Số mẫu huấn luyện**: {num_samples}")

            if "optimal_params" not in st.session_state:
                st.session_state["optimal_params"] = get_optimal_params(num_samples)
            params = st.session_state.get("training_params", st.session_state["optimal_params"].copy())

            #### Cấu hình mô hình
            st.subheader("⚙️ Cấu hình Mô hình")
            with st.expander("Tham số Tham khảo", expanded=False):
                st.markdown("""
                | Số mẫu       | Số lớp ẩn | Kích thước lớp ẩn | Tốc độ học | Số lần lặp | Hàm kích hoạt | Trình tối ưu | Kích thước batch |
                |--------------|-----------|-------------------|------------|------------|---------------|--------------|------------------|
                | ≤ 1,000      | 1         | 32                | 0.001      | 30         | ReLU          | Adam         | 32               |
                | ≤ 10,000     | 2         | (64, 32)          | 0.0005     | 50         | ReLU          | Adam         | 64               |
                | ≤ 50,000     | 2         | (128, 64)         | 0.0003     | 70         | ReLU          | Adam         | 128              |
                | > 50,000     | 3         | (128, 64, 32)     | 0.0001     | 100        | ReLU          | Adam         | 256              |
                """, unsafe_allow_html=True)
                st.info(f"Tham số tối ưu cho {num_samples} mẫu: {st.session_state['optimal_params']}")

            col_param1, col_param2 = st.columns(2)
            with col_param1:
                with st.expander("🧠 Cấu trúc Mạng", expanded=True):
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
                with st.expander("🔧 Tối ưu hóa", expanded=True):
                    params["learning_rate"] = st.number_input("Tốc độ học", min_value=0.0, step=0.0001, 
                                                              value=params["learning_rate"], format="%.4f")
                    params["epochs"] = st.number_input("Số lần lặp (Epochs)", min_value=1, value=params["epochs"])
                    params["batch_size"] = st.number_input("Kích thước batch", min_value=1, value=params["batch_size"])
                    params["solver"] = st.selectbox("Trình tối ưu", ["adam", "sgd"], 
                                                    index=["adam", "sgd"].index(params["solver"]))
                    early_stopping = st.checkbox("Dừng sớm (Early Stopping)", value=False)

            #### Chế độ huấn luyện
            st.subheader("🔄 Chế độ Huấn luyện")
            training_mode = st.selectbox("Chọn chế độ huấn luyện", ["Standard", "Pseudo Labelling"])
            pseudo_params = {}
            if training_mode == "Pseudo Labelling":
                with st.expander("⚙️ Cấu hình Pseudo Labelling", expanded=True):
                    pseudo_params["labeled_pct"] = st.number_input("Tỷ lệ dữ liệu có nhãn ban đầu mỗi lớp (%)", 
                                                                  min_value=0.1, max_value=100.0, value=1.0)
                    pseudo_params["threshold"] = st.number_input("Ngưỡng tin cậy", min_value=0.0, max_value=1.0, value=0.95)
                    pseudo_params["max_iterations"] = st.number_input("Số lần lặp tối đa", min_value=1, value=10)

            if st.button("Bắt đầu Huấn luyện", type="primary"):
                with st.spinner(f"Đang huấn luyện mô hình ({training_mode})..."):
                    start_time = time.time()
                    progress_bar = st.progress(0)
                    status_text = st.empty()

                    class ProgressCallback(callbacks.Callback):
                        def on_epoch_end(self, epoch, logs=None):
                            progress = (epoch + 1) / params["epochs"]
                            progress_bar.progress(min(progress, 1.0))
                            status_text.text(f"Epoch {epoch+1}/{params['epochs']}, Loss: {logs['loss']:.4f}, Accuracy: {logs['accuracy']:.4f}")

                    callbacks_list = [ProgressCallback()]
                    if early_stopping and training_mode == "Standard":
                        callbacks_list.append(callbacks.EarlyStopping(monitor='val_loss', patience=10))

                    model_name = f"Model_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                    if training_mode == "Standard":
                        model = build_model(params)
                        history = model.fit(X_train, y_train, epochs=params["epochs"], batch_size=params["batch_size"],
                                            validation_data=(X_valid, y_valid), callbacks=callbacks_list, verbose=0)
                        y_valid_pred = np.argmax(model.predict(X_valid, verbose=0), axis=1)
                        y_test_pred = np.argmax(model.predict(X_test, verbose=0), axis=1)
                        acc_valid = accuracy_score(y_valid, y_valid_pred)
                        acc_test = accuracy_score(y_test, y_test_pred)
                        cm_valid = confusion_matrix(y_valid, y_valid_pred)
                        cm_test = confusion_matrix(y_test, y_test_pred)

                        with mlflow.start_run(experiment_id=EXPERIMENT_ID, run_name=model_name) as run:
                            mlflow.log_params(params)
                            mlflow.log_metric("accuracy_val", acc_valid)
                            mlflow.log_metric("accuracy_test", acc_test)
                            mlflow.log_metric("training_time", time.time() - start_time)
                            mlflow.keras.log_model(model, "model")

                        results = {
                            'accuracy_val': acc_valid, 'accuracy_test': acc_test,
                            'cm_valid': cm_valid, 'cm_test': cm_test,
                            'run_name': model_name, 'run_id': run.info.run_id,
                            'params': params, 'training_time': time.time() - start_time,
                            'loss_history': history.history['loss'],
                            'val_loss_history': history.history.get('val_loss', []),
                            'accuracy_history': history.history['accuracy'],
                            'val_accuracy_history': history.history.get('val_accuracy', [])
                        }

                    else:  # Pseudo Labelling
                        labeled_indices = []
                        unlabeled_indices = []
                        for digit in range(10):
                            digit_indices = np.where(y_train == digit)[0]
                            labeled_digit, unlabeled_digit = train_test_split(digit_indices, 
                                                                              train_size=pseudo_params["labeled_pct"]/100, 
                                                                              random_state=42)
                            labeled_indices.extend(labeled_digit)
                            unlabeled_indices.extend(unlabeled_digit)
                        labeled_indices = np.array(labeled_indices)
                        unlabeled_indices = np.array(unlabeled_indices)
                        X_labeled = X_train[labeled_indices]
                        y_labeled = y_train[labeled_indices]

                        test_accuracies = []
                        pseudo_counts = []
                        with mlflow.start_run(experiment_id=EXPERIMENT_ID, run_name=model_name) as run:
                            mlflow.log_params({**params, **pseudo_params})
                            model = build_model(params)
                            history = model.fit(X_labeled, y_labeled, epochs=params["epochs"], 
                                                batch_size=params["batch_size"], callbacks=callbacks_list, verbose=0)
                            y_test_pred = np.argmax(model.predict(X_test, verbose=0), axis=1)
                            acc_test = accuracy_score(y_test, y_test_pred)
                            test_accuracies.append(acc_test)
                            pseudo_counts.append(0)
                            mlflow.log_metric("test_accuracy", acc_test, step=0)

                            for iteration in range(pseudo_params["max_iterations"]):
                                if len(unlabeled_indices) == 0:
                                    break
                                X_unlabeled = X_train[unlabeled_indices]
                                predictions = model.predict(X_unlabeled, verbose=0)
                                max_probs = np.max(predictions, axis=1)
                                pseudo_mask = max_probs > pseudo_params["threshold"]
                                if not np.any(pseudo_mask):
                                    break
                                pseudo_indices = unlabeled_indices[pseudo_mask]
                                pseudo_labels = np.argmax(predictions[pseudo_mask], axis=1)
                                labeled_indices = np.concatenate((labeled_indices, pseudo_indices))
                                y_labeled = np.concatenate((y_labeled, pseudo_labels))
                                unlabeled_indices = unlabeled_indices[~pseudo_mask]
                                X_labeled = X_train[labeled_indices]
                                model = build_model(params)
                                history = model.fit(X_labeled, y_labeled, epochs=params["epochs"], 
                                                    batch_size=params["batch_size"], callbacks=callbacks_list, verbose=0)
                                y_test_pred = np.argmax(model.predict(X_test, verbose=0), axis=1)
                                acc_test = accuracy_score(y_test, y_test_pred)
                                test_accuracies.append(acc_test)
                                pseudo_counts.append(len(pseudo_indices))
                                mlflow.log_metric("test_accuracy", acc_test, step=iteration+1)
                                mlflow.log_metric("pseudo_labeled_count", len(pseudo_indices), step=iteration+1)
                                progress_bar.progress((iteration + 1) / pseudo_params["max_iterations"])

                            y_valid_pred = np.argmax(model.predict(X_valid, verbose=0), axis=1)
                            y_test_pred = np.argmax(model.predict(X_test, verbose=0), axis=1)
                            acc_valid = accuracy_score(y_valid, y_valid_pred)
                            acc_test = accuracy_score(y_test, y_test_pred)
                            cm_valid = confusion_matrix(y_valid, y_valid_pred)
                            cm_test = confusion_matrix(y_test, y_test_pred)
                            mlflow.log_metric("accuracy_val", acc_valid)
                            mlflow.log_metric("accuracy_test", acc_test)
                            mlflow.log_metric("training_time", time.time() - start_time)
                            mlflow.keras.log_model(model, "model")

                            results = {
                                'accuracy_val': acc_valid, 'accuracy_test': acc_test,
                                'cm_valid': cm_valid, 'cm_test': cm_test,
                                'run_name': model_name, 'run_id': run.info.run_id,
                                'params': params, 'training_time': time.time() - start_time,
                                'test_accuracies': test_accuracies,
                                'pseudo_counts': pseudo_counts
                            }

                    st.session_state['training_results'] = results
                    st.success(f"Đã huấn luyện xong! Thời gian: {results['training_time']:.2f} giây")

            #### Hiển thị kết quả
            if 'training_results' in st.session_state:
                results = st.session_state['training_results']
                st.subheader("📊 Kết quả Huấn luyện")
                col1, col2, col3 = st.columns(3)
                col1.metric("Thời gian huấn luyện", f"{results['training_time']:.2f} giây")
                col2.metric("Độ chính xác Validation", f"{results['accuracy_val']*100:.2f}%")
                col3.metric("Độ chính xác Test", f"{results['accuracy_test']*100:.2f}%")

                st.subheader("Ma trận Nhầm lẫn")
                col_cm1, col_cm2 = st.columns(2)
                with col_cm1:
                    fig, ax = plt.subplots()
                    sns.heatmap(results['cm_valid'], annot=True, fmt="d", cmap="Blues", ax=ax)
                    ax.set_title("Validation")
                    st.pyplot(fig)
                    plt.close(fig)
                with col_cm2:
                    fig, ax = plt.subplots()
                    sns.heatmap(results['cm_test'], annot=True, fmt="d", cmap="Blues", ax=ax)
                    ax.set_title("Test")
                    st.pyplot(fig)
                    plt.close(fig)

                if 'test_accuracies' in results:
                    st.subheader("Kết quả Pseudo-Labeling")
                    col_acc, col_count = st.columns(2)
                    with col_acc:
                        fig, ax = plt.subplots()
                        ax.plot(results['test_accuracies'], marker='o')
                        ax.set_title("Độ chính xác Test qua các vòng")
                        ax.set_xlabel("Vòng lặp")
                        ax.set_ylabel("Độ chính xác")
                        st.pyplot(fig)
                        plt.close(fig)
                    with col_count:
                        fig, ax = plt.subplots()
                        ax.bar(range(len(results['pseudo_counts'])), results['pseudo_counts'])
                        ax.set_title("Số mẫu gán nhãn mỗi vòng")
                        ax.set_xlabel("Vòng lặp")
                        ax.set_ylabel("Số mẫu")
                        st.pyplot(fig)
                        plt.close(fig)

    ### Tab 6: Demo dự đoán
    with tab_demo:
        st.markdown('<div class="section-title">Demo Dự đoán Chữ số</div>', unsafe_allow_html=True)
        if 'split_data' not in st.session_state:
            st.warning("Vui lòng chia dữ liệu trước!")
        else:
            runs = client.search_runs(experiment_ids=[EXPERIMENT_ID], order_by=["attributes.start_time DESC"])
            model_options = {run.info.run_id: run.data.tags.get('mlflow.runName', run.info.run_id) for run in runs}
            if not model_options:
                st.info("Chưa có mô hình nào được huấn luyện.")
            else:
                selected_run_id = st.selectbox("Chọn mô hình:", list(model_options.keys()), 
                                               format_func=lambda x: model_options[x])
                if st.button("Tải mô hình"):
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
                            image_array = np.array(image).reshape(1, 784) / 255.0
                            if st.button("Dự đoán"):
                                pred = model.predict(image_array, verbose=0)
                                st.write(f"Dự đoán: {np.argmax(pred)} (Độ tin cậy: {np.max(pred)*100:.2f}%)")

    ### Tab 7: Thông tin huấn luyện
    with tab_log_info:
        st.markdown('<div class="section-title">Thông tin Huấn luyện</div>', unsafe_allow_html=True)
        runs = client.search_runs(experiment_ids=[EXPERIMENT_ID], order_by=["attributes.start_time DESC"])
        if not runs:
            st.info("Chưa có lần chạy nào.")
        else:
            run_options = {run.info.run_id: run.data.tags.get('mlflow.runName', run.info.run_id) for run in runs}
            selected_run_id = st.selectbox("Chọn lần chạy:", list(run_options.keys()), 
                                           format_func=lambda x: run_options[x])
            run = client.get_run(selected_run_id)

            st.write(f"**Tên:** {run.data.tags.get('mlflow.runName', run.info.run_id)}")
            st.write(f"**Tham số:** {run.data.params}")
            st.write(f"**Số liệu:** {run.data.metrics}")

            new_name = st.text_input("Đổi tên run:", value=run.data.tags.get('mlflow.runName', run.info.run_id))
            if st.button("Cập nhật tên"):
                client.set_tag(selected_run_id, "mlflow.runName", new_name)
                st.success("Đã cập nhật tên!")
                st.rerun()

            if st.button("Xóa run"):
                client.delete_run(selected_run_id)
                st.success("Đã xóa run!")
                st.rerun()

if __name__ == "__main__":
    run_mnist_neural_network_app()