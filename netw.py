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

    # Tải dữ liệu MNIST tự động khi khởi động
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

    # Tạo các tab
    tab_names = ["Thông tin", "Chọn số lượng dữ liệu", "Xử lý dữ liệu", "Chia dữ liệu", "Huấn luyện/Đánh giá", "Demo dự đoán", "Thông tin huấn luyện"]
    tab_info, tab_load, tab_preprocess, tab_split, tab_train_eval, tab_demo, tab_log_info = st.tabs(tab_names)

    # Tab 1: Thông tin
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

                **Thông tin cơ bản**:  
                - **$784$ đặc trưng**: Mỗi ảnh được biểu diễn dưới dạng vector $784$ chiều (giá trị pixel từ $0$ đến $255$).  
                - **$70,000$ mẫu**: Tổng số ảnh, được chia thành tập huấn luyện và kiểm tra.  
                - **Nhiệm vụ**: Dự đoán nhãn ($0$-$9$) dựa trên đặc trưng pixel.  
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

                **Ý nghĩa**:  
                - Là bài toán cơ bản để kiểm tra khả năng phân loại của các mô hình học máy.  
                - Đơn giản nhưng đủ phức tạp để đánh giá khả năng phân biệt các lớp tương tự (ví dụ: "$4$" và "$9$").  
                - Phù hợp cho cả người mới bắt đầu và nghiên cứu mô hình phức tạp.  
                """, unsafe_allow_html=True)

                st.subheader("📷 Minh họa dữ liệu MNIST")
                st.markdown("""
                Dưới đây là ảnh minh họa $10$ chữ số từ $0$ đến $9$ từ tập dữ liệu MNIST để bạn hình dung. Mỗi chữ số được biểu diễn dưới dạng ma trận $28 \\times 28$ pixel.
                """, unsafe_allow_html=True)
                try:
                    mnist_image = Image.open("mnist.png")
                    st.image(mnist_image, caption="Ảnh minh họa $10$ chữ số từ $0$ đến $9$ trong MNIST", width=800)
                except FileNotFoundError:
                    st.error("Không tìm thấy file `mnist.png`. Vui lòng kiểm tra đường dẫn.")
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

            st.subheader("🔧 Các tham số huấn luyện")
            st.markdown("""
            Dưới đây là các tham số chính trong quá trình huấn luyện Neural Network, được giải thích chi tiết với ý nghĩa, cách hoạt động, công thức (nếu có), ví dụ minh họa và lưu ý khi điều chỉnh:

            ---

            #### **1. Số lớp ẩn (Number of Hidden Layers)**  
            - **Ý nghĩa**: Quyết định độ sâu của mạng, tức là số lượng lớp nơ-ron nằm giữa lớp đầu vào và lớp đầu ra.  
            - **Hoạt động**:  
              - Mỗi lớp ẩn giúp mạng học được các đặc trưng phức tạp hơn từ dữ liệu đầu vào.  
              - Với bài toán đơn giản như MNIST, 1-2 lớp ẩn thường đủ; bài toán phức tạp hơn (như nhận diện ảnh tự nhiên) có thể cần nhiều lớp hơn.  
            - **Công thức**: Không có công thức cụ thể, thường được chọn dựa trên kinh nghiệm hoặc thử nghiệm.  
            - **Ví dụ**:  
              - **1 lớp ẩn**: Đủ để học các đặc trưng cơ bản trong bài toán tuyến tính hoặc gần tuyến tính.  
              - **2-3 lớp ẩn**: Phù hợp cho bài toán phi tuyến tính như phân loại chữ số MNIST.  
              - **5 lớp ẩn trở lên**: Thường dùng cho mạng sâu (deep learning) với dữ liệu phức tạp hơn.  
            - **Lưu ý**:  
              - Quá ít lớp ẩn có thể khiến mô hình không học được các đặc trưng đủ phức tạp (underfitting).  
              - Quá nhiều lớp ẩn làm tăng nguy cơ overfitting (mô hình học quá mức dữ liệu huấn luyện) và khó hội tụ nếu không đủ dữ liệu hoặc tài nguyên tính toán.  
              - Trong ứng dụng này, bạn có thể chọn từ 1 đến 5 lớp ẩn trong tab "Huấn luyện/Đánh giá".  

            ---

            #### **2. Số nơ-ron mỗi lớp ẩn (Number of Neurons per Hidden Layer)**  
            - **Ý nghĩa**: Quyết định độ rộng của mạng, tức là số lượng nơ-ron trong mỗi lớp ẩn, ảnh hưởng đến khả năng biểu diễn thông tin.  
            - **Hoạt động**:  
              - Nhiều nơ-ron hơn cho phép mạng học được nhiều đặc trưng hơn từ dữ liệu, nhưng cũng tăng chi phí tính toán và nguy cơ overfitting.  
              - Số nơ-ron thường giảm dần qua các lớp (ví dụ: 128 → 64 → 32) để học từ các đặc trưng chung đến cụ thể.  
            - **Công thức**: Không có công thức cố định, thường chọn là lũy thừa của 2 (16, 32, 64, 128, v.v.) để tối ưu hóa tính toán trên phần cứng như GPU.  
            - **Ví dụ**:  
              - **32 nơ-ron**: Phù hợp cho mạng nhỏ hoặc dữ liệu đơn giản.  
              - **128 nơ-ron**: Thường dùng cho lớp ẩn đầu tiên trong mạng sâu để học nhiều đặc trưng từ dữ liệu thô (như ảnh MNIST).  
              - **64 → 32**: Một cấu hình phổ biến cho mạng 2 lớp ẩn khi xử lý MNIST.  
            - **Lưu ý**:  
              - Quá nhiều nơ-ron có thể làm mô hình phức tạp không cần thiết, dẫn đến overfitting hoặc yêu cầu nhiều tài nguyên hơn.  
              - Quá ít nơ-ron khiến mô hình không học đủ đặc trưng, gây underfitting.  
              - Trong ứng dụng này, bạn có thể tùy chỉnh số nơ-ron cho từng lớp ẩn trong tab "Huấn luyện/Đánh giá".  

            ---

            #### **3. Tốc độ học (Learning Rate - η)**  
            - **Ý nghĩa**: Điều chỉnh mức độ thay đổi của trọng số và bias trong mỗi lần cập nhật, ảnh hưởng đến tốc độ và chất lượng hội tụ của mô hình.  
            - **Hoạt động**:  
              - Giá trị nhỏ (ví dụ: 0.0001) giúp mô hình học chậm nhưng ổn định, ít vượt qua điểm tối ưu của hàm mất mát.  
              - Giá trị lớn (ví dụ: 0.01) làm mô hình học nhanh hơn nhưng có thể dao động hoặc không hội tụ.  
            - **Công thức**:  
              $$ W_{t+1} = W_t - \\eta \\cdot \\frac{\\partial L}{\\partial W_t} $$  
              $$ b_{t+1} = b_t - \\eta \\cdot \\frac{\\partial L}{\\partial b_t} $$  
              - $W_{t+1}$, $b_{t+1}$: Trọng số và bias sau khi cập nhật.  
              - $W_t$, $b_t$: Trọng số và bias hiện tại.  
              - $\\eta$: Tốc độ học.  
              - $\\frac{\\partial L}{\\partial W_t}$, $\\frac{\\partial L}{\\partial b_t}$: Gradient của hàm mất mát theo trọng số và bias.  
            - **Ví dụ**:  
              - **$\\eta = 0.001$**: Phù hợp cho bài toán phức tạp như MNIST, cần hội tụ chậm và ổn định.  
              - **$\\eta = 0.01$**: Có thể dùng cho bài toán đơn giản hoặc khi muốn thử nghiệm nhanh.  
              - **$\\eta = 0.0001$**: Thích hợp khi mạng sâu hoặc dữ liệu lớn, tránh dao động quá mức.  
            - **Lưu ý**:  
              - Tốc độ học quá cao khiến mô hình không hội tụ, dao động quanh điểm tối ưu.  
              - Tốc độ học quá thấp làm quá trình huấn luyện chậm, tốn thời gian.  
              - Trong ứng dụng này, giá trị mặc định thường là 0.001, nhưng bạn có thể điều chỉnh trong tab "Huấn luyện/Đánh giá".  

            ---

            #### **4. Số lần lặp (Epochs)**  
            - **Ý nghĩa**: Số lần toàn bộ dữ liệu huấn luyện được đưa qua mạng, quyết định mức độ tinh chỉnh của mô hình.  
            - **Hoạt động**:  
              - Mỗi epoch là một lần mạng học từ toàn bộ dữ liệu, giúp cập nhật trọng số và bias để giảm hàm mất mát.  
              - Tăng số epoch cải thiện hiệu suất, nhưng quá nhiều có thể dẫn đến overfitting nếu không kiểm soát.  
            - **Công thức**: Không có công thức cụ thể, là tham số do người dùng chọn.  
            - **Ví dụ**:  
              - **10 epochs**: Phù hợp cho thử nghiệm nhanh hoặc dữ liệu lớn khi tài nguyên hạn chế.  
              - **50 epochs**: Thường dùng cho huấn luyện cơ bản với dữ liệu vừa phải (như 10,000 mẫu MNIST).  
              - **100 epochs**: Dùng cho huấn luyện chuyên sâu để đạt độ chính xác cao (như 70,000 mẫu MNIST).  
            - **Lưu ý**:  
              - Quá ít epoch khiến mô hình chưa học đủ, dẫn đến underfitting.  
              - Quá nhiều epoch làm tăng nguy cơ overfitting, đặc biệt nếu không dùng kỹ thuật như Early Stopping.  
              - Trong ứng dụng này, bạn có thể chọn từ 10 đến 200 epochs, và nên dùng Early Stopping để dừng khi mô hình không cải thiện thêm.  

            ---

            #### **5. Kích thước batch (Batch Size)**  
            - **Ý nghĩa**: Số mẫu dữ liệu được xử lý trong một lần lan truyền thuận và ngược trước khi cập nhật trọng số.  
            - **Hoạt động**:  
              - **Batch nhỏ** (ví dụ: 16): Cập nhật trọng số thường xuyên, giúp học chi tiết hơn nhưng chậm và có thể dao động.  
              - **Batch lớn** (ví dụ: 256): Cập nhật ít thường xuyên hơn, tăng tốc huấn luyện nhưng cần nhiều bộ nhớ và có thể bỏ qua chi tiết.  
            - **Công thức**: Không có công thức cố định, thường chọn là lũy thừa của 2 (16, 32, 64, 128, 256, v.v.) để tối ưu hóa tính toán trên phần cứng.  
            - **Ví dụ**:  
              - **Batch size = 32**: Phù hợp cho dữ liệu nhỏ hoặc thử nghiệm nhanh (như 1,000 mẫu MNIST).  
              - **Batch size = 128**: Thường dùng cho dữ liệu vừa (như 50,000 mẫu MNIST) để cân bằng tốc độ và độ chính xác.  
              - **Batch size = 256**: Dùng cho dữ liệu lớn (như 70,000 mẫu MNIST) để tăng tốc huấn luyện.  
            - **Lưu ý**:  
              - Batch quá nhỏ làm huấn luyện không ổn định, dễ dao động quanh điểm tối ưu.  
              - Batch quá lớn có thể khiến mô hình không học được các đặc trưng chi tiết, đặc biệt với dữ liệu phức tạp.  
              - Trong ứng dụng này, giá trị mặc định phụ thuộc vào số lượng mẫu (32, 64, 128, hoặc 256), nhưng bạn có thể tùy chỉnh.  

            ---

            #### **6. Hàm kích hoạt (Activation Function)**  
            - **Ý nghĩa**: Quyết định cách nơ-ron "kích hoạt" đầu ra dựa trên đầu vào, giúp mạng học được các mối quan hệ phi tuyến tính.  
            - **Hoạt động**:  
              - Chuyển đổi giá trị tuyến tính (tổng trọng số) thành phi tuyến để mô hình học được các đặc trưng phức tạp.  
              - Được áp dụng sau mỗi lớp (trừ lớp đầu ra trong một số trường hợp).  
            - **Các hàm kích hoạt phổ biến**:  
              - **ReLU (Rectified Linear Unit)**:  
                - **Ý nghĩa**: Đơn giản, nhanh, giúp tránh vấn đề biến mất gradient trong mạng sâu.  
                - **Hoạt động**: Chỉ cho phép giá trị dương đi qua, đặt tất cả giá trị âm về 0.  
                - **Công thức**:  
                  $$ f(x) = \\max(0, x) $$  
                - **Ví dụ**:  
                  - Nếu $x = 3$, thì $f(3) = 3$.  
                  - Nếu $x = -1$, thì $f(-1) = 0$.  
                - **Lưu ý**:  
                  - Thường dùng cho lớp ẩn vì hiệu quả và đơn giản.  
                  - Có thể gây "dead neurons" (nơ-ron không hoạt động) nếu đầu vào luôn âm.  
              - **Tanh (Hyperbolic Tangent)**:  
                - **Ý nghĩa**: Chuẩn hóa đầu ra về khoảng [-1, 1], phù hợp khi cần cân bằng giá trị âm và dương.  
                - **Hoạt động**: Tạo đầu ra phi tuyến, nhưng dễ gặp vấn đề biến mất gradient trong mạng sâu.  
                - **Công thức**:  
                  $$ f(x) = \\frac{e^x - e^{-x}}{e^x + e^{-x}} $$  
                - **Ví dụ**:  
                  - Nếu $x = 0$, thì $f(0) = 0$.  
                  - Nếu $x = 1$, thì $f(1) \\approx 0.76$.  
                  - Nếu $x = -1$, thì $f(-1) \\approx -0.76$.  
                - **Lưu ý**:  
                  - Ít dùng hơn ReLU do vấn đề biến mất gradient, nhưng vẫn hữu ích trong một số trường hợp.  
              - **Softmax**:  
                - **Ý nghĩa**: Dùng ở lớp đầu ra để chuyển đổi đầu ra thành xác suất cho bài toán phân loại đa lớp (như MNIST).  
                - **Hoạt động**: Chuẩn hóa tổng các đầu ra thành 1, giúp chọn lớp có xác suất cao nhất.  
                - **Công thức**:  
                  $$ f(x_i) = \\frac{e^{x_i}}{\\sum_{j=0}^{k} e^{x_j}} $$  
                  - $x_i$: Đầu vào của nơ-ron thứ $i$.  
                  - $k$: Số lớp (ở đây là 10, từ 0-9).  
                - **Ví dụ**:  
                  - Nếu $x = [1, 2, 3]$, thì $f(x) \\approx [0.09, 0.24, 0.67]$.  
                  - Tổng xác suất luôn bằng 1.  
                - **Lưu ý**:  
                  - Bắt buộc dùng ở lớp đầu ra cho bài toán phân loại đa lớp như MNIST.  
            - **Lưu ý chung**:  
              - ReLU là lựa chọn mặc định cho lớp ẩn trong ứng dụng này vì tính hiệu quả và phổ biến.  
              - Softmax luôn được dùng ở lớp đầu ra để dự đoán chữ số từ 0-9.  
              - Bạn có thể chọn giữa ReLU, Tanh, hoặc Softmax trong tab "Huấn luyện/Đánh giá" cho lớp ẩn.  

            ---

            #### **7. Trình tối ưu (Optimizer)**  
            - **Ý nghĩa**: Thuật toán điều chỉnh trọng số và bias để giảm hàm mất mát, quyết định cách mô hình học.  
            - **Hoạt động**:  
              - Dùng gradient (đạo hàm của hàm mất mát) để cập nhật tham số, với cách tiếp cận khác nhau tùy thuật toán.  
            - **Các trình tối ưu phổ biến**:  
              - **SGD (Stochastic Gradient Descent)**:  
                - **Ý nghĩa**: Cập nhật trọng số dựa trên gradient của một mẫu hoặc mini-batch, là phiên bản ngẫu nhiên của Gradient Descent.  
                - **Hoạt động**: Tính gradient cho từng batch và điều chỉnh tham số theo hướng giảm mất mát.  
                - **Công thức**:  
                  $$ W_{t+1} = W_t - \\eta \\cdot \\frac{\\partial L}{\\partial W_t} $$  
                  $$ b_{t+1} = b_t - \\eta \\cdot \\frac{\\partial L}{\\partial b_t} $$  
                  - $W_t$, $b_t$: Trọng số và bias hiện tại.  
                  - $\\eta$: Tốc độ học.  
                  - $\\frac{\\partial L}{\\partial W_t}$, $\\frac{\\partial L}{\\partial b_t}$: Gradient.  
                - **Ví dụ**:  
                  - Với $\\eta = 0.01$, nếu gradient $\\frac{\\partial L}{\\partial W_t} = 0.5$, thì $W_{t+1} = W_t - 0.01 \\cdot 0.5 = W_t - 0.005$.  
                - **Ưu điểm**: Đơn giản, hiệu quả với dữ liệu lớn khi dùng mini-batch.  
                - **Nhược điểm**: Dao động quanh điểm tối ưu, hội tụ chậm nếu không điều chỉnh tốt.  
              - **Adam (Adaptive Moment Estimation)**:  
                - **Ý nghĩa**: Kết hợp phương pháp động lượng và RMSProp, thích nghi tốc độ học cho từng tham số.  
                - **Hoạt động**: Dùng hai moment (bậc 1 và bậc 2) của gradient để điều chỉnh cập nhật, giúp hội tụ nhanh và ổn định hơn SGD.  
                - **Công thức**:  
                  1. $m_t = \\beta_1 \\cdot m_{t-1} + (1 - \\beta_1) \\cdot g_t$ (moment bậc 1 - trung bình động của gradient).  
                  2. $v_t = \\beta_2 \\cdot v_{t-1} + (1 - \\beta_2) \\cdot g_t^2$ (moment bậc 2 - trung bình động của bình phương gradient).  
                  3. $\\hat{m}_t = \\frac{m_t}{1 - \\beta_1^t}$, $\\hat{v}_t = \\frac{v_t}{1 - \\beta_2^t}$ (hiệu chỉnh bias).  
                  4. $W_{t+1} = W_t - \\eta \\cdot \\frac{\\hat{m}_t}{\\sqrt{\\hat{v}_t} + \\epsilon}$ (cập nhật trọng số).  
                  - $g_t$: Gradient tại bước $t$.  
                  - $\\beta_1 \\approx 0.9$, $\\beta_2 \\approx 0.999$: Hệ số giảm dần.  
                  - $\\epsilon \\approx 10^{-8}$: Hằng số nhỏ để tránh chia cho 0.  
                - **Ví dụ**:  
                  - Với gradient $g_t = 0.5$, Adam tự động điều chỉnh tốc độ học dựa trên $m_t$ và $v_t$, giúp cập nhật ổn định hơn SGD.  
                - **Ưu điểm**: Nhanh, ổn định, hiệu quả với hầu hết bài toán, đặc biệt là mạng sâu.  
                - **Nhược điểm**: Phức tạp hơn SGD, đôi khi kém hiệu quả trên hàm mất mát không lồi.  
            - **Lưu ý**:  
              - **Adam** là lựa chọn mặc định trong ứng dụng này vì khả năng hội tụ nhanh và ổn định.  
              - **SGD** phù hợp khi bạn muốn kiểm soát chi tiết quá trình huấn luyện hoặc khi làm việc với dữ liệu rất lớn.  
              - Bạn có thể chọn giữa SGD và Adam trong tab "Huấn luyện/Đánh giá".  

            ---

            Các tham số trên được điều chỉnh trong tab **"Huấn luyện/Đánh giá"** của ứng dụng này. Việc hiểu rõ ý nghĩa và cách hoạt động của chúng sẽ giúp bạn tối ưu hóa mô hình Neural Network để đạt hiệu suất tốt nhất trên tập dữ liệu MNIST!
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

            status_text.text("Đã tải xong! 100%")
            time.sleep(0.5)
            status_text.empty()
            progress_bar.empty()

    # Tab 2: Chọn số lượng dữ liệu
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
        selected_option = st.selectbox("Chọn số lượng mẫu:", list(sample_options.keys()), help="Chọn số lượng mẫu có sẵn hoặc nhập tùy chỉnh")
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

            col1, col2 = st.columns(2)
            with col1:
                test_pct = st.slider("Tỷ lệ Test (%)", 0, 50, 20, help="Tỷ lệ dữ liệu dùng để kiểm tra mô hình")
            with col2:
                valid_pct = st.slider("Tỷ lệ Validation (%)", 0, 50, 20, help="Tỷ lệ dữ liệu dùng để xác thực mô hình")

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

    # Tab 5: Huấn luyện/Đánh giá
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
                st.error("Dữ liệu huấn luyện chứa giá trị NaN. Đang xử lý...")
                X_train = np.nan_to_num(X_train, nan=0.0)
                y_train = np.nan_to_num(y_train, nan=0.0)
                st.success("Đã thay thế NaN bằng 0 trong dữ liệu huấn luyện!")

            num_samples = len(X_train)
            st.write(f"**Số mẫu huấn luyện**: {num_samples}")
            if X_train.shape[0] != y_train.shape[0]:
                st.error("Số mẫu của X_train và y_train không khớp!")
                st.stop()

            if "optimal_params" not in st.session_state:
                st.session_state["optimal_params"] = get_optimal_params(num_samples)
            
            params = st.session_state.get("training_params", st.session_state["optimal_params"].copy())

            # Bố cục chuyên nghiệp
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
                    st.markdown("**Tùy chỉnh số lớp ẩn và nơ-ron**", unsafe_allow_html=True)
                    num_hidden_layers = st.number_input("Số lớp ẩn", min_value=1, value=len(params["hidden_layer_sizes"]), 
                                                       help="Chọn số lớp ẩn để điều chỉnh độ phức tạp của mô hình.")
                    hidden_sizes = []
                    for i in range(num_hidden_layers):
                        default_value = params["hidden_layer_sizes"][i] if i < len(params["hidden_layer_sizes"]) else 32
                        hidden_size = st.number_input(f"Số nơ-ron lớp ẩn {i+1}", min_value=1, value=default_value, 
                                                      help=f"Số nơ-ron cho lớp ẩn thứ {i+1}.")
                        hidden_sizes.append(hidden_size)
                    params["hidden_layer_sizes"] = tuple(hidden_sizes)
                    params["activation"] = st.selectbox("Hàm kích hoạt (lớp ẩn)", ["relu", "tanh", "softmax"], 
                                                        index=["relu", "tanh", "softmax"].index(params["activation"]) if params["activation"] in ["relu", "tanh", "softmax"] else 0,
                                                        help="Chọn hàm kích hoạt cho lớp ẩn.")
            
            with col_param2:
                with st.expander("🔧 Tối ưu hóa", expanded=True):
                    st.markdown("**Cấu hình huấn luyện**", unsafe_allow_html=True)
                    params["learning_rate"] = st.number_input("Tốc độ học", min_value=0.0, step=0.0001, value=params["learning_rate"], 
                                                              format="%.4f", help="Tốc độ học càng nhỏ càng ổn định nhưng chậm.")
                    params["epochs"] = st.number_input("Số lần lặp (Epochs)", min_value=1, value=params["epochs"], 
                                                       help="Số lần lặp qua toàn bộ dữ liệu.")
                    params["batch_size"] = st.number_input("Kích thước batch", min_value=1, value=params["batch_size"], 
                                                           help="Số mẫu mỗi lần cập nhật trọng số.")
                    params["solver"] = st.selectbox("Trình tối ưu", ["adam", "sgd"], 
                                                    index=["adam", "sgd"].index(params["solver"]),
                                                    help="Adam (nhanh, hiệu quả), SGD (đơn giản, chậm hơn).")
                    early_stopping = st.checkbox("Dừng sớm (Early Stopping)", value=False, 
                                                 help="Dừng huấn luyện nếu không cải thiện trên tập validation sau 10 epochs.")

            col_reset, col_empty = st.columns([1, 3])
            with col_reset:
                if st.button("🔄 Khôi phục tham số tối ưu", key="reset_params"):
                    st.session_state["training_params"] = st.session_state["optimal_params"].copy()
                    st.success("Đã khôi phục tham số tối ưu!")
                    st.rerun()

            st.session_state["training_params"] = params

            # Phần huấn luyện
            st.subheader("🚀 Huấn luyện Mô hình")
            with st.container():
                if 'model_name' not in st.session_state:
                    st.session_state['model_name'] = f"Model_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                model_name = st.text_input("Đặt tên cho mô hình:", value=st.session_state['model_name'], 
                                           help="Đặt tên trước khi huấn luyện để lưu trữ trên MLflow.")
                st.session_state['model_name'] = model_name

                if st.button("Bắt đầu Huấn luyện", type="primary", key="start_training"):
                    try:
                        with st.spinner("Đang huấn luyện mô hình..."):
                            start_time = time.time()

                            model = models.Sequential()
                            model.add(layers.Input(shape=(784,)))
                            for neurons in params["hidden_layer_sizes"]:
                                model.add(layers.Dense(neurons, activation=params["activation"]))
                            model.add(layers.Dense(10, activation='softmax'))

                            optimizer = tf.keras.optimizers.Adam(learning_rate=params["learning_rate"]) if params["solver"] == "adam" else tf.keras.optimizers.SGD(learning_rate=params["learning_rate"])

                            model.compile(optimizer=optimizer,
                                          loss='sparse_categorical_crossentropy',
                                          metrics=['accuracy'])

                            progress_bar = st.progress(0)
                            status_text = st.empty()

                            class ProgressCallback(callbacks.Callback):
                                def on_epoch_end(self, epoch, logs=None):
                                    progress = (epoch + 1) / params["epochs"]
                                    progress_bar.progress(min(progress, 1.0))
                                    status_text.text(f"Epoch {epoch+1}/{params['epochs']}, Loss: {logs['loss']:.4f}, Accuracy: {logs['accuracy']:.4f}, Val Loss: {logs.get('val_loss', 'N/A'):.4f}, Val Accuracy: {logs.get('val_accuracy', 'N/A'):.4f}")

                            callbacks_list = [ProgressCallback()]
                            if early_stopping:
                                callbacks_list.append(callbacks.EarlyStopping(monitor='val_loss', patience=10))

                            history = model.fit(X_train, y_train, epochs=params["epochs"], batch_size=params["batch_size"],
                                                validation_data=(X_valid, y_valid), callbacks=callbacks_list, verbose=0)

                            y_valid_pred = np.argmax(model.predict(X_valid, verbose=0), axis=1)
                            y_test_pred = np.argmax(model.predict(X_test, verbose=0), axis=1)
                            acc_valid = accuracy_score(y_valid, y_valid_pred)
                            acc_test = accuracy_score(y_test, y_test_pred)
                            cm_valid = confusion_matrix(y_valid, y_valid_pred)
                            cm_test = confusion_matrix(y_test, y_test_pred)

                            with mlflow.start_run(experiment_id=EXPERIMENT_ID, run_name=model_name) as run:
                                mlflow.log_params({k: v for k, v in params.items() if k in ['hidden_layer_sizes', 'learning_rate', 'epochs', 'batch_size', 'activation', 'solver']})
                                mlflow.log_metric("accuracy_val", acc_valid)
                                mlflow.log_metric("accuracy_test", acc_test)
                                mlflow.log_metric("training_time", time.time() - start_time)
                                mlflow.log_metric("n_iter_actual", len(history.history['loss']))
                                mlflow.keras.log_model(model, "model")

                            st.session_state['model'] = model
                            st.session_state['training_results'] = {
                                'accuracy_val': acc_valid, 'accuracy_test': acc_test,
                                'cm_valid': cm_valid, 'cm_test': cm_test,
                                'run_name': model_name, 'run_id': run.info.run_id,
                                'params': params, 'training_time': time.time() - start_time,
                                'loss_history': history.history['loss'],
                                'val_loss_history': history.history['val_loss'] if 'val_loss' in history.history else [],
                                'accuracy_history': history.history['accuracy'],
                                'val_accuracy_history': history.history['val_accuracy'] if 'val_accuracy' in history.history else [],
                                'n_iter_actual': len(history.history['loss'])
                            }
                            st.session_state['latest_run_id'] = run.info.run_id  # Lưu run_id mới nhất

                            st.success(f"Đã huấn luyện xong! Thời gian: {time.time() - start_time:.2f} giây, Số lần lặp thực tế: {len(history.history['loss'])}")
                            tf.keras.backend.clear_session()
                            del X_train, y_train, X_valid, y_valid, X_test, y_test, split_data, history
                            gc.collect()

                    except Exception as e:
                        st.error(f"Lỗi trong quá trình huấn luyện: {e}")

            # Kết quả huấn luyện
            if 'training_results' in st.session_state:
                results = st.session_state['training_results']
                st.subheader("📊 Kết quả Huấn luyện")
                with st.container():
                    col_result1, col_result2, col_result3 = st.columns(3)
                    with col_result1:
                        st.metric("Thời gian huấn luyện", f"{results['training_time']:.2f} giây")
                    with col_result2:
                        st.metric("Độ chính xác Validation", f"{results['accuracy_val']*100:.2f}%")
                    with col_result3:
                        st.metric("Độ chính xác Test", f"{results['accuracy_test']*100:.2f}%")

                    st.markdown("#### 📈 Ma trận Nhầm lẫn")
                    st.markdown("""
                    - Ma trận nhầm lẫn cho thấy số lượng dự đoán đúng và sai của mô hình cho từng lớp ($0$-$9$):  
                      - **Hàng**: Nhãn thực tế.  
                      - **Cột**: Nhãn dự đoán.  
                      - **Số trên đường chéo**: Số mẫu dự đoán đúng.  
                      - **Số ngoài đường chéo**: Số mẫu dự đoán sai (nhầm lẫn giữa các lớp).  
                    """, unsafe_allow_html=True)
                    col_cm1, col_cm2 = st.columns(2)
                    with col_cm1:
                        fig, ax = plt.subplots(figsize=(6, 5))
                        sns.heatmap(results['cm_valid'], annot=True, fmt="d", cmap="Blues", ax=ax)
                        ax.set_title("Validation")
                        st.pyplot(fig)
                        plt.close(fig)
                    with col_cm2:
                        fig, ax = plt.subplots(figsize=(6, 5))
                        sns.heatmap(results['cm_test'], annot=True, fmt="d", cmap="Blues", ax=ax)
                        ax.set_title("Test")
                        st.pyplot(fig)
                        plt.close(fig)

                    st.markdown("#### 📉 Biểu đồ Kết quả Huấn luyện")
                    st.markdown("""
                    - **Biểu đồ Loss**: Thể hiện giá trị hàm mất mát qua các epoch, giúp đánh giá mức độ hội tụ của mô hình. Loss giảm đều cho thấy mô hình học tốt.  
                    - **Biểu đồ Accuracy**: Thể hiện độ chính xác qua các epoch, phản ánh khả năng phân loại của mô hình trên tập huấn luyện và validation.  
                    """, unsafe_allow_html=True)
                    col_loss, col_acc = st.columns(2)
                    with col_loss:
                        if results['loss_history']:
                            epochs = list(range(1, len(results['loss_history']) + 1))
                            fig, ax = plt.subplots(figsize=(6, 4))
                            ax.plot(epochs, results['loss_history'], label='Training Loss', color='blue', linewidth=2)
                            if results['val_loss_history']:
                                ax.plot(epochs, results['val_loss_history'], label='Validation Loss', color='orange', linestyle='--', linewidth=2)
                            ax.set_xlabel("Epochs")
                            ax.set_ylabel("Loss")
                            ax.set_title("Loss qua các Epoch")
                            ax.legend()
                            ax.grid(True)
                            st.pyplot(fig)
                            plt.close(fig)
                    with col_acc:
                        if results['accuracy_history']:
                            epochs = list(range(1, len(results['accuracy_history']) + 1))
                            fig, ax = plt.subplots(figsize=(6, 4))
                            ax.plot(epochs, results['accuracy_history'], label='Training Accuracy', color='green', linewidth=2)
                            if results['val_accuracy_history']:
                                ax.plot(epochs, results['val_accuracy_history'], label='Validation Accuracy', color='red', linestyle='--', linewidth=2)
                            ax.set_xlabel("Epochs")
                            ax.set_ylabel("Accuracy")
                            ax.set_title("Accuracy qua các Epoch")
                            ax.legend()
                            ax.grid(True)
                            st.pyplot(fig)
                            plt.close(fig)

                    st.markdown("#### 📋 Tóm tắt Kết quả Huấn luyện")
                    full_data = {
                        "Epoch": list(range(1, len(results['loss_history']) + 1)),
                        "Loss": results['loss_history'],
                        "Accuracy": results['accuracy_history'],
                    }
                    if results['val_loss_history']:
                        full_data["Val Loss"] = results['val_loss_history']
                        full_data["Val Accuracy"] = results['val_accuracy_history']
                    df_full = pd.DataFrame(full_data)

                    if 'display_epochs' not in st.session_state:
                        st.session_state['display_epochs'] = 5

                    st.table(df_full.head(st.session_state['display_epochs']))

                    if len(results['loss_history']) > st.session_state['display_epochs']:
                        if st.button("Xem thêm 10 epoch", key="show_more"):
                            st.session_state['display_epochs'] += 10
                            st.rerun()

                    if st.session_state['display_epochs'] > 5:
                        if st.button("Thu gọn", key="collapse"):
                            st.session_state['display_epochs'] = 5
                            st.rerun()

                    with st.expander("Xem chi tiết", expanded=False):
                        st.markdown("**Thông tin lần chạy:**")
                        st.write(f"- Tên: {results['run_name']}")
                        st.write(f"- ID: {results['run_id']}")
                        st.write(f"- Thời gian huấn luyện: {results['training_time']:.2f} giây")
                        st.write(f"- Số lần lặp thực tế: {results['n_iter_actual']}")
                        st.write(f"- Độ chính xác Validation: {results['accuracy_val']*100:.2f}%")
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
                            "Dừng sớm": early_stopping
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
                        # Khởi tạo session state cho canvas và lịch sử dự đoán
                        if 'canvas_key' not in st.session_state:
                            st.session_state['canvas_key'] = 0
                        if 'predictions' not in st.session_state:
                            st.session_state['predictions'] = []

                        # Nút làm mới canvas
                        if st.button("Xóa Canvas"):
                            st.session_state['canvas_key'] += 1
                            st.session_state['predictions'] = []  # Xóa lịch sử dự đoán

                        canvas_result = st_canvas(
                            stroke_width=20,
                            stroke_color="#FFFFFF",
                            background_color="#000000",
                            height=280,
                            width=280,
                            drawing_mode="freedraw",
                            key=f"canvas_{st.session_state['canvas_key']}"
                        )

                        if canvas_result.image_data is not None:
                            image = Image.fromarray(canvas_result.image_data).convert('L').resize((28, 28))
                            st.image(image, caption="Hình ảnh vẽ tay", width=100)
                            image_array = np.array(image).reshape(1, 784) / 255.0
                            if st.button("Dự đoán"):
                                pred = model.predict(image_array, verbose=0)
                                prediction = f"Dự đoán: {np.argmax(pred)} (Độ tin cậy: {np.max(pred)*100:.2f}%)"
                                st.session_state['predictions'].append(prediction)
                                st.write(prediction)

                        # Hiển thị lịch sử dự đoán
                        if st.session_state['predictions']:
                            st.subheader("Lịch sử dự đoán")
                            for p in st.session_state['predictions']:
                                st.write(p)


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
                    col_loss, col_acc = st.columns(2)
                    with col_loss:
                        if 'training_results' in st.session_state and selected_run_id == st.session_state['training_results']['run_id']:
                            results = st.session_state['training_results']
                            if results['loss_history']:
                                fig, ax = plt.subplots(figsize=(6, 4))
                                ax.plot(range(1, len(results['loss_history']) + 1), results['loss_history'], 
                                        label='Training Loss', color='blue', linewidth=2)
                                if results['val_loss_history']:
                                    ax.plot(range(1, len(results['val_loss_history']) + 1), results['val_loss_history'], 
                                            label='Validation Loss', color='orange', linestyle='--', linewidth=2)
                                ax.set_xlabel("Epochs")
                                ax.set_ylabel("Loss")
                                ax.set_title("Lịch sử Mất mát")
                                ax.legend()
                                ax.grid(True)
                                st.pyplot(fig)
                                plt.close(fig)
                    with col_acc:
                        if 'training_results' in st.session_state and selected_run_id == st.session_state['training_results']['run_id']:
                            results = st.session_state['training_results']
                            if results['accuracy_history']:
                                fig, ax = plt.subplots(figsize=(6, 4))
                                ax.plot(range(1, len(results['accuracy_history']) + 1), results['accuracy_history'], 
                                        label='Training Accuracy', color='green', linewidth=2)
                                if results['val_accuracy_history']:
                                    ax.plot(range(1, len(results['val_accuracy_history']) + 1), results['val_accuracy_history'], 
                                            label='Validation Accuracy', color='red', linestyle='--', linewidth=2)
                                ax.set_xlabel("Epochs")
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
                                "Accuracy Val": run.data.metrics.get('accuracy_val', 'N/A'),
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
    run_mnist_neural_network_app()