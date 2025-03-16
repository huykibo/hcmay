import os
import mlflow
import streamlit as st
import openml
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
import cv2  # Thêm để xử lý ảnh trong Demo

# Hàm tải dữ liệu MNIST
def fetch_mnist_data():
    mnist = openml.datasets.get_dataset(554)
    X, y, _, _ = mnist.get_data(target=mnist.default_target_attribute)
    return X, y

# Hàm kiểm tra và chuẩn hóa dữ liệu pixel về [0, 255]
def validate_and_fix_pixels(X, name="dữ liệu"):
    X = np.array(X, dtype=np.float64)
    invalid_mask = (X < 0) | (X > 255)
    if np.any(invalid_mask):
        st.warning(f"Phát hiện giá trị pixel không hợp lệ trong {name} (ngoài [0, 255]). Đang chuẩn hóa...")
        X_fixed = np.clip(X, 0, 255)
        return X_fixed, True
    return X, False

# Cache mô hình để tăng tốc độ
@st.cache_resource
def load_model(model):
    return model

# Hàm xử lý ảnh upload
def preprocess_uploaded_image(image):
    try:
        # Chuyển sang thang độ xám
        gray = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
        # Đảo ngược màu (nếu nền trắng, chữ đen -> giống MNIST)
        gray = 255 - gray
        # Resize về 28x28
        resized = cv2.resize(gray, (28, 28), interpolation=cv2.INTER_AREA)
        # Chuẩn hóa về [0, 1]
        normalized = resized / 255.0
        # Flatten thành vector 784 chiều
        flattened = normalized.flatten()
        return flattened
    except Exception as e:
        st.error(f"Lỗi xử lý ảnh: {e}")
        return None

# Hàm xử lý ảnh từ canvas
def preprocess_canvas_image(canvas_image):
    try:
        # Lấy dữ liệu ảnh từ canvas (RGBA)
        img_data = canvas_image[:, :, 3]  # Kênh alpha (độ sáng của nét vẽ)
        # Đảo ngược màu (nền đen, chữ trắng -> giống MNIST)
        img_data = 255 - img_data
        # Resize về 28x28
        resized = cv2.resize(img_data, (28, 28), interpolation=cv2.INTER_AREA)
        # Chuẩn hóa về [0, 1]
        normalized = resized / 255.0
        # Flatten thành vector 784 chiều
        flattened = normalized.flatten()
        return flattened
    except Exception as e:
        st.error(f"Lỗi xử lý ảnh canvas: {e}")
        return None

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

    st.title("Phân loại Chữ số MNIST với Neural Network (TensorFlow)")

    # CSS tùy chỉnh
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
            .stTabs [role="tabpanel"] {
                min-height: auto !important;
                height: auto !important;
            }
            .stTabs [data-testid="stVerticalBlock"] {
                min-height: auto !important;
                height: auto !important;
                padding-bottom: 0px !important;
            }
            .stTabs [data-testid="stVerticalBlock"] > div {
                min-height: auto !important;
                height: auto !important;
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
                - **Lớp ẩn (Hidden Layers)**: Xử lý thông tin thông qua các phép tính tuyến tính và phi tuyến (sử dụng hàm kích hoạt).  
                - **Lớp đầu ra (Output Layer)**: Đưa ra dự đoán (10 lớp, tương ứng với các chữ số $0$-$9$).  
                """, unsafe_allow_html=True)

                st.subheader("🔧 Quy trình hoạt động")
                st.markdown("""
                Neural Network hoạt động qua các bước sau, được tối ưu hóa dựa trên các tham số bạn có thể điều chỉnh trong tab **Huấn luyện/Đánh giá**:
                """, unsafe_allow_html=True)

                st.subheader("1. Khởi tạo mô hình")
                st.markdown("""
                - Xác định cấu trúc mạng (số lớp ẩn, số nơ-ron mỗi lớp) và khởi tạo **trọng số** ($W$) và **bias** ($b$) ngẫu nhiên (thường từ phân phối Gaussian).  
                - **Tham số liên quan**:  
                  - **Số lớp ẩn**: Được chọn từ $1$ đến $2$ trong giao diện huấn luyện.  
                  - **Số nơ-ron mỗi lớp**: Có thể điều chỉnh từ $16$ đến $128$.  
                - Mục đích: Thiết lập cấu trúc ban đầu để bắt đầu quá trình học.
                """, unsafe_allow_html=True)
                try:
                    st.image(os.path.join("plnw", "step1_init.png"), caption="Minh họa: Khởi tạo mô hình", width=600)
                except FileNotFoundError:
                    st.error("Không tìm thấy ảnh minh họa cho Bước 1.")
                except Exception as e:
                    st.error(f"Lỗi khi tải ảnh: {e}")

                st.subheader("2. Lan truyền thuận (Feedforward)")
                st.markdown("""
                - Tính toán đầu ra dự đoán ($\\hat{Y}$) từ đầu vào $X$ qua các lớp:  
                  $$ Z^{(l)} = A^{(l-1)} \\cdot W^{(l)} + b^{(l)} $$  
                  $$ A^{(l)} = \\sigma(Z^{(l)}) $$  
                - **Giải thích**:  
                  - $X$: Ma trận đầu vào, kích thước $N \\times 784$ ($N$ là số mẫu).  
                  - $A^{(l-1)}$: Đầu ra của lớp trước, với $A^{(0)} = X$.  
                  - $W^{(l)}$: Ma trận trọng số của lớp $l$, kích thước phụ thuộc số nơ-ron của lớp $l-1$ và $l$.  
                  - $b^{(l)}$: Vector bias của lớp $l$.  
                  - $Z^{(l)}$: Tổng trọng số tuyến tính của lớp $l$.  
                  - $\\sigma$: Hàm kích hoạt (ví dụ: ReLU, Sigmoid, Tanh).  
                  - $\\hat{Y}$: Đầu ra cuối cùng, kích thước $N \\times 10$ (10 lớp).  
                - **Ví dụ với Sigmoid**:  
                  $$ \\sigma(z) = \\frac{1}{1 + e^{-z}} $$  
                - Mục đích: Tạo dự đoán ban đầu từ dữ liệu đầu vào qua các lớp nơ-ron.
                """, unsafe_allow_html=True)
                try:
                    st.image(os.path.join("plnw", "step2_feedforward.png"), caption="Minh họa: Lan truyền thuận", width=600)
                except FileNotFoundError:
                    st.error("Không tìm thấy ảnh minh họa cho Bước 2.")
                except Exception as e:
                    st.error(f"Lỗi khi tải ảnh: {e}")

                st.subheader("3. Tính hàm mất mát (Loss Function)")
                st.markdown("""
                - Đo độ sai lệch giữa dự đoán ($\\hat{Y}$) và nhãn thực ($Y$) bằng **Cross-Entropy**:  
                  $$ L = -\\frac{1}{N} \\sum_{i=1}^{N} \\sum_{j=0}^{9} y_{ij} \\cdot \\log(\\hat{y}_{ij}) $$  
                - **Giải thích**:  
                  - $N$: Số mẫu trong tập dữ liệu.  
                  - $y_{ij}$: Nhãn thực tế (one-hot encoded), $1$ nếu mẫu $i$ thuộc lớp $j$, $0$ nếu không.  
                  - $\\hat{y}_{ij}$: Xác suất dự đoán mẫu $i$ thuộc lớp $j$.  
                  - $\\sum_{i=1}^{N}$: Tổng trên tất cả mẫu.  
                  - $\\sum_{j=0}^{9}$: Tổng trên tất cả lớp (0 đến 9).  
                - Mục đích: Định lượng sai lệch để điều chỉnh mô hình trong bước tiếp theo.
                """, unsafe_allow_html=True)
                try:
                    st.image(os.path.join("plnw", "step3_loss.png"), caption="Minh họa: Tính hàm mất mát", width=600)
                except FileNotFoundError:
                    st.error("Không tìm thấy ảnh minh họa cho Bước 3.")
                except Exception as e:
                    st.error(f"Lỗi khi tải ảnh: {e}")

                st.subheader("4. Lan truyền ngược (Backpropagation)")
                st.markdown("""
                - Tính đạo hàm của $L$ để cập nhật $W^{(l)}$ và $b^{(l)}$:  
                  - Lớp đầu ra:  
                    $$ \\delta^{(L)} = \\hat{Y} - Y $$  
                  - Lớp ẩn:  
                    $$ \\delta^{(l)} = (\\delta^{(l+1)} \\cdot (W^{(l+1)})^T) \\odot \\sigma'(Z^{(l)}) $$  
                  - Đạo hàm:  
                    $$ \\frac{\\partial L}{\\partial W^{(l)}} = (A^{(l-1)})^T \\cdot \\delta^{(l)} $$  
                    $$ \\frac{\\partial L}{\\partial b^{(l)}} = \\sum_{i=1}^{N} \\delta^{(l)}_i $$  
                - **Giải thích**:  
                  - $\\delta^{(L)}$: Sai số tại lớp đầu ra.  
                  - $\\delta^{(l)}$: Sai số tại lớp $l$, lan truyền ngược từ lớp sau.  
                  - $(W^{(l+1)})^T$: Ma trận chuyển vị của trọng số lớp tiếp theo.  
                  - $\\odot$: Nhân từng phần tử (Hadamard product).  
                  - $\\sigma'(Z^{(l)})$: Đạo hàm của hàm kích hoạt tại $Z^{(l)}$ (ví dụ: Sigmoid: $\\sigma'(z) = \\sigma(z) \\cdot (1 - \\sigma(z))$).  
                  - $\\frac{\\partial L}{\\partial W^{(l)}}$: Gradient của mất mát theo trọng số.  
                  - $\\frac{\\partial L}{\\partial b^{(l)}}$: Gradient của mất mát theo bias.  
                - Mục đích: Xác định hướng điều chỉnh tham số dựa trên sai số.
                """, unsafe_allow_html=True)
                try:
                    st.image(os.path.join("plnw", "step4_backprop.png"), caption="Minh họa: Lan truyền ngược", width=600)
                except FileNotFoundError:
                    st.error("Không tìm thấy ảnh minh họa cho Bước 4.")
                except Exception as e:
                    st.error(f"Lỗi khi tải ảnh: {e}")

                st.subheader("5. Cập nhật tham số (Gradient Descent)")
                st.markdown("""
                - Điều chỉnh $W^{(l)}$ và $b^{(l)}$ để giảm mất mát:  
                  $$ W^{(l)} = W^{(l)} - \\eta \\cdot \\frac{\\partial L}{\\partial W^{(l)}} $$  
                  $$ b^{(l)} = b^{(l)} - \\eta \\cdot \\frac{\\partial L}{\\partial b^{(l)}} $$  
                - **Giải thích**:  
                  - $\\eta$: Tốc độ học (learning rate), điều chỉnh kích thước bước cập nhật.  
                  - $\\frac{\\partial L}{\\partial W^{(l)}}$: Gradient của mất mát theo trọng số lớp $l$.  
                  - $\\frac{\\partial L}{\\partial b^{(l)}}$: Gradient của mất mát theo bias lớp $l$.  
                - Mục đích: Tối ưu hóa tham số để giảm sai số dự đoán.
                """, unsafe_allow_html=True)
                try:
                    st.image(os.path.join("plnw", "step5_gradient.png"), caption="Minh họa: Cập nhật tham số", width=600)
                except FileNotFoundError:
                    st.error("Không tìm thấy ảnh minh họa cho Bước 5.")
                except Exception as e:
                    st.error(f"Lỗi khi tải ảnh: {e}")

                st.subheader("6. Lặp lại")
                st.markdown("""
                - Lặp lại từ bước 2 qua nhiều **epoch** (số lần lặp tối đa, từ $10$ đến $100$) cho đến khi mất mát $L$ hội tụ.  
                - Mục đích: Tinh chỉnh mô hình qua nhiều vòng lặp để đạt hiệu suất tối ưu.
                """, unsafe_allow_html=True)
                try:
                    st.image(os.path.join("plnw", "step6_repeat_improved.png"), caption="Minh họa: Lặp lại", width=600)
                except FileNotFoundError:
                    st.error("Không tìm thấy ảnh minh họa cho Bước 6.")
                except Exception as e:
                    st.error(f"Lỗi khi tải ảnh: {e}")

                st.subheader("⚙️ Các tham số chính và ứng dụng")
                st.markdown("""
                Các tham số được sử dụng trong tab **Huấn luyện/Đánh giá** ảnh hưởng trực tiếp đến hiệu suất của Neural Network. Dưới đây là mô tả chi tiết từng tham số:
                """, unsafe_allow_html=True)

                st.subheader("1. Số lớp ẩn")
                st.markdown("""
                - Quy định số lượng lớp ẩn trong mạng, ảnh hưởng đến độ sâu và khả năng học các đặc trưng phức tạp.  
                - **Phạm vi/Giá trị mặc định**: Từ $1$ đến $2$.  
                - **Công thức liên quan**:  
                  $$ A^{(l)} = \\sigma(W^{(l)} \\cdot A^{(l-1)} + b^{(l)}), \quad l = 1, 2, ..., L_h $$  
                - **Giải thích**:  
                  - $L_h$: Số lớp ẩn, quyết định số lần biến đổi phi tuyến.  
                  - $A^{(l)}$: Đầu ra của lớp $l$.  
                  - $W^{(l)}$: Trọng số của lớp $l$.  
                  - $b^{(l)}$: Bias của lớp $l$.  
                  - $\\sigma$: Hàm kích hoạt.  
                - **Chú thích**: Giá trị $1$ phù hợp cho dữ liệu đơn giản, $2$ tăng khả năng học các mẫu phức tạp như MNIST.  
                """, unsafe_allow_html=True)

                st.subheader("2. Số nơ-ron mỗi lớp")
                st.markdown("""
                - Số đơn vị xử lý (nơ-ron) trong mỗi lớp ẩn, ảnh hưởng đến dung lượng biểu diễn của mạng.  
                - **Phạm vi/Giá trị mặc định**: Từ $16$ đến $128$.  
                - **Công thức liên quan**:  
                  $$ W^{(l)} \in \mathbb{R}^{n_{l-1} \times n_l} $$  
                - **Giải thích**:  
                  - $n_{l-1}$: Số nơ-ron của lớp trước.  
                  - $n_l$: Số nơ-ron của lớp hiện tại.  
                  - $W^{(l)}$: Ma trận trọng số giữa lớp $l-1$ và $l$.  
                - **Chú thích**: Giá trị lớn (ví dụ: $128$) tăng khả năng học nhưng có thể dẫn đến overfitting.  
                """, unsafe_allow_html=True)

                st.subheader("3. Tốc độ học (Learning Rate)")
                st.markdown("""
                - Tốc độ cập nhật trọng số trong Gradient Descent, kiểm soát bước nhảy khi tối ưu hóa mất mát.  
                - **Phạm vi/Giá trị mặc định**: $[0.01, 0.005, 0.001, 0.0005]$.  
                - **Công thức liên quan**:  
                  $$ W^{(l)} = W^{(l)} - \\eta \\cdot \\frac{\\partial L}{\\partial W^{(l)}} $$  
                - **Giải thích**:  
                  - $\\eta$: Tốc độ học.  
                  - $\\frac{\\partial L}{\\partial W^{(l)}}$: Gradient của hàm mất mát theo trọng số.  
                - **Chú thích**: $\\eta = 0.01$ học nhanh nhưng dễ vượt qua cực trị, $\\eta = 0.0005$ học chậm nhưng ổn định.  
                """, unsafe_allow_html=True)

                st.subheader("4. Số lần lặp (Max Iterations)")
                st.markdown("""
                - Số epoch tối đa để huấn luyện, quyết định số vòng lặp tối ưu hóa mất mát.  
                - **Phạm vi/Giá trị mặc định**: Từ $10$ đến $100$.  
                - **Công thức liên quan**:  
                  $$ \text{Tổng cập nhật} = E \\cdot \\frac{N}{B} $$  
                - **Giải thích**:  
                  - $E$: Số epoch.  
                  - $N$: Số mẫu.  
                  - $B$: Kích thước batch.  
                - **Chú thích**: Giá trị lớn (ví dụ: $100$) tăng cơ hội hội tụ nhưng tốn thời gian.  
                """, unsafe_allow_html=True)

                st.subheader("5. Hàm kích hoạt")
                st.markdown("""
                - Hàm phi tuyến áp dụng trên mỗi nơ-ron, giúp mạng học các mối quan hệ phi tuyến.  
                - **Phạm vi/Giá trị mặc định**: ReLU, Sigmoid, Tanh.  
                - **Công thức liên quan**:  
                  - ReLU:  
                    $$ \\sigma(z) = \max(0, z) $$  
                  - Sigmoid:  
                    $$ \\sigma(z) = \\frac{1}{1 + e^{-z}} $$  
                  - Tanh:  
                    $$ \\sigma(z) = \\tanh(z) $$  
                - **Giải thích**:  
                  - $\\sigma(z)$: Đầu ra của hàm kích hoạt ứng với đầu vào $z$.  
                - **Chú thích**: ReLU tránh gradient vanishing, Sigmoid phù hợp với đầu ra xác suất, Tanh cân bằng âm/dương.  
                """, unsafe_allow_html=True)

                st.subheader("6. Kích thước batch")
                st.markdown("""
                - Số mẫu xử lý cùng lúc trong mỗi lần cập nhật trọng số, ảnh hưởng đến hiệu suất và độ ổn định.  
                - **Phạm vi/Giá trị mặc định**: Từ $32$ đến $256$.  
                - **Công thức liên quan**:  
                  $$ \\frac{\\partial L}{\\partial W^{(l)}} = \\frac{1}{B} \\sum_{i=1}^{B} \\frac{\\partial L_i}{\\partial W^{(l)}} $$  
                - **Giải thích**:  
                  - $B$: Kích thước batch.  
                  - $\\frac{\\partial L_i}{\\partial W^{(l)}}$: Gradient của mất mát cho mẫu $i$.  
                - **Chú thích**: $B = 32$ giảm nhiễu nhưng chậm, $B = 256$ nhanh nhưng ít ổn định.  
                """, unsafe_allow_html=True)

                st.subheader("7. Trình tối ưu (Solver)")
                st.markdown("""
                - Phương pháp tối ưu hóa trọng số, ảnh hưởng đến tốc độ và hiệu quả hội tụ.  
                - **Phạm vi/Giá trị mặc định**: Adam, SGD.  
                - **Công thức liên quan**:  
                  - SGD:  
                    $$ W^{(l)} = W^{(l)} - \\eta \\cdot \\frac{\\partial L}{\\partial W^{(l)}} $$  
                  - Adam:  
                    $$ m_t = \\beta_1 m_{t-1} + (1 - \\beta_1) \\cdot g_t $$  
                    $$ v_t = \\beta_2 v_{t-1} + (1 - \\beta_2) \\cdot g_t^2 $$  
                    $$ W^{(l)}_{t+1} = W^{(l)}_t - \\eta \\cdot \\frac{m_t}{\\sqrt{v_t} + \epsilon} $$  
                - **Giải thích**:  
                  - $g_t$: Gradient tại bước $t$.  
                  - $m_t$: Động lượng (momentum).  
                  - $v_t$: Bình phương gradient (RMSProp).  
                  - $\\beta_1, \\beta_2$: Hằng số điều chỉnh (thường là $0.9$ và $0.999$).  
                  - $\\epsilon$: Giá trị nhỏ tránh chia cho $0$ (thường là $10^{-8}$).  
                - **Chú thích**: Adam nhanh và hiệu quả với dữ liệu lớn, SGD đơn giản nhưng chậm với dữ liệu phức tạp.  
                """, unsafe_allow_html=True)

                st.subheader("🟪 Ưu điểm và nhược điểm")
                st.markdown("""
                - **✅ Ưu điểm**:  
                  - Học được các đặc trưng phức tạp từ dữ liệu hình ảnh như MNIST.  
                  - Linh hoạt với nhiều tham số để tối ưu hóa.  
                - **❌ Nhược điểm**:  
                  - Tốn thời gian huấn luyện nếu số mẫu lớn hoặc cấu trúc mạng phức tạp.  
                  - Yêu cầu điều chỉnh tham số cẩn thận để đạt hiệu quả tối ưu.  
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
                st.subheader("📘 4. Công thức đánh giá độ chính xác (Accuracy)")
                st.markdown("""
                - Độ chính xác (**Accuracy**) đo tỷ lệ dự đoán đúng:  
                  $$ \\text{Accuracy} = \\frac{\\text{Số mẫu dự đoán đúng}}{\\text{Tổng số mẫu}} $$  
                - **Giải thích**:  
                  - $\\text{Số mẫu dự đoán đúng}$: Số lần mô hình dự đoán nhãn chính xác so với nhãn thực tế.  
                  - $\\text{Tổng số mẫu}$: Tổng số mẫu trong tập dữ liệu kiểm tra.  
                - **Ví dụ**: Dự đoán đúng $92/100$ ảnh → $\\text{Accuracy} = 0.92$ (tức $92\%$).  
                - Mục đích: Đo lường khả năng phân loại đúng các chữ số của Neural Network dựa trên đặc trưng pixel học được.
                """, unsafe_allow_html=True)
                status_text.text("Đã tải xong! 100%")
                time.sleep(0.5)
                status_text.empty()
                progress_bar.empty()

    # Tab 2: Tải dữ liệu (Đã cập nhật để tự động chọn tham số tối ưu)
    with tab_load:
        st.markdown('<div class="section-title">Tải và Chuẩn bị Dữ liệu</div>', unsafe_allow_html=True)

        st.markdown("""
        **Tập dữ liệu MNIST**: Gồm $70,000$ ảnh chữ số ($0$-$9$) với kích thước $28 \\times 28$ pixel. Bạn có thể tải toàn bộ dữ liệu và chọn số lượng mẫu phù hợp để huấn luyện.
        """, unsafe_allow_html=True)

        with st.container():
            st.subheader("Tải dữ liệu")
            if st.button("Tải dữ liệu MNIST từ OpenML", type="primary", help="Tải toàn bộ tập dữ liệu MNIST từ OpenML"):
                with st.spinner("Đang tải dữ liệu..."):
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    for i in range(0, 101, 20):
                        progress_bar.progress(i)
                        status_text.text(f"Đang tải dữ liệu từ OpenML... {i}%")
                        time.sleep(0.1)
                    try:
                        X, y = fetch_mnist_data()
                        X = np.array(X, dtype=np.float64)
                        y = np.array(y, dtype=np.int32)
                        st.session_state['full_data'] = (X, y)
                        with mlflow.start_run(experiment_id=EXPERIMENT_ID, run_name="Data_Load"):
                            mlflow.log_param("total_samples", X.shape[0])
                        st.success("Tải dữ liệu thành công!")
                        st.write(f"Kích thước dữ liệu: {X.shape[0]} mẫu, mỗi mẫu {X.shape[1]} đặc trưng")
                        status_text.text("Đã tải xong! 100%")
                        time.sleep(0.5)
                        status_text.empty()
                        progress_bar.empty()
                    except Exception as e:
                        st.error(f"Không thể tải dữ liệu: {e}")
                        status_text.empty()
                        progress_bar.empty()

        if 'full_data' in st.session_state:
            X_full, y_full = st.session_state['full_data']

            st.subheader("Chọn số lượng mẫu")
            st.markdown("""
            - **100 mẫu**: Huấn luyện nhanh, độ chính xác thấp, phù hợp để thử nghiệm.  
            - **1,000 mẫu**: Huấn luyện khá nhanh, độ chính xác trung bình, phù hợp để kiểm tra cơ bản.  
            - **10,000 mẫu**: Huấn luyện lâu hơn, độ chính xác khá, cân bằng giữa tốc độ và hiệu suất.  
            - **50,000 mẫu**: Huấn luyện lâu nhất, độ chính xác cao, phù hợp cho huấn luyện chuyên sâu.  
            """, unsafe_allow_html=True)

            col1, col2 = st.columns(2)
            with col1:
                sample_options = {
                    "100 mẫu (Thử nghiệm nhanh)": 100,
                    "1,000 mẫu (Kiểm tra cơ bản)": 1000,
                    "10,000 mẫu (Cân bằng hiệu suất)": 10000,
                    "50,000 mẫu (Huấn luyện chuyên sâu)": 50000
                }
                selected_option = st.selectbox("Chọn số lượng mẫu:", list(sample_options.keys()), help="Chọn số lượng mẫu có sẵn")
                num_samples = sample_options[selected_option]

                if st.button("Xác nhận số lượng (tùy chọn có sẵn)", type="primary"):
                    with st.spinner(f"Đang lấy {num_samples} mẫu..."):
                        progress_bar = st.progress(0)
                        status_text = st.empty()
                        for i in range(0, 101, 20):
                            progress_bar.progress(i)
                            status_text.text(f"Đang chọn {num_samples} mẫu... {i}%")
                            time.sleep(0.1)
                        indices = np.random.choice(len(X_full), size=num_samples, replace=False)
                        X_sampled = X_full[indices]
                        y_sampled = y_full[indices]
                        st.session_state['data'] = (X_sampled, y_sampled)
                        # Tự động cập nhật tham số tối ưu dựa trên num_samples
                        st.session_state['optimal_params'] = get_optimal_params(num_samples)
                        with mlflow.start_run(experiment_id=EXPERIMENT_ID, run_name="Data_Sample"):
                            mlflow.log_param("num_samples", num_samples)
                        st.success(f"Đã chọn {num_samples} mẫu! Tham số tối ưu đã được cập nhật.")
                        status_text.text("Đã xử lý xong! 100%")
                        time.sleep(0.5)
                        status_text.empty()
                        progress_bar.empty()

            with col2:
                custom_num_samples = st.number_input("Nhập số lượng tùy ý (tối đa $70,000$):", min_value=1, max_value=70000, value=1000, step=100, help="Nhập số lượng mẫu tùy chỉnh")
                if st.button("Xác nhận số lượng (tùy ý)", type="primary"):
                    if custom_num_samples <= 70000:
                        with st.spinner(f"Đang lấy {custom_num_samples} mẫu..."):
                            progress_bar = st.progress(0)
                            status_text = st.empty()
                            for i in range(0, 101, 20):
                                progress_bar.progress(i)
                                status_text.text(f"Đang chọn {custom_num_samples} mẫu... {i}%")
                                time.sleep(0.1)
                            indices = np.random.choice(len(X_full), size=custom_num_samples, replace=False)
                            X_sampled = X_full[indices]
                            y_sampled = y_full[indices]
                            st.session_state['data'] = (X_sampled, y_sampled)
                            # Tự động cập nhật tham số tối ưu dựa trên custom_num_samples
                            st.session_state['optimal_params'] = get_optimal_params(custom_num_samples)
                            with mlflow.start_run(experiment_id=EXPERIMENT_ID, run_name="Data_Sample_Custom"):
                                mlflow.log_param("num_samples", custom_num_samples)
                            st.success(f"Đã chọn {custom_num_samples} mẫu! Tham số tối ưu đã được cập nhật.")
                            status_text.text("Đã xử lý xong! 100%")
                            time.sleep(0.5)
                            status_text.empty()
                            progress_bar.empty()
                    else:
                        st.error("Số lượng mẫu vượt quá $70,000$. Vui lòng nhập số nhỏ hơn hoặc bằng $70,000$!")

    # Tab 3: Xử lý dữ liệu
    with tab_preprocess:
        st.markdown('<div class="section-title">Xử lý Dữ liệu</div>', unsafe_allow_html=True)

        if 'data' not in st.session_state:
            st.info("Vui lòng tải và chọn số lượng mẫu trước.")
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

            col1, col2 = st.columns([3, 1])
            with col1:
                if st.button("Chuẩn hóa dữ liệu (Normalization)", type="primary", help="Chuẩn hóa dữ liệu về thang [0, 1]"):
                    with st.spinner("Đang chuẩn hóa dữ liệu về [0, 1]..."):
                        progress_bar = st.progress(0)
                        status_text = st.empty()
                        for i in range(0, 101, 20):
                            progress_bar.progress(i)
                            status_text.text(f"Đang chuẩn hóa dữ liệu... {i}%")
                            time.sleep(0.1)
                        X_norm = X / 255.0
                        st.session_state["data_processed"] = (X_norm, y)
                        st.success("Đã xử lý dữ liệu!")
                        status_text.text("Đã xử lý xong! 100%")
                        time.sleep(0.5)
                        status_text.empty()
                        progress_bar.empty()
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
            st.info("Vui lòng tải và xử lý dữ liệu trước.")
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
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    for i in range(0, 101, 20):
                        progress_bar.progress(i)
                        status_text.text(f"Đang chia dữ liệu... {i}%")
                        time.sleep(0.1)
                    st.session_state['split_data'] = {
                        "X_train": X_train, "y_train": y_train,
                        "X_valid": X_valid, "y_valid": y_valid,
                        "X_test": X_test, "y_test": y_test
                    }
                    st.success("Đã chia dữ liệu thành công!")
                    status_text.text("Đã xử lý xong! 100%")
                    time.sleep(0.5)
                    status_text.empty()
                    progress_bar.empty()

    # Tab 5: Huấn luyện/Đánh giá (Đã cập nhật để sử dụng tham số tối ưu tự động)
    with tab_train_eval:
        st.markdown('<div class="section-title">Huấn luyện và Đánh giá Mô hình</div>', unsafe_allow_html=True)

        if 'split_data' not in st.session_state:
            st.info("Vui lòng chia dữ liệu trước.")
        else:
            X_train = st.session_state['split_data']["X_train"]
            y_train = st.session_state['split_data']["y_train"]
            X_valid = st.session_state['split_data']["X_valid"]
            y_valid = st.session_state['split_data']["y_valid"]
            X_test = st.session_state['split_data']["X_test"]
            y_test = st.session_state['split_data']["y_test"]

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
            if np.any(np.isnan(X_valid)):
                X_valid = np.nan_to_num(X_valid, nan=0.0)
            if np.any(np.isnan(X_test)):
                X_test = np.nan_to_num(X_test, nan=0.0)

            st.session_state['split_data'] = {
                "X_train": X_train, "y_train": y_train,
                "X_valid": X_valid, "y_valid": y_valid,
                "X_test": X_test, "y_test": y_test
            }

            num_samples = len(X_train)
            st.write(f"**Số mẫu huấn luyện**: {num_samples}")
            st.write(f"Kích thước X_train: {X_train.shape}, dtype: {X_train.dtype}")
            st.write(f"Kích thước y_train: {y_train.shape}, dtype: {y_train.dtype}")
            if X_train.shape[0] != y_train.shape[0]:
                st.error("Số mẫu của X_train và y_train không khớp!")
                st.stop()

            def get_optimal_params(num_samples):
                if num_samples <= 100:
                    return {
                        "hidden_layer_sizes": (32,),
                        "learning_rate": 0.01,
                        "epochs": 15,
                        "activation": "relu",
                        "solver": "sgd",
                        "batch_size": 64
                    }
                elif num_samples <= 1000:
                    return {
                        "hidden_layer_sizes": (64,),
                        "learning_rate": 0.005,
                        "epochs": 30,
                        "activation": "relu",
                        "solver": "adam",
                        "batch_size": 128
                    }
                elif num_samples <= 10000:
                    return {
                        "hidden_layer_sizes": (100, 50),
                        "learning_rate": 0.001,
                        "epochs": 50,
                        "activation": "relu",
                        "solver": "adam",
                        "batch_size": 256
                    }
                else:
                    return {
                        "hidden_layer_sizes": (128, 64),
                        "learning_rate": 0.001,
                        "epochs": 75,
                        "activation": "relu",
                        "solver": "adam",
                        "batch_size": 512
                    }

            # Đảm bảo optimal_params đã được khởi tạo
            if "optimal_params" not in st.session_state:
                st.session_state["optimal_params"] = get_optimal_params(num_samples)
            
            # Sử dụng optimal_params làm mặc định, cho phép người dùng chỉnh sửa
            params = st.session_state.get("training_params", st.session_state["optimal_params"].copy())

            st.subheader("⚙️ Cấu hình Tham số Mô hình")
            st.markdown("""
            | Số mẫu       | Số lớp ẩn | Kích thước lớp ẩn | Tốc độ học | Số lần lặp | Hàm kích hoạt | Trình tối ưu | Kích thước batch |
            |--------------|-----------|-------------------|------------|------------|---------------|--------------|------------------|
            | ≤ 100        | 1         | 32                | 0.01       | 15         | ReLU          | SGD          | 64              |
            | ≤ 1,000      | 1         | 64                | 0.005      | 30         | ReLU          | Adam         | 128             |
            | ≤ 10,000     | 2         | (100, 50)         | 0.001      | 50         | ReLU          | Adam         | 256             |
            | > 10,000     | 2         | (128, 64)         | 0.001      | 75         | ReLU          | Adam         | 512             |
            """, unsafe_allow_html=True)
            st.info(f"Tham số tối ưu cho {num_samples} mẫu: {st.session_state['optimal_params']}")

            col_param1, col_param2 = st.columns(2)
            with col_param1:
                with st.expander("🧠 Cấu trúc Mạng", expanded=True):
                    st.markdown("**Tùy chỉnh số lớp ẩn và nơ-ron**", unsafe_allow_html=True)
                    num_hidden_layers = st.number_input("Số lớp ẩn", min_value=1, max_value=2, value=len(params["hidden_layer_sizes"]), 
                                                       help="Chọn 1 hoặc 2 lớp ẩn để điều chỉnh độ phức tạp của mô hình.")
                    hidden_sizes = list(params["hidden_layer_sizes"])  # Chuyển tuple thành list để chỉnh sửa
                    
                    if num_hidden_layers == 1:
                        hidden_size_1 = st.number_input("Số nơ-ron lớp ẩn 1", min_value=16, max_value=128, 
                                                        value=hidden_sizes[0], 
                                                        help="Số nơ-ron cho lớp ẩn duy nhất (16-128).")
                        hidden_sizes = [hidden_size_1]
                    elif num_hidden_layers == 2:
                        hidden_size_1 = st.number_input("Số nơ-ron lớp ẩn 1", min_value=16, max_value=128, 
                                                        value=hidden_sizes[0] if len(hidden_sizes) > 0 else 100, 
                                                        help="Số nơ-ron cho lớp ẩn đầu tiên (16-128).")
                        hidden_size_2 = st.number_input("Số nơ-ron lớp ẩn 2", min_value=16, max_value=128, 
                                                        value=hidden_sizes[1] if len(hidden_sizes) > 1 else 50, 
                                                        help="Số nơ-ron cho lớp ẩn thứ hai (16-128).")
                        hidden_sizes = [hidden_size_1, hidden_size_2]
                    
                    params["hidden_layer_sizes"] = tuple(hidden_sizes)
                    params["activation"] = st.selectbox("Hàm kích hoạt", ["relu", "sigmoid", "tanh"], 
                                                        index=["relu", "sigmoid", "tanh"].index(params["activation"]),
                                                        help="Chọn hàm kích hoạt: ReLU (nhanh), Sigmoid (xác suất), Tanh (cân bằng).")
            
            with col_param2:
                with st.expander("🔧 Tối ưu hóa", expanded=True):
                    st.markdown("**Cấu hình huấn luyện**", unsafe_allow_html=True)
                    params["learning_rate"] = st.selectbox("Tốc độ học", [0.01, 0.005, 0.001, 0.0005], 
                                                           index=[0.01, 0.005, 0.001, 0.0005].index(params["learning_rate"]),
                                                           help="Tốc độ học càng nhỏ càng ổn định nhưng chậm.")
                    params["epochs"] = st.number_input("Số lần lặp (Epochs)", min_value=10, max_value=100, value=params["epochs"], 
                                                       help="Số lần lặp qua toàn bộ dữ liệu (10-100).")
                    params["batch_size"] = st.number_input("Kích thước batch", min_value=32, max_value=512, value=params["batch_size"], 
                                                           help="Số mẫu mỗi lần cập nhật trọng số (32-512).")
                    params["solver"] = st.selectbox("Trình tối ưu", ["adam", "sgd"], 
                                                    index=["adam", "sgd"].index(params["solver"]),
                                                    help="Adam (nhanh, hiệu quả), SGD (đơn giản, chậm hơn).")
                    early_stopping = st.checkbox("Dừng sớm (Early Stopping)", value=True, 
                                                 help="Dừng huấn luyện nếu không cải thiện trên tập validation sau 10 epochs.")

            col_reset, col_train = st.columns([1, 3])
            with col_reset:
                if st.button("🔄 Khôi phục tham số tối ưu", key="reset_params"):
                    st.session_state["training_params"] = st.session_state["optimal_params"].copy()
                    st.success("Đã khôi phục tham số tối ưu!")
                    st.rerun()

            st.session_state["training_params"] = params

            with col_train:
                if st.button("🚀 Bắt đầu Huấn luyện", type="primary", key="start_training"):
                    try:
                        with st.spinner("Đang huấn luyện mô hình..."):
                            progress_bar = st.progress(0)
                            status_text = st.empty()
                            start_time = time.time()

                            status_text.text("Đang chuẩn bị dữ liệu... 20%")
                            progress_bar.progress(20)
                            time.sleep(0.1)

                            # Xây dựng mô hình TensorFlow
                            model = models.Sequential()
                            model.add(layers.Input(shape=(784,)))
                            for neurons in params["hidden_layer_sizes"]:
                                model.add(layers.Dense(neurons, activation=params["activation"]))
                            model.add(layers.Dense(10, activation='softmax'))

                            # Chọn optimizer
                            optimizer = tf.keras.optimizers.Adam(learning_rate=params["learning_rate"]) if params["solver"] == "adam" else tf.keras.optimizers.SGD(learning_rate=params["learning_rate"])

                            # Biên dịch mô hình
                            model.compile(optimizer=optimizer,
                                          loss='sparse_categorical_crossentropy',
                                          metrics=['accuracy'])

                            # Callback để theo dõi tiến trình
                            class ProgressCallback(callbacks.Callback):
                                def on_epoch_end(self, epoch, logs=None):
                                    progress = (epoch + 1) / params["epochs"] * 100
                                    progress_bar.progress(int(progress))
                                    status_text.text(f"Đang huấn luyện... {int(progress)}%")

                            callbacks_list = [ProgressCallback()]
                            if early_stopping:
                                callbacks_list.append(callbacks.EarlyStopping(monitor='val_loss', patience=10))

                            status_text.text("Đang huấn luyện mô hình... 50%")
                            progress_bar.progress(50)

                            # Huấn luyện mô hình
                            history = model.fit(X_train, y_train, epochs=params["epochs"], batch_size=params["batch_size"],
                                                validation_data=(X_valid, y_valid), callbacks=callbacks_list, verbose=0)

                            status_text.text("Đang đánh giá mô hình... 90%")
                            progress_bar.progress(90)
                            time.sleep(0.1)

                            # Đánh giá mô hình
                            y_valid_pred = np.argmax(model.predict(X_valid, verbose=0), axis=1)
                            y_test_pred = np.argmax(model.predict(X_test, verbose=0), axis=1)
                            acc_valid = accuracy_score(y_valid, y_valid_pred)
                            acc_test = accuracy_score(y_test, y_test_pred)
                            cm_valid = confusion_matrix(y_valid, y_valid_pred)
                            cm_test = confusion_matrix(y_test, y_test_pred)

                            status_text.text("Đang lưu kết quả... 100%")
                            progress_bar.progress(100)
                            time.sleep(0.1)

                            # Lưu kết quả vào MLflow
                            run_name = f"NeuralNetwork_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                            with mlflow.start_run(experiment_id=EXPERIMENT_ID, run_name=run_name) as run:
                                mlflow.log_params({**params, "early_stopping": early_stopping})
                                mlflow.log_metric("accuracy_val", acc_valid)
                                mlflow.log_metric("accuracy_test", acc_test)
                                mlflow.log_metric("training_time", time.time() - start_time)
                                mlflow.log_metric("n_iter_actual", len(history.history['loss']))
                                for epoch, (loss, acc) in enumerate(zip(history.history['loss'], history.history['accuracy']), 1):
                                    mlflow.log_metric(f"loss_epoch_{epoch}", loss)
                                    mlflow.log_metric(f"accuracy_epoch_{epoch}", acc)
                                for epoch, val_acc in enumerate(history.history['val_accuracy'], 1):
                                    mlflow.log_metric(f"val_accuracy_epoch_{epoch}", val_acc)

                                st.session_state['model'] = model
                                st.session_state['training_results'] = {
                                    'accuracy_val': acc_valid, 'accuracy_test': acc_test,
                                    'cm_valid': cm_valid, 'cm_test': cm_test,
                                    'run_name': run_name, 'run_id': run.info.run_id,
                                    'params': params, 'training_time': time.time() - start_time,
                                    'loss_history': history.history['loss'],
                                    'val_loss_history': history.history['val_loss'],
                                    'accuracy_history': history.history['accuracy'],
                                    'val_accuracy_history': history.history['val_accuracy'],
                                    'n_iter_actual': len(history.history['loss'])
                                }

                            st.success(f"Đã huấn luyện xong! Thời gian: {time.time() - start_time:.2f} giây, Số lần lặp thực tế: {len(history.history['loss'])}")
                            status_text.text("Đã hoàn tất huấn luyện! 100%")
                            time.sleep(0.5)
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
                with col_cm2:
                    fig, ax = plt.subplots(figsize=(6, 5))
                    sns.heatmap(results['cm_test'], annot=True, fmt="d", cmap="Blues", ax=ax)
                    ax.set_title("Test")
                    st.pyplot(fig)

                st.subheader("📉 Biểu đồ Kết quả Huấn luyện")
                # Biểu đồ Loss
                if results['loss_history']:
                    fig, ax = plt.subplots(figsize=(8, 4))
                    ax.plot(range(1, len(results['loss_history']) + 1), results['loss_history'], 
                            label='Training Loss', marker='o', linestyle='-')
                    if results['val_loss_history']:
                        ax.plot(range(1, len(results['val_loss_history']) + 1), results['val_loss_history'], 
                                label='Validation Loss', marker='s', linestyle='--')
                    ax.set_xlabel("Epochs")
                    ax.set_ylabel("Loss")
                    ax.set_title("Training & Validation Loss")
                    ax.legend()
                    ax.grid(True)
                    st.pyplot(fig)
                    st.markdown("""
                    **Giải thích biểu đồ Loss:**
                    - **Train Loss (Mất mát huấn luyện):** Đại diện cho sai số giữa dự đoán và nhãn thực tế trên tập huấn luyện. Giá trị giảm dần qua các epoch cho thấy mô hình đang học tốt hơn.
                    - **Val Loss (Mất mát validation):** Đo lường sai số trên tập validation (nếu có), giúp đánh giá khả năng tổng quát hóa. Nếu Val Loss ổn định hoặc giảm chậm, mô hình không bị overfitting.
                    - Hai đường này nên có xu hướng tương tự; nếu Val Loss tăng trong khi Train Loss giảm, đó là dấu hiệu của overfitting.
                    """)
                    st.markdown("---")

                # Biểu đồ Accuracy
                if results['accuracy_history']:
                    fig, ax = plt.subplots(figsize=(8, 4))
                    ax.plot(range(1, len(results['accuracy_history']) + 1), results['accuracy_history'], 
                            label='Training Accuracy', marker='o', linestyle='-')
                    if results['val_accuracy_history'] and any(v is not None for v in results['val_accuracy_history']):
                        ax.plot(range(1, len(results['val_accuracy_history']) + 1), results['val_accuracy_history'], 
                                label='Validation Accuracy', marker='s', linestyle='--')
                    ax.set_xlabel("Epochs")
                    ax.set_ylabel("Accuracy")
                    ax.set_title("Training & Validation Accuracy")
                    ax.legend()
                    ax.grid(True)
                    st.pyplot(fig)
                    st.markdown("""
                    **Giải thích biểu đồ Accuracy:**
                    - **Train Accuracy (Độ chính xác huấn luyện):** Tỷ lệ dự đoán đúng trên tập huấn luyện, thường tăng qua các epoch khi mô hình học.
                    - **Val Accuracy (Độ chính xác validation):** Tỷ lệ dự đoán đúng trên tập validation (nếu có), phản ánh khả năng tổng quát hóa. Giá trị cao và ổn định cho thấy mô hình hoạt động tốt trên dữ liệu mới.
                    - Sự khác biệt giữa Train Accuracy và Val Accuracy không quá lớn là dấu hiệu của một mô hình cân bằng.
                    """)

                st.subheader("ℹ️ Thông tin Chi tiết")
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

    # Tab 6: Demo dự đoán
    with tab_demo:
        st.markdown('<div class="section-title">Demo Dự đoán Chữ số</div>', unsafe_allow_html=True)

        if 'model' not in st.session_state:
            st.info("Vui lòng huấn luyện mô hình trước khi sử dụng Demo.")
        else:
            model = st.session_state['model']
            st.markdown("""
            Hãy thử dự đoán chữ số bằng cách **tải ảnh lên** hoặc **vẽ tay**!  
            - **Upload ảnh**: Chọn file ảnh chứa chữ số viết tay (nền trắng, chữ đen).  
            - **Vẽ tay**: Sử dụng bảng vẽ để viết chữ số.  
            """, unsafe_allow_html=True)

            # Tabs con cho Upload và Vẽ
            demo_tabs = st.tabs(["📷 Upload Ảnh", "✏️ Vẽ Tay"])
            tab_upload, tab_draw = demo_tabs

            # Tab Upload Ảnh
            with tab_upload:
                st.markdown('<div class="mode-title">Dự đoán từ Ảnh Upload</div>', unsafe_allow_html=True)
                uploaded_file = st.file_uploader("Chọn ảnh chữ số (JPG, PNG)...", type=["jpg", "png", "jpeg"],
                                                help="Tải lên ảnh chứa chữ số viết tay, nền trắng, chữ đen.")
                
                if uploaded_file is not None:
                    # Hiển thị ảnh gốc
                    image = Image.open(uploaded_file)
                    st.image(image, caption="Ảnh đã tải lên", width=200)

                    # Xử lý ảnh và dự đoán
                    processed_image = preprocess_uploaded_image(image)
                    if processed_image is not None:
                        # Hiển thị ảnh đã xử lý (28x28)
                        st.image(processed_image.reshape(28, 28), caption="Ảnh sau xử lý (28x28)", width=100, clamp=True)
                        
                        # Dự đoán
                        with st.spinner("Đang dự đoán..."):
                            prediction = model.predict(processed_image.reshape(1, 784), verbose=0)
                            predicted_digit = np.argmax(prediction)
                            probabilities = prediction[0]
                            
                            # Hiển thị kết quả
                            st.markdown('<div class="prediction-box">', unsafe_allow_html=True)
                            st.write(f"**Dự đoán**: {predicted_digit}")
                            st.write("**Xác suất từng lớp**:")
                            prob_df = pd.DataFrame({
                                "Chữ số": range(10),
                                "Xác suất": [f"{p*100:.2f}%" for p in probabilities]
                            })
                            st.dataframe(prob_df, height=200)
                            st.markdown('</div>', unsafe_allow_html=True)

            # Tab Vẽ Tay
            with tab_draw:
                st.markdown('<div class="mode-title">Dự đoán từ Vẽ Tay</div>', unsafe_allow_html=True)
                st.markdown("""
                - Vẽ chữ số trên bảng dưới đây (nền đen, nét trắng).  
                - Nhấn "Dự đoán" để xem kết quả.  
                """, unsafe_allow_html=True)

                # Canvas để vẽ
                canvas_result = st_canvas(
                    fill_color="rgba(0, 0, 0, 1)",  # Nền đen
                    stroke_width=20,
                    stroke_color="#FFFFFF",  # Nét trắng
                    background_color="#000000",
                    height=280,
                    width=280,
                    drawing_mode="freedraw",
                    key="canvas",
                    display_toolbar=True,
                )

                if st.button("Dự đoán từ bản vẽ", type="primary"):
                    if canvas_result.image_data is not None:
                        # Xử lý ảnh từ canvas
                        processed_image = preprocess_canvas_image(canvas_result.image_data)
                        if processed_image is not None:
                            # Hiển thị ảnh đã xử lý (28x28)
                            st.image(processed_image.reshape(28, 28), caption="Ảnh sau xử lý (28x28)", width=100, clamp=True)
                            
                            # Dự đoán
                            with st.spinner("Đang dự đoán..."):
                                prediction = model.predict(processed_image.reshape(1, 784), verbose=0)
                                predicted_digit = np.argmax(prediction)
                                probabilities = prediction[0]
                                
                                # Hiển thị kết quả
                                st.markdown('<div class="prediction-box">', unsafe_allow_html=True)
                                st.write(f"**Dự đoán**: {predicted_digit}")
                                st.write("**Xác suất từng lớp**:")
                                prob_df = pd.DataFrame({
                                    "Chữ số": range(10),
                                    "Xác suất": [f"{p*100:.2f}%" for p in probabilities]
                                })
                                st.dataframe(prob_df, height=200)
                                st.markdown('</div>', unsafe_allow_html=True)
                    else:
                        st.warning("Vui lòng vẽ chữ số trước khi dự đoán!")

    # Tab 7: Thông tin huấn luyện
    with tab_log_info:
        st.markdown('<div class="section-title">Theo dõi Kết quả</div>', unsafe_allow_html=True)
        try:
            with st.spinner("Đang tải thông tin huấn luyện..."):
                progress_bar = st.progress(0)
                status_text = st.empty()
                for i in range(0, 101, 20):
                    progress_bar.progress(i)
                    status_text.text(f"Đang tải thông tin huấn luyện... {i}%")
                    time.sleep(0.1)
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

                status_text.text("Đã tải xong! 100%")
                time.sleep(0.5)
                status_text.empty()
                progress_bar.empty()
        except Exception as e:
            st.error(f"Lỗi kết nối MLflow: {e}")

if __name__ == "__main__":
    run_mnist_neural_network_app()