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
                    st.image(os.path.join("plnw", "step1_init.png"), caption="Minh họa: Khởi tạo mô hình", width=600)
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
                    st.image(os.path.join("plnw", "step2_feedforward.png"), caption="Minh họa: Lan truyền thuận", width=600)
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
                    st.image(os.path.join("plnw", "step3_loss.png"), caption="Minh họa: Tính hàm mất mát", width=600)
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
                - **Chú thích**:  
                  - $\\eta$: Tốc độ học (learning rate), kiểm soát mức độ thay đổi của $W$ và $b$.  
                  - $\\frac{\\partial L}{\\partial W^{(l)}}$: Gradient của $L$ theo $W^{(l)}$.  
                  - $\\frac{\\partial L}{\\partial b^{(l)}}$: Gradient của $L$ theo $b^{(l)}$.  
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
                - Lặp lại từ bước 2 qua nhiều **epoch** cho đến khi mất mát $L$ hội tụ.  
                - **Chú thích**:  
                  - **Epoch**: Một lần lặp qua toàn bộ tập dữ liệu huấn luyện.  
                - Mục đích: Tinh chỉnh mô hình qua nhiều vòng lặp để đạt hiệu suất tối ưu.  
                """, unsafe_allow_html=True)
                try:
                    st.image(os.path.join("plnw", "step6_repeat_improved.png"), caption="Minh họa: Lặp lại", width=600)
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
                   - **Hoạt động**: Quyết định cách mạng hội tụ đến điểm tối ưu.  
                   - **Ví dụ phổ biến:**  
                     - **Adam**: Kết hợp động lượng và RMSProp, thích nghi với tốc độ học, nhanh và hiệu quả.  
                     - **SGD (Stochastic Gradient Descent)**: Cập nhật trọng số dựa trên gradient, đơn giản nhưng chậm hơn Adam.  
                   - **Công thức (SGD)**:  
                     $$ W_{t+1} = W_t - \\eta \\cdot \\frac{\\partial L}{\\partial W_t} $$  
                     - $W_{t+1}$: Trọng số sau khi cập nhật.  
                     - $W_t$: Trọng số tại bước hiện tại.  
                     - $\\eta$: Tốc độ học.  
                     - $\\frac{\\partial L}{\\partial W_t}$: Gradient của mất mát theo trọng số.  
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