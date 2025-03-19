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

# Hàm chọn tham số tối ưu dựa trên số mẫu cho Pseudo-Labeling với Neural Network
def get_optimal_params(num_samples):
    if num_samples <= 1000:
        return {
            "hidden_layer_sizes": (32,),
            "learning_rate": 0.001,
            "epochs": 30,
            "activation": "relu",
            "solver": "adam",
            "batch_size": 32,
            "threshold": 0.9,  # Ngưỡng tin cậy thấp hơn vì ít dữ liệu
            "max_iterations": 3  # Ít vòng lặp hơn vì dữ liệu nhỏ
        }
    elif num_samples <= 10000:
        return {
            "hidden_layer_sizes": (64, 32),
            "learning_rate": 0.0005,
            "epochs": 50,
            "activation": "relu",
            "solver": "adam",
            "batch_size": 64,
            "threshold": 0.95,  # Ngưỡng trung bình
            "max_iterations": 5  # Số vòng lặp trung bình
        }
    elif num_samples <= 50000:
        return {
            "hidden_layer_sizes": (128, 64),
            "learning_rate": 0.0003,
            "epochs": 70,
            "activation": "relu",
            "solver": "adam",
            "batch_size": 128,
            "threshold": 0.97,  # Ngưỡng cao hơn để đảm bảo chất lượng
            "max_iterations": 7  # Tăng số vòng lặp
        }
    else:  # > 50,000
        return {
            "hidden_layer_sizes": (128, 64, 32),
            "learning_rate": 0.0001,
            "epochs": 100,
            "activation": "relu",
            "solver": "adam",
            "batch_size": 256,
            "threshold": 0.98,  # Ngưỡng rất cao cho dữ liệu lớn
            "max_iterations": 10  # Số vòng lặp tối đa
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

    tabs = st.tabs(["Thông tin", "Chọn dữ liệu", "Xử lý dữ liệu", "Chia dữ liệu", "Huấn luyện/Đánh giá", "Demo dự đoán", "Thông tin huấn luyện"])
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

            # Bước 1: Lấy 1% số lượng ảnh cho mỗi class (0-9) để làm tập train ban đầu
            st.subheader("Tạo tập dữ liệu ban đầu (1% mỗi lớp)")
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

            st.write(f"**Tập dữ liệu huấn luyện ban đầu lấy (1%)**: {len(X_train_initial)} mẫu")
            st.write(f"**Tập dữ liệu huấn luyện chưa gắn nhãn (99%)**: {len(X_unlabeled)} mẫu")

            # Lưu trữ tập dữ liệu ban đầu
            st.session_state['pseudo_data'] = {
                'X_train_initial': X_train_initial.copy(),
                'y_train_initial': y_train_initial.copy(),
                'X_unlabeled': X_unlabeled.copy(),
                'y_unlabeled_indices': y_unlabeled_indices,
                'X_test': X_test.copy(),
                'y_test': y_test.copy()
            }

            if "optimal_params" not in st.session_state:
                st.session_state["optimal_params"] = get_optimal_params(num_samples)
            
            params = st.session_state.get("training_params", st.session_state["optimal_params"].copy())

            st.subheader("⚙️ Cấu hình tham khảo Tham số Mô hình")
            st.markdown(f"""
            Dựa trên số mẫu huấn luyện ban đầu ({num_samples} mẫu), bảng dưới đây gợi ý các tham số tối ưu cho bài toán **Pseudo-Labeling với Neural Network**:

            | Số mẫu       | Số lớp ẩn | Kích thước lớp ẩn | Tốc độ học | Số lần lặp | Hàm kích hoạt | Trình tối ưu | Kích thước batch | Ngưỡng tin cậy | Số vòng lặp tối đa |
            |--------------|-----------|-------------------|------------|------------|---------------|--------------|------------------|----------------|-------------------|
            | ≤ 1,000      | 1         | 32                | 0.001      | 30         | ReLU          | Adam         | 32               | 0.9            | 3                 |
            | ≤ 10,000     | 2         | (64, 32)          | 0.0005     | 50         | ReLU          | Adam         | 64               | 0.95           | 5                 |
            | ≤ 50,000     | 2         | (128, 64)         | 0.0003     | 70         | ReLU          | Adam         | 128              | 0.97           | 7                 |
            | > 50,000     | 3         | (128, 64, 32)     | 0.0001     | 100        | ReLU          | Adam         | 256              | 0.98           | 10                |
            """, unsafe_allow_html=True)
            st.info(f"Tham số tối ưu gợi ý cho {num_samples} mẫu: {st.session_state['optimal_params']}")

            col_param1, col_param2 = st.columns(2)
            with col_param1:
                with st.expander("🧠 Cấu trúc Mạng", expanded=True):
                    st.markdown("**Tùy chỉnh số lớp ẩn và nơ-ron**", unsafe_allow_html=True)
                    num_hidden_layers = st.number_input("Số lớp ẩn", min_value=1, max_value=3, value=len(params["hidden_layer_sizes"]), 
                                                       help="Chọn 1, 2 hoặc 3 lớp ẩn để điều chỉnh độ phức tạp của mô hình.")
                    hidden_sizes = list(params["hidden_layer_sizes"])
                    
                    if num_hidden_layers == 1:
                        hidden_size_1 = st.number_input("Số nơ-ron lớp ẩn 1", min_value=16, max_value=128, 
                                                        value=hidden_sizes[0] if len(hidden_sizes) > 0 else 32, 
                                                        help="Số nơ-ron cho lớp ẩn duy nhất (16-128).")
                        hidden_sizes = [hidden_size_1]
                    elif num_hidden_layers == 2:
                        hidden_size_1 = st.number_input("Số nơ-ron lớp ẩn 1", min_value=16, max_value=128, 
                                                        value=hidden_sizes[0] if len(hidden_sizes) > 0 else 64, 
                                                        help="Số nơ-ron cho lớp ẩn đầu tiên (16-128).")
                        hidden_size_2 = st.number_input("Số nơ-ron lớp ẩn 2", min_value=16, max_value=128, 
                                                        value=hidden_sizes[1] if len(hidden_sizes) > 1 else 32, 
                                                        help="Số nơ-ron cho lớp ẩn thứ hai (16-128).")
                        hidden_sizes = [hidden_size_1, hidden_size_2]
                    elif num_hidden_layers == 3:
                        hidden_size_1 = st.number_input("Số nơ-ron lớp ẩn 1", min_value=16, max_value=128, 
                                                        value=hidden_sizes[0] if len(hidden_sizes) > 0 else 128, 
                                                        help="Số nơ-ron cho lớp ẩn đầu tiên (16-128).")
                        hidden_size_2 = st.number_input("Số nơ-ron lớp ẩn 2", min_value=16, max_value=128, 
                                                        value=hidden_sizes[1] if len(hidden_sizes) > 1 else 64, 
                                                        help="Số nơ-ron cho lớp ẩn thứ hai (16-128).")
                        hidden_size_3 = st.number_input("Số nơ-ron lớp ẩn 3", min_value=16, max_value=128, 
                                                        value=hidden_sizes[2] if len(hidden_sizes) > 2 else 32, 
                                                        help="Số nơ-ron cho lớp ẩn thứ ba (16-128).")
                        hidden_sizes = [hidden_size_1, hidden_size_2, hidden_size_3]
                    
                    params["hidden_layer_sizes"] = tuple(hidden_sizes)
                    params["activation"] = st.selectbox("Hàm kích hoạt", ["relu", "sigmoid", "tanh"], 
                                                        index=["relu", "sigmoid", "tanh"].index(params["activation"]),
                                                        help="Chọn hàm kích hoạt: ReLU (nhanh), Sigmoid (xác suất), Tanh (cân bằng).")
            
            with col_param2:
                with st.expander("🔧 Tối ưu hóa", expanded=True):
                    st.markdown("**Cấu hình huấn luyện**", unsafe_allow_html=True)
                    params["learning_rate"] = st.selectbox("Tốc độ học", [0.01, 0.005, 0.001, 0.0005, 0.0003, 0.0001], 
                                                           index=[0.01, 0.005, 0.001, 0.0005, 0.0003, 0.0001].index(params["learning_rate"]),
                                                           help="Tốc độ học càng nhỏ càng ổn định nhưng chậm.")
                    params["epochs"] = st.number_input("Số lần lặp (Epochs)", min_value=10, max_value=100, value=params["epochs"], 
                                                       help="Số lần lặp qua toàn bộ dữ liệu (10-100).")
                    params["batch_size"] = st.number_input("Kích thước batch", min_value=32, max_value=256, value=params["batch_size"], 
                                                           help="Số mẫu mỗi lần cập nhật trọng số (32-256).")
                    params["solver"] = st.selectbox("Trình tối ưu", ["adam", "sgd"], 
                                                    index=["adam", "sgd"].index(params["solver"]),
                                                    help="Adam (nhanh, hiệu quả), SGD (đơn giản, chậm hơn).")
                    threshold_default = st.session_state.get("optimal_params", {}).get("threshold", 0.95)
                    threshold = st.slider("Ngưỡng tin cậy Pseudo-Label", 0.5, 1.0, 
                                          threshold_default, 
                                          help="Ngưỡng để gán nhãn giả cho dữ liệu không có nhãn.")
                    max_iterations = st.number_input("Số vòng lặp tối đa", min_value=1, max_value=10, 
                                                     value=st.session_state["optimal_params"]["max_iterations"], 
                                                     help="Số lần lặp tối đa cho quá trình Pseudo-Labeling.")

            col_reset, col_train = st.columns([1, 3])
            with col_reset:
                if st.button("🔄 Khôi phục tham số tối ưu", key="reset_params"):
                    st.session_state["training_params"] = st.session_state["optimal_params"].copy()
                    st.success("Đã khôi phục tham số tối ưu!")
                    st.rerun()

            st.session_state["training_params"] = params

            with col_train:
                if st.button("🚀 Bắt đầu Huấn luyện với Pseudo-Labeling", type="primary", key="start_training"):
                    try:
                        with st.spinner("Đang thực hiện quy trình Pseudo-Labeling..."):
                            start_time = time.time()

                            # Khởi tạo tập dữ liệu huấn luyện
                            X_train = st.session_state['pseudo_data']['X_train_initial'].copy()
                            y_train = st.session_state['pseudo_data']['y_train_initial'].copy()
                            X_unlabeled = st.session_state['pseudo_data']['X_unlabeled'].copy()
                            y_unlabeled_indices = st.session_state['pseudo_data']['y_unlabeled_indices'].copy()

                            iteration = 0
                            pseudo_labeled_history = []
                            accuracy_test_history = []

                            # Tạo các container cố định để hiển thị thông tin vòng lặp
                            iteration_container = st.empty()  # Container cho tiêu đề vòng lặp
                            progress_bar_container = st.empty()  # Container cho progress bar
                            status_container = st.empty()  # Container cho thông tin epoch
                            pseudo_container = st.empty()  # Container cho số mẫu gán nhãn giả
                            accuracy_container = st.empty()  # Container cho độ chính xác

                            while iteration < max_iterations and len(X_unlabeled) > 0:
                                iteration += 1
                                # Hiển thị tiêu đề vòng lặp trong container cố định
                                iteration_container.markdown(f"**Vòng lặp {iteration}/{max_iterations}**")

                                # Bước 2: Huấn luyện mô hình trên tập dữ liệu hiện tại
                                model = models.Sequential()
                                model.add(layers.Input(shape=(784,)))
                                for neurons in params["hidden_layer_sizes"]:
                                    model.add(layers.Dense(neurons, activation=params["activation"]))
                                model.add(layers.Dense(10, activation='softmax'))

                                optimizer = tf.keras.optimizers.Adam(learning_rate=params["learning_rate"]) if params["solver"] == "adam" else tf.keras.optimizers.SGD(learning_rate=params["learning_rate"])

                                model.compile(optimizer=optimizer,
                                              loss='sparse_categorical_crossentropy',
                                              metrics=['accuracy'])

                                # Tạo progress bar trong container cố định
                                with progress_bar_container:
                                    progress_bar = st.progress(0)

                                class ProgressCallback(callbacks.Callback):
                                    def on_epoch_end(self, epoch, logs=None):
                                        progress = (epoch + 1) / params["epochs"] * 100
                                        progress_bar.progress(int(progress))
                                        # Cập nhật thông tin epoch trong container cố định
                                        status_container.markdown(
                                            f"**Vòng lặp {iteration}/{max_iterations} - Epoch {epoch+1}/{params['epochs']}**: "
                                            f"Loss: {logs['loss']:.4f}, Accuracy: {logs['accuracy']:.4f}"
                                        )

                                history = model.fit(X_train, y_train, epochs=params["epochs"], batch_size=params["batch_size"],
                                                    callbacks=[ProgressCallback()], verbose=0)

                                # Bước 3: Dự đoán nhãn cho tập dữ liệu chưa gắn nhãn
                                predictions = model.predict(X_unlabeled, verbose=0)
                                predicted_labels = np.argmax(predictions, axis=1)
                                confidences = np.max(predictions, axis=1)

                                # Bước 4: Gán nhãn giả với ngưỡng tin cậy
                                pseudo_mask = confidences >= threshold
                                X_pseudo = X_unlabeled[pseudo_mask]
                                y_pseudo = predicted_labels[pseudo_mask]

                                # Hiển thị số mẫu được gán nhãn giả trong container cố định
                                pseudo_container.markdown(f"**Số mẫu được gán nhãn giả trong vòng {iteration}**: {len(X_pseudo)}")

                                # Cập nhật tập dữ liệu huấn luyện
                                X_train = np.concatenate([X_train, X_pseudo], axis=0)
                                y_train = np.concatenate([y_train, y_pseudo], axis=0)

                                # Loại bỏ các mẫu đã được gán nhãn giả khỏi tập unlabeled
                                remaining_mask = ~pseudo_mask
                                X_unlabeled = X_unlabeled[remaining_mask]

                                # Ghi lại số lượng mẫu được gán nhãn giả
                                pseudo_labeled_history.append(len(X_pseudo))

                                # Đánh giá trên tập test
                                y_test_pred = np.argmax(model.predict(X_test, verbose=0), axis=1)
                                acc_test = accuracy_score(y_test, y_test_pred)
                                accuracy_test_history.append(acc_test)
                                # Hiển thị độ chính xác trong container cố định
                                accuracy_container.markdown(f"**Độ chính xác trên tập Test sau vòng {iteration}**: {acc_test*100:.2f}%")

                                tf.keras.backend.clear_session()
                                del model, predictions, predicted_labels, confidences, pseudo_mask, X_pseudo, y_pseudo, remaining_mask
                                gc.collect()

                            # Huấn luyện lần cuối trên toàn bộ tập dữ liệu đã gắn nhãn
                            iteration_container.markdown("**Huấn luyện lần cuối trên toàn bộ tập dữ liệu đã gắn nhãn**")
                            model = models.Sequential()
                            model.add(layers.Input(shape=(784,)))
                            for neurons in params["hidden_layer_sizes"]:
                                model.add(layers.Dense(neurons, activation=params["activation"]))
                            model.add(layers.Dense(10, activation='softmax'))

                            optimizer = tf.keras.optimizers.Adam(learning_rate=params["learning_rate"]) if params["solver"] == "adam" else tf.keras.optimizers.SGD(learning_rate=params["learning_rate"])

                            model.compile(optimizer=optimizer,
                                          loss='sparse_categorical_crossentropy',
                                          metrics=['accuracy'])

                            with progress_bar_container:
                                progress_bar_final = st.progress(0)
                            status_container_final = st.empty()

                            class ProgressCallback(callbacks.Callback):
                                def on_epoch_end(self, epoch, logs=None):
                                    progress = (epoch + 1) / params["epochs"] * 100
                                    progress_bar_final.progress(int(progress))
                                    status_container_final.markdown(
                                        f"**Huấn luyện cuối - Epoch {epoch+1}/{params['epochs']}**: "
                                        f"Loss: {logs['loss']:.4f}, Accuracy: {logs['accuracy']:.4f}"
                                    )

                            history = model.fit(X_train, y_train, epochs=params["epochs"], batch_size=params["batch_size"],
                                                callbacks=[ProgressCallback()], verbose=0)

                            # Đánh giá trên tập test
                            y_test_pred = np.argmax(model.predict(X_test, verbose=0), axis=1)
                            acc_test = accuracy_score(y_test, y_test_pred)
                            cm_test = confusion_matrix(y_test, y_test_pred)

                            run_name = f"PseudoLabeling_NN_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                            with mlflow.start_run(experiment_id=EXPERIMENT_ID, run_name=run_name) as run:
                                mlflow.log_params({
                                    'hidden_layer_sizes': params["hidden_layer_sizes"],
                                    'learning_rate': params["learning_rate"],
                                    'epochs': params["epochs"],
                                    'batch_size': params["batch_size"],
                                    'activation': params["activation"],
                                    'solver': params["solver"],
                                    'threshold': threshold,
                                    'max_iterations': max_iterations
                                })
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

                            # Hiển thị thông tin vòng lặp cuối cùng
                            final_iteration = iteration
                            iteration_container.markdown(f"**Vòng lặp {final_iteration}/{max_iterations}**")
                            pseudo_container.markdown(f"**Số mẫu được gán nhãn giả trong vòng {final_iteration}**: {pseudo_labeled_history[-1] if pseudo_labeled_history else 0}")
                            accuracy_container.markdown(f"**Độ chính xác trên tập Test sau vòng {final_iteration}**: {acc_test*100:.2f}%")

                            st.success(f"Đã hoàn thành Pseudo-Labeling! Thời gian: {time.time() - start_time:.2f} giây, Tổng số vòng lặp: {final_iteration}")
                            tf.keras.backend.clear_session()
                            del X_train, y_train, X_unlabeled, X_test, y_test, split_data, history
                            gc.collect()
                            st.rerun()

                    except Exception as e:
                        st.error(f"Lỗi trong quá trình huấn luyện với Pseudo-Labeling: {e}")

            if 'training_results' in st.session_state:
                results = st.session_state['training_results']
                st.subheader("📊 Kết quả Huấn luyện với Pseudo-Labeling")
                col_result1, col_result2, col_result3 = st.columns(3)
                with col_result1:
                    st.metric("Thời gian huấn luyện", f"{results['training_time']:.2f} giây")
                with col_result2:
                    st.metric("Độ chính xác Test", f"{results['accuracy_test']*100:.2f}%")
                with col_result3:
                    st.metric("Tổng số vòng lặp", f"{results['total_iterations']}")

                st.subheader("📈 Ma trận Nhầm lẫn trên tập Test")
                fig, ax = plt.subplots(figsize=(6, 5))
                sns.heatmap(results['cm_test'], annot=True, fmt="d", cmap="Blues", ax=ax)
                ax.set_title("Test")
                st.pyplot(fig)
                plt.close(fig)

                st.subheader("📉 Biểu đồ Kết quả Huấn luyện")
                col_chart1, col_chart2 = st.columns(2)
                with col_chart1:
                    if results['loss_history']:
                        fig, ax = plt.subplots(figsize=(6, 3))
                        ax.plot(range(1, len(results['loss_history']) + 1), results['loss_history'], 
                                label='Loss', linestyle='-', color='blue', linewidth=2)
                        ax.set_xlabel("Epochs")
                        ax.set_ylabel("Loss")
                        ax.set_title("Training Loss (Final)")
                        ax.legend()
                        ax.grid(True)
                        st.pyplot(fig)
                        plt.close(fig)
                    st.markdown("**Giải thích:** Biểu đồ này thể hiện mức độ mất mát (loss) của mô hình trong 10 epoch cuối cùng của lần huấn luyện cuối, cho thấy khả năng hội tụ.")

                    if results['pseudo_labeled_history']:
                        fig, ax = plt.subplots(figsize=(6, 3))
                        ax.plot(range(1, len(results['pseudo_labeled_history']) + 1), results['pseudo_labeled_history'], 
                                label='Số mẫu', linestyle='-', color='purple', linewidth=2)
                        ax.set_xlabel("Vòng lặp")
                        ax.set_ylabel("Số mẫu")
                        ax.set_title("Số mẫu Pseudo-Label")
                        ax.legend()
                        ax.grid(True)
                        st.pyplot(fig)
                        plt.close(fig)
                    st.markdown("**Giải thích:** Hiển thị số lượng mẫu được gán nhãn giả qua từng vòng lặp Pseudo-Labeling, phản ánh mức độ mở rộng tập dữ liệu.")

                with col_chart2:
                    if results['accuracy_history']:
                        fig, ax = plt.subplots(figsize=(6, 3))
                        ax.plot(range(1, len(results['accuracy_history']) + 1), results['accuracy_history'], 
                                label='Accuracy', linestyle='-', color='green', linewidth=2)
                        ax.set_xlabel("Epochs")
                        ax.set_ylabel("Accuracy")
                        ax.set_title("Training Accuracy (Final)")
                        ax.legend()
                        ax.grid(True)
                        st.pyplot(fig)
                        plt.close(fig)
                    st.markdown("**Giải thích:** Biểu đồ này cho thấy độ chính xác huấn luyện trong 10 epoch cuối cùng của lần huấn luyện cuối, đánh giá hiệu suất mô hình.")

                    if results['accuracy_test_history']:
                        fig, ax = plt.subplots(figsize=(6, 3))
                        ax.plot(range(1, len(results['accuracy_test_history']) + 1), results['accuracy_test_history'], 
                                label='Test Accuracy', linestyle='-', color='red', linewidth=2)
                        ax.set_xlabel("Vòng lặp")
                        ax.set_ylabel("Accuracy")
                        ax.set_title("Test Accuracy qua vòng lặp")
                        ax.legend()
                        ax.grid(True)
                        st.pyplot(fig)
                        plt.close(fig)
                    st.markdown("**Giải thích:** Thể hiện độ chính xác trên tập kiểm tra qua các vòng lặp, đánh giá sự cải thiện nhờ Pseudo-Labeling.")

                with st.expander("Xem chi tiết", expanded=False):
                    st.markdown("**Thông tin lần chạy:**")
                    st.write(f"- Tên: {results['run_name']}")
                    st.write(f"- ID: {results['run_id']}")
                    st.write(f"- Thời gian huấn luyện: {results['training_time']:.2f} giây")
                    st.write(f"- Tổng số vòng lặp: {results['total_iterations']}")
                    st.write(f"- Độ chính xác Test: {results['accuracy_test']*100:.2f}%")
                    st.markdown("**Tham số đã chọn:**")
                    st.json({
                        "Số lớp ẩn": len(results['params']['hidden_layer_sizes']),
                        "Số nơ-ron mỗi lớp": results['params']['hidden_layer_sizes'],
                        "Tốc độ học": results['params']['learning_rate'],
                        "Số lần lặp mỗi vòng (Epochs)": results['params']['epochs'],
                        "Kích thước batch": results['params']['batch_size'],
                        "Hàm kích hoạt": results['params']['activation'],
                        "Trình tối ưu": results['params']['solver'],
                        "Ngưỡng tin cậy": threshold,
                        "Số vòng lặp tối đa": max_iterations
                    })
     # Tab 6: Demo dự đoán
    with tab_demo:
        st.markdown('<div class="section-title">Demo Dự đoán Chữ số</div>', unsafe_allow_html=True)
        st.header("Dự đoán số viết tay")
        st.write("Chọn cách nhập liệu: tải lên hình ảnh, sử dụng dữ liệu Test hoặc vẽ trực tiếp.")

        if 'split_data' not in st.session_state:
            st.warning("⚠️ Vui lòng chia dữ liệu trước trong tab 'Chia dữ liệu'!")
        else:
            # Khởi tạo client MLflow chỉ một lần
            if 'mlflow_client' not in st.session_state:
                st.session_state['mlflow_client'] = MlflowClient()

            # Lấy danh sách runs một lần và lưu vào session_state
            if 'model_options' not in st.session_state or st.button("Làm mới danh sách mô hình"):
                with st.spinner("Đang tải danh sách mô hình..."):
                    runs = st.session_state['mlflow_client'].search_runs(
                        experiment_ids=[EXPERIMENT_ID], 
                        order_by=["attributes.start_time DESC"]
                    )
                    st.session_state['model_options'] = {
                        run.info.run_id: run.data.tags.get('mlflow.runName', f"Run_{run.info.run_id}") 
                        for run in runs if 'mlflow.runName' in run.data.tags
                    }

            model_options = st.session_state['model_options']

            if model_options:
                # Tự động chọn model mới nhất sau khi huấn luyện
                if 'latest_run_id' in st.session_state:
                    default_run_id = st.session_state['latest_run_id']
                else:
                    default_run_id = list(model_options.keys())[0]
                
                default_model_name = model_options.get(default_run_id, list(model_options.values())[0])
                
                # Chọn mô hình
                selected_model_name = st.selectbox(
                    "Chọn mô hình:", 
                    list(model_options.values()), 
                    index=list(model_options.values()).index(default_model_name),
                    key="model_select"
                )
                selected_run_id = [k for k, v in model_options.items() if v == selected_model_name][0]

                # Tải mô hình một lần và lưu vào session_state
                if 'selected_model' not in st.session_state or st.session_state['selected_run_id'] != selected_run_id:
                    with st.spinner("Đang tải mô hình..."):
                        model_uri = f"runs:/{selected_run_id}/model"
                        try:
                            model = mlflow.keras.load_model(model_uri)
                            st.session_state['selected_model'] = model
                            st.session_state['selected_run_id'] = selected_run_id
                        except Exception as e:
                            st.error(f"Không thể tải mô hình từ MLflow: {e}")
                            model = None
                else:
                    model = st.session_state['selected_model']

                if model is not None:
                    st.write(f"**Mô hình hiện tại**: {selected_model_name}")

                    input_method = st.selectbox(
                        "Chọn phương thức nhập liệu", 
                        ["Tải ảnh lên", "Dữ liệu Test", "Vẽ trực tiếp"],
                        key="input_method"
                    )
                    is_normalized = 'data_processed' in st.session_state

                    def preprocess_input(data, is_normalized):
                        if not is_normalized:
                            data = data / 255.0
                        return data

                    if input_method == "Tải ảnh lên":
                        st.markdown('<p class="mode-title">Dự đoán từ Ảnh Tải lên</p>', unsafe_allow_html=True)
                        uploaded_file = st.file_uploader("Tải lên hình ảnh", type=["png", "jpg", "jpeg"])
                        if uploaded_file is not None:
                            image = Image.open(uploaded_file).convert('L')
                            image = image.resize((28, 28))
                            st.image(image, caption="Hình ảnh đầu vào", width=100)

                            if st.button("Dự đoán", key="predict_upload_button"):
                                with st.spinner("Đang xử lý ảnh..."):
                                    image_array = np.array(image, dtype=np.float32)
                                    image_array = image_array.reshape(1, 784)
                                    image_processed = preprocess_input(image_array, is_normalized)
                                    prediction = model.predict(image_processed, verbose=0)[0]
                                    predicted_class = np.argmax(prediction)
                                    confidence = prediction[predicted_class] * 100
                                    st.markdown(f"""
                                        <div>
                                            <strong>Dự đoán:</strong> {predicted_class}<br>
                                            <strong>Độ tin cậy:</strong> {confidence:.2f}%
                                        </div>
                                    """, unsafe_allow_html=True)
                                    fig, ax = plt.subplots(figsize=(6, 4))
                                    ax.bar(range(10), prediction * 100, color='blue')
                                    ax.set_xlabel("Chữ số")
                                    ax.set_ylabel("Xác suất (%)")
                                    ax.set_title("Phân bố xác suất")
                                    st.pyplot(fig)
                                    plt.close(fig)
                                    st.success("Dự đoán hoàn tất!")
                                    del image, image_array, image_processed, prediction
                                    gc.collect()

                    elif input_method == "Dữ liệu Test":
                        st.markdown('<p class="mode-title">Dự đoán từ Dữ liệu Test</p>', unsafe_allow_html=True)
                        X_test = st.session_state['split_data']["X_test"]
                        y_test = st.session_state['split_data']["y_test"]
                        if len(X_test) == 0:
                            st.warning("Tập Test rỗng. Vui lòng chia lại dữ liệu với tỷ lệ Test > 0%.")
                        else:
                            col_select, col_display = st.columns([3, 2])
                            with col_select:
                                idx = st.slider("Chọn mẫu Test", 0, min(len(X_test) - 1, 100), 0)
                            with col_display:
                                st.write("**Ảnh mẫu Test:**")
                                fig, ax = plt.subplots(figsize=(2, 2))
                                ax.imshow(X_test[idx].reshape(28, 28), cmap='gray')
                                ax.axis('off')
                                st.pyplot(fig)
                                plt.close(fig)
                                st.write(f"**Nhãn thực tế:** {y_test[idx]}")

                            if st.button("🔍 Dự đoán", key="predict_test"):
                                with st.spinner("Đang dự đoán..."):
                                    sample = X_test[idx].reshape(1, -1)
                                    sample_processed = preprocess_input(sample, is_normalized)
                                    prediction = model.predict(sample_processed, verbose=0)[0]
                                    predicted_class = np.argmax(prediction)
                                    confidence = prediction[predicted_class] * 100
                                    st.markdown(f"""
                                        <div class="prediction-box">
                                            <strong>Dự đoán:</strong> {predicted_class}<br>
                                            <strong>Độ tin cậy:</strong> {confidence:.2f}%<br>
                                            <strong>Nhãn thực tế:</strong> {y_test[idx]}
                                        </div>
                                    """, unsafe_allow_html=True)
                                    fig, ax = plt.subplots(figsize=(6, 4))
                                    ax.bar(range(10), prediction * 100, color='blue')
                                    ax.set_xlabel("Chữ số")
                                    ax.set_ylabel("Xác suất (%)")
                                    ax.set_title("Phân bố xác suất")
                                    st.pyplot(fig)
                                    plt.close(fig)
                                    st.success("Dự đoán hoàn tất!")
                                    del sample, sample_processed, prediction
                                    gc.collect()

                    elif input_method == "Vẽ trực tiếp":
                        st.markdown('<p class="mode-title">Vẽ trực tiếp</p>', unsafe_allow_html=True)
                        st.write("Vẽ chữ số từ 0-9 (nét trắng trên nền đen):")

                        # Sử dụng key cố định cho canvas
                        if 'canvas_result' not in st.session_state:
                            st.session_state['canvas_result'] = None

                        canvas_result = st_canvas(
                            fill_color="rgba(255, 165, 0, 0.3)",
                            stroke_width=20,
                            stroke_color="#FFFFFF",
                            background_color="#000000",
                            height=280,
                            width=280,
                            drawing_mode="freedraw",
                            key="canvas_fixed_key",  # Key cố định
                            update_streamlit=False  # Ngăn rerender tự động
                        )

                        # Lưu kết quả canvas vào session_state
                        if canvas_result.image_data is not None:
                            st.session_state['canvas_result'] = canvas_result

                        col_pred, col_clear = st.columns([2, 1])
                        with col_pred:
                            if st.button("Dự đoán", key="predict_button"):
                                if st.session_state['canvas_result'] is not None:
                                    with st.spinner("Đang xử lý hình vẽ..."):
                                        image = Image.fromarray(
                                            st.session_state['canvas_result'].image_data.astype('uint8'), 'RGBA'
                                        ).convert('L')
                                        image_resized = image.resize((28, 28))
                                        image_array = np.array(image_resized, dtype=np.float32).reshape(1, 784)
                                        image_processed = preprocess_input(image_array, is_normalized)
                                        prediction = model.predict(image_processed, verbose=0)[0]
                                        predicted_class = np.argmax(prediction)
                                        confidence = prediction[predicted_class] * 100
                                        st.markdown(f"""
                                            <div>
                                                <strong>Dự đoán:</strong> {predicted_class}<br>
                                                <strong>Độ tin cậy:</strong> {confidence:.2f}%
                                            </div>
                                        """, unsafe_allow_html=True)
                                        fig, ax = plt.subplots(figsize=(6, 4))
                                        ax.bar(range(10), prediction * 100, color='blue')
                                        ax.set_xlabel("Chữ số")
                                        ax.set_ylabel("Xác suất (%)")
                                        ax.set_title("Phân bố xác suất")
                                        st.pyplot(fig)
                                        plt.close(fig)
                                        st.success("Dự đoán hoàn tất!")
                                        del image, image_resized, image_array, image_processed, prediction
                                        gc.collect()
                                else:
                                    st.warning("Vui lòng vẽ trước khi dự đoán!")

            else:
                st.warning("Chưa có mô hình nào được lưu trong MLflow.")

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
                    col_history1, col_history2 = st.columns(2)
                    with col_history1:
                        if 'training_results' in st.session_state and selected_run_id == st.session_state['training_results']['run_id']:
                            results = st.session_state['training_results']
                            if results['loss_history']:
                                fig, ax = plt.subplots(figsize=(6, 3))
                                ax.plot(range(1, len(results['loss_history']) + 1), results['loss_history'], 
                                        label='Loss', linestyle='-', color='blue', linewidth=2)
                                ax.set_xlabel("Epochs")
                                ax.set_ylabel("Loss")
                                ax.set_title("Training Loss (Final)")
                                ax.legend()
                                ax.grid(True)
                                st.pyplot(fig)
                                plt.close(fig)
                            st.markdown("**Giải thích:** Biểu đồ Loss của 10 epoch cuối trong lần huấn luyện cuối.")

                            if results['pseudo_labeled_history']:
                                fig, ax = plt.subplots(figsize=(6, 3))
                                ax.plot(range(1, len(results['pseudo_labeled_history']) + 1), results['pseudo_labeled_history'], 
                                        label='Số mẫu', linestyle='-', color='purple', linewidth=2)
                                ax.set_xlabel("Vòng lặp")
                                ax.set_ylabel("Số mẫu")
                                ax.set_title("Số mẫu Pseudo-Label")
                                ax.legend()
                                ax.grid(True)
                                st.pyplot(fig)
                                plt.close(fig)
                            st.markdown("**Giải thích:** Số mẫu được gán nhãn giả qua các vòng lặp.")

                    with col_history2:
                        if 'training_results' in st.session_state and selected_run_id == st.session_state['training_results']['run_id']:
                            results = st.session_state['training_results']
                            if results['accuracy_history']:
                                fig, ax = plt.subplots(figsize=(6, 3))
                                ax.plot(range(1, len(results['accuracy_history']) + 1), results['accuracy_history'], 
                                        label='Accuracy', linestyle='-', color='green', linewidth=2)
                                ax.set_xlabel("Epochs")
                                ax.set_ylabel("Accuracy")
                                ax.set_title("Training Accuracy (Final)")
                                ax.legend()
                                ax.grid(True)
                                st.pyplot(fig)
                                plt.close(fig)
                            st.markdown("**Giải thích:** Độ chính xác huấn luyện của 10 epoch cuối trong lần huấn luyện cuối.")

                            if results['accuracy_test_history']:
                                fig, ax = plt.subplots(figsize=(6, 3))
                                ax.plot(range(1, len(results['accuracy_test_history']) + 1), results['accuracy_test_history'], 
                                        label='Test Accuracy', linestyle='-', color='red', linewidth=2)
                                ax.set_xlabel("Vòng lặp")
                                ax.set_ylabel("Accuracy")
                                ax.set_title("Test Accuracy qua vòng lặp")
                                ax.legend()
                                ax.grid(True)
                                st.pyplot(fig)
                                plt.close(fig)
                            st.markdown("**Giải thích:** Độ chính xác trên tập test qua các vòng lặp.")

                    mlflow_ui_link = f"{mlflow_tracking_uri}/#/experiments/{EXPERIMENT_ID}"
                    st.markdown("---")
                    st.markdown(f"📊 **Xem chi tiết trên MLflow UI**: [Nhấn vào đây]({mlflow_ui_link})", unsafe_allow_html=True)

        except Exception as e:
            st.error(f"Lỗi khi tải thông tin huấn luyện: {e}")

if __name__ == "__main__":
    run_mnist_pseudo_labeling_app()