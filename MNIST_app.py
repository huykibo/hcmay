import os
import mlflow
import streamlit as st
import openml
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from mlflow.tracking import MlflowClient
from streamlit_drawable_canvas import st_canvas
from datetime import datetime
import time
import requests
import gc

# Hàm tải dữ liệu MNIST
def fetch_mnist_data():
    try:
        mnist = openml.datasets.get_dataset(554)
        X, y, _, _ = mnist.get_data(target=mnist.default_target_attribute)
        return X, y
    except Exception as e:
        st.error(f"Không tải được MNIST từ OpenML: {e}")
        return None, None

# Hàm kiểm tra và chuẩn hóa pixel
def validate_and_fix_pixels(X, name="dữ liệu"):
    X = np.array(X, dtype=np.float64)
    invalid_mask = (X < 0) | (X > 255)
    if np.any(invalid_mask):
        st.warning(f"Phát hiện giá trị pixel không hợp lệ trong {name}. Đang chuẩn hóa...")
        X = np.clip(X, 0, 255)
        return X, True
    return X, False

def run_mnist_classification_app():
    # Thiết lập MLflow
    mlflow_tracking_uri = "https://dagshub.com/huykibo/streamlit_mlflow.mlflow"
    try:
        os.environ["MLFLOW_TRACKING_USERNAME"] = st.secrets["mlflow"]["MLFLOW_TRACKING_USERNAME"]
        os.environ["MLFLOW_TRACKING_PASSWORD"] = st.secrets["mlflow"]["MLFLOW_TRACKING_PASSWORD"]
        mlflow.set_tracking_uri(mlflow_tracking_uri)
        mlflow.set_experiment("MNIST")
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

    EXPERIMENT_ID = mlflow.get_experiment_by_name("MNIST").experiment_id

    st.title("Phân loại Chữ số MNIST với Decision Tree và SVM")

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
        st.header("Giới thiệu về Ứng dụng và Các Mô hình Phân loại MNIST")
        st.markdown("""
        Chào bạn! Đây là ứng dụng phân loại chữ số viết tay từ tập dữ liệu **MNIST** bằng **Decision Tree** và **SVM**. Hãy khám phá các tính năng và cách hoạt động của nó nhé!
        """, unsafe_allow_html=True)

        st.subheader("Chọn thông tin để xem")
        info_option = st.selectbox(
            "",
            [
                "Ứng dụng này là gì và mục tiêu của nó?",
                "Tập dữ liệu MNIST: Đặc điểm và ý nghĩa",
                "Decision Tree – Mô hình cây quyết định",
                "SVM – Máy vector hỗ trợ",
                "So sánh Decision Tree và SVM",
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
                Đây là một ứng dụng phân loại chữ số viết tay dựa trên tập dữ liệu **MNIST**, sử dụng **Decision Tree** và **SVM**.  
                - **MNIST**: Tập dữ liệu gồm $70,000$ ảnh chữ số từ $0$ đến $9$, mỗi ảnh kích thước $28 \\times 28$ pixel (tổng cộng $784$ đặc trưng).  
                - **Mục tiêu**:  
                  - Xây dựng và huấn luyện hai mô hình học máy để nhận diện chính xác các chữ số.  
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

        elif info_option == "Decision Tree – Mô hình cây quyết định":
            with st.spinner("Đang tải thông tin..."):
                progress_bar = st.progress(0)
                status_text = st.empty()
                for i in range(0, 101, 10):
                    progress_bar.progress(i)
                    status_text.text(f"Đang tải thông tin... {i}%")
                    time.sleep(0.05)
                st.subheader("📘 3. Decision Tree – Mô hình cây quyết định")
                st.markdown("""
                **Decision Tree (Cây quyết định)** xây dựng một cấu trúc phân cấp giống như cây, trong đó dữ liệu được chia nhỏ dần dựa trên các đặc trưng (pixel trong MNIST) để đưa ra dự đoán cuối cùng. Trong bài toán này, tham số quan trọng như **Max Depth** được sử dụng để kiểm soát độ phức tạp của cây, tránh hiện tượng quá khớp (overfitting).
                """, unsafe_allow_html=True)

                st.subheader("🔧 Quy trình hoạt động")
                st.markdown("""
                Decision Tree hoạt động qua các bước sau, được tối ưu hóa dựa trên các tham số bạn có thể điều chỉnh trong tab **Huấn luyện/Đánh giá**:
                """, unsafe_allow_html=True)

                st.subheader("1. Nút gốc (Root Node)")
                st.markdown("""
                - Thuật toán bắt đầu với toàn bộ dữ liệu MNIST ($70,000$ mẫu, nhãn $0$-$9$) và chọn một pixel quan trọng, ví dụ: "Pixel 5 > 100?" (giả sử Pixel 5 là giá trị tại vị trí [0, 5] trong ảnh $28 \\times 28$).  
                - Dữ liệu được chia thành hai nhánh: nhánh "Yes" nếu Pixel 5 > 100, nhánh "No" nếu Pixel 5 ≤ 100.  
                """, unsafe_allow_html=True)
                try:
                    tree_step_1 = Image.open("illustrations/tree_step_1.png")
                    st.image(tree_step_1, caption="Bước 1: Nút gốc với toàn bộ dữ liệu MNIST", width=600)
                except FileNotFoundError:
                    st.error("Không tìm thấy file `illustrations/tree_step_1.png`.")
                except Exception as e:
                    st.error(f"Lỗi khi tải ảnh: {e}")

                st.subheader("2. Chia nhánh đầu tiên (Splitting)")
                st.markdown("""
                - Từ nút gốc, nhánh "Yes" (Pixel 5 > 100) chứa các mẫu có giá trị pixel sáng hơn, ví dụ: chữ số "$1$" hoặc "$7$" (thường có nét dày ở đầu).  
                - Nhánh "No" (Pixel 5 ≤ 100) chứa các mẫu tối hơn, ví dụ: chữ số "$0$" hoặc "$2$".  
                """, unsafe_allow_html=True)
                try:
                    tree_step_2 = Image.open("illustrations/tree_step_2.png")
                    st.image(tree_step_2, caption="Bước 2: Chia nhánh đầu tiên", width=600)
                except FileNotFoundError:
                    st.error("Không tìm thấy file `illustrations/tree_step_2.png`.")
                except Exception as e:
                    st.error(f"Lỗi khi tải ảnh: {e}")

                st.subheader("3. Chia nhánh tiếp theo")
                st.markdown("""
                - Từ nhánh "Yes" (Pixel 5 > 100), tiếp tục chia dựa trên "Pixel 10 > 50?" (giả sử Pixel 10 là vị trí [0, 10]).  
                - Nhánh "Yes" (Pixel 10 > 50) chứa các mẫu có nét dày hơn, ví dụ: "$1$" (nét đứng rõ rệt).  
                - Nhánh "No" (Pixel 10 ≤ 50) chứa các mẫu mỏng hơn, ví dụ: "$7$".  
                """, unsafe_allow_html=True)
                try:
                    tree_step_3 = Image.open("illustrations/tree_step_3.png")
                    st.image(tree_step_3, caption="Bước 3: Chia nhánh tiếp theo dựa trên Pixel 10 > 50", width=600)
                except FileNotFoundError:
                    st.error("Không tìm thấy file `illustrations/tree_step_3.png`.")
                except Exception as e:
                    st.error(f"Lỗi khi tải ảnh: {e}")

                st.subheader("4. Nút lá và tiêu chí dừng")
                st.markdown("""
                - Quá trình dừng khi nhóm dữ liệu thuần nhất (tất cả mẫu trong nhánh thuộc cùng một nhãn) hoặc đạt **Max Depth** (độ sâu tối đa của cây).  
                - Ví dụ: Nhánh "Yes" của "Pixel 10 > 50" → Nhãn "$1$" (thuần nhất).  
                - Nhánh "No" của "Pixel 10 ≤ 50" → Nhãn "$9$" (thuần nhất).  
                - Nhánh "No" của "Pixel 5 > 100" → Nhãn "$0$" (đạt Max Depth).  
                """, unsafe_allow_html=True)
                try:
                    tree_step_4 = Image.open("illustrations/tree_step_4.png")
                    st.image(tree_step_4, caption="Bước 4: Nút lá với nhãn dự đoán (1, 9, 0)", width=600)
                except FileNotFoundError:
                    st.error("Không tìm thấy file `illustrations/tree_step_4.png`.")
                except Exception as e:
                    st.error(f"Lỗi khi tải ảnh: {e}")

                st.subheader("5. Dự đoán")
                st.markdown("""
                - Với một mẫu mới có Pixel 5 = 150 (> 100) và Pixel 10 = 60 (> 50), thuật toán đi qua nhánh "Yes" rồi "Yes", dẫn đến nhãn "$1$".  
                - Kết quả dự đoán: "$1$" với độ tin cậy cao dựa trên các điều kiện pixel.  
                """, unsafe_allow_html=True)
                try:
                    tree_step_5 = Image.open("illustrations/tree_step_5.png")
                    st.image(tree_step_5, caption="Bước 5: Dự đoán nhãn '1' cho mẫu mới", width=600)
                except FileNotFoundError:
                    st.error("Không tìm thấy file `illustrations/tree_step_5.png`.")
                except Exception as e:
                    st.error(f"Lỗi khi tải ảnh: {e}")

                st.subheader("⚙️ Các tham số chính và ứng dụng")
                st.markdown("""
                - **Tiêu chí lựa chọn đặc trưng**:  
                  - **Entropy**: Đo mức độ "hỗn loạn" của dữ liệu dựa trên phân bố nhãn:  
                    $$ Entropy(S) = -\\sum_{i=0}^{9} p_i \\log_2(p_i) $$  
                    - $p_i$: Tỷ lệ mẫu thuộc nhãn $i$.  
                  - **Gini Index**: Đo độ "tinh khiết" của nhóm:  
                    $$ Gini(S) = 1 - \\sum_{i=0}^{9} p_i^2 $$  
                - **Max Depth**:  
                  - Là tham số giới hạn số mức chia tối đa của cây (độ sâu).  
                  - Trong bài toán MNIST, nếu không giới hạn Max Depth, cây có thể phát triển quá sâu (ví dụ: $784$ mức tương ứng $784$ pixel), dẫn đến overfitting.  
                  - Giá trị thường dùng:  
                    - Dữ liệu nhỏ (<$1000$ mẫu): $5$-$10$.  
                    - Dữ liệu trung bình ($1000$-$5000$ mẫu): $10$-$20$.  
                    - Dữ liệu lớn (>$5000$ mẫu): $20$-$50$.  
                  - Ví dụ: Với Max Depth = $10$, cây dừng sau $10$ lần chia, ngay cả khi dữ liệu chưa hoàn toàn thuần nhất.

                **Áp dụng với MNIST**:  
                - Decision Tree chia dữ liệu dựa trên giá trị pixel (ví dụ: Pixel 5, Pixel 10) để phân biệt nhãn ($0$-$9$).  
                - Tham số Max Depth giúp cân bằng giữa độ chính xác và khả năng khái quát hóa, đặc biệt với dữ liệu phức tạp như MNIST ($784$ đặc trưng).
                """, unsafe_allow_html=True)

                st.subheader("🟪 Ưu điểm và nhược điểm")
                st.markdown("""
                - **✅ Ưu điểm**:  
                  - Dễ hiểu, trực quan như một biểu đồ cây hỏi đáp.  
                  - Nhanh với dữ liệu nhỏ, không yêu cầu chuẩn hóa dữ liệu.  
                - **❌ Nhược điểm**:  
                  - Dễ bị **overfitting** nếu Max Depth quá lớn, đặc biệt khi dữ liệu phức tạp như MNIST.  
                  - Khó xử lý các mẫu có đặc trưng tương tự (ví dụ: "$3$" và "$8$").  
                """, unsafe_allow_html=True)
                status_text.text("Đã tải xong! 100%")
                time.sleep(0.5)
                status_text.empty()
                progress_bar.empty()

        elif info_option == "SVM – Máy vector hỗ trợ":
            with st.spinner("Đang tải thông tin..."):
                progress_bar = st.progress(0)
                status_text = st.empty()
                for i in range(0, 101, 10):
                    progress_bar.progress(i)
                    status_text.text(f"Đang tải thông tin... {i}%")
                    time.sleep(0.05)
                st.subheader("📘 4. SVM – Máy vector hỗ trợ")
                st.markdown("""
                **SVM (Support Vector Machine)** tìm một **siêu phẳng** trong không gian đặc trưng ($784$ chiều với MNIST) để phân tách các lớp nhãn sao cho khoảng cách từ siêu phẳng đến các mẫu gần nhất (support vectors) là lớn nhất. Nếu dữ liệu không phân tách tuyến tính, nó sử dụng **kernel** để chuyển dữ liệu lên không gian cao hơn.
                """, unsafe_allow_html=True)

                st.subheader("🔧 Quy trình hoạt động")
                st.markdown("""
                SVM hoạt động qua các bước sau, với các tham số bạn có thể điều chỉnh trong tab **Huấn luyện/Đánh giá**:
                """, unsafe_allow_html=True)

                st.subheader("1. Siêu phẳng")
                st.markdown("""
                - Siêu phẳng là một ranh giới phân tách trong không gian cao chiều, được định nghĩa bởi:  
                  $$ f(x) = w \\cdot x + b $$  
                  - $w$: Vector trọng số, xác định hướng của siêu phẳng.  
                  - $x$: Vector đặc trưng ($784$ pixel).  
                  - $b$: Độ lệch, điều chỉnh vị trí siêu phẳng.  
                - Mẫu nằm ở phía nào của siêu phẳng được xác định bởi dấu của $f(x)$.  
                """, unsafe_allow_html=True)
                try:
                    svm_step_1 = Image.open("illustrations/svm_step_1.png")
                    st.image(svm_step_1, caption="Bước 1: Siêu phẳng phân tách dữ liệu", width=600)
                except FileNotFoundError:
                    st.error("Không tìm thấy file `illustrations/svm_step_1.png`.")
                except Exception as e:
                    st.error(f"Lỗi khi tải ảnh: {e}")

                st.subheader("2. Tối ưu hóa lề")
                st.markdown("""
                - Khoảng cách lề (margin) từ siêu phẳng đến các điểm gần nhất được tính là:  
                  $$ Margin = \\frac{2}{\\|w\\|} $$  
                  - $\\|w\\|$: Độ dài vector $w$.  
                - Mục tiêu tối ưu hóa:  
                  $$ \\min_{w,b} \\frac{1}{2} \\|w\\|^2 $$  
                  Với ràng buộc:  
                  $$ y_i (w \\cdot x_i + b) \\geq 1 $$  
                  - $y_i$: Nhãn thực tế (+1 hoặc -1 cho phân loại nhị phân).  
                  - $x_i$: Vector đặc trưng của mẫu.  
                """, unsafe_allow_html=True)
                try:
                    svm_step_2 = Image.open("illustrations/svm_step_2.png")
                    st.image(svm_step_2, caption="Bước 2: Siêu phẳng tối ưu với lề lớn nhất và support vectors", width=600)
                except FileNotFoundError:
                    st.error("Không tìm thấy file `illustrations/svm_step_2.png`.")
                except Exception as e:
                    st.error(f"Lỗi khi tải ảnh: {e}")

                st.subheader("3. Soft Margin")
                st.markdown("""
                - Khi dữ liệu không phân tách hoàn hảo (có lẫn lộn giữa các lớp), SVM cho phép sai số:  
                  $$ \\min_{w,b,\\xi} \\frac{1}{2} \\|w\\|^2 + C \\sum \\xi_i $$  
                  - $\\xi_i$: Biến "lỏng" (slack variable), đo mức độ vi phạm của mẫu.  
                  - $C$: Tham số điều chỉnh, cân bằng giữa việc tối đa hóa lề và giảm thiểu lỗi.  
                - $C$ lớn ưu tiên ít lỗi hơn, $C$ nhỏ ưu tiên lề lớn hơn.  
                """, unsafe_allow_html=True)
                try:
                    svm_step_4 = Image.open("illustrations/svm_step_4.png")
                    st.image(svm_step_4, caption="Bước 3: Soft Margin với dữ liệu lẫn lộn", width=600)
                except FileNotFoundError:
                    st.error("Không tìm thấy file `illustrations/svm_step_4.png`.")
                except Exception as e:
                    st.error(f"Lỗi khi tải ảnh: {e}")

                st.subheader("4. Kernel Trick")
                st.markdown("""
                - Khi dữ liệu không phân tách tuyến tính trong không gian ban đầu, SVM ánh xạ dữ liệu lên không gian cao hơn thông qua hàm kernel:  
                  $$ K(x_i, x_j) = \\phi(x_i) \\cdot \\phi(x_j) $$  
                  - $\\phi$: Hàm ánh xạ (không cần tính trực tiếp).  
                - Các loại kernel:  
                  - **Linear**: $K(x_i, x_j) = x_i \\cdot x_j$.  
                  - **Polynomial**: $K(x_i, x_j) = (x_i \\cdot x_j + c)^d$.  
                  - **RBF**: $K(x_i, x_j) = \\exp(-\\gamma \\|x_i - x_j\\|^2)$, thường dùng cho dữ liệu phi tuyến.  
                - Kernel giúp tìm ranh giới phân tách trong không gian mới mà không cần tính toán tọa độ trực tiếp.  
                """, unsafe_allow_html=True)
                try:
                    svm_step_3 = Image.open("illustrations/svm_step_3.png")
                    st.image(svm_step_3, caption="Bước 4: Kernel nâng dữ liệu lên không gian cao hơn", width=600)
                except FileNotFoundError:
                    st.error("Không tìm thấy file `illustrations/svm_step_3.png`.")
                except Exception as e:
                    st.error(f"Lỗi khi tải ảnh: {e}")

                st.subheader("5. Dự đoán")
                st.markdown("""
                - Với mẫu mới $x$:  
                  $$ f(x) = \\text{sign} \\left( \\sum_{i} \\alpha_i y_i K(x_i, x) + b \\right) $$  
                  - $\\alpha_i$: Trọng số xác định từ quá trình huấn luyện, chỉ khác 0 với support vectors.  
                  - $K(x_i, x)$: Đo độ tương đồng giữa mẫu mới và support vectors.  
                - Với phân loại đa lớp ($0$-$9$), SVM áp dụng chiến lược như "One-vs-Rest" hoặc "One-vs-One".  
                """, unsafe_allow_html=True)
                try:
                    svm_step_5 = Image.open("illustrations/svm_step_5.png")
                    st.image(svm_step_5, caption="Bước 5: Dự đoán điểm mới dựa trên siêu phẳng và support vectors", width=600)
                except FileNotFoundError:
                    st.error("Không tìm thấy file `illustrations/svm_step_5.png`.")
                except Exception as e:
                    st.error(f"Lỗi khi tải ảnh: {e}")

                st.subheader("⚙️ Các tham số chính và ứng dụng")
                st.markdown("""
                - **C**: Tham số điều chỉnh mức độ sai số và kích thước lề.  
                - **Kernel**: Quy định cách dữ liệu được ánh xạ để phân tách.  

                **Áp dụng với MNIST**:  
                - SVM tìm ranh giới phân tách dựa trên toàn bộ đặc trưng pixel, tận dụng kernel để xử lý các mẫu phi tuyến.  
                """, unsafe_allow_html=True)

                st.subheader("🟪 Ưu điểm và nhược điểm")
                st.markdown("""
                - **✅ Ưu điểm**:  
                  - Hiệu quả với dữ liệu phức tạp, chính xác cao khi có kernel phù hợp.  
                  - Tốt cho việc phân biệt các chữ số như "$4$" và "$9$".  
                - **❌ Nhược điểm**:  
                  - Tốn thời gian tính toán với dữ liệu lớn.  
                  - Yêu cầu chuẩn hóa dữ liệu trước để đạt hiệu quả tối ưu.  
                """, unsafe_allow_html=True)
                status_text.text("Đã tải xong! 100%")
                time.sleep(0.5)
                status_text.empty()
                progress_bar.empty()

        elif info_option == "So sánh Decision Tree và SVM":
            with st.spinner("Đang tải thông tin..."):
                progress_bar = st.progress(0)
                status_text = st.empty()
                for i in range(0, 101, 10):
                    progress_bar.progress(i)
                    status_text.text(f"Đang tải thông tin... {i}%")
                    time.sleep(0.05)
                st.subheader("📘 5. So sánh Decision Tree và SVM")
                st.markdown("""
                | **Yếu tố**             | **Decision Tree**                                  | **SVM**                                      |
                |-------------------------|---------------------------------------------------|---------------------------------------------|
                | **Nguyên lý**          | Chia dữ liệu bằng các điều kiện logic             | Tìm siêu phẳng tối ưu phân tách lớp         |
                | **Quyết định**         | Dựa trên Entropy hoặc Gini tại mỗi bước          | Dựa trên khoảng cách lề và support vectors  |
                | **Không gian**         | Làm việc trực tiếp trên dữ liệu gốc               | Có thể ánh xạ lên không gian cao hơn (kernel)|
                | **Độ phức tạp**        | Tăng theo độ sâu cây                              | Tăng theo số lượng support vectors          |

                **Kết luận**:  
                - **Decision Tree**: Nhanh, dễ hiểu, phù hợp với dữ liệu nhỏ hoặc đơn giản, nhưng dễ bị overfitting.  
                - **SVM**: Chính xác hơn với dữ liệu phức tạp, phi tuyến như MNIST, nhưng chậm hơn và cần chuẩn hóa dữ liệu.
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
                st.subheader("📘 6. Công thức đánh giá độ chính xác (Accuracy)")
                st.markdown("""
                - Độ chính xác (**Accuracy**) đo tỷ lệ dự đoán đúng:  
                  $$ \\text{Accuracy} = \\frac{\\text{Số mẫu dự đoán đúng}}{\\text{Tổng số mẫu}} $$  
                - **Ví dụ**: Dự đoán đúng $92/100$ ảnh → $\\text{Accuracy} = 0.92$ (tức $92\%$).  

                **Ý nghĩa**:  
                - **Decision Tree**: Đo khả năng chia nhóm đúng dựa trên các đặc trưng pixel.  
                - **SVM**: Đo hiệu quả của siêu phẳng trong việc phân tách các lớp.
                """, unsafe_allow_html=True)
                status_text.text("Đã tải xong! 100%")
                time.sleep(0.5)
                status_text.empty()
                progress_bar.empty()

    # Tab 2: Tải dữ liệu
    with tab_load:
        st.markdown('<div class="section-title">Tải Dữ liệu</div>', unsafe_allow_html=True)
        st.markdown("""
        **Tập dữ liệu MNIST**: Được tải từ OpenML. Bạn có thể chọn số lượng mẫu phù hợp để huấn luyện.
        """, unsafe_allow_html=True)

        if 'full_data' not in st.session_state:
            if st.button("Tải dữ liệu MNIST từ OpenML", type="primary"):
                with st.spinner("Đang tải dữ liệu MNIST..."):
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    for i in range(0, 101, 20):
                        progress_bar.progress(i)
                        status_text.text(f"Đang tải dữ liệu... {i}%")
                        time.sleep(0.1)
                    X, y = fetch_mnist_data()
                    if X is not None:
                        X = np.array(X, dtype=np.float64)
                        y = np.array(y, dtype=np.int32)
                        st.session_state['full_data'] = (X, y)
                        with mlflow.start_run(experiment_id=EXPERIMENT_ID, run_name="Data_Load"):
                            mlflow.log_param("total_samples", X.shape[0])
                        st.success("Đã tải dữ liệu thành công!")
                        st.write(f"Kích thước dữ liệu: {X.shape[0]} mẫu, {X.shape[1]} đặc trưng")
                        status_text.text("Đã tải xong! 100%")
                        time.sleep(0.5)
                        status_text.empty()
                        progress_bar.empty()
                        st.rerun()
        else:
            X_full, y_full = st.session_state['full_data']
            st.subheader("Chọn số lượng mẫu")
            st.markdown("""
            - **100 mẫu**: Huấn luyện rất nhanh, độ chính xác thấp, phù hợp để thử nghiệm.  
            - **1,000 mẫu**: Huấn luyện nhanh, độ chính xác trung bình, phù hợp để kiểm tra cơ bản.  
            - **10,000 mẫu**: Huấn luyện khá nhanh, độ chính xác khá, cân bằng giữa tốc độ và hiệu suất.  
            - **50,000 mẫu**: Huấn luyện chậm, độ chính xác cao, phù hợp cho huấn luyện chuyên sâu.  
            """, unsafe_allow_html=True)

            col1, col2 = st.columns(2)
            with col1:
                sample_options = {
                    "100 mẫu (Huấn luyện rất nhanh)": 100,
                    "1,000 mẫu (Huấn luyện nhanh)": 1000,
                    "10,000 mẫu (Huấn luyện trung bình)": 10000,
                    "50,000 mẫu (Huấn luyện chậm)": 50000
                }
                selected_option = st.selectbox("Chọn số lượng mẫu:", list(sample_options.keys()), help="Chọn số lượng mẫu có sẵn")
                num_samples = min(sample_options[selected_option], len(X_full))

                if st.button("Xác nhận số lượng (tùy chọn có sẵn)", type="primary"):
                    with st.spinner(f"Đang lấy {num_samples} mẫu..."):
                        indices = np.random.choice(len(X_full), size=num_samples, replace=False)
                        X_sampled = X_full[indices]
                        y_sampled = y_full[indices]
                        st.session_state['data'] = (X_sampled.copy(), y_sampled.copy())
                        with mlflow.start_run(experiment_id=EXPERIMENT_ID, run_name="Data_Sample"):
                            mlflow.log_param("num_samples", num_samples)
                        st.success(f"Đã chọn {num_samples} mẫu!")
                        del X_full, y_full, X_sampled, y_sampled
                        gc.collect()

            with col2:
                custom_num_samples = st.number_input("Nhập số lượng tùy ý (tối đa 70,000):", min_value=1, max_value=70000, value=1000, step=100, help="Nhập số lượng mẫu tùy chỉnh")
                if st.button("Xác nhận số lượng (tùy ý)", type="primary"):
                    if custom_num_samples <= len(X_full):
                        with st.spinner(f"Đang lấy {custom_num_samples} mẫu..."):
                            indices = np.random.choice(len(X_full), size=custom_num_samples, replace=False)
                            X_sampled = X_full[indices]
                            y_sampled = y_full[indices]
                            st.session_state['data'] = (X_sampled.copy(), y_sampled.copy())
                            with mlflow.start_run(experiment_id=EXPERIMENT_ID, run_name="Data_Sample_Custom"):
                                mlflow.log_param("num_samples", custom_num_samples)
                            st.success(f"Đã chọn {custom_num_samples} mẫu!")
                            del X_full, y_full, X_sampled, y_sampled
                            gc.collect()
                    else:
                        st.error("Số lượng mẫu vượt quá dữ liệu hiện có. Vui lòng nhập số nhỏ hơn hoặc bằng 70,000!")

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
                            Công dụng: Đảm bảo thang đo đồng nhất, đặc biệt cần cho SVM.
                        </span>
                    </div>
                """, unsafe_allow_html=True)

            if "data_processed" in st.session_state:
                X_processed, y_processed = st.session_state["data_processed"]
                st.success("Đã xử lý dữ liệu!")
                st.subheader("Dữ liệu đã xử lý")
                fig, axes = plt.subplots(2, 5, figsize=(10, 4))
                for i, ax in enumerate(axes.flat):
                    ax.imshow(X_processed[i].reshape(28, 28), cmap='gray')
                    ax.set_title(f"Label: {y_processed[i]}")
                    ax.axis("off")
                st.pyplot(fig)
                plt.close(fig)

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

            X_train = np.array(X_train, dtype=np.float64)
            y_train = np.array(y_train, dtype=np.int32)
            X_valid = np.array(X_valid, dtype=np.float64)
            y_valid = np.array(y_valid, dtype=np.int32)
            X_test = np.array(X_test, dtype=np.float64)
            y_test = np.array(y_test, dtype=np.int32)

            num_samples = len(X_train)
            st.write(f"**Số mẫu huấn luyện**: {num_samples}")

            model_choice = st.selectbox("Chọn mô hình:", ["Decision Tree", "SVM"], help="Chọn Decision Tree hoặc SVM để huấn luyện.")

            def get_optimal_params(num_samples, model_choice):
                if model_choice == "Decision Tree":
                    if num_samples <= 1000:
                        return {"criterion": "gini", "max_depth": 5}
                    elif num_samples <= 5000:
                        return {"criterion": "gini", "max_depth": 10}
                    elif num_samples <= 50000:
                        return {"criterion": "gini", "max_depth": 20}
                    else:
                        return {"criterion": "gini", "max_depth": 30}
                else:  # SVM
                    if num_samples <= 1000:
                        return {"C": 0.1, "kernel": "rbf"}
                    elif num_samples <= 5000:
                        return {"C": 1.0, "kernel": "rbf"}
                    elif num_samples <= 50000:
                        return {"C": 5.0, "kernel": "rbf"}
                    else:
                        return {"C": 10.0, "kernel": "rbf"}

            if f"optimal_params_{model_choice}" not in st.session_state:
                st.session_state[f"optimal_params_{model_choice}"] = get_optimal_params(num_samples, model_choice)
            params = st.session_state.get(f"training_params_{model_choice}", st.session_state[f"optimal_params_{model_choice}"].copy())

            st.subheader("⚙️ Cấu hình Tham số Mô hình")
            if model_choice == "Decision Tree":
                st.markdown("""
                | Số mẫu       | Criterion | Max Depth |
                |--------------|-----------|-----------|
                | ≤ 1,000      | gini      | 5         |
                | 1,000-5,000  | gini      | 10        |
                | 5,000-50,000 | gini      | 20        |
                | > 50,000     | gini      | 30        |
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                | Số mẫu       | C    | Kernel |
                |--------------|------|--------|
                | ≤ 1,000      | 0.1  | rbf    |
                | 1,000-5,000  | 1.0  | rbf    |
                | 5,000-50,000 | 5.0  | rbf    |
                | > 50,000     | 10.0 | rbf    |
                """, unsafe_allow_html=True)
            st.info(f"Tham số tối ưu cho {num_samples} mẫu: {st.session_state[f'optimal_params_{model_choice}']}")

            col_param1, col_param2 = st.columns(2)
            with col_param1:
                with st.expander("🧠 Cấu trúc Mô hình", expanded=True):
                    st.markdown(f"**Tùy chỉnh tham số cho {model_choice}**", unsafe_allow_html=True)
                    if model_choice == "Decision Tree":
                        params["criterion"] = st.selectbox("Criterion", ["gini", "entropy"], 
                                                          index=["gini", "entropy"].index(params["criterion"]),
                                                          help="Chọn tiêu chí chia nhánh: Gini hoặc Entropy.")
                        params["max_depth"] = st.number_input("Max Depth", min_value=1, max_value=100, value=params["max_depth"],
                                                             help="Độ sâu tối đa của cây (1-100).")
                    else:
                        params["C"] = st.number_input("C", min_value=0.01, max_value=100.0, value=params["C"],
                                                     help="Tham số điều chỉnh mức độ sai số và lề.")
                        params["kernel"] = st.selectbox("Kernel", ["linear", "rbf", "poly", "sigmoid"],
                                                       index=["linear", "rbf", "poly", "sigmoid"].index(params["kernel"]),
                                                       help="Loại kernel cho SVM: Linear, RBF, Polynomial, Sigmoid.")

            with col_param2:
                with st.expander("🔧 Tối ưu hóa", expanded=True):
                    st.markdown("**Cấu hình huấn luyện**", unsafe_allow_html=True)
                    if st.button("🔄 Khôi phục tham số tối ưu", key=f"reset_params_{model_choice}"):
                        st.session_state[f"training_params_{model_choice}"] = st.session_state[f"optimal_params_{model_choice}"].copy()
                        st.success("Đã khôi phục tham số tối ưu!")
                        st.rerun()

            st.session_state[f"training_params_{model_choice}"] = params

            col_reset, col_train = st.columns([1, 3])
            with col_train:
                if st.button("🚀 Bắt đầu Huấn luyện", type="primary", key="start_training"):
                    try:
                        with st.spinner("Đang huấn luyện mô hình..."):
                            start_time = time.time()
                            progress_bar = st.progress(0)
                            status_text = st.empty()

                            status_text.text("Đang chuẩn bị dữ liệu... 20%")
                            progress_bar.progress(20)
                            time.sleep(0.1)

                            if model_choice == "Decision Tree":
                                pipeline = Pipeline([
                                    ('imputer', SimpleImputer(strategy='mean')),
                                    ('classifier', DecisionTreeClassifier(**params))
                                ])
                            else:
                                pipeline = Pipeline([
                                    ('imputer', SimpleImputer(strategy='mean')),
                                    ('classifier', SVC(probability=True, **params))
                                ])

                            status_text.text("Đang huấn luyện mô hình... 50%")
                            progress_bar.progress(50)
                            pipeline.fit(X_train, y_train)

                            status_text.text("Đang đánh giá mô hình... 90%")
                            progress_bar.progress(90)
                            y_valid_pred = pipeline.predict(X_valid)
                            y_test_pred = pipeline.predict(X_test)
                            acc_valid = accuracy_score(y_valid, y_valid_pred)
                            acc_test = accuracy_score(y_test, y_test_pred)
                            cm_valid = confusion_matrix(y_valid, y_valid_pred)
                            cm_test = confusion_matrix(y_test, y_test_pred)

                            run_name = f"{model_choice}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                            with mlflow.start_run(experiment_id=EXPERIMENT_ID, run_name=run_name) as run:
                                mlflow.log_params(params)
                                mlflow.log_metric("accuracy_val", acc_valid)
                                mlflow.log_metric("accuracy_test", acc_test)
                                mlflow.log_metric("training_time", time.time() - start_time)
                                mlflow.sklearn.log_model(pipeline, "model")

                                st.session_state['model'] = pipeline
                                st.session_state['training_results'] = {
                                    'accuracy_val': acc_valid, 'accuracy_test': acc_test,
                                    'cm_valid': cm_valid, 'cm_test': cm_test,
                                    'run_name': run_name, 'run_id': run.info.run_id,
                                    'params': params, 'training_time': time.time() - start_time,
                                    'model_choice': model_choice
                                }

                            status_text.text("Đã hoàn tất huấn luyện! 100%")
                            progress_bar.progress(100)
                            st.success(f"Đã huấn luyện xong! Thời gian: {time.time() - start_time:.2f} giây")
                            del X_train, y_train, X_valid, y_valid, X_test, y_test, split_data
                            gc.collect()
                            st.rerun()

                    except Exception as e:
                        st.error(f"Lỗi trong quá trình huấn luyện: {e}")

            if 'training_results' in st.session_state and st.session_state['training_results']['model_choice'] == model_choice:
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
                    plt.close(fig)
                with col_cm2:
                    fig, ax = plt.subplots(figsize=(6, 5))
                    sns.heatmap(results['cm_test'], annot=True, fmt="d", cmap="Blues", ax=ax)
                    ax.set_title("Test")
                    st.pyplot(fig)
                    plt.close(fig)

                with st.expander("Xem chi tiết", expanded=False):
                    st.markdown("**Thông tin lần chạy:**")
                    st.write(f"- Tên: {results['run_name']}")
                    st.write(f"- ID: {results['run_id']}")
                    st.write(f"- Thời gian huấn luyện: {results['training_time']:.2f} giây")
                    st.write(f"- Độ chính xác Validation: {results['accuracy_val']*100:.2f}%")
                    st.write(f"- Độ chính xác Test: {results['accuracy_test']*100:.2f}%")
                    st.markdown("**Tham số đã chọn:**")
                    st.json(results['params'])

    # Tab 6: Demo dự đoán
    with tab_demo:
        st.markdown('<div class="section-title">Demo Dự đoán Chữ số</div>', unsafe_allow_html=True)
        st.header("Dự đoán số viết tay")
        st.write("Chọn cách nhập liệu: tải lên hình ảnh, sử dụng dữ liệu Test hoặc vẽ trực tiếp.")

        if 'split_data' not in st.session_state or 'model' not in st.session_state:
            st.warning("⚠️ Vui lòng huấn luyện mô hình trước trong tab 'Huấn luyện/Đánh giá'!")
        else:
            model_choice = st.session_state['training_results']['model_choice']
            st.write(f"**Mô hình hiện tại**: {model_choice}")
            model = st.session_state['model']

            input_method = st.selectbox("Chọn phương thức nhập liệu", ["Tải ảnh lên", "Dữ liệu Test", "Vẽ trực tiếp"])
            is_normalized = 'data_processed' in st.session_state

            def preprocess_input(data, is_normalized):
                data, fixed = validate_and_fix_pixels(data)
                if fixed:
                    st.success("Đã chuẩn hóa dữ liệu về [0, 255]!")
                if not is_normalized:
                    data = data / 255.0
                return data

            if input_method == "Tải ảnh lên":
                st.markdown('<p class="mode-title">Dự đoán từ Ảnh Tải lên</p>', unsafe_allow_html=True)
                uploaded_images = st.file_uploader("Tải lên hình ảnh", type=["png", "jpg"], accept_multiple_files=True)
                if uploaded_images:
                    for i, uploaded_image in enumerate(uploaded_images):
                        try:
                            img = Image.open(uploaded_image).convert('L').resize((28, 28))
                            img_array = np.array(img).flatten().reshape(1, -1)
                            col_img, col_btn = st.columns([1, 2])
                            with col_img:
                                st.image(img, caption=f"Ảnh {i+1}", width=150)
                            with col_btn:
                                if st.button(f"Dự đoán ảnh {i+1}", key=f"predict_upload_{i}"):
                                    with st.spinner(f"Đang xử lý ảnh {i+1}..."):
                                        img_processed = preprocess_input(img_array, is_normalized)
                                        prediction = model.predict(img_processed)[0]
                                        proba = model.predict_proba(img_processed)[0]
                                        max_proba = np.max(proba) * 100
                                        st.markdown(f"""
                                            <div class="prediction-box">
                                                <strong>Dự đoán:</strong> {prediction}<br>
                                                <strong>Độ tin cậy:</strong> {max_proba:.2f}%
                                            </div>
                                        """, unsafe_allow_html=True)
                                        st.success(f"Dự đoán ảnh {i+1} hoàn tất!")
                                        del img, img_array, img_processed
                                        gc.collect()
                        except Exception as e:
                            st.error(f"Lỗi khi xử lý ảnh {i+1}: {e}")

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
                            prediction = model.predict(sample_processed)[0]
                            proba = model.predict_proba(sample_processed)[0]
                            max_proba = np.max(proba) * 100
                            st.markdown(f"""
                                <div class="prediction-box">
                                    <strong>Dự đoán:</strong> {prediction}<br>
                                    <strong>Độ tin cậy:</strong> {max_proba:.2f}%<br>
                                    <strong>Nhãn thực tế:</strong> {y_test[idx]}
                                </div>
                            """, unsafe_allow_html=True)
                            st.success("Dự đoán hoàn tất!")
                            del sample, sample_processed
                            gc.collect()

            elif input_method == "Vẽ trực tiếp":
                st.markdown('<p class="mode-title">Vẽ trực tiếp</p>', unsafe_allow_html=True)
                st.write("Vẽ chữ số từ 0-9 (nét trắng trên nền đen):")

                if 'canvas_key' not in st.session_state:
                    st.session_state['canvas_key'] = 0

                canvas_result = st_canvas(
                    fill_color="rgba(255, 165, 0, 0.3)",
                    stroke_width=20,
                    stroke_color="#FFFFFF",
                    background_color="#000000",
                    height=280,
                    width=280,
                    drawing_mode="freedraw",
                    key=f"canvas_{st.session_state['canvas_key']}"
                )

                if canvas_result.image_data is not None:
                    image = Image.fromarray(canvas_result.image_data[:, :, 3].astype('uint8'), 'L')
                    image_resized = image.resize((28, 28))
                    st.image(image_resized, caption="Hình ảnh bạn vẽ (resize 28x28)", width=100)

                    col_pred, col_clear = st.columns([2, 1])
                    with col_pred:
                        if st.button("Dự đoán", key="predict_button"):
                            with st.spinner("Đang xử lý hình vẽ..."):
                                image_array = np.array(image_resized, dtype=np.float32).flatten().reshape(1, -1)
                                image_processed = preprocess_input(image_array, is_normalized)
                                prediction = model.predict(image_processed)[0]
                                proba = model.predict_proba(image_processed)[0]
                                predicted_class = int(prediction)
                                confidence = proba[predicted_class] * 100
                                st.markdown(f"""
                                    <div class="prediction-box">
                                        <strong>Dự đoán:</strong> {predicted_class}<br>
                                        <strong>Độ tin cậy:</strong> {confidence:.2f}%
                                    </div>
                                """, unsafe_allow_html=True)
                                st.success("Dự đoán hoàn tất!")
                                del image, image_resized, image_array, image_processed
                                gc.collect()

                    with col_clear:
                        if st.button("Xóa bản vẽ", key="clear_button"):
                            st.session_state['canvas_key'] += 1
                            st.rerun()

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

                    mlflow_ui_link = f"{mlflow_tracking_uri}/#/experiments/{EXPERIMENT_ID}"
                    st.markdown("---")
                    st.markdown(f"📊 **Xem chi tiết trên MLflow UI**: [Nhấn vào đây]({mlflow_ui_link})", unsafe_allow_html=True)

        except Exception as e:
            st.error(f"Lỗi khi tải thông tin huấn luyện: {e}")

if __name__ == "__main__":
    run_mnist_classification_app()