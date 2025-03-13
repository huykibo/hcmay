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

def run_mnist_classification_app():
    # Thiết lập MLflow
    try:
        os.environ["MLFLOW_TRACKING_USERNAME"] = st.secrets["mlflow"]["MLFLOW_TRACKING_USERNAME"]
        os.environ["MLFLOW_TRACKING_PASSWORD"] = st.secrets["mlflow"]["MLFLOW_TRACKING_PASSWORD"]
        mlflow.set_tracking_uri(st.secrets["mlflow"]["MLFLOW_TRACKING_URI"])
        mlflow.set_experiment("MNIST")
    except KeyError as e:
        st.error(f"Lỗi: Không tìm thấy khóa {e} trong st.secrets. Vui lòng cấu hình secrets trong Streamlit.")
        st.stop()

    st.title("Ứng dụng Phân loại Chữ số MNIST")

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
    tabs = st.tabs(["Thông tin", "Tải dữ liệu", "Xử lí dữ liệu", "Chia dữ liệu", "Huấn luyện/Đánh Giá", "Demo dự đoán", "Thông tin huấn luyện"])
    tab_info, tab_load, tab_preprocess, tab_split, tab_train_eval, tab_demo, tab_log_info = tabs

    # Tab 1: Thông tin
    with tab_info:
        st.header("Giới thiệu về Ứng dụng và Các Mô hình Phân loại MNIST")
        info_option = st.selectbox(
            "Chọn thông tin để xem:",
            [
                "Ứng dụng này là gì và mục tiêu của nó?",
                "Tập dữ liệu MNIST: Đặc điểm và ý nghĩa",
                "Decision Tree – Mô hình cây quyết định",
                "SVM – Máy vector hỗ trợ",
                "So sánh Decision Tree và SVM",
                "Công thức đánh giá độ chính xác (Accuracy)"
            ],
            index=0,
            key="info_selectbox"
        )

        if info_option == "Ứng dụng này là gì và mục tiêu của nó?":
            st.subheader("1. Ứng dụng này là gì và mục tiêu của nó?")
            st.markdown("""
            Đây là một ứng dụng phân loại chữ số viết tay dựa trên tập dữ liệu MNIST – một trong những tập dữ liệu nổi tiếng nhất trong lĩnh vực học máy. MNIST bao gồm 70,000 ảnh chữ số từ 0 đến 9, mỗi ảnh có kích thước 28x28 pixel, tương đương với 784 đặc trưng (pixel). Mục tiêu của ứng dụng là xây dựng và huấn luyện các mô hình học máy (SVM và Decision Tree) để nhận diện chính xác các chữ số này, từ đó cung cấp một công cụ trực quan cho việc học tập, thử nghiệm và đánh giá hiệu quả của các thuật toán phân loại.

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
            - Là bài toán cơ bản để kiểm tra hiệu quả của các thuật toán phân loại.  
            - Dữ liệu đơn giản nhưng đủ phức tạp để đánh giá khả năng phân biệt giữa các lớp tương tự (ví dụ: "4" và "9").  
            - Phù hợp cho cả người mới bắt đầu và các nhà nghiên cứu muốn thử nghiệm các mô hình phức tạp hơn.
            """)
            st.subheader("Minh họa dữ liệu MNIST")
            with st.spinner("Đang tải ảnh minh họa..."):
                try:
                    mnist_image = Image.open("mnist.png")
                    st.image(mnist_image, caption="Ảnh minh họa 10 chữ số từ 0 đến 9 trong MNIST", width=800)
                except FileNotFoundError:
                    st.error("Không tìm thấy file `mnist.png`. Vui lòng đảm bảo file nằm trong cùng thư mục với code hoặc cung cấp đường dẫn chính xác.")
                except Exception as e:
                    st.error(f"Lỗi khi tải ảnh: {e}")

        elif info_option == "Decision Tree – Mô hình cây quyết định":
            st.subheader("3. Decision Tree – Mô hình cây quyết định")
            st.markdown("""
            **Decision Tree (Cây quyết định)** xây dựng một cấu trúc phân cấp giống như cây, trong đó dữ liệu được chia nhỏ dần dựa trên các đặc trưng (pixel trong MNIST) để đưa ra dự đoán cuối cùng.
            """)

            st.subheader("Cách hoạt động chi tiết:")
            st.markdown("""
            1. **Nút gốc (Root Node)**:  
               - Thuật toán bắt đầu với toàn bộ dữ liệu MNIST (70,000 mẫu, nhãn 0-9) và chọn một pixel quan trọng, ví dụ: "Pixel 5 > 100?" (giả sử Pixel 5 là giá trị tại vị trí [0, 5] trong ảnh 28x28).  
               - Dữ liệu được chia thành hai nhánh: nhánh "Yes" nếu Pixel 5 > 100, nhánh "No" nếu Pixel 5 ≤ 100.  
            """)
            try:
                tree_step_1 = Image.open("illustrations/tree_step_1.png")
                st.image(tree_step_1, caption="Bước 1: Nút gốc với toàn bộ dữ liệu MNIST", width=500)
            except FileNotFoundError:
                st.error("Không tìm thấy file `illustrations/tree_step_1.png`. Vui lòng đảm bảo file đã được tạo.")
            except Exception as e:
                st.error(f"Lỗi khi tải ảnh: {e}")

            st.markdown("""
            2. **Chia nhánh đầu tiên (Splitting)**:  
               - Từ nút gốc, nhánh "Yes" (Pixel 5 > 100) chứa các mẫu có giá trị pixel sáng hơn, ví dụ: chữ số "1" hoặc "7" (thường có nét dày ở đầu).  
               - Nhánh "No" (Pixel 5 ≤ 100) chứa các mẫu tối hơn, ví dụ: chữ số "0" hoặc "2".  
            """)
            try:
                tree_step_2 = Image.open("illustrations/tree_step_2.png")
                st.image(tree_step_2, caption="Bước 2: Chia nhánh đầu tiên dựa trên Pixel 5 > 100", width=500)
            except FileNotFoundError:
                st.error("Không tìm thấy file `illustrations/tree_step_2.png`. Vui lòng đảm bảo file đã được tạo.")
            except Exception as e:
                st.error(f"Lỗi khi tải ảnh: {e}")

            st.markdown("""
            3. **Chia nhánh tiếp theo**:  
               - Từ nhánh "Yes" (Pixel 5 > 100), tiếp tục chia dựa trên "Pixel 10 > 50?" (giả sử Pixel 10 là vị trí [0, 10]).  
               - Nhánh "Yes" (Pixel 10 > 50) chứa các mẫu có nét dày hơn, ví dụ: "1" (nét đứng rõ rệt).  
               - Nhánh "No" (Pixel 10 ≤ 50) chứa các mẫu mỏng hơn, ví dụ: "7".  
            """)
            try:
                tree_step_3 = Image.open("illustrations/tree_step_3.png")
                st.image(tree_step_3, caption="Bước 3: Chia nhánh tiếp theo dựa trên Pixel 10 > 50", width=500)
            except FileNotFoundError:
                st.error("Không tìm thấy file `illustrations/tree_step_3.png`. Vui lòng đảm bảo file đã được tạo.")
            except Exception as e:
                st.error(f"Lỗi khi tải ảnh: {e}")

            st.markdown("""
            4. **Nút lá và tiêu chí dừng**:  
               - Quá trình dừng khi nhóm dữ liệu thuần nhất hoặc đạt độ sâu tối đa.  
               - Ví dụ: Nhánh "Yes" của "Pixel 10 > 50" → Nhãn "1" (thuần nhất).  
               - Nhánh "No" của "Pixel 10 ≤ 50" → Nhãn "9" (thuần nhất).  
               - Nhánh "No" của "Pixel 5 > 100" → Nhãn "0" (độ sâu tối đa).  
            """)
            try:
                tree_step_4 = Image.open("illustrations/tree_step_4.png")
                st.image(tree_step_4, caption="Bước 4: Nút lá với nhãn dự đoán (1, 9, 0)", width=500)
            except FileNotFoundError:
                st.error("Không tìm thấy file `illustrations/tree_step_4.png`. Vui lòng đảm bảo file đã được tạo.")
            except Exception as e:
                st.error(f"Lỗi khi tải ảnh: {e}")

            st.markdown("""
            5. **Dự đoán**:  
               - Với một mẫu mới có Pixel 5 = 150 (> 100) và Pixel 10 = 60 (> 50), thuật toán đi qua nhánh "Yes" rồi "Yes", dẫn đến nhãn "1".  
               - Kết quả dự đoán: "1" với độ tin cậy cao dựa trên các điều kiện pixel.  
            """)
            try:
                tree_step_5 = Image.open("illustrations/tree_step_5.png")
                st.image(tree_step_5, caption="Bước 5: Dự đoán nhãn '1' cho mẫu mới", width=500)
            except FileNotFoundError:
                st.error("Không tìm thấy file `illustrations/tree_step_5.png`. Vui lòng đảm bảo file đã được tạo.")
            except Exception as e:
                st.error(f"Lỗi khi tải ảnh: {e}")

            st.markdown("""
            ### Tiêu chí lựa chọn đặc trưng và ngưỡng:  
            - **Entropy**: Đo mức độ "hỗn loạn" của dữ liệu dựa trên phân bố nhãn:  
              $$ Entropy(S) = -\\sum_{i=0}^{9} p_i \\log_2(p_i) $$  
              - $p_i$: Tỷ lệ mẫu thuộc nhãn $i$.  
            - **Gini Index**: Đo độ "tinh khiết" của nhóm:  
              $$ Gini(S) = 1 - \\sum_{i=0}^{9} p_i^2 $$  

            ### Áp dụng với MNIST:
            - Decision Tree chia dữ liệu dựa trên giá trị pixel (ví dụ: Pixel 5, Pixel 10) để phân biệt nhãn (0-9).  

            ### Ưu điểm:
            - Dễ hiểu, trực quan như một biểu đồ cây hỏi đáp.  
            - Nhanh với dữ liệu nhỏ, không yêu cầu chuẩn hóa dữ liệu.  

            ### Nhược điểm:
            - Dễ bị **overfitting** nếu cây quá sâu, đặc biệt khi dữ liệu phức tạp như MNIST.  
            - Khó xử lý các mẫu có đặc trưng tương tự (ví dụ: "3" và "8").  
            """)

        elif info_option == "SVM – Máy vector hỗ trợ":
            st.subheader("4. SVM – Máy vector hỗ trợ")
            st.markdown("""
            **SVM (Support Vector Machine)** tìm một **siêu phẳng** trong không gian đặc trưng (784 chiều với MNIST) để phân tách các lớp nhãn sao cho khoảng cách từ siêu phẳng đến các mẫu gần nhất (support vectors) là lớn nhất. Nếu dữ liệu không phân tách tuyến tính, nó sử dụng **kernel** để chuyển dữ liệu lên không gian cao hơn.

            ### Cách hoạt động chi tiết:
            1. **Siêu phẳng**:  
               - Siêu phẳng là một ranh giới phân tách trong không gian cao chiều, được định nghĩa bởi:  
                 $$ f(x) = w \\cdot x + b $$  
                 - $w$: Vector trọng số, xác định hướng của siêu phẳng.  
                 - $x$: Vector đặc trưng (784 pixel).  
                 - $b$: Độ lệch, điều chỉnh vị trí siêu phẳng.  
               - Mẫu nằm ở phía nào của siêu phẳng được xác định bởi dấu của $f(x)$.  
            """)
            try:
                svm_step_1 = Image.open("illustrations/svm_step_1.png")
                st.image(svm_step_1, caption="Bước 1: Siêu phẳng phân tách dữ liệu", width=500)
            except FileNotFoundError:
                st.error("Không tìm thấy file `illustrations/svm_step_1.png`. Vui lòng chạy code tạo ảnh trước hoặc kiểm tra đường dẫn.")
            except Exception as e:
                st.error(f"Lỗi khi tải ảnh: {e}")

            st.markdown("""
            2. **Tối ưu hóa lề**:  
               - Khoảng cách lề (margin) từ siêu phẳng đến các điểm gần nhất được tính là:  
                 $$ Margin = \\frac{2}{\\|w\\|} $$  
                 - $\\|w\\|$: Độ dài vector $w$.  
               - Mục tiêu tối ưu hóa:  
                 $$ \\min_{w,b} \\frac{1}{2} \\|w\\|^2 $$  
                 Với ràng buộc:  
                 $$ y_i (w \\cdot x_i + b) \\geq 1 $$  
                 - $y_i$: Nhãn thực tế (+1 hoặc -1 cho phân loại nhị phân).  
                 - $x_i$: Vector đặc trưng của mẫu.  
            """)
            try:
                svm_step_2 = Image.open("illustrations/svm_step_2.png")
                st.image(svm_step_2, caption="Bước 2: Siêu phẳng tối ưu với lề lớn nhất và support vectors", width=500)
            except FileNotFoundError:
                st.error("Không tìm thấy file `illustrations/svm_step_2.png`. Vui lòng chạy code tạo ảnh trước hoặc kiểm tra đường dẫn.")
            except Exception as e:
                st.error(f"Lỗi khi tải ảnh: {e}")

            st.markdown("""
            3. **Soft Margin**:  
               - Khi dữ liệu không phân tách hoàn hảo (có lẫn lộn giữa các lớp), SVM cho phép sai số:  
                 $$ \\min_{w,b,\\xi} \\frac{1}{2} \\|w\\|^2 + C \\sum \\xi_i $$  
                 - $\\xi_i$: Biến "lỏng" (slack variable), đo mức độ vi phạm của mẫu.  
                 - $C$: Tham số điều chỉnh, cân bằng giữa việc tối đa hóa lề và giảm thiểu lỗi.  
               - $C$ lớn ưu tiên ít lỗi hơn, $C$ nhỏ ưu tiên lề lớn hơn.  
            """)
            try:
                svm_step_4 = Image.open("illustrations/svm_step_4.png")
                st.image(svm_step_4, caption="Bước 3: Soft Margin với dữ liệu lẫn lộn", width=500)
            except FileNotFoundError:
                st.error("Không tìm thấy file `illustrations/svm_step_4.png`. Vui lòng chạy code tạo ảnh trước hoặc kiểm tra đường dẫn.")
            except Exception as e:
                st.error(f"Lỗi khi tải ảnh: {e}")

            st.markdown("""
            4. **Kernel Trick**:  
               - Khi dữ liệu không phân tách tuyến tính trong không gian ban đầu, SVM ánh xạ dữ liệu lên không gian cao hơn thông qua hàm kernel:  
                 $$ K(x_i, x_j) = \\phi(x_i) \\cdot \\phi(x_j) $$  
                 - $\\phi$: Hàm ánh xạ (không cần tính trực tiếp).  
               - Các loại kernel:  
                 - **Linear**: $K(x_i, x_j) = x_i \\cdot x_j$.  
                 - **Polynomial**: $K(x_i, x_j) = (x_i \\cdot x_j + c)^d$.  
                 - **RBF**: $K(x_i, x_j) = \\exp(-\\gamma \\|x_i - x_j\\|^2)$, thường dùng cho dữ liệu phi tuyến.  
               - Kernel giúp tìm ranh giới phân tách trong không gian mới mà không cần tính toán tọa độ trực tiếp.  
            """)
            try:
                svm_step_3 = Image.open("illustrations/svm_step_3.png")
                st.image(svm_step_3, caption="Bước 4: Kernel nâng dữ liệu lên không gian cao hơn", width=500)
            except FileNotFoundError:
                st.error("Không tìm thấy file `illustrations/svm_step_3.png`. Vui lòng chạy code tạo ảnh trước hoặc kiểm tra đường dẫn.")
            except Exception as e:
                st.error(f"Lỗi khi tải ảnh: {e}")

            st.markdown("""
            5. **Dự đoán**:  
               - Với mẫu mới $x$:  
                 $$ f(x) = \\text{sign} \\left( \\sum_{i} \\alpha_i y_i K(x_i, x) + b \\right) $$  
                 - $\\alpha_i$: Trọng số xác định từ quá trình huấn luyện, chỉ khác 0 với support vectors.  
                 - $K(x_i, x)$: Đo độ tương đồng giữa mẫu mới và support vectors.  
               - Với phân loại đa lớp (0-9), SVM áp dụng chiến lược như "One-vs-Rest" hoặc "One-vs-One".  
            """)
            try:
                svm_step_5 = Image.open("illustrations/svm_step_5.png")
                st.image(svm_step_5, caption="Bước 5: Dự đoán điểm mới dựa trên siêu phẳng và support vectors", width=500)
            except FileNotFoundError:
                st.error("Không tìm thấy file `illustrations/svm_step_5.png`. Vui lòng chạy code tạo ảnh trước hoặc kiểm tra đường dẫn.")
            except Exception as e:
                st.error(f"Lỗi khi tải ảnh: {e}")

            st.markdown("""
            ### Áp dụng với MNIST:
            - SVM tìm ranh giới phân tách dựa trên toàn bộ đặc trưng pixel, tận dụng kernel để xử lý các mẫu phi tuyến.  

            ### Ưu điểm:
            - Hiệu quả với dữ liệu phức tạp, chính xác cao khi có kernel phù hợp.  
            - Tốt cho việc phân biệt các chữ số như "4" và "9".  

            ### Nhược điểm:
            - Tốn thời gian tính toán với dữ liệu lớn.  
            - Yêu cầu chuẩn hóa dữ liệu trước để đạt hiệu quả tối ưu.  
            """)

        elif info_option == "So sánh Decision Tree và SVM":
            st.subheader("5. So sánh Decision Tree và SVM")
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
            """)

        elif info_option == "Công thức đánh giá độ chính xác (Accuracy)":
            st.subheader("6. Công thức đánh giá độ chính xác (Accuracy)")
            st.markdown("""
            Độ chính xác (Accuracy) đo tỷ lệ dự đoán đúng:  
            $$ Accuracy = \\frac{\\text{Số mẫu dự đoán đúng}}{\\text{Tổng số mẫu}} $$  
            - **Ví dụ**: Dự đoán đúng 92/100 ảnh → Accuracy = 92%.  

            **Ý nghĩa**:  
            - **Decision Tree**: Đo khả năng chia nhóm đúng dựa trên các đặc trưng pixel.  
            - **SVM**: Đo hiệu quả của siêu phẳng trong việc phân tách các lớp.
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
                    st.success("Tải dữ liệu thành công!")
                    st.write("Kích thước dữ liệu gốc:", X.shape)
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

    # Tab 3: Xử lí dữ liệu (Updated)
    with tab_preprocess:
        st.header("Xử lí Dữ liệu")
        if 'data' not in st.session_state:
            st.info("Vui lòng tải và chốt số lượng mẫu trước.")
        else:
            if "data_original" not in st.session_state:
                X, y = st.session_state['data']
                st.session_state["data_original"] = (X.copy(), y.copy())
            
            # Retrieve current_data with fallback
            current_data = st.session_state.get("data_processed", st.session_state["data_original"])
            
            # Validate current_data
            if not isinstance(current_data, (tuple, list)) or len(current_data) != 2:
                st.error("Lỗi: Dữ liệu không hợp lệ trong st.session_state. Vui lòng kiểm tra bước tải hoặc xử lý dữ liệu.")
            else:
                try:
                    X_current, y_current = current_data
                except (ValueError, TypeError) as e:
                    st.error(f"Lỗi khi truy xuất dữ liệu: {e}. Dữ liệu có thể không ở định dạng đúng (X, y).")
                else:
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
                                    Công dụng: Đảm bảo thang đo đồng nhất, đặc biệt quan trọng cho SVM.
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

                st.write(f"Train: {len(X_train)} mẫu, Validation: {len(X_valid)} mẫu, Test: {len(X_test)} mẫu")

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
            model_choice = st.selectbox("Chọn mô hình", ["Decision Tree", "SVM"])
            X_train = st.session_state['split_data']["X_train"]
            num_samples = len(X_train)
            st.write(f"Số lượng mẫu huấn luyện: {num_samples}")

            st.subheader("Bảng gợi ý tham số tối ưu dựa trên số lượng mẫu")
            if model_choice == "Decision Tree":
                st.markdown("""
                | Số lượng mẫu | Criterion          | Max Depth |
                |--------------|--------------------|-----------|
                | <1000        | gini hoặc entropy  | 5-10      |
                | 1000-5000    | gini hoặc entropy  | 10-20     |
                | 5000-50000   | gini hoặc entropy  | 20-30     |
                | >50000       | gini hoặc entropy  | 30-50     |
                """)
                st.markdown("""
                - **criterion**: "gini" đo độ tinh khiết, "entropy" đo độ hỗn loạn.  
                - **max_depth**: Giới hạn độ sâu để tránh overfitting.
                """)
            else:
                st.markdown("""
                | Số lượng mẫu | C         | Kernel     |
                |--------------|-----------|------------|
                | <1000        | 0.1-1.0   | rbf        |
                | 1000-5000    | 1.0-5.0   | rbf        |
                | 5000-50000   | 5.0-10.0  | rbf        |
                | >50000       | 10.0-50.0 | rbf hoặc poly |
                """)
                st.markdown("""
                - **C**: Điều chỉnh giữa lề lớn và lỗi phân loại.  
                - **kernel**: "rbf" phù hợp với dữ liệu phi tuyến như MNIST.
                """)

            params = {}
            if num_samples < 1000:
                if model_choice == "Decision Tree":
                    params["criterion"] = "gini"
                    params["max_depth"] = 5
                else:
                    params["C"] = 0.1
                    params["kernel"] = "rbf"
            elif 1000 <= num_samples <= 5000:
                if model_choice == "Decision Tree":
                    params["criterion"] = "gini"
                    params["max_depth"] = 10
                else:
                    params["C"] = 1.0
                    params["kernel"] = "rbf"
            elif 5000 < num_samples <= 50000:
                if model_choice == "Decision Tree":
                    params["criterion"] = "gini"
                    params["max_depth"] = 20
                else:
                    params["C"] = 5.0
                    params["kernel"] = "rbf"
            else:
                if model_choice == "Decision Tree":
                    params["criterion"] = "gini"
                    params["max_depth"] = 30
                else:
                    params["C"] = 10.0
                    params["kernel"] = "rbf"

            st.markdown("#### Tham số mô hình (đã đặt tự động, có thể điều chỉnh)")
            if model_choice == "Decision Tree":
                params["criterion"] = st.selectbox("Criterion", ["gini", "entropy"], index=["gini", "entropy"].index(params["criterion"]))
                params["max_depth"] = st.number_input("Max Depth", min_value=1, max_value=100, value=params["max_depth"])
            else:
                params["C"] = st.number_input("C", min_value=0.01, max_value=100.0, value=params["C"])
                params["kernel"] = st.selectbox("Kernel", ["linear", "rbf", "poly", "sigmoid"], index=["linear", "rbf", "poly", "sigmoid"].index(params["kernel"]))

            if st.button("Thực hiện Huấn luyện"):
                with st.spinner("Đang huấn luyện mô hình..."):
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    start_time = time.time()
                   
                    X_train = st.session_state['split_data']["X_train"]
                    y_train = st.session_state['split_data']["y_train"]
                    X_valid = st.session_state['split_data']["X_valid"]
                    y_valid = st.session_state['split_data']["y_valid"]
                    X_test = st.session_state['split_data']["X_test"]
                    y_test = st.session_state['split_data']["y_test"]

                    run_name = f"{model_choice}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                    with mlflow.start_run(run_name=run_name) as run:
                        for i in range(0, 11, 2):
                            progress_bar.progress(i)
                            status_text.text(f"Đang khởi tạo mô hình {i}%{i % 4 * '.'}")
                            time.sleep(0.1)
                       
                        if model_choice == "Decision Tree":
                            pipeline = Pipeline([
                                ('imputer', SimpleImputer(strategy='mean')),
                                ('classifier', DecisionTreeClassifier(**params))
                            ])
                            pipeline.fit(X_train, y_train)
                            model = pipeline
                        else:
                            pipeline = Pipeline([
                                ('imputer', SimpleImputer(strategy='mean')),
                                ('classifier', SVC(probability=True, **params))
                            ])
                            pipeline.fit(X_train, y_train)
                            model = pipeline

                        for i in range(10, 51, 5):
                            progress_bar.progress(i)
                            status_text.text(f"Đang huấn luyện {i}%{i % 4 * '.'}")
                            time.sleep(0.1)

                        mlflow.log_params(params)
                        y_valid_pred = model.predict(X_valid)
                        accuracy_val = accuracy_score(y_valid, y_valid_pred)
                        mlflow.log_metric("accuracy_val", accuracy_val)
                        cm_valid = confusion_matrix(y_valid, y_valid_pred)

                        for i in range(50, 76, 5):
                            progress_bar.progress(i)
                            status_text.text(f"Đang đánh giá validation {i}%{i % 4 * '.'}")
                            time.sleep(0.1)

                        y_test_pred = model.predict(X_test)
                        accuracy_test = accuracy_score(y_test, y_test_pred)
                        mlflow.log_metric("accuracy_test", accuracy_test)
                        cm_test = confusion_matrix(y_test, y_test_pred)
                        training_time = time.time() - start_time
                        mlflow.log_metric("training_time_seconds", training_time)
                        mlflow.sklearn.log_model(model, "model")

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
                            'model_choice': model_choice,
                            'params': params,
                            'num_samples': len(X_train),
                            'run_name': run_name,
                            'run_id': run_id
                        }

                        status_text.empty()
                        progress_bar.empty()

            if 'training_results' in st.session_state:
                st.success(f"Huấn luyện hoàn tất. Thời gian thực hiện: {st.session_state['training_results']['training_time']:.2f} giây.")
                st.write(f"Accuracy Validation: {st.session_state['training_results']['accuracy_val']:.4f}")
                st.write(f"Accuracy Test: {st.session_state['training_results']['accuracy_test']:.4f}")

                fig, ax = plt.subplots()
                sns.heatmap(st.session_state['training_results']['cm_valid'], annot=True, fmt="d", cmap="Blues", ax=ax)
                ax.set_title("Confusion Matrix - Validation")
                st.pyplot(fig)

                fig, ax = plt.subplots()
                sns.heatmap(st.session_state['training_results']['cm_test'], annot=True, fmt="d", cmap="Blues", ax=ax)
                ax.set_title("Confusion Matrix - Test")
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
                idx = st.slider("Chọn mẫu từ Test", 0, len(X_test)-1, 0)
                if st.button("Dự đoán"):
                    with st.spinner("Đang dự đoán..."):
                        for i in range(0, 51, 5):
                            progress_bar.progress(i)
                            status_text.text(f"Đang xử lý {i}%{i % 4 * '.'}")
                            time.sleep(0.1)
                        
                        sample = X_test.iloc[idx].values.reshape(1, -1)
                        if not is_normalized:
                            sample = preprocess_input(sample)
                        
                        prediction = st.session_state['model'].predict(sample)[0]
                        proba = st.session_state['model'].predict_proba(sample)[0]
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
                            img_array = np.array(img).flatten().reshape(1, -1)
                            if not is_normalized:
                                img_array = preprocess_input(img_array)
                            
                            prediction = st.session_state['model'].predict(img_array)[0]
                            proba = st.session_state['model'].predict_proba(img_array)[0]
                            confidence = max(proba) * 100
                            
                            for j in range(50, 101, 5):
                                progress_bar.progress(j)
                                status_text.text(f"Đang dự đoán ảnh {i+1} - {j}%{j % 4 * '.'}")
                                time.sleep(0.1)
                            
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
                            
                            img = Image.fromarray((image_data * 255).astype(np.uint8)).convert('L').resize((28, 28))
                            img_array = np.array(img).flatten().reshape(1, -1)
                            if not is_normalized:
                                img_array = preprocess_input(img_array)
                            
                            prediction = st.session_state['model'].predict(img_array)[0]
                            proba = st.session_state['model'].predict_proba(img_array)[0]
                            confidence = max(proba) * 100
                            
                            for i in range(50, 101, 5):
                                progress_bar.progress(i)
                                status_text.text(f"Đang dự đoán {i}%{i % 4 * '.'}")
                                time.sleep(0.1)
                            
                            st.success(f"Dự đoán: **{prediction}** | Confidence: **{confidence:.2f}%**")
                            
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
            experiment = client.get_experiment_by_name("MNIST")
            if not experiment:
                st.error("Không tìm thấy experiment 'MNIST'. Vui lòng kiểm tra lại MLflow tracking URI.")
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
    run_mnist_classification_app()