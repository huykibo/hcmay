import os
import mlflow
import streamlit as st
import openml
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.neural_network import MLPClassifier  # Neural Network
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from mlflow.tracking import MlflowClient
from streamlit_drawable_canvas import st_canvas
from datetime import datetime
import time

def run_mnist_neural_network_app():
    # Thiết lập MLflow
    try:
        os.environ["MLFLOW_TRACKING_USERNAME"] = st.secrets["mlflow"]["MLFLOW_TRACKING_USERNAME"]
        os.environ["MLFLOW_TRACKING_PASSWORD"] = st.secrets["mlflow"]["MLFLOW_TRACKING_PASSWORD"]
        mlflow.set_tracking_uri(st.secrets["mlflow"]["MLFLOW_TRACKING_URI"])
        mlflow.set_experiment("MNIST_NeuralNetwork")
    except KeyError as e:
        st.error(f"Lỗi: Không tìm thấy khóa {e} trong st.secrets. Vui lòng cấu hình secrets trong Streamlit.")
        st.stop()

    st.title("Ứng dụng Phân loại Chữ số MNIST với Neural Network")

    # CSS cho MathJax và Tooltip
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

        elif info_option == "Tập dữ liệu MNIST: Đặc điểm và ý nghĩa":
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
            with st.spinner("Đang tải ảnh minh họa..."):
                try:
                    mnist_image = Image.open("mnist.png")
                    st.image(mnist_image, caption="Ảnh minh họa $10$ chữ số từ $0$ đến $9$ trong MNIST", width=800)
                except FileNotFoundError:
                    st.error("Không tìm thấy file `mnist.png`. Vui lòng kiểm tra đường dẫn.")
                except Exception as e:
                    st.error(f"Lỗi khi tải ảnh: {e}")

        elif info_option == "Neural Network – Mạng nơ-ron nhân tạo":
            st.subheader("📊 3. Neural Network – Mạng nơ-ron nhân tạo")
            st.markdown("""
            **Neural Network (Mạng nơ-ron nhân tạo)** là một mô hình học máy mô phỏng cách hoạt động của mạng nơ-ron sinh học trong não người.  
            - **Cấu trúc**: Gồm các **nơ-ron nhân tạo** (nodes) được tổ chức thành các **lớp (layers)**:  
              - **Lớp đầu vào (Input Layer)**: Nhận dữ liệu ($784$ pixel từ ảnh MNIST).  
              - **Lớp ẩn (Hidden Layers)**: Xử lý thông tin bằng cách kết hợp tuyến tính và áp dụng hàm kích hoạt phi tuyến.  
              - **Lớp đầu ra (Output Layer)**: Đưa ra dự đoán (nhãn từ $0$-$9$).  

            Neural Network đặc biệt hiệu quả với bài toán MNIST nhờ khả năng học các đặc trưng phức tạp từ dữ liệu hình ảnh.
            """, unsafe_allow_html=True)

            st.subheader("🛠️ Các bước thực hiện trong Neural Network")
            st.markdown("""
            1. **Khởi tạo mô hình**:  
               - Xác định cấu trúc mạng (số lớp ẩn, số nơ-ron mỗi lớp).  
               - Khởi tạo **trọng số** $W$ và **bias** $b$ ngẫu nhiên hoặc bằng $0$.  
            """, unsafe_allow_html=True)
            try:
                st.image(os.path.join("plnw", "step1_init.png"), caption="Minh họa Bước 1: Khởi tạo mô hình", width=600)
            except FileNotFoundError:
                st.error("Không tìm thấy ảnh minh họa cho Bước 1. Vui lòng chạy mã tạo ảnh trước.")

            st.markdown("""
            2. **Lan truyền thuận (Feedforward)**:  
               - Tính giá trị dự đoán $\\hat{Y}$ từ dữ liệu đầu vào $X$:  
                 - **Lớp đầu vào**: $A^{(0)} = X$ (ma trận $N \\times 784$, $N$ là số mẫu).  
                 - **Cho mỗi lớp $l$**:  
                   - Tổng tuyến tính:  
                     $$ Z^{(l)} = A^{(l-1)} \\cdot W^{(l)} + b^{(l)} $$  
                   - Áp dụng hàm kích hoạt:  
                     $$ A^{(l)} = \\sigma(Z^{(l)}) $$  
                 - **Lớp đầu ra**: $\\hat{Y} = A^{(L)}$ (ma trận $N \\times 10$).  
               - Ví dụ hàm kích hoạt **sigmoid**:  
                 $$ \\sigma(z) = \\frac{1}{1 + e^{-z}} $$
            """, unsafe_allow_html=True)
            try:
                st.image(os.path.join("plnw", "step2_feedforward.png"), caption="Minh họa Bước 2: Lan truyền thuận", width=600)
            except FileNotFoundError:
                st.error("Không tìm thấy ảnh minh họa cho Bước 2. Vui lòng chạy mã tạo ảnh trước.")

            st.markdown("""
            3. **Tính hàm mất mát (Loss Function)**:  
               - Đo độ sai lệch giữa $\\hat{Y}$ và $Y$ (giá trị thực). Với MNIST, dùng **Cross-Entropy**:  
                 $$ L = -\\frac{1}{N} \\sum_{i=1}^{N} \\sum_{j=0}^{9} y_{ij} \\cdot \\log(\\hat{y}_{ij}) $$  
               - Trong đó:  
                 - $y_{ij}$: Nhãn thực (dạng one-hot encoded).  
                 - $\\hat{y}_{ij}$: Xác suất dự đoán cho lớp $j$.  
            """, unsafe_allow_html=True)
            try:
                st.image(os.path.join("plnw", "step3_loss.png"), caption="Minh họa Bước 3: Tính hàm mất mát", width=600)
            except FileNotFoundError:
                st.error("Không tìm thấy ảnh minh họa cho Bước 3. Vui lòng chạy mã tạo ảnh trước.")

            st.markdown("""
            4. **Lan truyền ngược (Backpropagation)**:  
               - Tính đạo hàm của $L$ theo $W^{(l)}$ và $b^{(l)}$ để cập nhật tham số:  
                 - Tại **Lớp đầu ra**:  
                   $$ \\delta^{(L)} = \\hat{Y} - Y $$  
                 - Tại **Lớp ẩn**:  
                   $$ \\delta^{(l)} = (\\delta^{(l+1)} \\cdot (W^{(l+1)})^T) \\odot \\sigma'(Z^{(l)}) $$  
                   - $\\sigma'(z)$: Đạo hàm hàm kích hoạt (với sigmoid: $\\sigma'(z) = \\sigma(z) \\cdot (1 - \\sigma(z))$).  
                 - Đạo hàm theo trọng số và bias:  
                   $$ \\frac{\\partial L}{\\partial W^{(l)}} = (A^{(l-1)})^T \\cdot \\delta^{(l)} $$  
                   $$ \\frac{\\partial L}{\\partial b^{(l)}} = \\sum_{i=1}^{N} \\delta^{(l)}_i $$
            """, unsafe_allow_html=True)
            try:
                st.image(os.path.join("plnw", "step4_backprop.png"), caption="Minh họa Bước 4: Lan truyền ngược", width=600)
            except FileNotFoundError:
                st.error("Không tìm thấy ảnh minh họa cho Bước 4. Vui lòng chạy mã tạo ảnh trước.")

            st.markdown("""
            5. **Cập nhật tham số (Gradient Descent)**:  
               - Điều chỉnh $W$ và $b$ để giảm mất mát:  
                 $$ W^{(l)} = W^{(l)} - \\eta \\cdot \\frac{\\partial L}{\\partial W^{(l)}} $$  
                 $$ b^{(l)} = b^{(l)} - \\eta \\cdot \\frac{\\partial L}{\\partial b^{(l)}} $$  
               - Trong đó: $\\eta$ là **tốc độ học (learning rate)**.  
            """, unsafe_allow_html=True)
            try:
                st.image(os.path.join("plnw", "step5_gradient.png"), caption="Minh họa Bước 5: Cập nhật tham số", width=600)
            except FileNotFoundError:
                st.error("Không tìm thấy ảnh minh họa cho Bước 5. Vui lòng chạy mã tạo ảnh trước.")

            st.markdown("""
            6. **Lặp lại**:  
               - Quay lại bước $2$ qua nhiều **epoch** cho đến khi $L$ hội tụ.  
            """, unsafe_allow_html=True)
            try:
                st.image(os.path.join("plnw", "step6_repeat_improved.png"), caption="Minh họa Bước 6: Lặp lại", width=600)
            except FileNotFoundError:
                st.error("Không tìm thấy ảnh minh họa cho Bước 6. Vui lòng chạy mã tạo ảnh trước.")

            st.subheader("⚙️ Các tham số cơ bản và công dụng")
            st.markdown("""
            Dưới đây là các tham số bạn sẽ sử dụng để điều chỉnh mô hình trong ứng dụng này:  
            - **hidden_layer_sizes**:  
              - **Ý nghĩa**: Số nơ-ron trong lớp ẩn (ví dụ: $128$).  
              - **Công dụng**: Quyết định sức mạnh của mô hình; nhiều nơ-ron hơn thì học được đặc trưng phức tạp hơn nhưng tốn thời gian hơn.  
            - **learning_rate_init**:  
              - **Ý nghĩa**: Tốc độ học ban đầu (ví dụ: $0.001$).  
              - **Công dụng**: Điều chỉnh tốc độ cập nhật trọng số; nhỏ hơn thì học chậm nhưng ổn định hơn.  
            - **max_iter**:  
              - **Ý nghĩa**: Số lần huấn luyện tối đa (ví dụ: $200$).  
              - **Công dụng**: Giới hạn số lần mô hình học qua dữ liệu để đạt độ chính xác mong muốn.  
            """, unsafe_allow_html=True)

            st.subheader("🟪 Ưu điểm và nhược điểm")
            st.markdown("""
            ##### ✅ **Ưu điểm**:  
            - Học được các đặc trưng phức tạp từ dữ liệu hình ảnh như MNIST.  
            - Dễ sử dụng với các tham số cơ bản được tối ưu sẵn.  

            ##### ❌ **Nhược điểm**:  
            - Tốn thời gian huấn luyện nếu số mẫu lớn hoặc số nơ-ron nhiều.  
            - Cần dữ liệu được chuẩn hóa để đạt hiệu quả tốt nhất.  
            """, unsafe_allow_html=True)

        elif info_option == "Công thức đánh giá độ chính xác (Accuracy)":
            st.subheader("📘 4. Công thức đánh giá độ chính xác (Accuracy)")
            st.markdown("""
            Độ chính xác (**Accuracy**) đo tỷ lệ dự đoán đúng:  
            $$ \\text{Accuracy} = \\frac{\\text{Số mẫu dự đoán đúng}}{\\text{Tổng số mẫu}} $$  
            - **Ví dụ**: Dự đoán đúng $92/100$ ảnh → $\\text{Accuracy} = 92\\%$.  
            - **Ý nghĩa**: Với Neural Network, Accuracy đo khả năng mô hình phân loại đúng các chữ số dựa trên đặc trưng pixel học được.  
            """, unsafe_allow_html=True)

    # Tab 2: Tải dữ liệu
    with tab_load:
        st.header("Tải Dữ liệu")
        if st.button("Tải dữ liệu MNIST từ OpenML"):
            with st.spinner("Đang tải dữ liệu từ OpenML..."):
                progress_bar = st.progress(0)
                status_text = st.empty()
                try:
                    mnist = openml.datasets.get_dataset(554)
                    progress_bar.progress(20)
                    status_text.text("Đã tải 20% - Đang lấy dữ liệu...")

                    X, y, _, _ = mnist.get_data(target=mnist.default_target_attribute)
                    progress_bar.progress(50)
                    status_text.text("Đã tải 50% - Đang xử lý dữ liệu...")

                    st.session_state['full_data'] = (X, y)
                    progress_bar.progress(90)
                    status_text.text(f"Đã tải 90% - Hoàn tất {X.shape[0]} mẫu...")

                    with mlflow.start_run(run_name="Data_Load"):
                        mlflow.log_param("total_samples", X.shape[0])

                    progress_bar.progress(100)
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
                with st.spinner(f"Đang lấy {num_samples} mẫu..."):
                    progress_bar = st.progress(0)
                    status_text = st.empty()

                    df = pd.concat([X_full, y_full.rename("label")], axis=1)
                    progress_bar.progress(30)
                    status_text.text("Đã xử lý 30% - Đang nối dữ liệu...")

                    sampled_df = df.sample(n=num_samples, random_state=42)
                    progress_bar.progress(70)
                    status_text.text("Đã xử lý 70% - Đang lấy mẫu...")

                    X_sampled = sampled_df.drop(columns=["label"])
                    y_sampled = sampled_df["label"]
                    st.session_state['data'] = (X_sampled, y_sampled)
                    progress_bar.progress(90)
                    status_text.text("Đã xử lý 90% - Đang lưu dữ liệu...")

                    with mlflow.start_run(run_name="Data_Sample"):
                        mlflow.log_param("num_samples", num_samples)

                    progress_bar.progress(100)
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
            X, y = st.session_state['data']
            if "data_original" not in st.session_state:
                st.session_state["data_original"] = (X.copy(), y.copy())

            # Xóa data_processed nếu không hợp lệ
            if "data_processed" in st.session_state:
                data_processed = st.session_state["data_processed"]
                if not (isinstance(data_processed, tuple) and len(data_processed) == 2):
                    st.session_state.pop("data_processed", None)

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
                    X_norm = X / 255.0
                    st.session_state["data_processed"] = (X_norm, y)
                    st.success("Đã chuẩn hóa dữ liệu!")
                    st.rerun()
            with col2:
                st.markdown("""
                    <div class="tooltip">
                        ?
                        <span class="tooltiptext">
                            Đưa dữ liệu về khoảng [0, 1] bằng cách chia cho 255.<br>
                            Công dụng: Đảm bảo thang đo đồng nhất, cải thiện hiệu suất của Neural Network.
                        </span>
                    </div>
                """, unsafe_allow_html=True)

            if "data_processed" in st.session_state:
                data_processed = st.session_state["data_processed"]
                if isinstance(data_processed, tuple) and len(data_processed) == 2:
                    try:
                        X_processed, y_processed = data_processed
                        st.subheader("Dữ liệu đã xử lý")
                        fig, axes = plt.subplots(2, 5, figsize=(10, 4))
                        for i, ax in enumerate(axes.flat):
                            ax.imshow(X_processed.iloc[i].values.reshape(28, 28), cmap='gray')
                            ax.set_title(f"Label: {y_processed.iloc[i]}")
                            ax.axis("off")
                        st.pyplot(fig)
                    except (ValueError, TypeError, AttributeError) as e:
                        st.error(f"Lỗi khi hiển thị dữ liệu đã xử lý: {e}. Vui lòng thử chuẩn hóa lại dữ liệu.")
                        st.session_state.pop("data_processed", None)
                else:
                    st.error("Dữ liệu đã xử lý không đúng định dạng. Vui lòng thử chuẩn hóa lại dữ liệu.")
                    st.session_state.pop("data_processed", None)
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
                
                test_size = int(total_samples * test_pct / 100)
                if test_size > 0:
                    X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=test_size / total_samples, random_state=42)
                else:
                    X_temp, y_temp = X, y
                    X_test, y_test = pd.DataFrame(), pd.Series()

                valid_size = int(len(X_temp) * valid_pct / 100)
                if valid_size > 0 and len(X_temp) > valid_size:
                    X_train, X_valid, y_train, y_valid = train_test_split(X_temp, y_temp, test_size=valid_size / len(X_temp), random_state=42)
                else:
                    X_train, y_train = X_temp, y_temp
                    X_valid, y_valid = pd.DataFrame(), pd.Series()

                st.write(f"Train: {len(X_train)} mẫu, Validation: {len(X_valid)} mẫu, Test: {len(X_test)} mẫu")
                if st.button("Xác nhận chia dữ liệu"):
                    st.session_state['split_data'] = {
                        "X_train": X_train, "y_train": y_train,
                        "X_valid": X_valid, "y_valid": y_valid,
                        "X_test": X_test, "y_test": y_test
                    }
                    st.success("Dữ liệu đã được chia!")

    # Tab 5: Huấn luyện/Đánh giá (Cập nhật chuyên nghiệp)
    with tab_train_eval:
        st.header("Huấn Luyện và Đánh Giá Mô Hình Neural Network")
        st.markdown("""
            <div style='text-align: justify;'>
                Phần này cho phép bạn huấn luyện mô hình <b>Mạng Nơ-ron Nhân tạo (Neural Network)</b> trên tập dữ liệu MNIST và đánh giá hiệu suất. 
                Chỉ cần thiết lập một vài tham số cơ bản, các cài đặt khác đã được tối ưu sẵn để phù hợp với bài toán phân loại chữ số viết tay.
            </div>
        """, unsafe_allow_html=True)

        if 'split_data' not in st.session_state:
            st.info("Vui lòng chia dữ liệu từ tab 'Chia dữ liệu' trước khi tiếp tục.")
        else:
            X_train = st.session_state['split_data']["X_train"]
            num_samples = len(X_train)
            st.markdown(f"""
                <div style='background-color: #f0f2f6; padding: 10px; border-radius: 5px;'>
                    <b>Số lượng mẫu huấn luyện:</b> {num_samples} mẫu
                </div>
            """, unsafe_allow_html=True)

            # Gợi ý tham số dựa trên số mẫu
            def suggest_parameters(num_samples):
                if num_samples < 1000:
                    return {"hidden_size": 64, "max_iter": 100, "lr": 0.01}
                elif 1000 <= num_samples <= 5000:
                    return {"hidden_size": 128, "max_iter": 200, "lr": 0.001}
                else:
                    return {"hidden_size": 256, "max_iter": 300, "lr": 0.001}

            suggested_params = suggest_parameters(num_samples)

            # Bảng gợi ý tham số
            st.subheader("📋 Gợi ý tham số tối ưu dựa trên số lượng mẫu")
            param_table = pd.DataFrame({
                "Tham số": ["Số nơ-ron lớp ẩn (hidden_size)", "Số lần huấn luyện tối đa (max_iter)", "Tốc độ học (learning_rate)"],
                "Giá trị gợi ý": [suggested_params["hidden_size"], suggested_params["max_iter"], suggested_params["lr"]],
                "Mô tả": [
                    "Số nơ-ron trong lớp ẩn, quyết định độ phức tạp của mô hình.",
                    "Số lần lặp tối đa để mô hình học dữ liệu.",
                    "Tốc độ cập nhật trọng số, ảnh hưởng đến sự ổn định và tốc độ học."
                ]
            })
            st.table(param_table)

            # Form nhập tham số
            st.subheader("⚙️ Thiết lập tham số mô hình")
            with st.form(key="training_form"):
                col1, col2, col3 = st.columns(3)
                with col1:
                    hidden_size = st.number_input(
                        "Số nơ-ron lớp ẩn",
                        min_value=10, max_value=500, value=suggested_params["hidden_size"], step=10,
                        help="Số lượng nơ-ron càng lớn, mô hình càng mạnh nhưng cần nhiều tài nguyên hơn."
                    )
                with col2:
                    max_iter = st.number_input(
                        "Số lần huấn luyện tối đa",
                        min_value=50, max_value=500, value=suggested_params["max_iter"], step=10,
                        help="Tăng số lần lặp để cải thiện độ chính xác, nhưng sẽ lâu hơn."
                    )
                with col3:
                    lr = st.selectbox(
                        "Tốc độ học",
                        options=[0.01, 0.001, 0.0001], index=[0.01, 0.001, 0.0001].index(suggested_params["lr"]),
                        help="Giá trị nhỏ hơn giúp mô hình học ổn định nhưng chậm hơn."
                    )
                submit_button = st.form_submit_button(label="Thực hiện Huấn luyện")

            # Huấn luyện mô hình
            if submit_button:
                with st.spinner("Đang huấn luyện mô hình Neural Network..."):
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    start_time = time.time()

                    X_train = st.session_state['split_data']["X_train"]
                    y_train = st.session_state['split_data']["y_train"]
                    X_valid = st.session_state['split_data']["X_valid"]
                    y_valid = st.session_state['split_data']["y_valid"]
                    X_test = st.session_state['split_data']["X_test"]
                    y_test = st.session_state['split_data']["y_test"]

                    # Định nghĩa pipeline
                    pipeline = Pipeline([
                        ('imputer', SimpleImputer(strategy='mean')),
                        ('classifier', MLPClassifier(
                            hidden_layer_sizes=(hidden_size,),
                            activation='relu',
                            solver='adam',
                            learning_rate_init=lr,
                            max_iter=max_iter,
                            random_state=42
                        ))
                    ])

                    # Tiến trình huấn luyện
                    for i in range(0, 51, 5):
                        progress_bar.progress(i)
                        status_text.text(f"Đang huấn luyện {i}%{i % 4 * '.'}")
                        time.sleep(0.1)

                    pipeline.fit(X_train, y_train)
                    model = pipeline

                    # Ghi log với MLflow
                    run_name = f"NeuralNetwork_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                    with mlflow.start_run(run_name=run_name) as run:
                        mlflow.log_param("hidden_size", hidden_size)
                        mlflow.log_param("max_iter", max_iter)
                        mlflow.log_param("learning_rate", lr)
                        mlflow.log_param("num_samples", num_samples)

                        # Đánh giá trên tập validation
                        y_valid_pred = model.predict(X_valid)
                        accuracy_val = accuracy_score(y_valid, y_valid_pred)
                        mlflow.log_metric("accuracy_val", accuracy_val)
                        cm_valid = confusion_matrix(y_valid, y_valid_pred)

                        for i in range(50, 76, 5):
                            progress_bar.progress(i)
                            status_text.text(f"Đang đánh giá validation {i}%{i % 4 * '.'}")
                            time.sleep(0.1)

                        # Đánh giá trên tập test
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
                            'model_choice': 'Neural Network',
                            'params': {'hidden_size': hidden_size, 'max_iter': max_iter, 'learning_rate': lr},
                            'num_samples': num_samples,
                            'run_name': run_name,
                            'run_id': run_id
                        }

                    status_text.empty()
                    progress_bar.empty()

            # Hiển thị kết quả huấn luyện
            if 'training_results' in st.session_state and st.session_state['training_results']['model_choice'] == 'Neural Network':
                st.subheader("📊 Kết Quả Huấn Luyện")
                results = st.session_state['training_results']
                st.markdown(f"""
                    <div style='background-color: #e6ffe6; padding: 15px; border-radius: 5px;'>
                        <b>Trạng thái:</b> Huấn luyện hoàn tất thành công!<br>
                        <b>Thời gian huấn luyện:</b> {results['training_time']:.2f} giây<br>
                        <b>Độ chính xác (Validation):</b> {results['accuracy_val']:.4f} ({results['accuracy_val']*100:.2f}%)<br>
                        <b>Độ chính xác (Test):</b> {results['accuracy_test']:.4f} ({results['accuracy_test']*100:.2f}%)<br>
                        <b>Tên lần chạy:</b> {results['run_name']}<br>
                        <b>ID lần chạy:</b> {results['run_id']}
                    </div>
                """, unsafe_allow_html=True)

                st.subheader("📉 Ma Trận Nhầm Lẫn")
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("**Tập Validation**")
                    fig, ax = plt.subplots(figsize=(5, 4))
                    sns.heatmap(results['cm_valid'], annot=True, fmt="d", cmap="Blues", ax=ax)
                    ax.set_xlabel("Dự đoán")
                    ax.set_ylabel("Thực tế")
                    st.pyplot(fig)
                with col2:
                    st.markdown("**Tập Test**")
                    fig, ax = plt.subplots(figsize=(5, 4))
                    sns.heatmap(results['cm_test'], annot=True, fmt="d", cmap="Blues", ax=ax)
                    ax.set_xlabel("Dự đoán")
                    ax.set_ylabel("Thực tế")
                    st.pyplot(fig)

                st.subheader("Thông tin Kết quả")
                with st.expander("Xem chi tiết kết quả", expanded=True):
                    st.markdown("#### Thông tin lần chạy:", unsafe_allow_html=True)
                    st.write(f"- **Tên lần chạy (Run Name)**: {results['run_name']}")
                    st.write(f"- **ID lần chạy (Run ID)**: {results['run_id']}")

                    st.markdown("#### Cài đặt bạn đã chọn:", unsafe_allow_html=True)
                    st.write(f"- **Mô hình**: Neural Network")
                    st.write(f"- **Tham số**:")
                    for key, value in results['params'].items():
                        st.write(f"  - {key}: {value}")
                    st.write(f"- **Thời gian chạy**: {results['training_time']:.2f} giây")
                    st.write(f"- **Số mẫu huấn luyện**: {results['num_samples']}")

                    st.markdown("#### Kết quả đạt được:", unsafe_allow_html=True)
                    st.markdown(f"""
                    - **Độ chính xác Validation**: {results['accuracy_val']*100:.2f}%  
                    - **Độ chính xác Test**: {results['accuracy_test']*100:.2f}%  
                    """, unsafe_allow_html=True)
            else:
                st.info("Chưa có kết quả huấn luyện cho mô hình này. Vui lòng nhấn 'Thực hiện Huấn luyện' để xem kết quả.")

    # Tab 6: Demo dự đoán
    with tab_demo:
        st.header("Demo Dự đoán")
        st.markdown("""
        Phần này cho phép bạn thử nghiệm dự đoán với dữ liệu Test, ảnh upload, hoặc số bạn vẽ.
        """, unsafe_allow_html=True)

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
                        
                        st.success(f"Dự đoán: **{prediction}** | Độ tin cậy: **{confidence:.2f}%** | Giá trị thực: **{y_true}**")
                        fig, ax = plt.subplots()
                        ax.imshow(X_test.iloc[idx].values.reshape(28, 28), cmap='gray')
                        ax.axis("off")
                        st.pyplot(fig)
                        
                        time.sleep(1)
                        progress_bar.empty()
                        status_text.empty()

            elif mode == "Upload ảnh mới":
                uploaded_images = st.file_uploader("Upload ảnh ($28\\times28$, grayscale)", type=["png", "jpg"], accept_multiple_files=True)
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
                            
                            st.success(f"Dự đoán: **{prediction}** | Độ tin cậy: **{confidence:.2f}%**")
                            st.image(img, caption=f"Ảnh {i+1} được upload", use_container_width=True)
                            
                            time.sleep(1)
                            progress_bar.empty()
                            status_text.empty()

            elif mode == "Vẽ số":
                st.write("Vẽ một chữ số từ $0$-$9$ trên canvas bên dưới ($28\\times28$ pixel):")
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
                            
                            img = Image.fromarray((canvas_result.image_data * 255).astype(np.uint8)).convert('L').resize((28, 28))
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
                            
                            st.success(f"Dự đoán: **{prediction}** | Độ tin cậy: **{confidence:.2f}%**")
                            
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
            experiment = client.get_experiment_by_name("MNIST_NeuralNetwork")
            if not experiment:
                st.error("Không tìm thấy experiment 'MNIST_NeuralNetwork'. Vui lòng kiểm tra lại MLflow tracking URI.")
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
            mlflow_url = "https://dagshub.com/huykibo/streamlit_mlflow.mlflow"  # Cập nhật URL nếu cần
            if st.button("Mở MLflow UI trên Dagshub"):
                st.markdown(f'[Click để mở MLflow UI]({mlflow_url})', unsafe_allow_html=True)
        except Exception as e:
            st.error(f"Lỗi kết nối MLflow: {e}. Vui lòng kiểm tra MLFLOW_TRACKING_URI và thông tin xác thực.")

if __name__ == "__main__":
    run_mnist_neural_network_app()