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
        </style>
    """, unsafe_allow_html=True)

    tabs = st.tabs(["Thông tin", "Tải dữ liệu", "Xử lý dữ liệu", "Chia dữ liệu", "Huấn luyện/Đánh giá", "Demo dự đoán", "Thông tin huấn luyện"])
    tab_info, tab_load, tab_preprocess, tab_split, tab_train_eval, tab_demo, tab_log_info = tabs

    with tab_info:
        st.header("Giới thiệu về Ứng dụng và Các Mô hình Phân loại")
        st.markdown("""
        Chào bạn! Đây là ứng dụng phân loại chữ số viết tay từ tập dữ liệu **MNIST** bằng **Decision Tree** và **SVM**. Hãy khám phá cách hoạt động của chúng nhé!
        """, unsafe_allow_html=True)

        info_option = st.selectbox(
            "",
            [
                "Ứng dụng này là gì và mục tiêu của nó?",
                "Tập dữ liệu MNIST: Đặc điểm và ý nghĩa",
                "Decision Tree – Cây Quyết định",
                "SVM – Máy Vector Hỗ trợ",
                "So sánh Decision Tree và SVM",
                "Công thức đánh giá độ chính xác (Accuracy)"
            ],
            label_visibility="collapsed",
            help="Chọn để xem chi tiết về ứng dụng, dữ liệu hoặc mô hình."
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
                Đây là ứng dụng phân loại chữ số viết tay dựa trên tập dữ liệu **MNIST**, sử dụng hai mô hình: **Decision Tree** và **SVM**.  
                - **MNIST**: Tập dữ liệu gồm $70,000$ ảnh chữ số từ $0$ đến $9$, mỗi ảnh kích thước $28 \\times 28$ pixel (tổng cộng $784$ đặc trưng).  
                - **Mục tiêu**:  
                  - Xây dựng và huấn luyện các mô hình để nhận diện chính xác các chữ số.  
                  - So sánh hiệu quả của Decision Tree và SVM trên bài toán này.  
                  - Cung cấp công cụ trực quan để học tập và thử nghiệm.  

                **Thông tin cơ bản**:  
                - **$784$ đặc trưng**: Mỗi ảnh là vector $784$ chiều (giá trị pixel từ $0$ đến $255$).  
                - **$70,000$ mẫu**: Tổng số ảnh, được chia thành tập huấn luyện, kiểm tra và xác thực.  
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
                try:
                    mnist_image = Image.open("mnist.png")
                    st.image(mnist_image, caption="Ảnh minh họa $10$ chữ số từ $0$ đến $9$ trong MNIST", width=800)
                except FileNotFoundError:
                    st.error("Không tìm thấy file `mnist.png`.")
                status_text.text("Đã tải xong! 100%")
                time.sleep(0.5)
                status_text.empty()
                progress_bar.empty()

        elif info_option == "Decision Tree – Cây Quyết định":
            with st.spinner("Đang tải thông tin..."):
                progress_bar = st.progress(0)
                status_text = st.empty()
                for i in range(0, 101, 10):
                    progress_bar.progress(i)
                    status_text.text(f"Đang tải thông tin... {i}%")
                    time.sleep(0.05)
                st.subheader("📘 3. Decision Tree – Cây Quyết định")
                st.markdown("""
                **Decision Tree (Cây Quyết định)** là mô hình phân loại xây dựng một cây phân cấp, trong đó mỗi nút đại diện cho một quyết định dựa trên đặc trưng (pixel trong MNIST) để phân loại dữ liệu thành các nhãn ($0$-$9$).  
                """, unsafe_allow_html=True)
                st.subheader("🔧 Quy trình hoạt động")
                st.markdown("""
                1. **Khởi tạo cây**: Bắt đầu từ nút gốc chứa toàn bộ dữ liệu.  
                2. **Chia nhánh**: Chọn đặc trưng (pixel) và ngưỡng tối ưu dựa trên tiêu chí như **Gini** hoặc **Entropy**.  
                3. **Lặp lại**: Tiếp tục chia cho đến khi đạt độ sâu tối đa (**Max Depth**) hoặc dữ liệu thuần nhất.  
                4. **Dự đoán**: Đi qua các nhánh để đến nút lá, trả về nhãn.  

                **Công thức**:  
                - **Gini**: $$ Gini = 1 - \\sum_{i=0}^{9} p_i^2 $$  
                - **Entropy**: $$ Entropy = -\\sum_{i=0}^{9} p_i \\log_2(p_i) $$  
                - $p_i$: Tỷ lệ mẫu thuộc lớp $i$.  

                **Tham số chính**:  
                - **Max Depth**: Độ sâu tối đa của cây, kiểm soát độ phức tạp (từ $5$ đến $30$).  
                - **Criterion**: Tiêu chí chia nhánh (Gini hoặc Entropy).  
                """, unsafe_allow_html=True)
                st.subheader("🟪 Ưu điểm và nhược điểm")
                st.markdown("""
                - **✅ Ưu điểm**: Dễ hiểu, nhanh với dữ liệu nhỏ, không cần chuẩn hóa.  
                - **❌ Nhược điểm**: Dễ bị overfitting nếu **Max Depth** lớn, kém hiệu quả với dữ liệu phức tạp như MNIST.  
                """, unsafe_allow_html=True)
                status_text.text("Đã tải xong! 100%")
                time.sleep(0.5)
                status_text.empty()
                progress_bar.empty()

        elif info_option == "SVM – Máy Vector Hỗ trợ":
            with st.spinner("Đang tải thông tin..."):
                progress_bar = st.progress(0)
                status_text = st.empty()
                for i in range(0, 101, 10):
                    progress_bar.progress(i)
                    status_text.text(f"Đang tải thông tin... {i}%")
                    time.sleep(0.05)
                st.subheader("📘 4. SVM – Máy Vector Hỗ trợ")
                st.markdown("""
                **SVM (Support Vector Machine)** tìm siêu phẳng tối ưu trong không gian $784$ chiều để phân tách các lớp ($0$-$9$). Nếu dữ liệu không tuyến tính, nó dùng **kernel** để ánh xạ lên không gian cao hơn.  
                """, unsafe_allow_html=True)
                st.subheader("🔧 Quy trình hoạt động")
                st.markdown("""
                1. **Tìm siêu phẳng**: $$ f(x) = w \\cdot x + b $$  
                   - $w$: Vector trọng số, $x$: Đặc trưng, $b$: Bias.  
                2. **Tối ưu lề**: $$ \\min \\frac{1}{2} \\|w\\|^2 + C \\sum \\xi_i $$  
                   - $C$: Tham số cân bằng lề và lỗi.  
                   - $\\xi_i$: Sai số cho phép (soft margin).  
                3. **Kernel**: Ví dụ RBF: $$ K(x_i, x_j) = \\exp(-\\gamma \\|x_i - x_j\\|^2) $$  
                4. **Dự đoán**: Dựa trên khoảng cách tới siêu phẳng.  

                **Tham số chính**:  
                - **C**: Độ nghiêm ngặt phân loại (từ $0.1$ đến $10$).  
                - **Kernel**: Loại ánh xạ (linear, rbf, poly).  
                """, unsafe_allow_html=True)
                st.subheader("🟪 Ưu điểm và nhược điểm")
                st.markdown("""
                - **✅ Ưu điểm**: Chính xác cao với dữ liệu phức tạp, hiệu quả với kernel phù hợp.  
                - **❌ Nhược điểm**: Chậm với dữ liệu lớn, cần chuẩn hóa dữ liệu.  
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
                | **Tiêu chí**          | **Decision Tree**                     | **SVM**                           |
                |-----------------------|---------------------------------------|-----------------------------------|
                | **Phương pháp**       | Chia dữ liệu theo điều kiện logic     | Tìm siêu phẳng tối ưu            |
                | **Tham số chính**     | Max Depth, Criterion                  | C, Kernel                        |
                | **Tốc độ**           | Nhanh với dữ liệu nhỏ                 | Chậm với dữ liệu lớn             |
                | **Chuẩn hóa**         | Không cần                             | Cần                              |
                | **Hiệu quả**          | Dễ overfitting, kém với dữ liệu phức tạp | Chính xác cao, tốt với dữ liệu phức tạp |

                **Kết luận**: Decision Tree nhanh và đơn giản, SVM mạnh mẽ hơn nhưng tốn tài nguyên.  
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
                $$ \\text{Accuracy} = \\frac{\\text{Số mẫu dự đoán đúng}}{\\text{Tổng số mẫu}} $$  
                - **Ví dụ**: Dự đoán đúng $92/100$ ảnh → $Accuracy = 0.92$ (92%).  
                - **Ý nghĩa**: Đo khả năng phân loại đúng của mô hình trên tập kiểm tra.  
                """, unsafe_allow_html=True)
                status_text.text("Đã tải xong! 100%")
                time.sleep(0.5)
                status_text.empty()
                progress_bar.empty()

    with tab_load:
        st.markdown('<div class="section-title">Tải và Chuẩn bị Dữ liệu</div>', unsafe_allow_html=True)
        st.markdown("""
        **Tập dữ liệu MNIST**: Gồm $70,000$ ảnh chữ số ($0$-$9$), mỗi ảnh $28 \\times 28$ pixel. Chọn số lượng mẫu để huấn luyện.  
        """, unsafe_allow_html=True)

        with st.container():
            st.subheader("Tải dữ liệu")
            if st.button("Tải dữ liệu MNIST từ OpenML", type="primary"):
                with st.spinner("Đang tải dữ liệu..."):
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
                        st.success("Tải dữ liệu thành công!")
                        st.write(f"Kích thước dữ liệu: {X.shape[0]} mẫu, {X.shape[1]} đặc trưng")
                        status_text.text("Đã tải xong! 100%")
                        time.sleep(0.5)
                        status_text.empty()
                        progress_bar.empty()

        if 'full_data' in st.session_state:
            X_full, y_full = st.session_state['full_data']
            st.subheader("Chọn số lượng mẫu")
            st.markdown("""
            - **100 mẫu**: Thử nghiệm nhanh (~vài giây).  
            - **1,000 mẫu**: Kiểm tra cơ bản (~10-20 giây).  
            - **10,000 mẫu**: Cân bằng hiệu suất (~1-2 phút).  
            - **50,000 mẫu**: Huấn luyện chuyên sâu (~5-10 phút).  
            """, unsafe_allow_html=True)

            col1, col2 = st.columns(2)
            with col1:
                sample_options = {
                    "100 mẫu (Thử nghiệm nhanh)": 100,
                    "1,000 mẫu (Kiểm tra cơ bản)": 1000,
                    "10,000 mẫu (Cân bằng hiệu suất)": 10000,
                    "50,000 mẫu (Huấn luyện chuyên sâu)": 50000
                }
                selected_option = st.selectbox("Chọn số lượng mẫu:", list(sample_options.keys()))
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
                        with mlflow.start_run(experiment_id=EXPERIMENT_ID, run_name="Data_Sample"):
                            mlflow.log_param("num_samples", num_samples)
                        st.success(f"Đã chọn {num_samples} mẫu!")
                        status_text.text("Đã xử lý xong! 100%")
                        time.sleep(0.5)
                        status_text.empty()
                        progress_bar.empty()

            with col2:
                custom_num_samples = st.number_input("Nhập số lượng tùy ý (tối đa $70,000$):", min_value=1, max_value=70000, value=1000, step=100)
                if st.button("Xác nhận số lượng (tùy ý)", type="primary"):
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
                        with mlflow.start_run(experiment_id=EXPERIMENT_ID, run_name="Data_Sample_Custom"):
                            mlflow.log_param("num_samples", custom_num_samples)
                        st.success(f"Đã chọn {custom_num_samples} mẫu!")
                        status_text.text("Đã xử lý xong! 100%")
                        time.sleep(0.5)
                        status_text.empty()
                        progress_bar.empty()

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
                if st.button("Chuẩn hóa dữ liệu (Normalization)", type="primary"):
                    with st.spinner("Đang chuẩn hóa dữ liệu về [0, 1]..."):
                        progress_bar = st.progress(0)
                        status_text = st.empty()
                        for i in range(0, 101, 20):
                            progress_bar.progress(i)
                            status_text.text(f"Đang chuẩn hóa dữ liệu... {i}%")
                            time.sleep(0.1)
                        X_norm = X / 255.0
                        st.session_state["data_processed"] = (X_norm, y)
                        st.success("Đã chuẩn hóa dữ liệu về [0, 1]!")
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
                            Công dụng: Đảm bảo thang đo đồng nhất, đặc biệt cần cho SVM.
                        </span>
                    </div>
                """, unsafe_allow_html=True)

            if "data_processed" in st.session_state:
                X_processed, y_processed = st.session_state["data_processed"]
                st.subheader("Dữ liệu đã xử lý")
                fig, axes = plt.subplots(2, 5, figsize=(10, 4))
                for i, ax in enumerate(axes.flat):
                    ax.imshow(X_processed[i].reshape(28, 28), cmap='gray')
                    ax.set_title(f"Label: {y_processed[i]}")
                    ax.axis("off")
                st.pyplot(fig)

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
                test_pct = st.slider("Tỷ lệ Test (%)", 0, 50, 20)
            with col2:
                valid_pct = st.slider("Tỷ lệ Validation (%)", 0, 50, 20)

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

            X_train = np.array(X_train, dtype=np.float64)
            y_train = np.array(y_train, dtype=np.int32)
            X_valid = np.array(X_valid, dtype=np.float64)
            y_valid = np.array(y_valid, dtype=np.int32)
            X_test = np.array(X_test, dtype=np.float64)
            y_test = np.array(y_test, dtype=np.int32)

            num_samples = len(X_train)
            st.write(f"**Số mẫu huấn luyện**: {num_samples}")

            model_choice = st.selectbox("Chọn mô hình:", ["Decision Tree", "SVM"])

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

            st.subheader("⚙️ Cấu hình tham số mô hình")
            st.markdown("""
            Dưới đây là bảng tham số tối ưu dựa trên số mẫu huấn luyện:
            """, unsafe_allow_html=True)
            if model_choice == "Decision Tree":
                st.markdown("""
                | Số mẫu       | Criterion | Max Depth |
                |--------------|-----------|-----------|
                | $\\leq 1,000$| gini      | $5$       |
                | $1,000-5,000$| gini      | $10$      |
                | $5,000-50,000$| gini     | $20$      |
                | $>50,000$    | gini      | $30$      |
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                | Số mẫu       | C     | Kernel |
                |--------------|-------|--------|
                | $\\leq 1,000$| $0.1$ | rbf    |
                | $1,000-5,000$| $1.0$ | rbf    |
                | $5,000-50,000$| $5.0$| rbf    |
                | $>50,000$    | $10.0$| rbf    |
                """, unsafe_allow_html=True)

            st.info(f"Tham số tối ưu cho {num_samples} mẫu: {st.session_state[f'optimal_params_{model_choice}']}")

            col_param1, col_param2 = st.columns(2)
            with col_param1:
                with st.expander("Cấu trúc mô hình"):
                    if model_choice == "Decision Tree":
                        params["criterion"] = st.selectbox("Criterion", ["gini", "entropy"], index=["gini", "entropy"].index(params["criterion"]))
                        params["max_depth"] = st.number_input("Max Depth", min_value=1, max_value=100, value=params["max_depth"])
                    else:
                        params["C"] = st.number_input("C", min_value=0.01, max_value=100.0, value=params["C"])
                        params["kernel"] = st.selectbox("Kernel", ["linear", "rbf", "poly", "sigmoid"], index=["linear", "rbf", "poly", "sigmoid"].index(params["kernel"]))
            with col_param2:
                if st.button("🔄 Khôi phục tham số tối ưu"):
                    st.session_state[f"training_params_{model_choice}"] = st.session_state[f"optimal_params_{model_choice}"].copy()
                    st.success("Đã khôi phục tham số tối ưu!")
                    st.rerun()

            st.session_state[f"training_params_{model_choice}"] = params

            if st.button("🚀 Bắt đầu Huấn luyện", type="primary"):
                with st.spinner("Đang huấn luyện mô hình..."):
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    start_time = time.time()

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
                    time.sleep(0.5)
                    status_text.empty()
                    progress_bar.empty()
                    st.rerun()

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

                st.subheader("ℹ️ Thông tin Chi tiết")
                with st.expander("Xem chi tiết", expanded=False):
                    st.markdown("**Thông tin lần chạy:**")
                    st.write(f"- Tên: {results['run_name']}")
                    st.write(f"- ID: {results['run_id']}")
                    st.write(f"- Thời gian huấn luyện: {results['training_time']:.2f} giây")
                    st.write(f"- Độ chính xác Validation: {results['accuracy_val']*100:.2f}%")
                    st.write(f"- Độ chính xác Test: {results['accuracy_test']*100:.2f}%")
                    st.markdown("**Tham số đã chọn:**")
                    st.json(results['params'])

    with tab_demo:
        st.markdown('<div class="section-title">Demo Dự đoán Chữ số</div>', unsafe_allow_html=True)
        if 'split_data' not in st.session_state or 'model' not in st.session_state:
            st.info("Vui lòng huấn luyện mô hình trước khi sử dụng Demo.")
        else:
            model_choice = st.session_state['training_results']['model_choice']
            st.write(f"**Mô hình hiện tại**: {model_choice}")

            mode = st.selectbox("Chọn phương thức dự đoán:", ["Dữ liệu Test", "Upload ảnh", "Vẽ số"])

            def preprocess_input(data, is_normalized):
                data, fixed = validate_and_fix_pixels(data)
                if fixed:
                    st.success("Đã chuẩn hóa dữ liệu về [0, 255]!")
                if not is_normalized:
                    data = data / 255.0
                return data

            is_normalized = 'data_processed' in st.session_state
            model = st.session_state['model']

            if mode == "Dữ liệu Test":
                st.subheader("Dự đoán từ Dữ liệu Test")
                X_test = st.session_state['split_data']["X_test"]
                y_test = st.session_state['split_data']["y_test"]
                if len(X_test) == 0:
                    st.warning("Tập Test rỗng. Vui lòng chia lại dữ liệu với tỷ lệ Test > 0%.")
                else:
                    col_select, col_display = st.columns([3, 2])
                    with col_select:
                        idx = st.slider("Chọn mẫu Test", 0, len(X_test) - 1, 0)
                    with col_display:
                        st.write("**Ảnh mẫu Test:**")
                        fig, ax = plt.subplots(figsize=(2, 2))
                        ax.imshow(X_test[idx].reshape(28, 28), cmap='gray')
                        ax.axis('off')
                        st.pyplot(fig)
                        st.write(f"**Nhãn thực tế:** {y_test[idx]}")

                    if st.button("🔍 Dự đoán", key="predict_test"):
                        with st.spinner("Đang dự đoán..."):
                            progress_bar = st.progress(0)
                            status_text = st.empty()
                            for i in range(0, 101, 20):
                                progress_bar.progress(i)
                                status_text.text(f"Đang dự đoán mẫu Test... {i}%")
                                time.sleep(0.1)
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
                            status_text.text("Đã dự đoán xong! 100%")
                            time.sleep(0.5)
                            status_text.empty()
                            progress_bar.empty()

            elif mode == "Upload ảnh":
                st.subheader("Dự đoán từ Ảnh Tải lên")
                uploaded_images = st.file_uploader("Chọn ảnh (PNG/JPG)", type=["png", "jpg"], accept_multiple_files=True)
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
                                        progress_bar = st.progress(0)
                                        status_text = st.empty()
                                        for j in range(0, 101, 20):
                                            progress_bar.progress(j)
                                            status_text.text(f"Đang xử lý ảnh {i+1}... {j}%")
                                            time.sleep(0.1)
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
                                        status_text.text(f"Đã dự đoán xong ảnh {i+1}! 100%")
                                        time.sleep(0.5)
                                        status_text.empty()
                                        progress_bar.empty()
                        except Exception as e:
                            st.error(f"Lỗi khi xử lý ảnh {i+1}: {e}")

            elif mode == "Vẽ số":
                st.subheader("Dự đoán từ Hình vẽ")
                canvas_result = st_canvas(fill_color="black", stroke_width=20, stroke_color="white", 
                                          background_color="black", width=280, height=280, drawing_mode="freedraw", key="canvas")
                col1, col2 = st.columns([1, 1])
                with col1:
                    if st.button("Dự đoán số đã vẽ"):
                        if canvas_result.image_data is not None and np.any(canvas_result.image_data):
                            with st.spinner("Đang xử lý..."):
                                progress_bar = st.progress(0)
                                status_text = st.empty()
                                for i in range(0, 101, 20):
                                    progress_bar.progress(i)
                                    status_text.text(f"Đang xử lý hình vẽ... {i}%")
                                    time.sleep(0.1)
                                img = Image.fromarray((canvas_result.image_data * 255).astype(np.uint8)).convert('L').resize((28, 28))
                                img_array = np.array(img).flatten().reshape(1, -1)
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
                                st.image(img, caption="Hình vẽ của bạn")
                                status_text.text("Đã dự đoán xong! 100%")
                                time.sleep(0.5)
                                status_text.empty()
                                progress_bar.empty()
                        else:
                            st.warning("Vui lòng vẽ trước!")
                with col2:
                    if st.button("Xóa Canvas"):
                        st.session_state['canvas_key'] = st.session_state.get('canvas_key', 0) + 1
                        st.rerun()

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
    run_mnist_classification_app()