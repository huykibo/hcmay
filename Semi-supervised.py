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
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Flatten, Dropout
from keras.utils import to_categorical
from matplotlib import font_manager

# Cài đặt font chuyên nghiệp (Roboto nếu có, nếu không dùng mặc định)
try:
    font_path = font_manager.findfont(font_manager.FontProperties(family='Roboto'))
    plt.rcParams['font.family'] = 'Roboto'
except:
    plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.size'] = 12
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12

# Hàm tạo ảnh minh họa chuyên nghiệp (giữ nguyên)
def create_illustration(step, X, y, save_path="plnw/plnw_step.png"):
    global X_labeled, y_labeled
    fig = plt.figure(figsize=(12, 8), facecolor='#f7f7f7')
    ax = plt.gca()
    ax.set_facecolor('#ffffff')
    
    plt.suptitle(f"Bước {step}", fontsize=20, fontweight='bold', color='#2c3e50', y=1.05)
    plt.title("Pseudo Labeling trên MNIST", fontsize=16, color='#34495e', pad=10)

    if step == 0:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        sizes = [len(X_train), len(X_test)]
        labels = ['Train (80%)', 'Test (20%)']
        colors = ['#1abc9c', '#3498db']
        explode = (0.05, 0)
        plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90, 
                textprops={'fontsize': 14, 'color': '#2c3e50'}, explode=explode, shadow=True)
        plt.axis('equal')
        plt.figtext(0.5, 0.01, "Chia dữ liệu thành tập huấn luyện (80%) và kiểm tra (20%)", 
                    ha='center', fontsize=12, color='#7f8c8d', bbox=dict(facecolor='white', alpha=0.8))

    elif step == 1:
        X_labeled = pd.DataFrame()
        y_labeled = pd.Series(dtype='str')
        for digit in range(10):
            X_digit = X[y == str(digit)]
            n_labeled = max(1, int(len(X_digit) * 0.01))
            X_labeled_digit = X_digit.sample(n=n_labeled, random_state=42)
            y_labeled_digit = pd.Series([str(digit)] * n_labeled)
            X_labeled = pd.concat([X_labeled, X_labeled_digit])
            y_labeled = pd.concat([y_labeled, y_labeled_digit])
        
        samples_per_class = y_labeled.value_counts().sort_index()
        bars = plt.bar(samples_per_class.index, samples_per_class.values, color='#e67e22', edgecolor='#d35400', linewidth=1.5)
        plt.xlabel('Chữ số (0-9)', fontsize=14, color='#2c3e50')
        plt.ylabel('Số lượng mẫu (1%)', fontsize=14, color='#2c3e50')
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2, height + 5, int(height), 
                     ha='center', fontsize=12, color='#2c3e50')
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.figtext(0.5, 0.01, "Lấy 1% dữ liệu có nhãn từ mỗi lớp (0-9) làm tập huấn luyện ban đầu", 
                    ha='center', fontsize=12, color='#7f8c8d', bbox=dict(facecolor='white', alpha=0.8))

    elif step == 2:
        if X_labeled is None or y_labeled is None:
            print("Lỗi: X_labeled hoặc y_labeled chưa được định nghĩa từ bước 1.")
            return None
        layers = ['Input\n(28x28)', 'Flatten', 'Dense\n(128, ReLU)', 'Dropout\n(0.3)', 'Output\n(10, Softmax)']
        x = range(len(layers))
        y = [1] * len(layers)
        plt.plot(x, y, 'o-', color='#e74c3c', linewidth=3, markersize=10, markerfacecolor='white', markeredgewidth=2)
        for i, layer in enumerate(layers):
            plt.text(i, 1.15, layer, ha='center', fontsize=12, color='#2c3e50', bbox=dict(facecolor='white', edgecolor='#bdc3c7', alpha=0.9))
        plt.ylim(0.5, 1.5)
        plt.yticks([])
        plt.xticks([])
        plt.figtext(0.5, 0.01, "Huấn luyện mạng nơ-ron trên 1% dữ liệu có nhãn ban đầu", 
                    ha='center', fontsize=12, color='#7f8c8d', bbox=dict(facecolor='white', alpha=0.8))

    elif step == 3:
        if X_labeled is None:
            print("Lỗi: X_labeled chưa được định nghĩa từ bước 1.")
            return None
        X_unlabeled = X[~X.index.isin(X_labeled.index)].sample(frac=1)
        plt.imshow(X_unlabeled.iloc[0].values.reshape(28, 28), cmap='gray')
        plt.axis('off')
        plt.figtext(0.5, 0.01, "Dự đoán nhãn cho 99% dữ liệu chưa gán nhãn bằng mô hình đã huấn luyện", 
                    ha='center', fontsize=12, color='#7f8c8d', bbox=dict(facecolor='white', alpha=0.8))
        plt.text(14, 32, "Dự đoán: ?", ha='center', fontsize=14, color='#e74c3c', fontweight='bold')

    elif step == 4:
        threshold = 0.95
        bars = plt.bar(['Dưới ngưỡng', 'Trên ngưỡng'], [30, 70], color=['#e74c3c', '#2ecc71'], edgecolor='#2c3e50', linewidth=1.5)
        plt.ylabel('Tỷ lệ mẫu (%)', fontsize=14, color='#2c3e50')
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2, height + 2, f'{int(height)}%', 
                     ha='center', fontsize=12, color='#2c3e50')
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.figtext(0.5, 0.01, f"Gán nhãn giả cho các mẫu có độ tin cậy trên ngưỡng {threshold}", 
                    ha='center', fontsize=12, color='#7f8c8d', bbox=dict(facecolor='white', alpha=0.8))

    elif step == 5:
        if X_labeled is None:
            print("Lỗi: X_labeled chưa được định nghĩa từ bước 1.")
            return None
        X_unlabeled = X[~X.index.isin(X_labeled.index)]
        X_new_labeled = pd.concat([X_labeled, X_unlabeled.sample(frac=0.01, random_state=42)])
        sizes = [len(X_labeled), len(X_new_labeled) - len(X_labeled)]
        labels = ['1% ban đầu', 'Nhãn giả mới']
        colors = ['#3498db', '#9b59b6']
        explode = (0.05, 0)
        plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90, 
                textprops={'fontsize': 14, 'color': '#2c3e50'}, explode=explode, shadow=True)
        plt.axis('equal')
        plt.figtext(0.5, 0.01, "Cập nhật tập dữ liệu bằng cách thêm nhãn giả vào tập huấn luyện", 
                    ha='center', fontsize=12, color='#7f8c8d', bbox=dict(facecolor='white', alpha=0.8))

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    save_path = save_path.replace("step", f"step{step}")
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close()

    return save_path

def run_mnist_pseudo_labeling_app():
    # Thiết lập MLflow (bỏ qua nếu không cần)
    mlflow.set_tracking_uri("file:./mlruns")  # Sử dụng local storage
    mlflow.set_experiment("MNIST_Pseudo_Labeling")

    st.title("Ứng dụng Pseudo Labeling với Neural Network trên MNIST")

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
    tabs = st.tabs(["Thông tin", "Tải dữ liệu", "Xử lý dữ liệu", "Chia dữ liệu", "Huấn luyện/Đánh Giá", "Demo dự đoán", "Thông tin huấn luyện"])
    tab_info, tab_load, tab_preprocess, tab_split, tab_train_eval, tab_demo, tab_log_info = tabs

    # Tab 1: Thông tin (Cập nhật nội dung mới về Neural Network)
    with tab_info:
        st.header("Giới thiệu về Neural Network")
        info_option = st.selectbox(
            "Chọn thông tin để xem:",
            [
                "Neural Network là gì?",
                "Sự khác biệt giữa não bộ và máy tính truyền thống",
                "Tại sao cần Neural Network?",
                "Các bước thực hiện và công thức trong Neural Network"
            ],
            index=0,
            key="info_selectbox"
        )

        content_placeholder = st.empty()

        with content_placeholder.container():
            if info_option == "Neural Network là gì?":
                st.subheader("1. Neural Network là gì?")
                st.markdown("""
                **Neural Network (Mạng nơ-ron nhân tạo)** là một hệ thống tính toán lấy cảm hứng từ cách hoạt động của các nơ-ron trong bộ não con người. Nó bao gồm các **node** (nút) được tổ chức thành các lớp:  
                - **Input Layer**: Lớp đầu vào nhận dữ liệu.  
                - **Hidden Layers**: Các lớp ẩn xử lý thông tin.  
                - **Output Layer**: Lớp đầu ra đưa ra dự đoán.  

                Mỗi node nhận dữ liệu đầu vào, thực hiện tính toán (tổng tuyến tính và áp dụng hàm kích hoạt), rồi truyền kết quả sang các node khác. Mục tiêu là mô phỏng khả năng học hỏi và xử lý thông tin của não bộ để giải quyết các bài toán phức tạp mà các phương pháp truyền thống khó thực hiện.
                """)

            elif info_option == "Sự khác biệt giữa não bộ và máy tính truyền thống":
                st.subheader("2. Sự khác biệt giữa não bộ và máy tính truyền thống")
                st.markdown("""
                Tại sao chó có thể nhận diện người quen, trẻ em phân biệt được động vật, nhưng máy tính lại gặp khó khăn với những việc này? Sự khác biệt nằm ở **cấu trúc và cách xử lý thông tin**:  

                - **Bộ não**:  
                  - Có hàng tỷ nơ-ron kết nối linh hoạt qua các synap.  
                  - Xử lý thông tin song song, nhận diện mẫu (pattern recognition), và học hỏi từ kinh nghiệm.  
                  - Giỏi giải quyết các bài toán không tuyến tính, phức tạp một cách tự nhiên.  

                - **Máy tính truyền thống**:  
                  - Hoạt động tuần tự, dựa trên các quy tắc logic cố định (rule-based).  
                  - Giỏi tính toán chính xác và nhanh với dữ liệu có cấu trúc, nhưng kém trong việc xử lý dữ liệu không rõ ràng hoặc nhận diện mẫu phức tạp như hình ảnh, âm thanh.  

                Neural Network ra đời để thu hẹp khoảng cách này bằng cách mô phỏng cách nơ-ron trong não kết nối và kích hoạt, giúp máy tính "học" từ dữ liệu thay vì chỉ thực hiện các lệnh cố định.
                """)

            elif info_option == "Tại sao cần Neural Network?":
                st.subheader("3. Tại sao cần Neural Network?")
                st.markdown("""
                Máy tính có nên mô phỏng bộ não không? **Có**, nhưng chỉ ở một mức độ nhất định. Neural Network không sao chép toàn bộ hoạt động sinh học của não, mà chỉ lấy cảm hứng từ cách các nơ-ron truyền tín hiệu để tạo ra một mô hình toán học hiệu quả.  

                **Lý do cần Neural Network**:  
                - Các phương pháp truyền thống (như logistic regression) chỉ giải quyết được bài toán tuyến tính đơn giản (ví dụ: phân chia dữ liệu bằng một đường thẳng).  
                - Neural Network với nhiều hidden layers có thể học các biểu diễn (representation) cấp cao của dữ liệu, giải quyết bài toán phi tuyến tính phức tạp hơn (như bài toán XOR).  

                Ví dụ: Trong bài toán XOR, logistic regression không thể phân chia dữ liệu không tuyến tính, nhưng thêm hidden layers giúp tạo ra ranh giới phi tuyến tính, giải quyết vấn đề hiệu quả.
                """)

            elif info_option == "Các bước thực hiện và công thức trong Neural Network":
                st.subheader("4. Các bước thực hiện và công thức trong Neural Network")
                st.markdown("""
                Dưới đây là các bước thực hiện và công thức tính toán trong Neural Network:  

                #### Các bước thực hiện  
                1. **Khởi tạo mô hình**: Xác định số layer, số node mỗi layer, khởi tạo \( W \) (trọng số) và \( b \) (bias).  
                2. **Lan truyền thuận (Feedforward)**: Tính \( \hat{Y} \) từ \( X \).  
                3. **Tính hàm mất mát (Loss Function)**: Đánh giá sai lệch giữa \( \hat{Y} \) và \( Y \).  
                4. **Lan truyền ngược (Backpropagation)**: Tính đạo hàm của hàm mất mát theo \( W \) và \( b \).  
                5. **Cập nhật tham số (Gradient Descent)**: Điều chỉnh \( W \) và \( b \) để giảm mất mát.  
                6. **Lặp lại**: Quay lại bước 2 cho đến khi mô hình hội tụ.  

                #### Công thức chi tiết  
                - **Lan truyền thuận**:  
                  \( Z^{(l)} = A^{(l-1)} \cdot W^{(l)} + b^{(l)} \)  
                  \( A^{(l)} = \sigma(Z^{(l)}) \)  
                  Với \( \sigma(z) = \frac{1}{1 + e^{-z}} \) (sigmoid).  
                  \( \hat{Y} = A^{(L)} \) (output layer).  

                - **Hàm mất mát (ví dụ Binary Cross-Entropy)**:  
                  \( L = -\frac{1}{N} \sum_{i=1}^{N} [y_i \cdot \log(\hat{y}_i) + (1 - y_i) \cdot \log(1 - \hat{y}_i)] \)  

                - **Lan truyền ngược**:  
                  Tại output layer:  
                  \( \delta^{(L)} = (\hat{Y} - Y) \odot \sigma'(Z^{(L)}) \)  
                  Tại hidden layer:  
                  \( \delta^{(l)} = (\delta^{(l+1)} \cdot (W^{(l+1)})^T) \odot \sigma'(Z^{(l)}) \)  
                  Đạo hàm:  
                  \( \frac{\partial L}{\partial W^{(l)}} = (A^{(l-1)})^T \cdot \delta^{(l)} \)  
                  \( \frac{\partial L}{\partial b^{(l)}} = \sum_{i=1}^{N} \delta^{(l)}_i \)  

                - **Gradient Descent**:  
                  \( W^{(l)} = W^{(l)} - \eta \cdot \frac{\partial L}{\partial W^{(l)}} \)  
                  \( b^{(l)} = b^{(l)} - \eta \cdot \frac{\partial L}{\partial b^{(l)}} \)  
                  \( \eta \): Learning rate.  

                Neural Network mạnh mẽ nhờ khả năng học từ dữ liệu qua các vòng lặp này!
                """)

    # Các tab khác giữ nguyên (Tab 2 đến Tab 7)
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
                    X = X / 255.0  # Chuẩn hóa ngay khi tải
                    y = y.astype(str)
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

    with tab_preprocess:
        st.header("Xử lý Dữ liệu")
        if 'data' not in st.session_state:
            st.info("Vui lòng tải và chốt số lượng mẫu trước.")
        else:
            X, y = st.session_state['data']
            if "data_original" not in st.session_state:
                st.session_state["data_original"] = (X.copy(), y.copy())

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
                            Công dụng: Đảm bảo thang đo đồng nhất, hữu ích cho Neural Network.
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
                test_size = int(total_samples * test_pct / 100)

                if st.button("Xác nhận chia dữ liệu"):
                    with st.spinner("Đang chia dữ liệu..."):
                        X_train_full, X_test, y_train_full, y_test = train_test_split(
                            X, y, test_size=test_size / total_samples, random_state=42
                        )

                        labeled_pct = 0.01  # 1%
                        X_labeled = pd.DataFrame()
                        y_labeled = pd.Series(dtype='str')
                        X_unlabeled = pd.DataFrame()

                        for digit in range(10):
                            X_digit = X_train_full[y_train_full == str(digit)]
                            y_digit = y_train_full[y_train_full == str(digit)]
                            n_labeled = max(1, int(len(X_digit) * labeled_pct))
                            X_labeled_digit, X_unlabeled_digit, y_labeled_digit, _ = train_test_split(
                                X_digit, y_digit, train_size=n_labeled, random_state=42
                            )
                            X_labeled = pd.concat([X_labeled, X_labeled_digit])
                            y_labeled = pd.concat([y_labeled, y_labeled_digit])
                            X_unlabeled = pd.concat([X_unlabeled, X_unlabeled_digit])

                        st.session_state['split_data'] = {
                            "X_train_labeled": X_labeled,
                            "y_train_labeled": y_labeled,
                            "X_train_unlabeled": X_unlabeled,
                            "X_test": X_test,
                            "y_test": y_test
                        }
                        st.success("Dữ liệu đã được chia!")
                        st.write(f"Train (Labeled): {len(X_labeled)} mẫu, Train (Unlabeled): {len(X_unlabeled)} mẫu, Test: {len(X_test)} mẫu")

    with tab_train_eval:
        st.header("Huấn luyện và Đánh Giá với Pseudo Labeling")
        if 'split_data' not in st.session_state:
            st.info("Vui lòng chia dữ liệu trước.")
        else:
            X_labeled = st.session_state['split_data']["X_train_labeled"]
            y_labeled = st.session_state['split_data']["y_train_labeled"]
            X_unlabeled = st.session_state['split_data']["X_train_unlabeled"]
            X_test = st.session_state['split_data']["X_test"]
            y_test = st.session_state['split_data']["y_test"]
            st.write(f"Số mẫu labeled ban đầu: {len(X_labeled)}, Số mẫu unlabeled: {len(X_unlabeled)}, Test: {len(X_test)}")

            st.subheader("Tham số mô hình")
            params = {
                "hidden_layers": st.number_input("Số lớp ẩn", 1, 5, 2),
                "neurons": st.number_input("Số nơ-ron mỗi lớp", 32, 1024, 128, step=32),
                "epochs": st.number_input("Số epochs mỗi vòng", 1, 100, 10),
                "dropout": st.slider("Dropout rate", 0.0, 0.5, 0.3, step=0.05),
                "threshold": st.slider("Ngưỡng gán nhãn giả", 0.5, 1.0, 0.95, step=0.01),
                "max_iterations": st.number_input("Số vòng lặp tối đa", 1, 20, 5)
            }

            if st.button("Thực hiện Huấn luyện"):
                with st.spinner("Đang thực hiện Pseudo Labeling..."):
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    start_time = time.time()

                    X_labeled_np = X_labeled.values.reshape(-1, 28, 28)
                    y_labeled_np = to_categorical(y_labeled)
                    X_unlabeled_np = X_unlabeled.values.reshape(-1, 28, 28)
                    X_test_np = X_test.values.reshape(-1, 28, 28)
                    y_test_np = to_categorical(y_test)

                    iteration = 0
                    accuracy_history = []

                    run_name = f"Pseudo_Labeling_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                    with mlflow.start_run(run_name=run_name) as run:
                        mlflow.log_params(params)
                        while iteration < params["max_iterations"] and len(X_unlabeled_np) > 0:
                            iteration += 1
                            st.write(f"### Vòng lặp {iteration}/{params['max_iterations']}")
                            progress_bar.progress(int((iteration / params["max_iterations"]) * 100))
                            status_text.text(f"Đang huấn luyện vòng {iteration}...")

                            model = Sequential()
                            model.add(Flatten(input_shape=(28, 28)))
                            for _ in range(params["hidden_layers"]):
                                model.add(Dense(params["neurons"], activation='relu'))
                                model.add(Dropout(params["dropout"]))
                            model.add(Dense(10, activation='softmax'))
                            model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

                            history = model.fit(X_labeled_np, y_labeled_np, epochs=params["epochs"], batch_size=32, verbose=0)

                            pseudo_probs = model.predict(X_unlabeled_np, verbose=0)
                            pseudo_labels = np.argmax(pseudo_probs, axis=1)
                            confidences = np.max(pseudo_probs, axis=1)

                            mask = confidences >= params["threshold"]
                            X_pseudo = X_unlabeled_np[mask]
                            y_pseudo = to_categorical(pseudo_labels[mask])

                            X_labeled_np = np.concatenate([X_labeled_np, X_pseudo])
                            y_labeled_np = np.concatenate([y_labeled_np, y_pseudo])
                            X_unlabeled_np = X_unlabeled_np[~mask]

                            st.write(f"Số mẫu được gán nhãn giả: {len(X_pseudo)}, Unlabeled còn lại: {len(X_unlabeled_np)}")

                            y_test_pred = np.argmax(model.predict(X_test_np, verbose=0), axis=1)
                            y_test_true = np.argmax(y_test_np, axis=1)
                            test_acc = accuracy_score(y_test_true, y_test_pred)
                            accuracy_history.append(test_acc)
                            st.write(f"Accuracy trên Test sau vòng {iteration}: {test_acc:.4f}")

                            mlflow.log_metric(f"test_accuracy_iter_{iteration}", test_acc)
                            mlflow.log_metric(f"labeled_samples_iter_{iteration}", len(X_labeled_np))
                            mlflow.log_metric(f"unlabeled_samples_iter_{iteration}", len(X_unlabeled_np))

                        model.fit(X_labeled_np, y_labeled_np, epochs=params["epochs"], batch_size=32, verbose=0)
                        y_test_pred = np.argmax(model.predict(X_test_np, verbose=0), axis=1)
                        y_test_true = np.argmax(y_test_np, axis=1)
                        final_acc = accuracy_score(y_test_true, y_test_pred)
                        cm_test = confusion_matrix(y_test_true, y_test_pred)
                        training_time = time.time() - start_time

                        mlflow.log_metric("final_test_accuracy", final_acc)
                        mlflow.log_metric("training_time_seconds", training_time)
                        mlflow.keras.log_model(model, "final_model")

                        st.session_state['model'] = model
                        st.session_state['training_results'] = {
                            'final_acc': final_acc,
                            'cm_test': cm_test,
                            'accuracy_history': accuracy_history,
                            'training_time': training_time,
                            'run_name': run_name,
                            'run_id': run.info.run_id
                        }

                        progress_bar.progress(100)
                        status_text.text("Hoàn tất!")
                        st.success(f"Huấn luyện hoàn tất! Thời gian: {training_time:.2f} giây")

            if 'training_results' in st.session_state:
                st.write(f"Accuracy trên Test: {st.session_state['training_results']['final_acc']:.4f}")

                st.markdown("### Confusion Matrix trên Test")
                fig, ax = plt.subplots(figsize=(8, 6))
                sns.heatmap(st.session_state['training_results']['cm_test'], annot=True, fmt="d", cmap="Blues", ax=ax)
                ax.set_title("Confusion Matrix - Test")
                st.pyplot(fig)

                st.markdown("### Biểu đồ Accuracy qua các vòng lặp")
                fig, ax = plt.subplots(figsize=(8, 4))
                ax.plot(range(1, len(st.session_state['training_results']['accuracy_history']) + 1), 
                        st.session_state['training_results']['accuracy_history'], marker='o')
                ax.set_title("Accuracy trên Test qua các vòng lặp")
                ax.set_xlabel("Vòng lặp")
                ax.set_ylabel("Accuracy")
                ax.grid(True)
                st.pyplot(fig)

    with tab_demo:
        st.header("Demo Dự đoán")
        if 'split_data' not in st.session_state or 'model' not in st.session_state:
            st.info("Vui lòng huấn luyện mô hình trước.")
        else:
            mode = st.radio("Chọn phương thức dự đoán:", ["Dữ liệu từ Test", "Upload ảnh mới", "Vẽ số"])
            progress_bar = st.progress(0)
            status_text = st.empty()

            X_test = st.session_state['split_data']["X_test"]
            y_test = st.session_state['split_data']["y_test"]
            is_normalized = "data_processed" in st.session_state

            if mode == "Dữ liệu từ Test":
                if X_test.empty or y_test.empty:
                    st.warning("Tập Test không có dữ liệu. Vui lòng chia dữ liệu với tỷ lệ Test > 0.")
                else:
                    idx = st.slider("Chọn mẫu từ Test", 0, len(X_test)-1, 0)
                    if st.button("Dự đoán"):
                        with st.spinner("Đang dự đoán..."):
                            for i in range(0, 51, 5):
                                progress_bar.progress(i)
                                status_text.text(f"Đang xử lý {i}%{i % 4 * '.'}")
                                time.sleep(0.1)
                            sample = X_test.iloc[idx].values.reshape(1, 28, 28)
                            prediction = np.argmax(st.session_state['model'].predict(sample, verbose=0), axis=1)[0]
                            proba = st.session_state['model'].predict(sample, verbose=0)[0]
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
                            img = Image.open(uploaded_image).convert('L').resize((28, 28))
                            img_array = np.array(img).reshape(1, 28, 28)
                            if not is_normalized:
                                img_array = img_array / 255.0
                            prediction = np.argmax(st.session_state['model'].predict(img_array, verbose=0), axis=1)[0]
                            proba = st.session_state['model'].predict(img_array, verbose=0)[0]
                            confidence = max(proba) * 100
                            st.success(f"Dự đoán: **{prediction}** | Confidence: **{confidence:.2f}%**")
                            st.image(img, caption=f"Ảnh {i+1} được upload", use_container_width=True)

            elif mode == "Vẽ số":
                st.write("Vẽ một chữ số từ 0-9 trên canvas (28x28 pixel):")
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
                        with st.spinner("Đang xử lý..."):
                            img = Image.fromarray((canvas_result.image_data[:, :, 0]).astype(np.uint8)).convert('L').resize((28, 28))
                            img_array = np.array(img).reshape(1, 28, 28)
                            if not is_normalized:
                                img_array = img_array / 255.0
                            prediction = np.argmax(st.session_state['model'].predict(img_array, verbose=0), axis=1)[0]
                            proba = st.session_state['model'].predict(img_array, verbose=0)[0]
                            confidence = max(proba) * 100
                            st.success(f"Dự đoán: **{prediction}** | Confidence: **{confidence:.2f}%**")
                            st.image(img, caption="Ảnh đã vẽ", width=150)

    with tab_log_info:
        st.header("Theo dõi kết quả")
        st.markdown("""
        Tab này cho phép bạn xem danh sách các lần huấn luyện đã thực hiện. Chọn một lần chạy để xem chi tiết, đổi tên hoặc xóa.
        """, unsafe_allow_html=True)
        
        try:
            client = MlflowClient()
            experiment = client.get_experiment_by_name("MNIST_Pseudo_Labeling")
            if not experiment:
                st.error("Không tìm thấy experiment 'MNIST_Pseudo_Labeling'. Vui lòng kiểm tra MLflow tracking URI.")
            else:
                experiment_id = experiment.experiment_id
                experiment_path = experiment.name
                st.write(f"**Path:** {experiment_path}")
                st.write(f"**Experiment ID:** {experiment_id}")

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
                        key="main_select"
                    )
                    selected_run_id = [k for k, v in run_options.items() if v == selected_run_name][0]
                    selected_run = client.get_run(selected_run_id)

                    st.subheader("Đổi tên Run")
                    new_run_name = st.text_input("Nhập tên mới:", value=selected_run_name, key="rename_input")
                    if st.button("Cập nhật tên", key="rename_button"):
                        if new_run_name.strip() and new_run_name.strip() != selected_run_name:
                            client.set_tag(selected_run_id, "mlflow.runName", new_run_name.strip())
                            st.success(f"Đã đổi tên thành: {new_run_name.strip()}")
                            st.rerun()

                    st.subheader("Xóa Run")
                    if st.button("Xóa lần chạy", key="delete_button"):
                        client.delete_run(selected_run_id)
                        st.success(f"Đã xóa: {selected_run_name}")
                        st.rerun()

                    st.subheader("Thông tin chi tiết của Run")
                    st.write(f"**Tên lần chạy:** {selected_run_name}")
                    st.write(f"**ID lần chạy:** {selected_run_id}")
                    st.write(f"**Thời gian bắt đầu:** {datetime.fromtimestamp(selected_run.info.start_time / 1000)}")
                    st.markdown("**Tham số:**", unsafe_allow_html=True)
                    if selected_run.data.params:
                        st.json(selected_run.data.params)
                    st.markdown("**Kết quả:**", unsafe_allow_html=True)
                    if selected_run.data.metrics:
                        st.json(selected_run.data.metrics)

        except Exception as e:
            st.error(f"Lỗi kết nối MLflow: {e}")

if __name__ == "__main__":
    run_mnist_pseudo_labeling_app()
