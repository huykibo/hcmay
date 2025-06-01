import os
import mlflow
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from mlflow.tracking import MlflowClient
from datetime import datetime
import time
import gc

#lưu í bật mlflow ui trước sau đó chạy streamlit run của file code...
# Hàm xử lý dữ liệu ảnh (nếu dữ liệu là ảnh)
def process_image_to_features(image_path):
    img = Image.open(image_path).convert('L').resize((64, 64))  # Giả sử ảnh hoa 64x64 grayscale
    img_array = np.array(img).flatten() / 255.0  # Chuẩn hóa về [0, 1]
    return img_array

# Hàm tải và xử lý dữ liệu từ CSV
def load_and_preprocess_data(uploaded_file):
    if uploaded_file.name.endswith('.csv'):
        data = pd.read_csv(uploaded_file)
        required_columns = ['Leaf_Length', 'Leaf_Width', 'Stem_Length', 'Petal_Size', 'Label']
        missing_columns = [col for col in required_columns if col not in data.columns]
        if missing_columns:
            st.error(f"❌ File CSV thiếu các cột cần thiết: {', '.join(missing_columns)}. Dữ liệu cần có các cột: {', '.join(required_columns)}.")
            return None, None
        
        X = data[['Leaf_Length', 'Leaf_Width', 'Stem_Length', 'Petal_Size']].values
        y = data['Label'].values
    else:
        st.error("❌ Hiện chỉ hỗ trợ file CSV. Nếu dùng ảnh, cần thêm logic xử lý riêng.")
        return None, None
    return X, y

# Ứng dụng chính
def run_flower_classification_app():
    # Thiết lập MLflow cục bộ
    mlflow.set_tracking_uri("http://localhost:5000")
    mlflow.set_experiment("Flower_Classification")
    EXPERIMENT_ID = mlflow.get_experiment_by_name("Flower_Classification").experiment_id

    # Tiêu đề chính
    st.title("🌸 Ứng dụng Phân loại Hoa với Logistic Regression và SVM")
    st.markdown("---")

    # Tạo các tab
    tab_preprocess, tab_train, tab_predict, tab_mlflow = st.tabs([
        "📊 Tiền xử lý dữ liệu", "🛠️ Huấn luyện mô hình", "🔍 Dự đoán", "📈 Thông tin MLflow"
    ])

    # Tab 1: Tiền xử lý dữ liệu
    with tab_preprocess:
        st.header("📊 Tiền xử lý dữ liệu")
        st.markdown("""
        **Hướng dẫn:** Tải lên file CSV chứa dữ liệu hoa với các cột:  
        - `Leaf_Length` (Chiều dài lá)  
        - `Leaf_Width` (Chiều rộng lá)  
        - `Stem_Length` (Chiều dài thân)  
        - `Petal_Size` (Kích thước cánh hoa)  
        - `Label` (Nhãn loài hoa: 0 hoặc 1)  
        Dữ liệu sẽ được chuẩn hóa và hiển thị dưới dạng biểu đồ phân tán.
        """)
        
        uploaded_file = st.file_uploader("📂 Tải lên file dữ liệu", type=["csv"])
        if uploaded_file:
            with st.spinner("⏳ Đang tải và xử lý dữ liệu..."):
                X, y = load_and_preprocess_data(uploaded_file)
                if X is not None:
                    scaler = StandardScaler()
                    X_scaled = scaler.fit_transform(X)
                    st.session_state['data'] = (X_scaled, y, scaler)
                    st.success(f"✅ Đã tải dữ liệu: {X.shape[0]} mẫu, {X.shape[1]} đặc trưng")

                    # Hiển thị biểu đồ phân tán
                    with st.container():
                        st.subheader("📈 Minh họa dữ liệu")
                        fig, ax = plt.subplots(figsize=(10, 6))
                        sample_indices = np.random.choice(len(X), min(100, len(X)), replace=False)
                        sns.scatterplot(x=X[sample_indices, 0], y=X[sample_indices, 1], hue=y[sample_indices], 
                                        palette="Set2", size=10, ax=ax)
                        ax.set_title("Phân bố dữ liệu hoa (Chiều dài lá và Chiều rộng lá)", fontsize=14, pad=15)
                        ax.set_xlabel("Chiều dài lá (Leaf_Length)", fontsize=12)
                        ax.set_ylabel("Chiều rộng lá (Leaf_Width)", fontsize=12)
                        ax.grid(True, linestyle='--', alpha=0.7)
                        ax.legend(title="Loài hoa (Label)", loc="best")
                        st.pyplot(fig)
                        plt.close(fig)

                    # Giải thích biểu đồ
                    with st.expander("ℹ️ Giải thích biểu đồ"):
                        st.markdown("""
                        Biểu đồ trên là một **đồ thị phân tán (scatter plot)** thể hiện phân bố của dữ liệu hoa dựa trên hai đặc trưng:  
                        - **Chiều dài lá (Leaf_Length)**: Trục X.  
                        - **Chiều rộng lá (Leaf_Width)**: Trục Y.  
                        - **Màu sắc**: Biểu thị các loài hoa khác nhau (nhãn `Label`: 0 hoặc 1).  
                        - **Mục đích**: Giúp hình dung mức độ phân tách giữa các loài hoa dựa trên hai đặc trưng này. Nếu các cụm màu sắc tách biệt rõ ràng, mô hình học máy sẽ dễ dàng phân loại hơn.
                        """)

    # Tab 2: Huấn luyện mô hình
    with tab_train:
        st.header("🛠️ Huấn luyện mô hình")
        if 'data' not in st.session_state:
            st.info("ℹ️ Vui lòng tiền xử lý dữ liệu trước khi huấn luyện.")
        else:
            X, y, scaler = st.session_state['data']
            st.write(f"**Tổng số mẫu dữ liệu**: {len(X)}")

            # Phân chia dữ liệu
            with st.container():
                st.subheader("📌 Phân chia dữ liệu")
                col1, col2 = st.columns(2)
                with col1:
                    valid_pct = st.slider("Tỷ lệ Validation (%)", 0, 100, 15, help="Phần trăm dữ liệu dùng để kiểm tra trong quá trình huấn luyện.")
                with col2:
                    test_pct = st.slider("Tỷ lệ Test (%)", 0, 100 - valid_pct, 15, help="Phần trăm dữ liệu dùng để đánh giá cuối cùng.")
                train_pct = 100 - valid_pct - test_pct
                st.write(f"**Phân bổ dữ liệu**: Train: {train_pct}%, Validation: {valid_pct}%, Test: {test_pct}%")

                if st.button("📊 Chia dữ liệu", type="primary"):
                    X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=test_pct/100, random_state=42)
                    X_train, X_valid, y_train, y_valid = train_test_split(X_temp, y_temp, 
                                                                          test_size=valid_pct/(100 - test_pct) if test_pct < 100 else 0, 
                                                                          random_state=42)
                    st.session_state['split_data'] = {
                        "X_train": X_train, "y_train": y_train,
                        "X_valid": X_valid, "y_valid": y_valid,
                        "X_test": X_test, "y_test": y_test
                    }
                    st.success("✅ Đã chia dữ liệu thành công!")

            # Huấn luyện mô hình
            if 'split_data' in st.session_state:
                split_data = st.session_state['split_data']
                X_train, y_train = split_data['X_train'], split_data['y_train']

                with st.container():
                    st.subheader("⚙️ Cấu hình huấn luyện")
                    model_choice = st.selectbox("Chọn mô hình", ["Logistic Regression", "SVM"], key="train_model_selectbox")
                    model_name = st.text_input("Tên mô hình", value=f"{model_choice}_Model", help="Đặt tên cho mô hình để dễ nhận diện.")
                    k_folds = st.slider("Số lần kiểm tra chéo (K-folds)", 2, 10, 5, help="Số lần kiểm tra chéo để đánh giá độ ổn định của mô hình.")

                    # Cấu hình tham số mô hình
                    params = {}
                    if model_choice == "Logistic Regression":
                        with st.expander("🔧 Tham số Logistic Regression"):
                            params["Độ điều chỉnh"] = st.number_input("Độ điều chỉnh (C)", min_value=0.01, max_value=100.0, value=1.0,
                                                                     help="Điều chỉnh mức độ regularization: Giá trị nhỏ tăng regularization, giảm overfitting.")
                            params["Số lần lặp tối đa"] = st.number_input("Số lần lặp tối đa", min_value=100, max_value=5000, value=1000,
                                                                         help="Số lần lặp tối đa để mô hình hội tụ.")
                            params["Hình phạt"] = st.selectbox("Hình phạt", ["l2", "l1"], index=0,
                                                               help="Loại regularization: 'l2' (mặc định) hoặc 'l1' (Lasso).")
                            params["Phương pháp giải"] = st.selectbox("Phương pháp giải", ["lbfgs", "liblinear", "saga"], index=0,
                                                                      help="Phương pháp tối ưu: 'lbfgs' (mặc định), 'liblinear' (nhỏ gọn), 'saga' (hỗ trợ l1).")
                    elif model_choice == "SVM":
                        with st.expander("🔧 Tham số SVM"):
                            params["Độ điều chỉnh"] = st.number_input("Độ điều chỉnh (C)", min_value=0.01, max_value=100.0, value=1.0,
                                                                     help="Điều chỉnh mức độ sai số và lề: Giá trị lớn ưu tiên phân loại chính xác hơn.")
                            params["Loại kernel"] = st.selectbox("Loại kernel", ["linear", "rbf", "poly", "sigmoid"], index=1,
                                                                 help="Loại kernel cho SVM: Linear (tuyến tính), RBF (phi tuyến), Polynomial (đa thức), Sigmoid.")

                    if st.button("🚀 Bắt đầu huấn luyện", type="primary"):
                        # Thanh tiến trình
                        progress_text = st.empty()
                        progress_bar = st.progress(0)
                        with st.spinner(f"⏳ Đang huấn luyện mô hình {model_name}..."):
                            start_time = time.time()
                            if model_choice == "Logistic Regression":
                                model = LogisticRegression(C=params["Độ điều chỉnh"], max_iter=params["Số lần lặp tối đa"], 
                                                           penalty=params["Hình phạt"], solver=params["Phương pháp giải"])
                            else:
                                model = SVC(C=params["Độ điều chỉnh"], kernel=params["Loại kernel"], probability=True)

                            pipeline = Pipeline([('classifier', model)])
                            pipeline.fit(X_train, y_train)

                            # Giả lập thanh tiến trình (dựa trên thời gian huấn luyện)
                            training_time = time.time() - start_time
                            for i in range(101):
                                time.sleep(training_time / 100)  # Chia nhỏ thời gian để hiển thị tiến trình
                                progress_text.text(f"Tiến trình huấn luyện: {i}%")
                                progress_bar.progress(i)
                            progress_text.empty()

                            cv_scores = cross_val_score(pipeline, X_train, y_train, cv=k_folds)
                            cv_mean = cv_scores.mean()
                            cv_std = cv_scores.std()

                            acc_valid = accuracy_score(split_data['y_valid'], pipeline.predict(split_data['X_valid']))
                            acc_test = accuracy_score(split_data['y_test'], pipeline.predict(split_data['X_test']))

                            run_name = f"{model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                            with mlflow.start_run(experiment_id=EXPERIMENT_ID, run_name=run_name):
                                mlflow.log_param("model", model_choice)
                                mlflow.log_param("k_folds", k_folds)
                                if model_choice == "Logistic Regression":
                                    mlflow.log_param("Độ điều chỉnh", params["Độ điều chỉnh"])
                                    mlflow.log_param("Số lần lặp tối đa", params["Số lần lặp tối đa"])
                                    mlflow.log_param("Hình phạt", params["Hình phạt"])
                                    mlflow.log_param("Phương pháp giải", params["Phương pháp giải"])
                                else:
                                    mlflow.log_param("Độ điều chỉnh", params["Độ điều chỉnh"])
                                    mlflow.log_param("Loại kernel", params["Loại kernel"])
                                mlflow.log_metric("cv_mean", cv_mean)
                                mlflow.log_metric("cv_std", cv_std)
                                mlflow.log_metric("accuracy_valid", acc_valid)
                                mlflow.log_metric("accuracy_test", acc_test)
                                mlflow.log_metric("training_time", time.time() - start_time)
                                mlflow.sklearn.log_model(pipeline, "model")

                            st.session_state[f'model_{model_name}'] = pipeline
                            st.success(f"✅ Đã huấn luyện mô hình {model_name}! Thời gian: {time.time() - start_time:.2f} giây")
                            st.write(f"**Kiểm tra chéo (Cross-Validation)**: Trung bình = {cv_mean:.4f}, Độ lệch chuẩn = {cv_std:.4f}")
                            st.write(f"**Độ chính xác**: Validation = {acc_valid:.4f}, Test = {acc_test:.4f}")

                            # Biểu đồ đánh giá chuyên nghiệp
                            with st.container():
                                st.subheader("📊 Đánh giá hiệu suất mô hình")
                                fig, ax = plt.subplots(figsize=(8, 5))
                                metrics = ['Validation', 'Test']
                                values = [acc_valid, acc_test]
                                colors = ['#66BB6A', '#42A5F5']  # Màu sắc chuyên nghiệp
                                bars = ax.bar(metrics, values, color=colors, width=0.5)
                                ax.set_ylim(0, 1.1)
                                ax.set_title(f"Hiệu suất của mô hình {model_name}" + 
                                             (f" (Kernel: {params['Loại kernel']})" if model_choice == "SVM" else ""), 
                                             fontsize=14, pad=15)
                                ax.set_ylabel("Độ chính xác", fontsize=12)
                                ax.set_xlabel("Tập dữ liệu", fontsize=12)
                                ax.grid(True, axis='y', linestyle='--', alpha=0.7)
                                for bar in bars:
                                    yval = bar.get_height()
                                    ax.text(bar.get_x() + bar.get_width()/2, yval + 0.02, f"{yval:.4f}", 
                                            ha='center', va='bottom', fontsize=10, color='black')
                                st.pyplot(fig)
                                plt.close(fig)

                            # Giải thích kết quả
                            with st.expander("ℹ️ Phân tích kết quả"):
                                if model_choice == "Logistic Regression":
                                    explanation = """
                                    **Phân tích hiệu suất Logistic Regression:**  
                                    - **Trung bình kiểm tra chéo (CV Mean)**: {cv_mean:.4f} – Độ chính xác trung bình từ {k_folds} lần kiểm tra chéo, thể hiện hiệu suất ổn định trên dữ liệu huấn luyện.  
                                    - **Độ lệch chuẩn CV (CV Std)**: {cv_std:.4f} – Độ biến thiên của độ chính xác, giá trị nhỏ cho thấy mô hình ổn định.  
                                    - **Độ chính xác Validation**: {acc_valid:.4f} – Khả năng dự đoán trên tập validation (dữ liệu chưa thấy trong huấn luyện).  
                                    - **Độ chính xác Test**: {acc_test:.4f} – Khả năng tổng quát hóa trên tập test (dữ liệu mới).  
                                    - **Nhận xét**: Nếu độ chính xác Validation và Test cao và gần nhau, mô hình có khả năng phân loại tốt, ít bị overfitting.  
                                    - **Tham số ảnh hưởng**:  
                                      - `Độ điều chỉnh (C) = {C}`: Điều chỉnh mức độ regularization.  
                                      - `Hình phạt = {penalty}`: Loại regularization.  
                                      - `Phương pháp giải = {solver}`: Ảnh hưởng đến tốc độ hội tụ.
                                    """
                                    st.markdown(explanation.format(k_folds=k_folds, cv_mean=cv_mean, cv_std=cv_std, 
                                                                   acc_valid=acc_valid, acc_test=acc_test, 
                                                                   C=params["Độ điều chỉnh"], penalty=params["Hình phạt"], 
                                                                   solver=params["Phương pháp giải"]))
                                else:  # SVM
                                    explanation = """
                                    **Phân tích hiệu suất SVM:**  
                                    - **Trung bình kiểm tra chéo (CV Mean)**: {cv_mean:.4f} – Độ chính xác trung bình từ {k_folds} lần kiểm tra chéo, thể hiện hiệu suất ổn định trên dữ liệu huấn luyện.  
                                    - **Độ lệch chuẩn CV (CV Std)**: {cv_std:.4f} – Độ biến thiên của độ chính xác, giá trị nhỏ cho thấy mô hình ổn định.  
                                    - **Độ chính xác Validation**: {acc_valid:.4f} – Khả năng dự đoán trên tập validation.  
                                    - **Độ chính xác Test**: {acc_test:.4f} – Khả năng tổng quát hóa trên tập test.  
                                    - **Nhận xét**: Nếu độ chính xác Validation và Test cao và gần nhau, mô hình có khả năng phân loại tốt, ít bị overfitting.  
                                    - **Tham số ảnh hưởng**:  
                                      - `Độ điều chỉnh (C) = {C}`: Điều chỉnh sai số và lề.  
                                      - `Loại kernel = {kernel}`: Quyết định cách phân tách dữ liệu, phù hợp với dữ liệu có phân bố {kernel_desc}.
                                    """
                                    kernel_desc = "tuyến tính" if params["Loại kernel"] == "linear" else "phi tuyến phức tạp"
                                    st.markdown(explanation.format(k_folds=k_folds, cv_mean=cv_mean, cv_std=cv_std, 
                                                                   acc_valid=acc_valid, acc_test=acc_test, 
                                                                   C=params["Độ điều chỉnh"], kernel=params["Loại kernel"], 
                                                                   kernel_desc=kernel_desc))

    # Tab 3: Dự đoán
    with tab_predict:
        st.header("🔍 Dự đoán loài hoa")
        if 'data' not in st.session_state or not any(key.startswith('model_') for key in st.session_state.keys()):
            st.warning("⚠️ Vui lòng tải dữ liệu và huấn luyện ít nhất một mô hình trước.")
        else:
            X, y, scaler = st.session_state['data']
            model_options = [key.replace('model_', '') for key in st.session_state.keys() if key.startswith('model_')]
            if not model_options:
                st.warning("⚠️ Chưa có mô hình nào được huấn luyện.")
            else:
                with st.container():
                    st.subheader("📌 Chọn mô hình và dữ liệu dự đoán")
                    model_choice = st.selectbox("Chọn mô hình", model_options, key="predict_model_selectbox")

                    prediction_method = st.radio("Chọn cách lấy dữ liệu dự đoán", ["Chọn từ tập dữ liệu", "Ngẫu nhiên", "Tải lên file CSV"])
                    
                    if prediction_method == "Chọn từ tập dữ liệu":
                        sample_idx = st.slider("Chọn mẫu dữ liệu", 0, len(X) - 1, 0, help="Chọn một mẫu từ tập dữ liệu đã tải.")
                        X_sample = X[sample_idx].reshape(1, -1)
                        true_label = y[sample_idx]
                    elif prediction_method == "Ngẫu nhiên":
                        sample_idx = np.random.randint(0, len(X))
                        X_sample = X[sample_idx].reshape(1, -1)
                        true_label = y[sample_idx]
                    else:
                        uploaded_sample = st.file_uploader("📂 Tải lên mẫu dữ liệu (CSV)", type=["csv"], key="predict_upload")
                        if uploaded_sample:
                            X_sample, _ = load_and_preprocess_data(uploaded_sample)
                            true_label = None
                        else:
                            X_sample = None

                    if X_sample is not None:
                        if st.button("🔍 Dự đoán", type="primary"):
                            with st.spinner("⏳ Đang dự đoán..."):
                                model_key = f'model_{model_choice}'
                                if model_key in st.session_state:
                                    model = st.session_state[model_key]
                                    pred_proba = model.predict_proba(X_sample)[0]
                                    pred_class = model.predict(X_sample)[0]
                                    confidence = pred_proba.max() * 100
                                    st.write(f"**Kết quả dự đoán**: {pred_class}")
                                    if true_label is not None:
                                        st.write(f"**Nhãn thực tế**: {true_label}")
                                    st.write(f"**Độ tin cậy**: {confidence:.2f}%")
                                    st.success("✅ Dự đoán hoàn tất!")
                                else:
                                    st.error(f"❌ Mô hình '{model_choice}' chưa được huấn luyện hoặc không tồn tại.")

    # Tab 4: Thông tin MLflow
    with tab_mlflow:
        st.header("📈 Quản lý thông tin MLflow")
        if st.button("🔄 Làm mới danh sách"):
            st.session_state.pop('mlflow_runs', None)
            st.rerun()

        try:
            with st.spinner("⏳ Đang tải thông tin từ MLflow..."):
                client = MlflowClient()
                runs = client.search_runs(experiment_ids=[EXPERIMENT_ID], order_by=["attributes.start_time DESC"])
                if not runs:
                    st.info("ℹ️ Chưa có lần chạy nào được ghi nhận.")
                else:
                    # Phần 1: Chi tiết lần chạy
                    with st.container():
                        st.subheader("📋 Chi tiết lần chạy")
                        run_options = {run.info.run_id: run.data.tags.get('mlflow.runName', f"Run_{run.info.run_id}") for run in runs}
                        selected_run_name = st.selectbox("Chọn lần chạy", list(run_options.values()))
                        selected_run_id = [k for k, v in run_options.items() if v == selected_run_name][0]
                        selected_run = client.get_run(selected_run_id)

                        new_run_name = st.text_input("Nhập tên mới", value=selected_run_name)
                        if st.button("✏️ Cập nhật tên"):
                            client.set_tag(selected_run_id, "mlflow.runName", new_run_name.strip())
                            st.success(f"✅ Đã đổi tên thành: {new_run_name.strip()}")
                            st.rerun()

                        if st.button("🗑️ Xóa lần chạy"):
                            client.delete_run(selected_run_id)
                            st.success(f"✅ Đã xóa: {selected_run_name}")
                            st.rerun()

                        st.write(f"**Tên lần chạy**: {selected_run_name}")
                        st.write(f"**ID**: {selected_run_id}")
                        st.write(f"**Thời gian bắt đầu**: {datetime.fromtimestamp(selected_run.info.start_time / 1000)}")
                        st.write("**Tham số**:")
                        st.json(selected_run.data.params)
                        st.write("**Số liệu**:")
                        st.json(selected_run.data.metrics)
                        st.markdown("🔗 Xem chi tiết tại: [MLflow UI](http://localhost:5000)")

                    # Phần 2: So sánh các mô hình
                    with st.container():
                        st.subheader("📊 So sánh hiệu suất các mô hình")
                        # Tạo danh sách các lần chạy để người dùng chọn
                        run_names = [run.data.tags.get('mlflow.runName', f"Run_{run.info.run_id}") for run in runs]
                        selected_runs = st.multiselect("Chọn các mô hình để so sánh", run_names)  # Không mặc định chọn

                        if not selected_runs:
                            st.warning("⚠️ Vui lòng chọn ít nhất một mô hình để so sánh.")
                        else:
                            # Lọc dữ liệu của các mô hình được chọn
                            comparison_data = []
                            for run in runs:
                                run_name = run.data.tags.get('mlflow.runName', f"Run_{run.info.run_id}")
                                if run_name in selected_runs:
                                    model_type = run.data.params.get('model', 'Không xác định')
                                    acc_valid = run.data.metrics.get('accuracy_valid', 0.0)
                                    acc_test = run.data.metrics.get('accuracy_test', 0.0)
                                    comparison_data.append({
                                        "Tên lần chạy": run_name,
                                        "Loại mô hình": model_type,
                                        "Độ chính xác Validation": acc_valid,
                                        "Độ chính xác Test": acc_test
                                    })

                            # Hiển thị bảng so sánh
                            if comparison_data:
                                comparison_df = pd.DataFrame(comparison_data)
                                st.write("**Bảng so sánh hiệu suất:**")
                                st.dataframe(comparison_df.style.format({
                                    "Độ chính xác Validation": "{:.4f}",
                                    "Độ chính xác Test": "{:.4f}"
                                }))

                                # Biểu đồ so sánh
                                st.write("**Biểu đồ so sánh độ chính xác:**")
                                fig, ax = plt.subplots(figsize=(10, 6))
                                bar_width = 0.35
                                index = np.arange(len(comparison_df))
                                bars1 = ax.bar(index, comparison_df["Độ chính xác Validation"], bar_width, label="Validation", color="#66BB6A")
                                bars2 = ax.bar(index + bar_width, comparison_df["Độ chính xác Test"], bar_width, label="Test", color="#42A5F5")
                                ax.set_xlabel("Tên lần chạy", fontsize=12)
                                ax.set_ylabel("Độ chính xác", fontsize=12)
                                ax.set_title("So sánh hiệu suất các mô hình", fontsize=14, pad=15)
                                ax.set_xticks(index + bar_width / 2)
                                ax.set_xticklabels(comparison_df["Tên lần chạy"], rotation=45, ha="right")
                                ax.set_ylim(0, 1.1)
                                ax.grid(True, axis='y', linestyle='--', alpha=0.7)
                                ax.legend()
                                for bar in bars1:
                                    yval = bar.get_height()
                                    ax.text(bar.get_x() + bar.get_width()/2, yval + 0.02, f"{yval:.4f}", ha='center', va='bottom', fontsize=8)
                                for bar in bars2:
                                    yval = bar.get_height()
                                    ax.text(bar.get_x() + bar.get_width()/2, yval + 0.02, f"{yval:.4f}", ha='center', va='bottom', fontsize=8)
                                st.pyplot(fig)
                                plt.close(fig)
                            else:
                                st.info("ℹ️ Không có dữ liệu để so sánh.")

        except Exception as e:
            st.error(f"❌ Lỗi khi tải thông tin MLflow: {e}")

if __name__ == "__main__":
    run_flower_classification_app()