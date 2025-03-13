import os
import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import PolynomialFeatures, MinMaxScaler, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score
import joblib
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
import time
from datetime import datetime

def run_titanic_app():
    try:
        os.environ["MLFLOW_TRACKING_USERNAME"] = st.secrets["mlflow"]["MLFLOW_TRACKING_USERNAME"]
        os.environ["MLFLOW_TRACKING_PASSWORD"] = st.secrets["mlflow"]["MLFLOW_TRACKING_PASSWORD"]
        mlflow.set_tracking_uri(st.secrets["mlflow"]["MLFLOW_TRACKING_URI"])
        mlflow.set_experiment("Titanic")
    except KeyError as e:
        st.error(f"Lỗi: Không tìm thấy khóa {e} trong st.secrets. Vui lòng cấu hình secrets trong Streamlit.")
        st.stop()

    # Khởi tạo session_state nếu chưa có
    if 'data_loaded' not in st.session_state:
        st.session_state.data_loaded = False
    if 'data_processed' not in st.session_state:
        st.session_state.data_processed = False
    if 'data_split' not in st.session_state:
        st.session_state.data_split = False
    if 'models_trained' not in st.session_state:
        st.session_state.models_trained = False
    if 'df' not in st.session_state:
        st.session_state.df = None
    if 'X_train' not in st.session_state:
        st.session_state.X_train = None
    if 'y_train' not in st.session_state:
        st.session_state.y_train = None
    if 'X_val' not in st.session_state:
        st.session_state.X_val = None
    if 'y_val' not in st.session_state:
        st.session_state.y_val = None
    if 'X_test' not in st.session_state:
        st.session_state.X_test = None
    if 'y_test' not in st.session_state:
        st.session_state.y_test = None
    if 'model' not in st.session_state:
        st.session_state.model = None
    if "run_id" not in st.session_state:
        st.session_state["run_id"] = None
    if "run_name" not in st.session_state:
        st.session_state["run_name"] = None
    if "selected_samples" not in st.session_state:
        st.session_state["selected_samples"] = None

    st.title("🚢 Dự đoán sống sót trên Titanic")

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
      padding: 8px;
      position: absolute;
      z-index: 1;
      right: 105%;
      top: 50%;
      transform: translateY(-50%);
      opacity: 0;
      transition: opacity 0.3s;
      border: 1px solid #ccc;
      font-size: 0.85em;
      line-height: 1.3;
    }
    .tooltip:hover .tooltiptext {
      visibility: visible;
      opacity: 1;
    }
    </style>
    """, unsafe_allow_html=True)

    # Tạo các tab
    tabs = st.tabs([
        "Thông tin ứng dụng",
        "Phân tích dữ liệu",
        "Huấn luyện mô hình",
        "Dự đoán",
        "Thông tin huấn luyện & MLflow UI"
    ])
    tab_info, tab_analysis, tab_train, tab_predict, tab_mlflow = tabs

    # ---------------- Tab 1: Thông tin ứng dụng ----------------
    with tab_info:
        st.header("Giới thiệu & Thông tin Ứng dụng")
        st.markdown(r"""
**Ứng dụng "Dự đoán sống sót trên Titanic"** giúp người dùng thực hiện các bước sau:
- **Phân tích dữ liệu:** Tải và kiểm tra dữ liệu Titanic.
- **Xử lý dữ liệu:** Loại bỏ cột không cần thiết, tự động điền giá trị thiếu, mã hóa biến phân loại và chuẩn hóa dữ liệu.
- **Chia dữ liệu:** Phân chia dữ liệu thành Train, Validation và Test.
- **Huấn luyện mô hình:** Huấn luyện mô hình hồi quy với Cross Validation.
- **Dự đoán:** Dự đoán khả năng sống sót kèm độ tin cậy (Confidence).
- **Thông tin huấn luyện & MLflow UI:** Xem chi tiết các run đã log, đổi tên, xóa và truy cập MLflow UI.
        """, unsafe_allow_html=True)

    # ---------------- Tab 2: Phân tích dữ liệu ----------------
    with tab_analysis:
        st.header("Phân tích và xử lý dữ liệu")
        with st.expander("📥 Tải dữ liệu", expanded=True):
            uploaded_file = st.file_uploader("Tải file CSV (Titanic dataset)", type=["csv"])
            if uploaded_file is not None:
                st.session_state.df = pd.read_csv(uploaded_file)
                st.write("Dữ liệu đã được tải lên:")
                st.write(st.session_state.df.head(10))
                st.session_state.data_loaded = True

        with st.expander("🔍 Kiểm tra dữ liệu"):
            if st.session_state.get("data_loaded", False):
                df = st.session_state.df
                st.write("**Giá trị thiếu:**")
                st.write(df.isnull().sum())
                st.write("**Kiểu dữ liệu:**")
                st.write(df.dtypes)
            else:
                st.warning("Vui lòng tải dữ liệu trước.")

        with st.expander("⚙️ Xử lý dữ liệu"):
            if st.session_state.get("data_loaded", False):
                df = st.session_state.df.copy()
                st.write("**Xử lý dữ liệu:** Loại bỏ cột không cần thiết, điền giá trị thiếu, mã hóa biến phân loại, và chuẩn hóa dữ liệu.")

                default_cols = ["PassengerId", "Name", "Ticket", "Cabin"]
                valid_default_cols = [col for col in default_cols if col in df.columns]
                dropped_cols = st.multiselect("Chọn cột cần loại bỏ:", 
                                              df.columns.tolist(), 
                                              default=valid_default_cols)
                df.drop(columns=dropped_cols, errors='ignore', inplace=True)
                st.write(f"Đã loại bỏ các cột: {', '.join(dropped_cols)}")

                st.write("**Điền giá trị thiếu:**")
                missing_cols = df.columns[df.isnull().any()].tolist()
                if missing_cols:
                    st.write(f"Các cột có giá trị thiếu: {', '.join(missing_cols)}")
                    fill_missing_cols = st.multiselect("Chọn cột để điền giá trị thiếu:", 
                                                       missing_cols, 
                                                       default=missing_cols)
                    fill_method = st.selectbox("Chọn phương pháp điền cho tất cả cột:", 
                                              ["trung vị (median)", "trung bình (mean)", "mode", "loại bỏ"],
                                              index=0)
                    for col in fill_missing_cols:
                        if df[col].dtype in ['float64', 'int64']:
                            if fill_method == "trung vị (median)":
                                df[col].fillna(df[col].median(), inplace=True)
                                st.write(f"- Đã điền cột {col} bằng trung vị: {df[col].median()}")
                            elif fill_method == "trung bình (mean)":
                                df[col].fillna(df[col].mean(), inplace=True)
                                st.write(f"- Đã điền cột {col} bằng trung bình: {df[col].mean():.2f}")
                            elif fill_method == "loại bỏ":
                                df.dropna(subset=[col], inplace=True)
                                st.write(f"- Đã loại bỏ các hàng thiếu giá trị ở cột {col}")
                        else:
                            if fill_method == "mode":
                                mode_value = df[col].mode()[0]
                                df[col].fillna(mode_value, inplace=True)
                                st.write(f"- Đã điền cột {col} bằng mode: {mode_value}")
                            elif fill_method == "loại bỏ":
                                df.dropna(subset=[col], inplace=True)
                                st.write(f"- Đã loại bỏ các hàng thiếu giá trị ở cột {col}")
                else:
                    st.info("Không có cột nào thiếu giá trị sau khi loại bỏ cột.")

                st.write("**Mã hóa các biến phân loại:**")
                categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
                allowed_cols = ["Sex", "Embarked"]
                cols_to_encode = st.multiselect("Chọn cột để mã hóa (chỉ Sex và Embarked được phép):", 
                                                categorical_cols, 
                                                default=[col for col in allowed_cols if col in categorical_cols])
                valid_cols = [col for col in cols_to_encode if col in allowed_cols]
                invalid_cols = [col for col in cols_to_encode if col not in allowed_cols]
                if invalid_cols:
                    st.error(f"Các cột sau không được phép mã hóa: {', '.join(invalid_cols)}.")
                for col in valid_cols:
                    df[col] = df[col].astype('category').cat.codes
                    st.write(f"- Đã mã hóa cột {col}: {dict(enumerate(df[col].astype('category').cat.categories))}")

                st.write("**Chuẩn hóa dữ liệu số:**")
                numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
                if "Survived" in numeric_cols:
                    numeric_cols.remove("Survived")
                norm_method = st.selectbox("Chọn phương pháp chuẩn hóa:", 
                                           ["Min-Max Scaling", "Standard Scaling"], 
                                           key="norm_method")
                if norm_method == "Min-Max Scaling":
                    scaler = MinMaxScaler(feature_range=(0, 1))
                    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
                    st.write(f"- Đã chuẩn hóa các cột số bằng Min-Max Scaling: {', '.join(numeric_cols)}")
                else:
                    scaler = StandardScaler()
                    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
                    st.write(f"- Đã chuẩn hóa các cột số bằng Standard Scaling: {', '.join(numeric_cols)}")

                st.success("Dữ liệu đã được xử lý!")
                st.write("Dữ liệu sau khi xử lý:")
                st.write(df.head(10))
                st.session_state.df = df
                st.session_state.data_processed = True
            else:
                st.warning("Vui lòng tải dữ liệu trước.")

        with st.expander("📊 Chia dữ liệu"):
            if st.session_state.get("data_processed", False) and st.session_state.df is not None:
                df = st.session_state.df.copy()
                if "Survived" not in df.columns:
                    st.error("Không tìm thấy cột mục tiêu 'Survived'.")
                else:
                    st.write("### Bước 1: Chọn số lượng mẫu")
                    total_samples = len(df)
                    st.write(f"Tổng số mẫu hiện có: {total_samples}")
                    num_samples = st.slider("Chọn số lượng mẫu để sử dụng:", 
                                            min_value=10, 
                                            max_value=total_samples, 
                                            value=min(100, total_samples), 
                                            step=1)
                    if st.button("Xác nhận số lượng mẫu"):
                        st.session_state["selected_samples"] = df.sample(n=num_samples, random_state=42)
                        st.success(f"Đã chốt {num_samples} mẫu để sử dụng!")

                    if st.session_state["selected_samples"] is not None:
                        selected_df = st.session_state["selected_samples"]
                        total_selected_samples = len(selected_df)
                        st.write(f"### Bước 2: Chia dữ liệu ({total_selected_samples} mẫu đã chốt)")
                        
                        test_pct = st.slider("Tỷ lệ tập Test (%)", 
                                             0, 100, 20)
                        test_size = int(total_selected_samples * test_pct / 100)
                        
                        remaining_df = selected_df.copy()
                        if test_size > 0:
                            X_temp, X_test, y_temp, y_test = train_test_split(
                                remaining_df.drop(columns=["Survived"]),
                                remaining_df["Survived"],
                                test_size=test_size / len(remaining_df),
                                random_state=42
                            )
                            remaining_df = pd.concat([X_temp, y_temp], axis=1)
                        else:
                            X_test, y_test = pd.DataFrame(), pd.Series()
                        
                        remaining_samples = len(remaining_df)
                        valid_pct = st.slider("Tỷ lệ tập Validation (%) từ phần còn lại", 
                                              0, 100, 20)
                        valid_size = int(remaining_samples * valid_pct / 100)
                        
                        if valid_size > 0 and len(remaining_df) > valid_size:
                            X_train, X_val, y_train, y_val = train_test_split(
                                remaining_df.drop(columns=["Survived"]),
                                remaining_df["Survived"],
                                test_size=valid_size / len(remaining_df),
                                random_state=42
                            )
                        else:
                            X_train, y_train = remaining_df.drop(columns=["Survived"]), remaining_df["Survived"]
                            X_val, y_val = pd.DataFrame(), pd.Series()

                        train_size = len(X_train)
                        train_pct = (train_size / total_selected_samples) * 100
                        st.write(f"**Train:** {train_size} mẫu ({train_pct:.1f}%)")
                        st.write(f"**Validation:** {valid_size} mẫu ({valid_pct}%)")
                        st.write(f"**Test:** {test_size} mẫu ({test_pct}%)")

                        if st.button("Xác nhận chia dữ liệu"):
                            min_samples = 10
                            if train_size < min_samples:
                                st.warning("Số mẫu tập Train quá ít.")
                            if valid_size < min_samples and valid_size > 0:
                                st.warning("Số mẫu tập Validation quá ít.")
                            if test_size < min_samples and test_size > 0:
                                st.warning("Số mẫu tập Test quá ít.")
                            st.session_state.X_train = X_train
                            st.session_state.y_train = y_train
                            st.session_state.X_val = X_val
                            st.session_state.y_val = y_val
                            st.session_state.X_test = X_test
                            st.session_state.y_test = y_test
                            st.session_state.data_split = True
                            st.success("Dữ liệu đã được chia thành công!")
            else:
                st.warning("Vui lòng tải dữ liệu trước.")

    # ---------------- Tab 3: Huấn luyện mô hình ----------------
    with tab_train:
        st.header("Huấn luyện & Kiểm thử mô hình")
        if st.session_state.get("data_split", False):
            col_model, col_model_tip = st.columns([0.8, 0.2])
            with col_model:
                model_choice_to_train = st.selectbox("Chọn mô hình để huấn luyện:", 
                                                    ["Hồi quy Đa biến", "Hồi quy Đa thức"])
            with col_model_tip:
                st.markdown("""
                <span class="tooltip">? 
                  <span class="tooltiptext">
                    <strong>Hồi quy Đa biến</strong>: \(\hat{y} = \beta_0 + \beta_1 x_1 + \dots\).<br>
                    <strong>Hồi quy Đa thức</strong>: \(\hat{y} = \beta_0 + \beta_1 x + \beta_2 x^2 + \dots\).
                  </span>
                </span>
                """, unsafe_allow_html=True)

            col_lr, col_lr_tip = st.columns([0.8, 0.2])
            with col_lr:
                lr_method = st.selectbox("Chọn phương pháp learning rate:", 
                                        ["constant", "invscaling", "adaptive"], 
                                        index=0)
            with col_lr_tip:
                st.markdown("""
                <span class="tooltip">? 
                  <span class="tooltiptext">
                    Learning rate (\(\eta\)) điều chỉnh tốc độ cập nhật trọng số \(\beta\).
                  </span>
                </span>
                """, unsafe_allow_html=True)

            eta0 = 0.01
            if lr_method == "constant":
                col_eta, col_eta_tip = st.columns([0.8, 0.2])
                with col_eta:
                    eta0 = st.number_input("Nhập giá trị eta0:", 
                                           value=0.01, min_value=0.0001, max_value=1.0, 
                                           step=0.0001, format="%.4f")
                with col_eta_tip:
                    st.markdown("""
                    <span class="tooltip">? 
                      <span class="tooltiptext">
                        <strong>eta0</strong>: Learning rate ban đầu trong "constant".
                      </span>
                    </span>
                    """, unsafe_allow_html=True)

            poly_degree = 1
            if model_choice_to_train == "Hồi quy Đa thức":
                col_poly, col_poly_tip = st.columns([0.8, 0.2])
                with col_poly:
                    poly_degree = st.number_input("Chọn bậc của đa thức:", 
                                                  min_value=1, max_value=5, value=2)
                with col_poly_tip:
                    st.markdown("""
                    <span class="tooltip">? 
                      <span class="tooltiptext">
                        <strong>Bậc đa thức</strong>: Số mũ cao nhất trong hồi quy đa thức.
                      </span>
                    </span>
                    """, unsafe_allow_html=True)

            col_fold, col_fold_tip = st.columns([0.8, 0.2])
            with col_fold:
                num_folds = st.number_input("Chọn số folds (KFold Cross-Validation):", 
                                            min_value=2, max_value=20, value=5, step=1)
            with col_fold_tip:
                st.markdown("""
                <span class="tooltip">? 
                  <span class="tooltiptext">
                    <strong>Cross Validation (K-Fold)</strong>: Chia dữ liệu thành \(K\) phần.
                  </span>
                </span>
                """, unsafe_allow_html=True)

            if st.button("Huấn luyện mô hình"):
                X_train = st.session_state.X_train
                y_train = st.session_state.y_train
                X_val = st.session_state.X_val
                y_val = st.session_state.y_val
                X_test = st.session_state.X_test
                y_test = st.session_state.y_test

                if X_train.isnull().values.any() or y_train.isnull().values.any():
                    st.error("Dữ liệu huấn luyện chứa giá trị NaN. Vui lòng kiểm tra lại bước xử lý dữ liệu.")
                    return
                if not np.isfinite(X_train.values).all() or not np.isfinite(y_train.values).all():
                    st.error("Dữ liệu huấn luyện chứa giá trị vô cực. Vui lòng kiểm tra lại.")
                    return

                # Khởi tạo thanh tiến trình và trạng thái
                progress_bar = st.progress(0)
                status_text = st.empty()
                start_time = time.time()

                with st.spinner("Đang huấn luyện mô hình với Cross Validation..."):
                    run_name = f"{model_choice_to_train}_Run_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}"
                    with mlflow.start_run(run_name=run_name) as run:
                        # Log tham số
                        params = {
                            "model_choice": model_choice_to_train,
                            "learning_rate_method": lr_method,
                            "eta0": eta0 if lr_method == "constant" else "N/A",
                            "poly_degree": poly_degree if model_choice_to_train == "Hồi quy Đa thức" else "N/A",
                            "num_folds": num_folds
                        }
                        mlflow.log_params(params)

                        # Huấn luyện mô hình
                        max_iter = 1000
                        tol = 1e-3

                        if model_choice_to_train == "Hồi quy Đa biến":
                            model = SGDRegressor(
                                learning_rate=lr_method,
                                eta0=eta0 if lr_method == "constant" else 0.01,
                                max_iter=max_iter,
                                tol=tol,
                                random_state=42
                            )
                        else:
                            model = Pipeline([
                                ('poly', PolynomialFeatures(degree=poly_degree, include_bias=False)),
                                ('sgd', SGDRegressor(
                                    learning_rate=lr_method,
                                    eta0=eta0 if lr_method == "constant" else 0.01,
                                    max_iter=max_iter,
                                    tol=tol,
                                    random_state=42
                                ))
                            ])

                        # Quy trình huấn luyện
                        status_text.text("Đang thực hiện Cross Validation...")
                        progress_bar.progress(0)
                        cv_scores = cross_val_score(model, X_train, y_train, cv=num_folds, scoring='r2')
                        progress_bar.progress(40)

                        status_text.text("Đang huấn luyện mô hình...")
                        model.fit(X_train, y_train)
                        progress_bar.progress(70)

                        status_text.text("Đang tính toán chỉ số...")
                        y_pred_val = model.predict(X_val)
                        mse_val = mean_squared_error(y_val, y_pred_val)
                        r2_val = r2_score(y_val, y_pred_val)
                        y_pred_val_class = [1 if pred >= 0.5 else 0 for pred in y_pred_val]
                        accuracy_val = accuracy_score(y_val, y_pred_val_class)

                        y_pred_test = model.predict(X_test)
                        mse_test = mean_squared_error(y_test, y_pred_test)
                        r2_test = r2_score(y_test, y_pred_test)
                        y_pred_test_class = [1 if pred >= 0.5 else 0 for pred in y_pred_test]
                        accuracy_test = accuracy_score(y_test, y_pred_test_class)
                        progress_bar.progress(90)

                        status_text.text("Đang lưu kết quả...")
                        training_time = time.time() - start_time

                        # Log metrics
                        metrics = {
                            "mean_cv_score_r2": float(np.mean(cv_scores)),
                            "mse_val": float(mse_val),
                            "r2_val": float(r2_val),
                            "accuracy_val": float(accuracy_val),
                            "mse_test": float(mse_test),
                            "r2_test": float(r2_test),
                            "accuracy_test": float(accuracy_test),
                            "training_time_seconds": float(training_time)
                        }
                        mlflow.log_metrics(metrics)

                        # Lưu model
                        mlflow.sklearn.log_model(model, "model")

                        # Lưu kết quả vào session_state
                        st.session_state["run_id"] = run.info.run_id
                        st.session_state["run_name"] = run_name
                        st.session_state["cv_scores"] = cv_scores
                        st.session_state["mse_val"] = mse_val
                        st.session_state["r2_val"] = r2_val
                        st.session_state["accuracy_val"] = accuracy_val
                        st.session_state["mse_test"] = mse_test
                        st.session_state["r2_test"] = r2_test
                        st.session_state["accuracy_test"] = accuracy_test
                        st.session_state["y_pred_val"] = y_pred_val
                        st.session_state["y_pred_test"] = y_pred_test
                        st.session_state["training_time"] = training_time
                        st.session_state["params"] = params
                        st.session_state["model"] = model
                        st.session_state["models_trained"] = True

                        progress_bar.progress(100)
                        status_text.text("Hoàn tất huấn luyện!")

            # Hiển thị kết quả nếu đã huấn luyện
            if st.session_state.get("models_trained", False):
                st.subheader("Thông tin Kết quả")
                with st.expander("Xem chi tiết kết quả", expanded=True):
                    st.markdown("#### Thông tin lần chạy:", unsafe_allow_html=True)
                    st.write(f"- **Tên lần chạy (Run Name)**: {st.session_state['run_name']}")
                    st.write(f"- **ID lần chạy (Run ID)**: {st.session_state['run_id']}")

                    st.markdown("#### Cài đặt bạn đã chọn:", unsafe_allow_html=True)
                    st.write(f"- **Phương pháp**: {st.session_state['params']['model_choice']}")
                    if st.session_state['params']['model_choice'] == "Hồi quy Đa thức":
                        st.write(f"- **Bậc đa thức**: {st.session_state['params']['poly_degree']}")
                    st.write(f"- **Phương pháp Learning Rate**: {st.session_state['params']['learning_rate_method']}")
                    if st.session_state['params']['learning_rate_method'] == "constant":
                        st.write(f"- **Eta0**: {st.session_state['params']['eta0']}")
                    st.write(f"- **Số folds Cross Validation**: {st.session_state['params']['num_folds']}")
                    st.write(f"- **Thời gian huấn luyện**: {st.session_state['training_time']:.2f} giây")

                    st.markdown("#### Kết quả đạt được:", unsafe_allow_html=True)
                    st.write(f"- **Mean CV Score (R²)**: {np.mean(st.session_state['cv_scores']):.2f}")
                    st.write(f"- **Validation MSE**: {st.session_state['mse_val']:.2f}")
                    st.write(f"- **Validation R²**: {st.session_state['r2_val']:.2f}")
                    st.write(f"- **Validation Accuracy (ngưỡng 0.5)**: {st.session_state['accuracy_val']:.2f}")
                    st.write(f"- **Test MSE**: {st.session_state['mse_test']:.2f}")
                    st.write(f"- **Test R²**: {st.session_state['r2_test']:.2f}")
                    st.write(f"- **Test Accuracy (ngưỡng 0.5)**: {st.session_state['accuracy_test']:.2f}")
                    st.markdown(f"""
                    - **Nhận xét**:  
                      *=> Mô hình đạt độ chính xác {st.session_state['accuracy_test']:.2f} trên tập test, cho thấy khả năng tổng quát hóa { 'tốt' if st.session_state['accuracy_test'] > 0.8 else 'trung bình' if st.session_state['accuracy_test'] > 0.6 else 'kém'}.*  
                      Mean CV Score (R²) gần 1 và MSE nhỏ cho thấy mô hình khớp tốt với dữ liệu.
                    """, unsafe_allow_html=True)

                st.markdown("### Biểu đồ Actual vs Predicted (Validation)")
                fig, ax = plt.subplots()
                sns.scatterplot(x=st.session_state.y_val, y=st.session_state['y_pred_val'], ax=ax)
                ax.plot([0, 1], [0, 1], 'r--')
                ax.set_xlabel("Thực tế")
                ax.set_ylabel("Dự đoán")
                st.pyplot(fig)

                st.markdown("### Biểu đồ Actual vs Predicted (Test)")
                fig2, ax2 = plt.subplots()
                sns.scatterplot(x=st.session_state.y_test, y=st.session_state['y_pred_test'], ax=ax2)
                ax2.plot([0, 1], [0, 1], 'r--')
                ax2.set_xlabel("Thực tế")
                ax2.set_ylabel("Dự đoán")
                st.pyplot(fig2)
        else:
            st.warning("Vui lòng chia tập dữ liệu trước.")

    # ---------------- Tab 4: Dự đoán ----------------
    with tab_predict:
        st.header("Demo Dự đoán")
        if st.session_state.get("models_trained", False):
            mode = st.radio("Chọn phương thức dự đoán:", ["Nhập thông tin thủ công", "Dữ liệu từ Test"])

            progress_bar = st.progress(0)
            status_text = st.empty()

            if mode == "Nhập thông tin thủ công":
                st.write("Nhập thông tin hành khách:")
                df = st.session_state.df
                features = df.drop(columns=["Survived"]).columns.tolist()
                input_values = []
                for feature in features:
                    if np.issubdtype(df[feature].dtype, np.number):
                        default_value = int(round(abs(df[feature].median())))  # Chuyển về số nguyên
                        value = st.number_input(f"{feature}:", value=default_value, step=1, format="%d", key=f"input_{feature}")
                    else:
                        options = list(sorted(df[feature].unique()))
                        value = st.selectbox(f"{feature}:", options, key=f"input_{feature}")
                    input_values.append(value)
                
                if st.button("Dự đoán"):
                    with st.spinner("Đang dự đoán..."):
                        for i in range(0, 51, 5):
                            progress_bar.progress(i)
                            status_text.text(f"Đang xử lý {i}%{i % 4 * '.'}")
                            time.sleep(0.1)

                        input_array = np.array(input_values).reshape(1, -1)
                        prediction = st.session_state.model.predict(input_array)[0]
                        prediction = np.clip(prediction, 0, 1)
                        
                        if prediction >= 0.5:
                            result = "Sống"
                            confidence = int(round(prediction * 100))  # Chuyển về số nguyên
                        else:
                            result = "Không sống"
                            confidence = int(round((1 - prediction) * 100))  # Chuyển về số nguyên

                        for i in range(50, 101, 5):
                            progress_bar.progress(i)
                            status_text.text(f"Đang hoàn tất {i}%{i % 4 * '.'}")
                            time.sleep(0.1)

                        st.success(f"Dự đoán: **{result}** | Độ tin cậy: **{confidence}%**")
                        
                        time.sleep(1)
                        progress_bar.empty()
                        status_text.empty()

            elif mode == "Dữ liệu từ Test":
                X_test = st.session_state['X_test']
                y_test = st.session_state['y_test']
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

                            sample = X_test.iloc[idx].values.reshape(1, -1)
                            prediction = st.session_state.model.predict(sample)[0]
                            prediction = np.clip(prediction, 0, 1)
                            
                            if prediction >= 0.5:
                                result = "Sống"
                                confidence = int(round(prediction * 100))  # Chuyển về số nguyên
                            else:
                                result = "Không sống"
                                confidence = int(round((1 - prediction) * 100))  # Chuyển về số nguyên

                            y_true = "Sống" if y_test.iloc[idx] == 1 else "Không sống"

                            for i in range(50, 101, 5):
                                progress_bar.progress(i)
                                status_text.text(f"Đang hoàn tất {i}%{i % 4 * '.'}")
                                time.sleep(0.1)

                            st.success(f"Dự đoán: **{result}** | Độ tin cậy: **{confidence}%** | Giá trị thực: **{y_true}**")
                            
                            time.sleep(1)
                            progress_bar.empty()
                            status_text.empty()

        else:
            st.warning("Vui lòng huấn luyện mô hình trước.")

    # ---------------- Tab 5: Thông tin huấn luyện & MLflow UI ----------------
    with tab_mlflow:
        st.header("Theo dõi kết quả")
        st.markdown("""
        Tab này cho phép bạn xem danh sách các lần huấn luyện, đổi tên, xóa và xem chi tiết tham số cùng kết quả của từng run.
        """, unsafe_allow_html=True)
        
        try:
            client = MlflowClient()
            experiment = client.get_experiment_by_name("Titanic")
            if not experiment:
                st.error("Không tìm thấy experiment 'Titanic'. Vui lòng kiểm tra lại MLflow tracking URI.")
            else:
                experiment_id = experiment.experiment_id
                runs = client.search_runs(experiment_ids=[experiment_id], order_by=["attributes.start_time DESC"])
                
                if not runs:
                    st.info("Chưa có lần chạy nào được ghi nhận.")
                else:
                    run_options = {run.info.run_id: run.data.tags.get('mlflow.runName', f"Run_{run.info.run_id}") for run in runs}
                    run_names = list(run_options.values())

                    default_run_name = st.session_state.get('run_name', run_names[0]) if 'run_name' in st.session_state else run_names[0]

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
                                if 'run_name' in st.session_state and st.session_state['run_id'] == selected_run_id:
                                    st.session_state['run_name'] = new_run_name.strip()
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
                            if 'run_id' in st.session_state and st.session_state['run_id'] == selected_run_id:
                                del st.session_state['run_id']
                                del st.session_state['run_name']
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
                        metrics_display = {
                            "Thời gian huấn luyện (giây)": f"{selected_run.data.metrics.get('training_time_seconds', 'N/A'):.2f}",
                            "Mean CV Score (R²)": f"{selected_run.data.metrics.get('mean_cv_score_r2', 'N/A'):.2f}",
                            "Validation MSE": f"{selected_run.data.metrics.get('mse_val', 'N/A'):.2f}",
                            "Validation R²": f"{selected_run.data.metrics.get('r2_val', 'N/A'):.2f}",
                            "Validation Accuracy": f"{selected_run.data.metrics.get('accuracy_val', 'N/A'):.2f}",
                            "Test MSE": f"{selected_run.data.metrics.get('mse_test', 'N/A'):.2f}",
                            "Test R²": f"{selected_run.data.metrics.get('r2_test', 'N/A'):.2f}",
                            "Test Accuracy": f"{selected_run.data.metrics.get('accuracy_test', 'N/A'):.2f}"
                        }
                        st.json(metrics_display, expanded=True)
                    else:
                        st.write("Không có kết quả được ghi nhận.")

                    st.subheader("Truy cập MLflow UI")
                    if st.button("Mở MLflow UI trên Dagshub"):
                        st.write(f"Đang chuyển hướng tới: https://dagshub.com/huykibo/streamlit_mlflow.mlflow")
                        st.markdown(f'<meta http-equiv="refresh" content="0;URL=https://dagshub.com/huykibo/streamlit_mlflow.mlflow">', unsafe_allow_html=True)
        except Exception as e:
            st.error(f"Lỗi kết nối MLflow: {e}. Vui lòng kiểm tra MLFLOW_TRACKING_URI và thông tin xác thực.")

if __name__ == "__main__":
    run_titanic_app()