import os
import mlflow
import streamlit as st
import openml
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans, DBSCAN
import matplotlib.pyplot as plt
from mlflow.tracking import MlflowClient
from datetime import datetime
from sklearn.impute import SimpleImputer
import time
import plotly.express as px
import plotly.graph_objects as go
from scipy.spatial import ConvexHull
from sklearn.decomposition import PCA
import streamlit.components.v1 as components

def run_mnist_clustering_app():
    st.title("Ứng dụng Phân cụm Dữ liệu MNIST")

    st.markdown("""
        <style>
            .inline-container {
                display: inline-flex;
                align-items: center;
                gap: 5px;
            }
        </style>
    """, unsafe_allow_html=True)

    tab_info, tab_load, tab_cluster, tab_log_info = st.tabs(["Thông tin", "Tải dữ liệu", "Phân cụm", "Theo dõi kết quả"])

    with tab_info:
        st.header("Giới thiệu về Phân cụm Dữ liệu MNIST")
        st.markdown("""
        Chào bạn! Đây là ứng dụng giúp bạn hiểu cách phân nhóm các chữ số viết tay từ tập dữ liệu **MNIST** – một tập hợp gồm $70,000$ ảnh, mỗi ảnh là một chữ số từ $0$ đến $9$. Chúng ta sẽ dùng hai phương pháp phân cụm chính: **K-means** và **DBSCAN**. Hãy cùng khám phá nhé!
        """, unsafe_allow_html=True)

        st.subheader("1. MNIST là gì? Tại sao cần phân cụm?")
        st.markdown("""
        - **MNIST**: Tập dữ liệu gồm $70,000$ ảnh chữ số viết tay, mỗi ảnh có kích thước $28 \\times 28$ pixel (tổng cộng $784$ đặc trưng mỗi ảnh).  
        - **Mục tiêu phân cụm**:  
          - Gom các chữ số giống nhau vào cùng một nhóm (ví dụ: tất cả số $1$ vào một nhóm).  
          - Trực quan hóa dữ liệu bằng biểu đồ $2D$ hoặc $3D$.  
          - Tiết kiệm thời gian phân tích, hỗ trợ các tác vụ như nhận diện chữ số sau này.  
        """, unsafe_allow_html=True)

        st.subheader("Minh họa dữ liệu MNIST")
        st.markdown("""
        Dưới đây là $10$ ảnh mẫu từ tập dữ liệu MNIST (từ $0$ đến $9$) để bạn hình dung. Mỗi ảnh là một chữ số viết tay được biểu diễn dưới dạng ma trận $28 \\times 28$ pixel.
        """, unsafe_allow_html=True)

        with st.spinner("Đang tải ảnh mẫu..."):
            mnist = openml.datasets.get_dataset(554)
            X, y, _, _ = mnist.get_data(target=mnist.default_target_attribute)
            
            sample_images = []
            sample_labels = []
            for digit in range(10):
                digit_indices = np.where(y == str(digit))[0]
                if len(digit_indices) > 0:
                    selected_idx = digit_indices[0]
                    sample_images.append(X.iloc[selected_idx].values)
                    sample_labels.append(y.iloc[selected_idx])
            
            fig, axes = plt.subplots(2, 5, figsize=(15, 6))
            for i, (img, label) in enumerate(zip(sample_images, sample_labels)):
                row = i // 5
                col = i % 5
                axes[row, col].imshow(img.reshape(28, 28), cmap='gray')
                axes[row, col].set_title(f'Nhãn: {label}')
                axes[row, col].axis('off')
            plt.tight_layout()
            st.pyplot(fig)
            
            st.markdown("""
            - **Ghi chú**: Mỗi ảnh là một ma trận $28 \\times 28$ pixel, với giá trị từ $0$ (trắng) đến $255$ (đen). Nhãn thực tế ($0$-$9$) chỉ được dùng để minh họa, không sử dụng trong phân cụm.
            """, unsafe_allow_html=True)

        st.subheader("2. Tìm hiểu về K-means và DBSCAN")
        st.markdown("Chọn một phần để xem chi tiết nhé:")
        info_option = st.selectbox(
            "",
            ["K-means là gì?", "DBSCAN là gì?", "So sánh K-means và DBSCAN"],
            label_visibility="collapsed",
            help="Chọn để xem thông tin chi tiết về từng phương pháp hoặc so sánh chúng."
        )
        if info_option == "K-means là gì?":
            st.subheader("📊 K-means – Thuật toán phân cụm dựa trên khoảng cách")
            st.markdown("""
            **K-means** là một thuật toán phân cụm không giám sát (unsupervised learning) phổ biến, giúp nhóm các điểm dữ liệu thành $K$ cụm dựa trên sự tương đồng về khoảng cách. Đây là một trong những thuật toán phân cụm đơn giản và hiệu quả nhất trong thực tế.
            """, unsafe_allow_html=True)

            st.subheader("📘 1. Khái niệm cơ bản")
            st.markdown("""
            ##### 🔹 **Tâm cụm (Centroid)**  
            Là điểm trung bình của tất cả các điểm trong cụm. Tâm cụm không nhất thiết phải là một điểm dữ liệu thực tế mà chỉ là điểm đại diện.

            ##### 🔹 **Khoảng cách Euclidean**  
            Khoảng cách phổ biến để đo độ gần giữa hai điểm:  
            $$ \\text{Distance}(p, q) = \\sqrt{(x_q - x_p)^2 + (y_q - y_p)^2} $$  
            - **Giải thích**:  
              - $\\text{Distance}(p, q)$: Khoảng cách giữa hai điểm $p$ và $q$.  
              - $p, q$: Hai điểm với tọa độ lần lượt là $(x_p, y_p)$ và $(x_q, y_q)$.  
              - $x_p, y_p, x_q, y_q$: Tọa độ $x$, $y$ của các điểm $p$ và $q$.  
              - $\\sqrt{}$: Căn bậc hai.  
            """, unsafe_allow_html=True)

            st.subheader("📷 2. Minh họa quá trình K-means")
            st.markdown("""
            Hình ảnh dưới đây minh họa cách K-means hoạt động: Dữ liệu được phân thành các cụm với tâm cụm được đánh dấu bằng ký hiệu $\\times$.
            """, unsafe_allow_html=True)
            st.image("1k.png", use_container_width=True)
            st.caption("Nguồn: [https://towardsdatascience.com/K-means-a-complete-introduction-1702af9cd8c](https://towardsdatascience.com/K-means-a-complete-introduction-1702af9cd8c)")

            st.subheader("🛠️ 3. Thuật toán K-means – Các bước thực hiện")
            st.markdown("""
            Thuật toán K-means thực hiện theo các bước sau:  
            1. **Khởi tạo**:  
               - Chọn số lượng cụm $K$.  
               - Chọn ngẫu nhiên $K$ điểm làm tâm cụm ban đầu (hoặc dùng phương pháp **K-means++** để tối ưu).  

            2. **Gán điểm vào cụm gần nhất**:  
               - Tính khoảng cách từ mỗi điểm đến từng tâm cụm.  
               - Gán điểm đó vào cụm có tâm gần nhất:  
               $$ C_i = \\{ x_j : \\text{Distance}(x_j, \\mu_i) \\leq \\text{Distance}(x_j, \\mu_k), \\ \\forall k \\} $$  
               - **Giải thích**:  
                 - $C_i$: Cụm thứ $i$.  
                 - $x_j$: Điểm dữ liệu.  
                 - $\\mu_i, \\mu_k$: Tâm cụm của cụm $i$ và cụm $k$.  
                 - $\\leq$: Nhỏ hơn hoặc bằng.  
                 - $\\forall k$: Với mọi $k$.  

            3. **Cập nhật lại tâm cụm**:  
               - Tính tâm cụm mới bằng trung bình tọa độ các điểm trong cụm:  
               $$ \\mu_i = \\frac{1}{|C_i|} \\sum_{x_j \\in C_i} x_j $$  
               - **Giải thích**:  
                 - $\\mu_i$: Tâm cụm của cụm $i$.  
                 - $|C_i|$: Số điểm trong cụm $C_i$.  
                 - $x_j$: Điểm dữ liệu trong cụm $C_i$.  
                 - $\\sum$: Ký hiệu tổng.  
                 - $\\in$: Thuộc về.  
               - Dịch chuyển tâm cụm đến vị trí trung bình mới.  

            4. **Lặp lại cho đến khi hội tụ**:  
               - Tiếp tục gán lại điểm vào cụm gần nhất.  
               - Cập nhật tâm cụm mới.  
               - **Kết thúc**: Khi tâm cụm không thay đổi hoặc thay đổi rất nhỏ sau mỗi lần cập nhật.  
            """, unsafe_allow_html=True)

            st.subheader("🟩 4. Đánh giá và lựa chọn số cụm $K$")
            st.markdown("""
            ##### 🔹 **Phương pháp Elbow**  
            - Chạy thuật toán với các giá trị $K$ khác nhau.  
            - Tính **Within-Cluster Sum of Squares (WCSS)** – tổng bình phương khoảng cách từ các điểm đến tâm cụm:  
            $$ \\text{WCSS} = \\sum_{i=1}^{K} \\sum_{x_j \\in C_i} \\| x_j - \\mu_i \\|^2 $$  
            - **Giải thích**:  
              - $\\text{WCSS}$: Tổng bình phương khoảng cách trong cụm.  
              - $K$: Số cụm.  
              - $x_j$: Điểm dữ liệu trong cụm $C_i$.  
              - $\\mu_i$: Tâm cụm của cụm $C_i$.  
              - $\\| \\cdot \\|^2$: Bình phương khoảng cách.  
            - Vẽ đồ thị $\\text{WCSS}$ theo từng giá trị $K$.  
            - Chọn $K$ tại điểm gấp khúc (elbow point) – nơi $\\text{WCSS}$ giảm chậm lại.  
            """, unsafe_allow_html=True)

            # Tạo dữ liệu mẫu để minh họa phương pháp Elbow
            np.random.seed(42)
            data = np.concatenate([
                np.random.normal([2, 2], 0.5, size=(30, 2)),
                np.random.normal([5, 5], 0.5, size=(30, 2)),
                np.random.normal([8, 2], 0.5, size=(30, 2))
            ])
            wcss = []
            for k in range(1, 11):
                kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
                kmeans.fit(data)
                wcss.append(kmeans.inertia_)
            fig5, ax5 = plt.subplots(figsize=(6, 4))
            ax5.plot(range(1, 11), wcss, marker='o')
            ax5.set_title("Phương pháp Elbow để chọn $K$")
            ax5.set_xlabel("Số cụm ($K$)")
            ax5.set_ylabel("$\\text{WCSS}$")
            ax5.grid(True)
            st.pyplot(fig5)

            st.subheader("🟪 5. Ưu điểm và nhược điểm")
            st.markdown("""
            ##### ✅ **Ưu điểm**:  
            - Dễ hiểu và dễ triển khai.  
            - Tính toán nhanh, ngay cả với tập dữ liệu lớn.  
            - Kết quả trực quan, dễ phân tích.  

            ##### ❌ **Nhược điểm**:  
            - Phụ thuộc vào giá trị $K$.  
            - Nhạy cảm với tâm cụm khởi tạo (có thể dùng **K-means++** để cải thiện).  
            - Không hiệu quả với các cụm không hình cầu hoặc có mật độ không đồng đều.  
            """, unsafe_allow_html=True)

            st.subheader("📘 6. Ứng dụng thực tế")
            st.markdown("""
            - **Phân khúc khách hàng**: Trong tiếp thị.  
            - **Phân loại văn bản**: Và tài liệu.  
            - **Nén ảnh**: Bằng cách giảm số lượng màu sắc.  
            - **Xử lý ảnh y tế**: Để phát hiện vùng bất thường.  
            """, unsafe_allow_html=True)

            st.subheader("📊 7. Tổng kết")
            st.markdown("""
            **K-means** là một thuật toán mạnh mẽ và linh hoạt trong bài toán phân cụm. Dù có nhược điểm khi gặp dữ liệu phức tạp hoặc chứa nhiễu, nhưng nhờ sự đơn giản và tốc độ nhanh, nó vẫn là lựa chọn hàng đầu trong nhiều bài toán thực tế. Khi kết hợp với các kỹ thuật như **Elbow Method**, **Silhouette Score**, hoặc dùng phiên bản cải tiến như **K-means++**, ta có thể tối ưu hóa kết quả phân cụm rất hiệu quả.
            """, unsafe_allow_html=True)

        elif info_option == "DBSCAN là gì?":
            st.subheader("📈 DBSCAN – Phân nhóm dựa trên mật độ")
            st.markdown("""
            **DBSCAN** (Density-Based Spatial Clustering of Applications with Noise) là một thuật toán phân cụm không giám sát, hoạt động dựa trên mật độ của các điểm dữ liệu. Nó nổi bật với khả năng tìm các cụm có hình dạng bất kỳ và loại bỏ nhiễu hiệu quả, không cần xác định trước số lượng cụm như K-means.
            """, unsafe_allow_html=True)

            st.subheader("📘 1. Khái niệm cơ bản")
            st.markdown("""
            DBSCAN dựa trên hai tham số chính: **eps ($\\epsilon$)** và **minPts**, cùng các khái niệm quan trọng sau:  

            ##### 🔹 **Vùng lân cận Epsilon ($\\epsilon$-neighborhood)**  
            Là tập hợp các điểm nằm trong bán kính $\\epsilon$ quanh một điểm $p$:  
            $$ N_{\\epsilon}(p) = \\{ q \\in D \\mid \\text{Distance}(p, q) \\leq \\epsilon \\} $$  
            - **Giải thích**:  
              - $N_{\\epsilon}(p)$: Vùng lân cận của điểm $p$.  
              - $p, q$: Điểm trung tâm $p$ và điểm khác $q$.  
              - $D$: Tập dữ liệu.  
              - $\\text{Distance}(p, q)$: Khoảng cách từ $p$ đến $q$.  
              - $\\epsilon$: Bán kính tối đa của vùng lân cận.  
              - $\\leq$: Nhỏ hơn hoặc bằng.  
              - $\\in$: Thuộc về.  

            ##### 🔹 **Điểm lõi (Core Point)**  
            Là điểm có ít nhất **minPts** điểm (bao gồm chính nó) trong vùng $N_{\\epsilon}(p)$.

            ##### 🔹 **Khả năng tiếp cận trực tiếp mật độ**  
            Một điểm $q$ được gọi là tiếp cận trực tiếp từ $p$ nếu:  
            - $q$ nằm trong $N_{\\epsilon}(p)$, và  
            - $p$ là điểm lõi (có $|N_{\\epsilon}(p)| \\geq \\text{minPts}$).

            ##### 🔹 **Khả năng tiếp cận mật độ**  
            Điểm $q$ tiếp cận mật độ từ $p$ nếu tồn tại chuỗi điểm $p_1, p_2, \\ldots, p_n$ sao cho:  
            - $p_1 = p$, $p_n = q$,  
            - Mỗi $p_{i+1}$ tiếp cận trực tiếp từ $p_i$.  
            Hai điểm có khả năng tiếp cận mật độ với nhau sẽ thuộc cùng một cụm.
            """, unsafe_allow_html=True)

            st.subheader("🔍 2. Phân loại điểm trong DBSCAN")
            st.markdown("""
            DBSCAN chia các điểm dữ liệu thành $3$ loại:  
            - **Điểm lõi (Core Point)**: Có ít nhất **minPts** điểm trong vùng $N_{\\epsilon}$.  
            - **Điểm biên (Border Point)**: Nằm trong $N_{\\epsilon}$ của một điểm lõi nhưng không đủ **minPts** để tự là điểm lõi.  
            - **Điểm nhiễu (Noise Point)**: Không thuộc vùng lân cận của bất kỳ điểm lõi nào.  
            """, unsafe_allow_html=True)
            st.image("2db.png", caption="Phân loại điểm - Vuông xanh: điểm lõi, Tròn đen: điểm biên, Tròn trắng: nhiễu (minPts = 3). Nguồn: [https://imgur.com/ohzPUif.png](https://imgur.com/ohzPUif.png)", use_container_width=True)

            st.subheader("🛠️ 3. Cách DBSCAN hoạt động")
            st.markdown("""
            DBSCAN sử dụng phương pháp lan truyền để tạo cụm. Các bước chính:  

            1. **Chọn điểm khởi tạo**: Lấy một điểm bất kỳ chưa duyệt trong tập dữ liệu.  
            2. **Xác định điểm lõi**:  
               - Tính $N_{\\epsilon}(p)$. Nếu $|N_{\\epsilon}(p)| \\geq \\text{minPts}$, $p$ là điểm lõi, khởi tạo cụm mới.  
               - Nếu không, đánh dấu $p$ là nhiễu (tạm thời).  
            3. **Lan truyền cụm**:  
               - Từ điểm lõi $p$, kiểm tra các điểm trong $N_{\\epsilon}(p)$.  
               - Với mỗi điểm $q$ trong $N_{\\epsilon}(p)$:  
                 - Nếu $q$ chưa duyệt, tính $N_{\\epsilon}(q)$. Nếu $q$ là điểm lõi, thêm $N_{\\epsilon}(q)$ vào cụm.  
               - Tiếp tục lan truyền cho đến khi không còn điểm lõi nào để mở rộng.  
            4. **Lặp lại**: Quay lại bước $1$ với điểm chưa duyệt để tạo cụm mới, cho đến khi duyệt hết dữ liệu.  
            """, unsafe_allow_html=True)
            st.image("3db.gif", caption="Quá trình lan truyền tạo cụm trong DBSCAN. Nguồn: [https://imgur.com/9D6aAF2.gif](https://imgur.com/9D6aAF2.gif)", use_container_width=True)

            st.subheader("⚙️ 4. Chọn tham số cho DBSCAN")
            st.markdown("""
            Hai tham số chính cần điều chỉnh:  

            ##### 🔹 **minPts**:  
            - Số điểm tối thiểu để một điểm trở thành điểm lõi.  
            - **Gợi ý**: $\\text{minPts} \\geq \\text{số chiều dữ liệu} + 1$. Với dữ liệu lớn/nhiễu, chọn giá trị lớn hơn (ví dụ: $5$-$10$).  

            ##### 🔹 **Epsilon ($\\epsilon$)**:  
            - Khoảng cách tối đa để các điểm được xem là "gần nhau".  
            - **Cách chọn**:  
              - Vẽ đồ thị **k-distance** (khoảng cách đến điểm láng giềng thứ $k$ gần nhất, với $k = \\text{minPts} - 1$).  
              - Chọn $\\epsilon$ tại "điểm khuỷu tay" (elbow point) – nơi khoảng cách tăng đột ngột.  
            - **Lưu ý**:  
              - $\\epsilon$ nhỏ → Nhiều cụm nhỏ, nhiều nhiễu.  
              - $\\epsilon$ lớn → Gộp các cụm thành một.  
            """, unsafe_allow_html=True)

            st.subheader("🟪 5. Ưu điểm và nhược điểm")
            st.markdown("""
            ##### ✅ **Ưu điểm**:  
            - Không cần chọn số cụm trước.  
            - Phát hiện nhiễu tốt (các điểm lẻ loi).  
            - Phù hợp với cụm có hình dạng bất kỳ, mật độ không đồng đều.  

            ##### ❌ **Nhược điểm**:  
            - Chạy chậm hơn K-means với dữ liệu lớn (độ phức tạp $O(n^2)$ nếu không tối ưu).  
            - Kết quả phụ thuộc lớn vào $\\epsilon$ và **minPts**, cần thử nghiệm để chọn giá trị phù hợp.  
            - Không hiệu quả nếu mật độ cụm quá khác biệt.  
            """, unsafe_allow_html=True)

            st.subheader("📊 6. Ứng dụng với MNIST")
            st.markdown("""
            - **Phân cụm chữ số**: Tìm các nhóm chữ số tương tự dựa trên đặc trưng hình ảnh.  
            - **Loại bỏ nhiễu**: Phát hiện các ảnh bất thường hoặc không rõ ràng.  
            - **Thử nghiệm tham số**: Với MNIST ($784$ chiều), cần giảm chiều (dùng PCA) trước khi áp dụng DBSCAN để tăng hiệu quả và chọn $\\epsilon$, **minPts** hợp lý.  
            """, unsafe_allow_html=True)

            st.subheader("📘 7. Tổng kết")
            st.markdown("""
            **DBSCAN** là lựa chọn mạnh mẽ khi bạn muốn phân cụm dữ liệu có nhiễu hoặc hình dạng phức tạp mà không cần biết trước số cụm. Tuy nhiên, việc chọn $\\epsilon$ và **minPts** là yếu tố then chốt để đạt kết quả tốt. Với dữ liệu lớn như MNIST, kết hợp DBSCAN với giảm chiều dữ liệu là cách tiếp cận hiệu quả.
            """, unsafe_allow_html=True)

        elif info_option == "So sánh K-means và DBSCAN":
            st.subheader("So sánh K-means và DBSCAN")
            st.markdown("""
            Dưới đây là bảng so sánh để bạn dễ hình dung sự khác biệt giữa K-means và DBSCAN:  
            | **Tiêu chí**            | **K-means**                          | **DBSCAN**                          |  
            |--------------------------|--------------------------------------|--------------------------------------|  
            | **Cách hoạt động**      | Chia dữ liệu thành $K$ nhóm cố định dựa trên khoảng cách. | Tìm các vùng có nhiều điểm gần nhau, bỏ qua điểm lẻ loi. |  
            | **Số nhóm**             | Phải chọn trước (ví dụ: $10$ nhóm).   | Tự động tìm, không cần chọn.        |  
            | **Tham số chính**       | Số nhóm ($K$).                      | Khoảng cách ($\\epsilon$), số điểm tối thiểu (**minPts**). |  
            | **Tốc độ**             | Nhanh, phù hợp dữ liệu lớn.         | Chậm hơn, đặc biệt với dữ liệu lớn. |  
            | **Xử lý nhiễu**         | Không, tất cả điểm đều thuộc nhóm.  | Có, loại bỏ điểm lẻ loi (nhiễu).    |  
            | **Ứng dụng với MNIST**  | Chia $10$ chữ số thành $10$ nhóm cố định. | Tìm nhóm chữ số bất thường, loại bỏ nhiễu. |  
            """, unsafe_allow_html=True)

    with tab_load:
        st.header("Tải Dữ liệu MNIST")
        st.markdown("""
        Phần này cho phép tải dữ liệu MNIST từ OpenML và chọn số lượng mẫu để phân cụm. Tổng cộng có $70,000$ mẫu, người dùng có thể chọn một phần nhỏ hơn để giảm thời gian xử lý.
        """, unsafe_allow_html=True)

        if st.button("Tải dữ liệu"):
            with st.spinner("Đang tải dữ liệu..."):
                progress_bar = st.progress(0)
                status_text = st.empty()

                mnist = openml.datasets.get_dataset(554)
                progress_bar.progress(20)
                status_text.text("Đã tải 20% - Đang truy xuất dữ liệu...")

                X, y, _, _ = mnist.get_data(target=mnist.default_target_attribute)
                progress_bar.progress(50)
                status_text.text("Đã tải 50% - Đang xử lý dữ liệu...")

                if X.isnull().values.any():
                    progress_bar.progress(70)
                    status_text.text("Đã tải 70% - Đang xử lý giá trị NaN...")
                    imputer = SimpleImputer(strategy='mean')
                    X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns, index=X.index)

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
                st.success("Tải dữ liệu thành công.")
                st.write("Kích thước dữ liệu gốc:", X.shape)

        if 'full_data' in st.session_state:
            X_full, y_full = st.session_state['full_data']
            num_samples = st.slider("Chọn số lượng mẫu:", 
                                    min_value=10, max_value=len(X_full), value=min(1000, len(X_full)), step=1)
            if st.button("Xác nhận số lượng mẫu"):
                with st.spinner(f"Đang xử lý {num_samples} mẫu..."):
                    progress_bar = st.progress(0)
                    status_text = st.empty()

                    df = pd.concat([X_full, y_full.rename("label")], axis=1)
                    progress_bar.progress(30)
                    status_text.text("Đã xử lý 30% - Đang kết hợp dữ liệu...")

                    sampled_df = df.sample(n=num_samples, random_state=42)
                    progress_bar.progress(70)
                    status_text.text("Đã xử lý 70% - Đang lấy mẫu ngẫu nhiên...")

                    X_sampled = sampled_df.drop(columns=["label"])
                    y_sampled = sampled_df["label"]
                    st.session_state['data'] = (X_sampled, y_sampled)
                    progress_bar.progress(90)
                    status_text.text("Đang xử lý 90% - Đang lưu trữ dữ liệu...")

                    with mlflow.start_run(run_name="Data_Sample"):
                        mlflow.log_param("num_samples", num_samples)
                    
                    progress_bar.progress(100)
                    status_text.text("Đã xử lý 100% - Hoàn tất!")
                    time.sleep(1)
                    status_text.empty()
                    progress_bar.empty()
                    st.success(f"Đã chọn {num_samples} mẫu để phân cụm.")

    with tab_cluster:
        st.header("Phân cụm Dữ liệu")
        st.markdown("""
        Phần này giúp bạn gom nhóm dữ liệu MNIST bằng K-means hoặc DBSCAN. Sau khi gom nhóm, bạn sẽ thấy kết quả trên biểu đồ $2D$.  
        **Lưu ý**: Đây là bài toán không giám sát (unsupervised learning), không sử dụng nhãn (chữ số thật) trong quá trình phân cụm.
        """, unsafe_allow_html=True)

        if 'data' not in st.session_state:
            st.info("Vui lòng tải dữ liệu từ tab 'Tải dữ liệu' trước khi thực hiện phân cụm.")
        else:
            X, y = st.session_state['data']
            num_samples = X.shape[0]
            st.write(f"Dữ liệu hiện tại: {num_samples} ảnh, mỗi ảnh có {X.shape[1]} đặc trưng.")

            st.subheader("Cấu hình Phân cụm")
            col1, col2 = st.columns([1, 1])

            with col1:
                cluster_method = st.selectbox(
                    "Chọn cách phân cụm:",
                    ["K-means", "DBSCAN"],
                    help="K-means cần chọn số nhóm trước; DBSCAN tự động tìm nhóm dựa trên mật độ."
                )

            params = {}
            with col2:
                suggestion_data = {
                    "Số lượng mẫu": ["nhỏ hơn 10,000", "10,000–30,000", "lớn hơn 30,000"],
                    "K-means (n_clusters)": ["5–10", "10–20", "20–50"],
                    "DBSCAN (epsilon)": ["2.0–4.0", "3.0–6.0", "5.0–10.0"],
                    "DBSCAN (minPts)": ["3–5", "5–10", "10–20"]
                }

                if num_samples < 10000:
                    range_idx = 0
                elif num_samples <= 30000:
                    range_idx = 1
                else:
                    range_idx = 2

                suggested_n_clusters = None
                suggested_eps = None
                suggested_min_samples = None

                if cluster_method == "K-means":
                    st.markdown("**Số nhóm ($n_{\\text{clusters}}$)**", unsafe_allow_html=True)
                    range_str = suggestion_data["K-means (n_clusters)"][range_idx]
                    start, end = map(int, range_str.split("–"))
                    suggested_n_clusters = (start + end) // 2
                    n_clusters = st.number_input(
                        "",
                        min_value=2, max_value=50, value=suggested_n_clusters, step=1,
                        label_visibility="collapsed",
                        help=f"Gợi ý: {range_str}. Giá trị tối ưu tự động: {suggested_n_clusters}"
                    )
                    params["n_clusters"] = n_clusters
                else:
                    st.markdown("**Khoảng cách tối đa ($\\epsilon$)**", unsafe_allow_html=True)
                    range_str_eps = suggestion_data["DBSCAN (epsilon)"][range_idx]
                    start_eps, end_eps = map(float, range_str_eps.split("–"))
                    suggested_eps = (start_eps + end_eps) / 2
                    eps = st.number_input(
                        "",
                        min_value=0.1, max_value=10.0, value=suggested_eps, step=0.1,
                        label_visibility="collapsed",
                        help=f"Gợi ý: {range_str_eps}. Giá trị tối ưu tự động: {suggested_eps}"
                    )

                    st.markdown("**Số điểm tối thiểu ($\\text{minPts}$)**", unsafe_allow_html=True)
                    range_str_minpts = suggestion_data["DBSCAN (minPts)"][range_idx]
                    start_minpts, end_minpts = map(int, range_str_minpts.split("–"))
                    suggested_min_samples = (start_minpts + end_minpts) // 2
                    min_samples = st.number_input(
                        "",
                        min_value=2, max_value=20, value=suggested_min_samples, step=1,
                        label_visibility="collapsed",
                        help=f"Gợi ý: {range_str_minpts}. Giá trị tối ưu tự động: {suggested_min_samples}"
                    )
                    params["eps"] = eps
                    params["min_samples"] = min_samples

            st.subheader("Gợi ý tham số tối ưu dựa trên số lượng dữ liệu")
            st.markdown(
                f"Dựa trên số lượng mẫu hiện tại (**{num_samples} mẫu**), dưới đây là gợi ý tham số tối ưu:",
                unsafe_allow_html=True
            )
            if cluster_method == "K-means":
                st.table({
                    "Số lượng mẫu": suggestion_data["Số lượng mẫu"],
                    "K-means ($n_{\\text{clusters}}$)": suggestion_data["K-means (n_clusters)"]
                })
            else:
                st.table({
                    "Số lượng mẫu": suggestion_data["Số lượng mẫu"],
                    "DBSCAN ($\\epsilon$)": suggestion_data["DBSCAN (epsilon)"],
                    "DBSCAN ($\\text{minPts}$)": suggestion_data["DBSCAN (minPts)"]
                })

            if st.button("Bắt đầu phân cụm", key="run_cluster"):
                progress_bar = st.progress(0)
                status_text = st.empty()
                start_time = time.time()

                X_processed = X / 255.0

                run_name = f"{cluster_method}_Run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                with mlflow.start_run(run_name=run_name) as run:
                    if cluster_method == "K-means":
                        progress_bar.progress(10)
                        status_text.text("Đang chạy K-means (10%)...")
                        model = KMeans(n_clusters=n_clusters, random_state=42)
                        cluster_labels = model.fit_predict(X_processed)
                        progress_bar.progress(100)
                        status_text.text("Hoàn tất K-means (100%)!")
                        inertia = model.inertia_
                        centroids = model.cluster_centers_
                        mlflow.log_metric("inertia", inertia)
                        mlflow.sklearn.log_model(model, "kmeans_model")
                    else:
                        progress_bar.progress(10)
                        status_text.text("Đang chạy DBSCAN (10%)...")
                        model = DBSCAN(eps=eps, min_samples=min_samples)
                        cluster_labels = model.fit_predict(X_processed)
                        progress_bar.progress(100)
                        status_text.text("Hoàn tất DBSCAN (100%)!")
                        n_clusters_est = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
                        n_noise = list(cluster_labels).count(-1)
                        mlflow.log_metric("n_clusters", n_clusters_est)
                        mlflow.log_metric("n_noise", n_noise)
                        mlflow.sklearn.log_model(model, "dbscan_model")

                    training_time = time.time() - start_time
                    mlflow.log_params(params)
                    mlflow.log_param("cluster_method", cluster_method)
                    mlflow.log_metric("training_time_seconds", training_time)

                    run_id = run.info.run_id
                    st.session_state['latest_run'] = {
                        'run_id': run_id,
                        'run_name': run_name
                    }
                    st.session_state['cluster_labels'] = cluster_labels
                    st.success(f"Phân cụm xong! Thời gian: {training_time:.2f} giây.")

                    st.subheader("Kết quả Phân cụm (Biểu đồ 2D)")
                    pca = PCA(n_components=2)
                    X_2d = pca.fit_transform(X_processed)
                    df_plot = pd.DataFrame({
                        'PCA1': X_2d[:, 0],
                        'PCA2': X_2d[:, 1],
                        'Cluster': cluster_labels
                    })

                    fig = go.Figure()
                    colors = ['blue', 'green', 'red', 'cyan', 'magenta', 'orange', 'purple', 'brown', 'pink', 'gray']
                    symbols = ['circle'] * len(colors)
                    unique_clusters = np.unique(cluster_labels)

                    for i, cluster in enumerate(unique_clusters):
                        if cluster == -1 and cluster_method == "DBSCAN":
                            cluster_data = df_plot[df_plot['Cluster'] == cluster]
                            fig.add_trace(go.Scatter(
                                x=cluster_data['PCA1'],
                                y=cluster_data['PCA2'],
                                mode='markers',
                                name='Nhiễu',
                                marker=dict(
                                    symbol='x',
                                    color='grey',
                                    size=8,
                                    opacity=0.5
                                ),
                                hovertemplate="PCA1: %{x:.2f}<br>PCA2: %{y:.2f}<br>Cụm: Nhiễu"
                            ))
                        else:
                            cluster_data = df_plot[df_plot['Cluster'] == cluster]
                            cluster_name = f'Cluster {cluster + 1}' if cluster >= 0 else f'Cluster {cluster}'
                            fig.add_trace(go.Scatter(
                                x=cluster_data['PCA1'],
                                y=cluster_data['PCA2'],
                                mode='markers',
                                name=cluster_name,
                                marker=dict(
                                    symbol=symbols[i % len(symbols)],
                                    color=colors[i % len(colors)],
                                    size=10,
                                    opacity=0.8
                                ),
                                customdata=[cluster_name] * len(cluster_data),
                                hovertemplate="PCA1: %{x:.2f}<br>PCA2: %{y:.2f}<br>Cụm: %{customdata}"
                            ))

                    if cluster_method == "K-means":
                        centroids_2d = pca.transform(centroids)
                        fig.add_trace(go.Scatter(
                            x=centroids_2d[:, 0],
                            y=centroids_2d[:, 1],
                            mode='markers',
                            name='Centroids',
                            marker=dict(
                                symbol='star',
                                color='yellow',
                                size=15,
                                opacity=1.0
                            )
                        ))

                    fig.update_layout(
                        title="Kết quả Phân cụm (PCA 2D)",
                        xaxis_title="PCA1",
                        yaxis_title="PCA2",
                        legend_title="Cụm",
                        template='plotly_white',
                        width=900,
                        height=600,
                        hovermode='closest',
                        showlegend=True,
                        margin=dict(l=50, r=50, t=50, b=50)
                    )
                    st.plotly_chart(fig, use_container_width=True)

                    st.subheader("Hiểu biểu đồ này như thế nào?")
                    if cluster_method == "K-means":
                        st.markdown(f"""
                        - **Biểu đồ**: Mỗi điểm là một ảnh chữ số, được giảm từ $784$ chiều xuống $2$ chiều (dùng PCA).  
                        - **Màu sắc**: Mỗi cụm có một màu riêng (ví dụ: Cluster 1 là xanh dương, Cluster 2 là xanh lá).  
                        - **Tâm cụm**: Điểm vàng (hình ngôi sao) là trung tâm của mỗi cụm, đại diện cho trung bình của các điểm trong cụm.  
                        - **Rê chuột**: Rê chuột vào điểm để xem giá trị PCA1, PCA2, và cụm.  
                        - **Ý nghĩa**: K-means chia dữ liệu thành ${n_clusters}$ cụm. Lý tưởng là mỗi cụm chứa các điểm dữ liệu tương tự nhau dựa trên đặc trưng hình ảnh.  
                        """, unsafe_allow_html=True)
                    else:
                        noise_percentage = (n_noise / num_samples * 100) if num_samples > 0 else 0
                        st.markdown(f"""
                        - **Biểu đồ**: Mỗi điểm là một ảnh chữ số, được giảm từ $784$ chiều xuống $2$ chiều (dùng PCA).  
                        - **Màu sắc**: Mỗi cụm có một màu riêng. Điểm nhiễu (không thuộc cụm nào) có màu xám, hình chữ 'x'.  
                        - **Rê chuột**: Rê chuột vào điểm để xem giá trị PCA1, PCA2, và cụm.  
                        - **Ý nghĩa**: DBSCAN tự tìm ${n_clusters_est}$ cụm và ${n_noise}$ điểm nhiễu (${noise_percentage:.2f}\\%$ tổng số). Lý tưởng là các cụm chứa các điểm dữ liệu tương tự nhau, nhiễu là các điểm bất thường.  
                        """, unsafe_allow_html=True)

                    st.subheader("Thông tin chi tiết")
                    with st.expander("Xem chi tiết kết quả", expanded=True):
                        st.markdown("**Thông tin lần chạy:**")
                        st.write(f"- Tên lần chạy: {run_name}")
                        st.write(f"- ID lần chạy: {run_id}")

                        st.markdown("**Cài đặt:**")
                        st.write(f"- Phương pháp: {cluster_method}")
                        if cluster_method == "K-means":
                            st.write(f"- Số nhóm: $ {n_clusters} $", unsafe_allow_html=True)
                        else:
                            st.write(f"- Khoảng cách tối đa ($\\epsilon$): $ {eps} $", unsafe_allow_html=True)
                            st.write(f"- Số điểm tối thiểu ($\\text{{minPts}}$): $ {min_samples} $", unsafe_allow_html=True)
                        st.write(f"- Thời gian chạy: $ {training_time:.2f} $ giây", unsafe_allow_html=True)
                        st.write(f"- Số ảnh đã phân cụm: $ {X.shape[0]} $", unsafe_allow_html=True)

                        st.markdown("**Kết quả chi tiết:**")
                        if cluster_method == "K-means":
                            st.write(f"- Độ chặt của cụm (inertia): $ {inertia:.2f} $ (số càng nhỏ, các điểm càng gần trung tâm cụm).", unsafe_allow_html=True)
                        else:
                            noise_percentage = (n_noise / num_samples * 100) if num_samples > 0 else 0
                            st.write(f"- Số cụm tìm được: $ {n_clusters_est} $", unsafe_allow_html=True)
                            st.write(f"- Số điểm nhiễu: $ {n_noise} $ ($ {noise_percentage:.2f}\\% $ tổng số ảnh).", unsafe_allow_html=True)

    with tab_log_info:
        st.header("Theo dõi kết quả")
        st.markdown("""
        Tab này cho phép bạn xem danh sách các lần phân cụm đã thực hiện. Chọn một lần chạy từ danh sách để xem chi tiết, đổi tên hoặc xóa.
        """, unsafe_allow_html=True)
        
        try:
            client = MlflowClient()
            experiment_id = "4"  # ID của experiment MNIST Clustering
            experiment = client.get_experiment(experiment_id)
            if not experiment:
                st.error(f"Không tìm thấy experiment với ID: {experiment_id}. Vui lòng kiểm tra lại MLflow tracking URI.")
            else:
                runs = client.search_runs(experiment_ids=[experiment_id], order_by=["attributes.start_time DESC"])
                
                if not runs:
                    st.info("Chưa có lần chạy nào được ghi nhận.")
                else:
                    run_options = {run.info.run_id: run.data.tags.get('mlflow.runName', f"Run_{run.info.run_id}") for run in runs}
                    run_names = list(run_options.values())

                    default_run_name = st.session_state.get('latest_run', {}).get('run_name', run_names[0]) if 'latest_run' in st.session_state else run_names[0]

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
                                if 'latest_run' in st.session_state and st.session_state['latest_run']['run_id'] == selected_run_id:
                                    st.session_state['latest_run']['run_name'] = new_run_name.strip()
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
                            if 'latest_run' in st.session_state and st.session_state['latest_run']['run_id'] == selected_run_id:
                                del st.session_state['latest_run']
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
                        reduce_method = selected_run.data.params.get("cluster_method", "Không xác định")
                        metrics_display = {}

                        training_time = selected_run.data.metrics.get("training_time_seconds", "N/A")
                        metrics_display["Thời gian thực hiện (giây)"] = f"{float(training_time):.2f}" if training_time != "N/A" else "N/A"

                        if reduce_method == "K-means":
                            inertia = selected_run.data.metrics.get("inertia", "N/A")
                            metrics_display["Tổng bình phương khoảng cách (K-means)"] = f"{float(inertia):.2f}" if inertia != "N/A" else "N/A"
                        elif reduce_method == "DBSCAN":
                            n_clusters = selected_run.data.metrics.get("n_clusters", "N/A")
                            n_noise = selected_run.data.metrics.get("n_noise", "N/A")
                            metrics_display["Số cụm tìm được (DBSCAN)"] = n_clusters
                            metrics_display["Số điểm nhiễu (DBSCAN)"] = n_noise

                        st.json(metrics_display, expanded=True)
                    else:
                        st.write("Không có kết quả được ghi nhận.")

                    # Thêm nút liên kết tới MLflow UI
                    st.subheader("Truy cập MLflow UI")
                    mlflow_url = "https://dagshub.com/huykibo/streamlit_mlflow.mlflow"  # Thay bằng URL MLflow của bạn nếu khác
                    if st.button("Mở MLflow UI trên Dagshub"):
                        st.markdown(f'[Click để mở MLflow UI]({mlflow_url})', unsafe_allow_html=True)
        except Exception as e:
            st.error(f"Lỗi kết nối MLflow: {e}. Vui lòng kiểm tra MLFLOW_TRACKING_URI và thông tin xác thực.")

if __name__ == "__main__":
    run_mnist_clustering_app()