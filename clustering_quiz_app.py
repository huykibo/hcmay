import streamlit as st

# Thiết lập cấu hình trang
st.set_page_config(page_title="Trắc nghiệm Học Máy", layout="centered")
st.title("🧠 Trắc nghiệm Học Máy")

# Sidebar để chọn phần câu hỏi
quiz_sections = [
    "Cơ bản về AI và Học Máy",
    "Thuật toán Phân cụm",
    "Giảm chiều dữ liệu",
    "Hồi quy Tuyến tính và Normal Equation",
    "Gradient Descent",
    "Mini-batch, Stochastic, Batch Gradient Descent",
    "Multiple Linear Regression",
    "Regularization (Ridge, Lasso, Elastic Net)",
    "Logistic Regression",
    "Softmax Regression",
    "Decision Trees",
    "k-Nearest Neighbors (k-NN)",
    "Naive Bayes",
    "Ứng dụng Thực tế",
    "Reinforcement Learning"
]
section = st.sidebar.radio("Chọn phần câu hỏi", quiz_sections)

# ==== DỮ LIỆU CÂU HỎI ====

# Phần 1: Cơ bản về AI và Học Máy
ai_ml_questions = [
    {
        "question": "Dữ liệu không cấu trúc dễ dàng xử lý hơn dữ liệu có cấu trúc.",
        "options": ["True", "False"],
        "answer": "False",
        "explanation": "Dữ liệu không cấu trúc (như ảnh, video) khó xử lý hơn dữ liệu có cấu trúc (như bảng tính) vì nó thiếu tổ chức rõ ràng."
    },
    {
        "question": "\"Mùa đông của AI\" là giai đoạn mà học máy phát triển mạnh mẽ nhất.",
        "options": ["True", "False"],
        "answer": "False",
        "explanation": "\"Mùa đông của AI\" là giai đoạn suy thoái trong nghiên cứu AI, không phải thời kỳ phát triển mạnh mẽ."
    },
    {
        "question": "Deep Learning là một bước tiến quan trọng trong quá trình phát triển của học máy.",
        "options": ["True", "False"],
        "answer": "True",
        "explanation": "Deep Learning là một nhánh quan trọng của học máy, sử dụng mạng nơ-ron sâu để giải quyết các bài toán phức tạp."
    },
    {
        "question": "Học máy là một nhánh của trí tuệ nhân tạo (AI).",
        "options": ["True", "False"],
        "answer": "True",
        "explanation": "Học máy là một phần của AI, tập trung vào việc phát triển các thuật toán tự học từ dữ liệu."
    },
    {
        "question": "Học sâu (Deep Learning) được sử dụng rộng rãi trong các lĩnh vực nào?",
        "options": ["Nhận dạng hình ảnh", "Dịch ngôn ngữ", "Xe tự lái", "Tất cả các đáp án trên"],
        "answer": "Tất cả các đáp án trên",
        "explanation": "Deep Learning được ứng dụng rộng rãi trong nhận dạng hình ảnh, dịch ngôn ngữ, xe tự lái và nhiều lĩnh vực khác."
    },
    {
        "question": "Dữ liệu nào sau đây được coi là dữ liệu có cấu trúc?",
        "options": ["Ảnh chụp", "Đoạn video", "Dữ liệu bán hàng trong một bảng tính", "Bài viết trên blog"],
        "answer": "Dữ liệu bán hàng trong một bảng tính",
        "explanation": "Dữ liệu có cấu trúc được tổ chức trong các định dạng như bảng tính, dễ dàng truy vấn và phân tích."
    },
    {
        "question": "Nhận dạng hình ảnh là một ứng dụng phổ biến của học máy.",
        "options": ["True", "False"],
        "answer": "True",
        "explanation": "Nhận dạng hình ảnh là một trong những ứng dụng nổi bật của học máy, đặc biệt trong học sâu."
    },
    {
        "question": "AI và Học máy là hai thuật ngữ hoàn toàn tách biệt, không có mối quan hệ với nhau.",
        "options": ["True", "False"],
        "answer": "False",
        "explanation": "Học máy là một nhánh của AI, giúp hệ thống học hỏi từ dữ liệu để thực hiện các nhiệm vụ thông minh."
    },
    {
        "question": "Ứng dụng nào sau đây không phải là một ứng dụng của học máy?",
        "options": ["Nhận diện khuôn mặt", "Tìm kiếm thông tin trên Google", "Vẽ tranh", "Dự đoán giá cổ phiếu"],
        "answer": "Vẽ tranh",
        "explanation": "Vẽ tranh không phải là ứng dụng trực tiếp của học máy, mặc dù học máy có thể hỗ trợ tạo nghệ thuật qua các mô hình như GAN."
    },
    {
        "question": "Mối quan hệ giữa Học máy và Trí tuệ nhân tạo là gì?",
        "options": ["Học máy là một phần của Trí tuệ nhân tạo", "Trí tuệ nhân tạo là một phần của Học máy", 
                    "Học máy và Trí tuệ nhân tạo không liên quan đến nhau", "Học máy chỉ là một khái niệm trong lý thuyết, không áp dụng vào thực tiễn"],
        "answer": "Học máy là một phần của Trí tuệ nhân tạo",
        "explanation": "Học máy là một lĩnh vực con của AI, tập trung vào các thuật toán học từ dữ liệu."
    },
    {
        "question": "Giai đoạn nào được xem là sự hồi sinh của học máy?",
        "options": ["Những năm 1950 - 1980", "Giai đoạn trước \"mùa đông của AI\"", "Từ năm 1990 trở đi", "Thời điểm xuất hiện máy tính cá nhân"],
        "answer": "Từ năm 1990 trở đi",
        "explanation": "Học máy hồi sinh từ những năm 1990 nhờ sự gia tăng sức mạnh tính toán và dữ liệu lớn."
    },
    {
        "question": "Ví dụ nào sau đây là một ứng dụng của Học máy?",
        "options": ["Gợi ý sản phẩm trên trang thương mại điện tử", "Lập kế hoạch cho các chiến dịch marketing", 
                    "Tạo giao diện người dùng", "Viết mã nguồn cho ứng dụng web"],
        "answer": "Gợi ý sản phẩm trên trang thương mại điện tử",
        "explanation": "Hệ thống gợi ý sản phẩm sử dụng học máy để phân tích hành vi người dùng và đề xuất sản phẩm phù hợp."
    },
    {
        "question": "Học máy là gì?",
        "options": ["Một nhánh của lập trình truyền thống", "Một nhánh của trí tuệ nhân tạo", 
                    "Một phương pháp lập trình không cần dữ liệu", "Một thuật toán tối ưu hóa"],
        "answer": "Một nhánh của trí tuệ nhân tạo",
        "explanation": "Học máy là một nhánh của AI, cho phép máy tính học từ dữ liệu mà không cần lập trình chi tiết."
    },
    {
        "question": "Xe tự lái là một ứng dụng của học sâu trong lĩnh vực ô tô.",
        "options": ["True", "False"],
        "answer": "True",
        "explanation": "Xe tự lái sử dụng học sâu để xử lý dữ liệu từ cảm biến và đưa ra quyết định lái xe."
    }
]

# Phần 2: Thuật toán Phân cụm
clustering_questions = [
    {
        "question": "Làm thế nào để phân cụm một tập dữ liệu có các cụm với hình dạng hỗn hợp?",
        "options": ["K-means rồi DBSCAN", "DBSCAN rồi K-means", "Agglomerative rồi K-means", "K-means và Agglomerative"],
        "answer": "DBSCAN rồi K-means",
        "explanation": "Bạn sẽ dùng DBSCAN trước để tìm cụm có hình dạng phức tạp, sau đó dùng K-means để tinh chỉnh."
    },
    {
        "question": "Tại sao DBSCAN gắn nhãn hầu hết các điểm là nhiễu?",
        "options": ["Eps nhỏ hoặc MinPts cao", "Cụm hình cầu", "Không xác định số cụm", "Tiêu chí liên kết kém"],
        "answer": "Eps nhỏ hoặc MinPts cao",
        "explanation": "DBSCAN gán nhiều điểm thành nhiễu nếu khoảng cách tối đa (Eps) nhỏ hoặc số điểm tối thiểu (MinPts) cao."
    },
    {
        "question": "Liên kết (linkage) đóng vai trò gì trong Agglomerative Clustering?",
        "options": ["Xác định số lượng cụm", "Xác định khoảng cách gộp", "Đặt ngưỡng mật độ", "Định vị tâm cụm"],
        "answer": "Xác định khoảng cách gộp",
        "explanation": "Liên kết (linkage) xác định cách tính khoảng cách để gộp cụm trong Agglomerative Clustering."
    },
    {
        "question": "Thuật toán nào ít phù hợp nhất với tập dữ liệu lớn?",
        "options": ["K-means", "DBSCAN", "Agglomerative Clustering", "DBSCAN và Agglomerative"],
        "answer": "Agglomerative Clustering",
        "explanation": "Agglomerative Clustering tính toán nhiều khoảng cách, không hiệu quả với tập dữ liệu lớn."
    },
    {
        "question": "Dendrogram của single linkage và complete linkage khác nhau như thế nào?",
        "options": ["Single tạo cụm chặt chẽ", "Complete tạo cụm dạng chuỗi", "Single tạo cụm dạng chuỗi", "Cả hai tạo cụm giống nhau"],
        "answer": "Single tạo cụm dạng chuỗi",
        "explanation": "Single linkage có xu hướng tạo cụm chuỗi dài, còn complete linkage tạo cụm chặt chẽ hơn."
    },
    {
        "question": "Thuật toán nào tốt nhất cho các cụm không xác định với nhiễu?",
        "options": ["K-means với K cố định", "DBSCAN", "Single-link Agglomerative", "Average-link Agglomerative"],
        "answer": "DBSCAN",
        "explanation": "DBSCAN tự động phát hiện cụm và phân biệt nhiễu tốt khi không biết trước số cụm."
    },
    {
        "question": "Thuật toán nào phù hợp với cụm hình cầu không có nhiễu?",
        "options": ["DBSCAN", "K-means", "Single-link Agglomerative", "Complete-link Agglomerative"],
        "answer": "K-means",
        "explanation": "K-means phù hợp với cụm hình cầu và dữ liệu ít nhiễu."
    },
    {
        "question": "Mục tiêu chính của K-means clustering là gì?",
        "options": ["Tối đa hóa mật độ điểm trong cụm", "Giảm khoảng cách tới tâm cụm", "Xây dựng hệ thống phân cấp cụm", "Liên kết cụm theo khoảng cách"],
        "answer": "Giảm khoảng cách tới tâm cụm",
        "explanation": "Mục tiêu chính của K-means là giảm tổng khoảng cách điểm tới tâm cụm."
    },
    {
        "question": "DBSCAN xử lý ngoại lai như thế nào so với K-means?",
        "options": ["Đánh dấu ngoại lai là nhiễu", "Gán ngoại lai vào cụm", "Xử lý ngoại lai giống nhau", "Tiền xử lý ngoại lai"],
        "answer": "Đánh dấu ngoại lai là nhiễu",
        "explanation": "DBSCAN đánh dấu điểm nhiễu thay vì gán chúng vào cụm."
    },
    {
        "question": "Loại liên kết nào trong Agglomerative Clustering tạo ra các cụm chặt chẽ?",
        "options": ["Single linkage", "Complete linkage", "Average linkage", "Centroid linkage"],
        "answer": "Complete linkage",
        "explanation": "Complete linkage tạo cụm chặt chẽ hơn so với single linkage."
    },
    {
        "question": "Điều gì định nghĩa phân cụm DBSCAN?",
        "options": ["Số cụm cố định", "Cụm dựa trên tâm", "Hình dạng cụm dựa trên mật độ", "Liên kết phân cấp cụm"],
        "answer": "Hình dạng cụm dựa trên mật độ",
        "explanation": "DBSCAN dựa vào mật độ điểm để xác định cụm có hình dạng phức tạp."
    },
    {
        "question": "Tại sao cụm K-means bị lệch bởi ngoại lai?",
        "options": ["Cụm không hình cầu", "Tiêu chí liên kết sai", "Số cụm thấp", "Ngoại lai làm lệch tâm cụm"],
        "answer": "Ngoại lai làm lệch tâm cụm",
        "explanation": "Ngoại lai làm lệch tâm cụm trong K-means dẫn đến kết quả không chính xác."
    },
    {
        "question": "Agglomerative Clustering tạo ra kết quả gì?",
        "options": ["Cụm với tâm cố định", "Một cụm thống nhất", "Cụm dựa trên mật độ", "Cây phân cấp dendrogram"],
        "answer": "Cây phân cấp dendrogram",
        "explanation": "Agglomerative Clustering tạo cây phân cấp cụm (dendrogram)."
    },
    {
        "question": "Để phát hiện nhiều cụm nhỏ hơn trong DBSCAN, bạn nên điều chỉnh gì?",
        "options": ["Tăng số cụm", "Giảm số điểm tối thiểu", "Tăng ngưỡng khoảng cách", "Sử dụng liên kết phân cấp"],
        "answer": "Giảm số điểm tối thiểu",
        "explanation": "Giảm số điểm tối thiểu để phát hiện thêm các cụm nhỏ hơn."
    },
    {
        "question": "Tại sao K-means phải chỉ định số cụm (K)?",
        "options": ["Nó gán điểm cho K tâm cụm", "Nó dùng tiêu chí mật độ", "Nó xây dựng hệ thống phân cấp cụm", "Nó xác định điểm nhiễu"],
        "answer": "Nó gán điểm cho K tâm cụm",
        "explanation": "K-means cần biết trước số cụm để gán điểm cho các tâm cụm."
    },
    {
        "question": "Làm thế nào để cải thiện K-means để chống lại ngoại lai?",
        "options": ["Dùng trung vị cho tâm cụm", "Dùng liên kết phân cấp", "Tự động tăng số cụm", "Chuyển sang phân cụm mật độ"],
        "answer": "Chuyển sang phân cụm mật độ",
        "explanation": "Chuyển sang thuật toán dựa trên mật độ như DBSCAN để chống nhiễu tốt hơn."
    }
]

# Phần 3: Giảm chiều dữ liệu
dimred_questions = [
    {
        "question": "PCA biến đổi dữ liệu thành tập hợp mới gồm:",
        "options": ["Đặc trưng tương quan", "Biến gốc", "Đặc trưng độc lập", "Thành phần trực giao"],
        "answer": "Thành phần trực giao",
        "explanation": "PCA biến đổi dữ liệu thành các thành phần trực giao."
    },
    {
        "question": "t-SNE chủ yếu được dùng để:",
        "options": ["Phân loại", "Hồi quy", "Giảm chiều dữ liệu để trực quan hóa", "Chọn đặc trưng"],
        "answer": "Giảm chiều dữ liệu để trực quan hóa",
        "explanation": "t-SNE chủ yếu được dùng để giảm chiều dữ liệu phục vụ trực quan hóa."
    },
    {
        "question": "Autoencoders thường được huấn luyện theo cách nào?",
        "options": ["Không giám sát", "Có giám sát", "Tăng cường", "Bán giám sát"],
        "answer": "Không giám sát",
        "explanation": "Autoencoder thường được huấn luyện không giám sát."
    },
    {
        "question": "Phương pháp nào sau đây là phương pháp lọc để chọn đặc trưng?",
        "options": ["Recursive Feature Elimination", "L1 Regularization", "Pearson Correlation Coefficient", "Forward Selection"],
        "answer": "Pearson Correlation Coefficient",
        "explanation": "Pearson là phương pháp lọc đơn giản để chọn đặc trưng."
    },
    {
        "question": "Kỹ thuật nào loại bỏ đặc trưng bằng cách xem xét đệ quy các tập đặc trưng nhỏ hơn?",
        "options": ["Forward Selection", "Backward Elimination", "PCA", "Recursive Feature Elimination"],
        "answer": "Recursive Feature Elimination",
        "explanation": "RFE loại bỏ các đặc trưng ít quan trọng từng bước một cách đệ quy."
    },
    {
        "question": "Phương pháp nào không phải là kỹ thuật giảm chiều?",
        "options": ["PCA", "t-SNE", "Lasso", "Decision Tree"],
        "answer": "Decision Tree",
        "explanation": "Decision Tree là mô hình học máy chứ không phải kỹ thuật giảm chiều."
    },
    {
        "question": "Phương pháp nhúng (embedded) kết hợp chọn đặc trưng với:",
        "options": ["Trực quan hóa dữ liệu", "Biến đổi đặc trưng", "Huấn luyện mô hình", "Chuẩn hóa dữ liệu"],
        "answer": "Huấn luyện mô hình",
        "explanation": "Embedded methods kết hợp chọn đặc trưng trong quá trình huấn luyện mô hình."
    },
    {
        "question": "L1 regularization chủ yếu được dùng để:",
        "options": ["Giảm overfitting bằng cách phạt trọng số lớn", "Khuyến khích độ thưa bằng cách đặt một số hệ số về 0", "Loại bỏ đa cộng tuyến", "Tăng độ phức tạp mô hình"],
        "answer": "Khuyến khích độ thưa bằng cách đặt một số hệ số về 0",
        "explanation": "L1 giúp chọn đặc trưng bằng cách đưa một số trọng số về 0."
    },
    {
        "question": "Phương pháp nào tập trung vào giảm chiều phi tuyến?",
        "options": ["PCA", "L1 Regularization", "t-SNE", "Forward Selection"],
        "answer": "t-SNE",
        "explanation": "t-SNE là kỹ thuật giảm chiều phi tuyến dùng cho trực quan hóa dữ liệu phức tạp."
    },
    {
        "question": "Autoencoders học giảm chiều bằng cách:",
        "options": ["Phân cụm các đặc trưng tương tự", "Dùng phép chiếu tuyến tính", "Giảm thiểu lỗi tái tạo", "Tối đa hóa độ chính xác phân loại"],
        "answer": "Giảm thiểu lỗi tái tạo",
        "explanation": "Autoencoder học nén bằng cách giảm lỗi tái tạo."
    },
    {
        "question": "So với PCA, t-SNE phù hợp hơn để:",
        "options": ["Tập dữ liệu lớn", "Thành phần có thể giải thích", "Trực quan hóa cụm đa chiều", "Xếp hạng đặc trưng"],
        "answer": "Trực quan hóa cụm đa chiều",
        "explanation": "t-SNE thường dùng để trực quan hóa cụm trong không gian cao chiều."
    },
    {
        "question": "Phương pháp điều chuẩn nào làm nhỏ tất cả hệ số nhưng hiếm khi đặt chúng về đúng 0?",
        "options": ["L1 Regularization", "L2 Regularization", "Forward Selection", "Recursive Feature Elimination"],
        "answer": "L2 Regularization",
        "explanation": "L2 làm nhỏ trọng số chứ không đưa về 0 như L1."
    },
    {
        "question": "Mục đích của decoder trong autoencoder là gì?",
        "options": ["Giảm overfitting", "Tạo nhiễu", "Tái tạo dữ liệu đầu vào", "Chọn đặc trưng quan trọng"],
        "answer": "Tái tạo dữ liệu đầu vào",
        "explanation": "Decoder của autoencoder tái tạo lại đầu vào từ vector mã hóa."
    },
    {
        "question": "Autoencoder gồm hai phần chính nào?",
        "options": ["Predictor và Transformer", "Encoder và Decoder", "Feature Selector và Reconstructor", "Generator và Discriminator"],
        "answer": "Encoder và Decoder",
        "explanation": "Autoencoder gồm phần mã hóa (encoder) và giải mã (decoder)."
    },
    {
        "question": "Trong PCA, thành phần chính đầu tiên nắm giữ:",
        "options": ["Phương sai nhỏ nhất trong dữ liệu", "Trung bình của các đặc trưng", "Phương sai lớn nhất trong dữ liệu", "Tổng phương sai"],
        "answer": "Phương sai lớn nhất trong dữ liệu",
        "explanation": "Thành phần chính đầu tiên giữ phương sai lớn nhất."
    },
    {
        "question": "Thông tin lẫn nhau (Mutual Information) đo lường điều gì?",
        "options": ["Mối quan hệ tuyến tính giữa các biến", "Chênh lệch trung bình giữa các đặc trưng", "Lượng thông tin chia sẻ giữa các biến", "Phương sai giải thích bởi đặc trưng"],
        "answer": "Lượng thông tin chia sẻ giữa các biến",
        "explanation": "MI đo lượng thông tin chia sẻ giữa hai biến."
    },
    {
        "question": "PCA giả định các thành phần là:",
        "options": ["Phụ thuộc tuyến tính", "Trực giao và sắp xếp theo phương sai", "Phân loại", "Chuẩn hóa về khoảng 0-1"],
        "answer": "Trực giao và sắp xếp theo phương sai",
        "explanation": "PCA tạo các thành phần chính trực giao và sắp xếp theo mức độ phương sai."
    },
    {
        "question": "Phương pháp nào bắt đầu với tập đặc trưng rỗng và thêm đặc trưng từng bước?",
        "options": ["Backward Selection", "RFE", "Forward Selection", "Lasso"],
        "answer": "Forward Selection",
        "explanation": "Forward Selection bắt đầu từ tập rỗng và thêm đặc trưng từng bước."
    },
    {
        "question": "Ý tưởng chính của t-SNE là gì?",
        "options": ["Giữ cấu trúc toàn cục", "Tìm phép chiếu tuyến tính", "Giữ tương đồng cục bộ trong dữ liệu", "Tối đa hóa phương sai"],
        "answer": "Giữ tương đồng cục bộ trong dữ liệu",
        "explanation": "t-SNE tập trung giữ lại sự tương đồng cục bộ trong dữ liệu."
    },
    {
        "question": "Encoder trong autoencoder làm gì?",
        "options": ["Thêm nhiễu vào dữ liệu", "Tăng chiều dữ liệu", "Nén đầu vào thành biểu diễn chiều thấp hơn", "Dự đoán nhãn lớp"],
        "answer": "Nén đầu vào thành biểu diễn chiều thấp hơn",
        "explanation": "Encoder mã hóa đầu vào thành biểu diễn có chiều thấp hơn."
    }
]

# Phần 4: Hồi quy Tuyến tính và Normal Equation
linear_regression_questions = [
    {
        "question": "Điều kiện nào cần thiết để Normal Equation có thể tính toán được?",
        "options": ["Ma trận X^TX phải là một ma trận vuông khả nghịch (có định thức khác 0)", 
                    "Số lượng biến đầu vào luôn phải nhỏ hơn số mẫu dữ liệu", 
                    "Giá trị đầu ra y phải có phân phối chuẩn", 
                    "Không có điều kiện nào, Normal Equation luôn có thể tính toán được"],
        "answer": "Ma trận X^TX phải là một ma trận vuông khả nghịch (có định thức khác 0)",
        "explanation": "Normal Equation yêu cầu ma trận X^TX khả nghịch để giải được hệ phương trình tuyến tính."
    },
    {
        "question": "Normal Equation là gì?",
        "options": ["Một phương pháp tối ưu hóa hồi quy tuyến tính bằng cách lặp liên tục để cập nhật trọng số", 
                    "Một phương pháp tính toán trọng số tối ưu trong hồi quy tuyến tính bằng cách giải hệ phương trình tuyến tính", 
                    "Một phương pháp chuẩn hóa dữ liệu trước khi áp dụng hồi quy tuyến tính", 
                    "Một phương pháp sử dụng cây quyết định để dự đoán biến mục tiêu"],
        "answer": "Một phương pháp tính toán trọng số tối ưu trong hồi quy tuyến tính bằng cách giải hệ phương trình tuyến tính",
        "explanation": "Normal Equation giải trực tiếp hệ phương trình để tìm trọng số tối ưu mà không cần lặp."
    },
    {
        "question": "Trong Normal Equation, nếu số mẫu dữ liệu m lớn hơn số lượng biến n, điều gì có thể xảy ra?",
        "options": ["Ma trận X^TX luôn khả nghịch", "Ma trận X^TX không khả nghịch", 
                    "Normal Equation không thể sử dụng được", "Không có gì đặc biệt xảy ra"],
        "answer": "Ma trận X^TX luôn khả nghịch",
        "explanation": "Khi m > n và các cột của X độc lập tuyến tính, X^TX thường khả nghịch."
    },
    {
        "question": "Normal Equation có thể hoạt động tốt với dữ liệu lớn không?",
        "options": ["Có, vì nó không cần tuning hyperparameters như Gradient Descent", 
                    "Không, vì nó đòi hỏi tính toán nghịch đảo ma trận (X^TX)^{−1} có độ phức tạp O(n^3)", 
                    "Có, vì nó nhanh hơn Gradient Descent trong mọi trường hợp", 
                    "Không, vì nó chỉ áp dụng được cho bài toán phân loại"],
        "answer": "Không, vì nó đòi hỏi tính toán nghịch đảo ma trận (X^TX)^{−1} có độ phức tạp O(n^3)",
        "explanation": "Việc tính nghịch đảo ma trận có độ phức tạp cao, khiến Normal Equation không hiệu quả với dữ liệu lớn."
    }
]

# Phần 5: Gradient Descent
gradient_descent_questions = [
    {
        "question": "Nếu tốc độ học quá nhỏ, quá trình hội tụ của Gradient Descent sẽ rất chậm.",
        "options": ["True", "False"],
        "answer": "True",
        "explanation": "Tốc độ học nhỏ làm các bước cập nhật nhỏ, dẫn đến hội tụ chậm."
    },
    {
        "question": "Gradient Descent hoạt động bằng cách cập nhật liên tục các hệ số dựa trên đạo hàm của hàm mất mát.",
        "options": ["True", "False"],
        "answer": "True",
        "explanation": "Gradient Descent sử dụng đạo hàm để điều chỉnh hệ số theo hướng giảm hàm mất mát."
    },
    {
        "question": "Gradient Descent luôn hội tụ về một nghiệm duy nhất bất kể learning rate (tốc độ học) được chọn như thế nào.",
        "options": ["True", "False"],
        "answer": "False",
        "explanation": "Tốc độ học không phù hợp có thể khiến Gradient Descent không hội tụ hoặc dao động."
    },
    {
        "question": "Mục tiêu chính của thuật toán Gradient Descent là gì?",
        "options": ["Tìm nghiệm chính xác của phương trình hồi quy", 
                    "Tìm cực tiểu của hàm lỗi để tối ưu hóa các hệ số của mô hình", 
                    "Tăng tốc độ dự đoán của mô hình", 
                    "Giảm thiểu số lượng biến trong mô hình"],
        "answer": "Tìm cực tiểu của hàm lỗi để tối ưu hóa các hệ số của mô hình",
        "explanation": "Gradient Descent tối ưu hóa hệ số bằng cách giảm dần hàm mất mát."
    },
    {
        "question": "Hàm mất mát nào thường được sử dụng trong Gradient Descent cho hồi quy tuyến tính đơn?",
        "options": ["Mean Absolute Error (MAE)", "Root Mean Squared Error (RMSE)", 
                    "Mean Squared Error (MSE)", "Hinge Loss"],
        "answer": "Mean Squared Error (MSE)",
        "explanation": "MSE thường được dùng vì nó có đạo hàm liên tục, phù hợp với Gradient Descent."
    },
    {
        "question": "Gradient Descent sẽ hội tụ nhanh hơn nếu:",
        "options": ["Dữ liệu có phương sai cao", 
                    "Tốc độ học được tăng lên mà không ảnh hưởng đến độ ổn định", 
                    "Số lần lặp giảm", 
                    "Hàm mất mát có dạng phi tuyến tính"],
        "answer": "Tốc độ học được tăng lên mà không ảnh hưởng đến độ ổn định",
        "explanation": "Tăng tốc độ học hợp lý giúp hội tụ nhanh hơn mà vẫn ổn định."
    },
    {
        "question": "Gradient của hàm mất mát được tính như thế nào trong thuật toán Gradient Descent?",
        "options": ["Là tổng bình phương của các sai số", 
                    "Là độ dốc của hàm lỗi đối với các hệ số w0 và w1", 
                    "Là khoảng cách giữa giá trị dự đoán và giá trị thực tế", 
                    "Là tổng các đạo hàm bậc hai của hàm lỗi"],
        "answer": "Là độ dốc của hàm lỗi đối với các hệ số w0 và w1",
        "explanation": "Gradient là đạo hàm riêng của hàm mất mát theo các tham số."
    },
    {
        "question": "Nếu tốc độ học (learning rate) được chọn quá lớn, điều gì có thể xảy ra trong quá trình Gradient Descent?",
        "options": ["Mô hình sẽ hội tụ nhanh hơn", 
                    "Mô hình sẽ mắc kẹt trong cực trị cục bộ", 
                    "Mô hình có thể dao động xung quanh nghiệm hoặc không hội tụ", 
                    "Mô hình sẽ tối ưu hóa sai số một cách chính xác"],
        "answer": "Mô hình có thể dao động xung quanh nghiệm hoặc không hội tụ",
        "explanation": "Tốc độ học lớn gây bước nhảy quá dài, dẫn đến dao động hoặc không hội tụ."
    },
    {
        "question": "Gradient Descent có thể bị mắc kẹt ở các cực trị cục bộ khi áp dụng cho bài toán hồi quy tuyến tính đơn.",
        "options": ["True", "False"],
        "answer": "False",
        "explanation": "Hồi quy tuyến tính đơn có hàm mất mát lồi, không có cực trị cục bộ."
    },
    {
        "question": "Thuật toán Gradient Descent được sử dụng để tìm các hệ số tối ưu w0 và w1 trong hồi quy tuyến tính.",
        "options": ["True", "False"],
        "answer": "True",
        "explanation": "Gradient Descent tối ưu hóa w0 và w1 để giảm sai số dự đoán."
    },
    {
        "question": "Trong thuật toán Gradient Descent, phương sai của dữ liệu không ảnh hưởng đến tốc độ hội tụ.",
        "options": ["True", "False"],
        "answer": "False",
        "explanation": "Phương sai lớn có thể làm Gradient Descent hội tụ chậm hơn nếu không chuẩn hóa."
    }
]

# Phần 6: Mini-batch, Stochastic, Batch Gradient Descent
gradient_variants_questions = [
    {
        "question": "Nếu kích thước batch bằng 1, Mini-batch Gradient Descent sẽ trở thành phương pháp nào?",
        "options": ["Stochastic Gradient Descent", "Mini-batch Gradient Descent", "Batch Gradient Descent", "Không thay đổi"],
        "answer": "Stochastic Gradient Descent",
        "explanation": "Batch size = 1 nghĩa là cập nhật gradient trên từng mẫu, tức là Stochastic GD."
    },
    {
        "question": "Phương pháp nào cung cấp ước lượng gradient chính xác nhất cho mỗi lần cập nhật tham số?",
        "options": ["Stochastic Gradient Descent", "Mini-batch Gradient Descent", "Batch Gradient Descent", "Tất cả đều như nhau"],
        "answer": "Batch Gradient Descent",
        "explanation": "Batch GD dùng toàn bộ dữ liệu, cho gradient chính xác nhất mỗi lần cập nhật."
    },
    {
        "question": "Phương pháp nào dưới đây sử dụng toàn bộ tập dữ liệu để tính gradient trong mỗi lần lặp?",
        "options": ["Stochastic Gradient Descent", "Mini-batch Gradient Descent", "Batch Gradient Descent", "Adaptive Gradient Descent"],
        "answer": "Batch Gradient Descent",
        "explanation": "Batch GD tính gradient trên toàn bộ tập dữ liệu mỗi lần lặp."
    },
    {
        "question": "Phương pháp nào nhạy cảm nhất với việc chọn learning rate không phù hợp?",
        "options": ["Stochastic Gradient Descent", "Mini-batch Gradient Descent", "Batch Gradient Descent", "Không có sự khác biệt"],
        "answer": "Stochastic Gradient Descent",
        "explanation": "Stochastic GD nhạy cảm hơn do gradient dao động lớn từ từng mẫu."
    },
    {
        "question": "Phương pháp nào tận dụng tốt nhất khả năng tính toán song song trên phần cứng như GPU?",
        "options": ["Stochastic Gradient Descent", "Mini-batch Gradient Descent", "Batch Gradient Descent", "Gradient Descent với learning rate thích nghi"],
        "answer": "Mini-batch Gradient Descent",
        "explanation": "Mini-batch GD cân bằng giữa tốc độ và khả năng song song trên GPU."
    },
    {
        "question": "Phương pháp nào vừa nhanh hơn Batch Gradient Descent, vừa ổn định hơn Stochastic Gradient Descent?",
        "options": ["Stochastic Gradient Descent", "Mini-batch Gradient Descent", "Batch Gradient Descent", "Batch Gradient Descent với learning rate nhỏ"],
        "answer": "Mini-batch Gradient Descent",
        "explanation": "Mini-batch GD kết hợp ưu điểm của cả hai phương pháp."
    },
    {
        "question": "Phương pháp nào cập nhật tham số dựa trên gradient của một mẫu dữ liệu ngẫu nhiên duy nhất trong mỗi lần lặp?",
        "options": ["Stochastic Gradient Descent", "Mini-batch Gradient Descent", "Batch Gradient Descent", "Gradient Descent với momentum"],
        "answer": "Stochastic Gradient Descent",
        "explanation": "Stochastic GD dùng một mẫu ngẫu nhiên mỗi lần cập nhật."
    },
    {
        "question": "Phương pháp nào đòi hỏi ít bộ nhớ nhất trong quá trình tính toán gradient?",
        "options": ["Stochastic Gradient Descent", "Mini-batch Gradient Descent", "Batch Gradient Descent", "Tất cả đều như nhau"],
        "answer": "Stochastic Gradient Descent",
        "explanation": "Stochastic GD chỉ cần lưu gradient cho một mẫu mỗi lần."
    },
    {
        "question": "Phương pháp Gradient Descent nào thường được sử dụng trong các mạng nơ-ron sâu để huấn luyện trên dữ liệu lớn?",
        "options": ["Stochastic Gradient Descent", "Mini-batch Gradient Descent", "Batch Gradient Descent", "Gradient Descent với bước nhảy cố định"],
        "answer": "Mini-batch Gradient Descent",
        "explanation": "Mini-batch GD hiệu quả và phù hợp với dữ liệu lớn trong deep learning."
    },
    {
        "question": "Phương pháp nào có thể không hội tụ chính xác đến điểm tối ưu toàn cục mà dao động quanh nó?",
        "options": ["Stochastic Gradient Descent", "Mini-batch Gradient Descent", "Batch Gradient Descent", "Cả Batch GD và Mini-batch GD"],
        "answer": "Stochastic Gradient Descent",
        "explanation": "Stochastic GD dao động do gradient dựa trên mẫu ngẫu nhiên."
    },
    {
        "question": "Nếu kích thước batch bằng số mẫu trong tập dữ liệu, Mini-batch Gradient Descent sẽ trở thành phương pháp nào?",
        "options": ["Stochastic Gradient Descent", "Mini-batch Gradient Descent", "Batch Gradient Descent", "Không thay đổi"],
        "answer": "Batch Gradient Descent",
        "explanation": "Batch size bằng kích thước dữ liệu biến Mini-batch GD thành Batch GD."
    },
    {
        "question": "Phương pháp Gradient Descent nào thường chậm nhất khi tập dữ liệu rất lớn?",
        "options": ["Stochastic Gradient Descent", "Mini-batch Gradient Descent", "Batch Gradient Descent", "Tất cả đều như nhau"],
        "answer": "Batch Gradient Descent",
        "explanation": "Batch GD chậm vì phải tính gradient trên toàn bộ dữ liệu mỗi lần lặp."
    },
    {
        "question": "Phương pháp nào có quá trình hội tụ ổn định nhất (ít dao động nhất)?",
        "options": ["Stochastic Gradient Descent", "Mini-batch Gradient Descent", "Batch Gradient Descent", "Tất cả đều như nhau"],
        "answer": "Batch Gradient Descent",
        "explanation": "Batch GD ổn định nhất do dùng toàn bộ dữ liệu để tính gradient."
    }
]

# Phần 7: Multiple Linear Regression
multiple_regression_questions = [
    {
        "question": "Trong trường hợp nào Adjusted R2 có thể âm?",
        "options": ["Khi mô hình giải thích tốt hơn giá trị trung bình", 
                    "Khi mô hình kém hơn so với chỉ dùng giá trị trung bình của y", 
                    "Khi R2 lớn hơn 1", 
                    "Khi tất cả biến độc lập không tương quan"],
        "answer": "Khi mô hình kém hơn so với chỉ dùng giá trị trung bình của y",
        "explanation": "Adjusted R2 âm khi mô hình không giải thích dữ liệu tốt hơn giá trị trung bình."
    },
    {
        "question": "Nếu SE của một hệ số w1 rất nhỏ, điều này ngụ ý gì?",
        "options": ["w1 không quan trọng trong mô hình", "w1 có ảnh hưởng lớn đến y", 
                    "Ước lượng của w1 rất chính xác", "R2 của mô hình rất cao"],
        "answer": "Ước lượng của w1 rất chính xác",
        "explanation": "SE nhỏ cho thấy độ không chắc chắn thấp, tức là ước lượng chính xác."
    },
    {
        "question": "Khi nào Adjusted R2 nhỏ hơn R2?",
        "options": ["Khi mô hình có ít biến độc lập", 
                    "Khi thêm biến không cải thiện khả năng giải thích của mô hình", 
                    "Khi tất cả biến độc lập đều không quan trọng", 
                    "Khi số lượng mẫu nhỏ"],
        "answer": "Khi thêm biến không cải thiện khả năng giải thích của mô hình",
        "explanation": "Adjusted R2 phạt khi thêm biến không hữu ích, làm nó nhỏ hơn R2."
    },
    {
        "question": "Multiple Linear Regression khác với Simple Linear Regression ở điểm nào?",
        "options": ["Chỉ có một biến độc lập", "Có nhiều biến độc lập", 
                    "Không có hệ số chặn (intercept)", "Chỉ dự đoán giá trị phân loại"],
        "answer": "Có nhiều biến độc lập",
        "explanation": "Multiple Linear Regression sử dụng nhiều biến độc lập, không chỉ một."
    },
    {
        "question": "Nếu R2 tăng nhưng Adjusted R2 giảm khi thêm một biến, điều này cho thấy gì?",
        "options": ["Biến mới không cải thiện mô hình đáng kể", "Biến mới làm tăng độ chính xác dự đoán", 
                    "Số lượng mẫu quá nhỏ", "Mô hình bị overfitting"],
        "answer": "Biến mới không cải thiện mô hình đáng kể",
        "explanation": "Adjusted R2 giảm cho thấy biến mới không mang lại giá trị giải thích đáng kể."
    },
    {
        "question": "Trong Multiple Linear Regression, hệ số chặn (β0) đại diện cho điều gì?",
        "options": ["Độ dốc của một biến độc lập", "Giá trị dự đoán khi tất cả biến độc lập bằng 0", 
                    "Phương sai của sai số", "Tỷ lệ giải thích của mô hình"],
        "answer": "Giá trị dự đoán khi tất cả biến độc lập bằng 0",
        "explanation": "β0 là giá trị y khi tất cả biến độc lập đều bằng 0."
    },
    {
        "question": "Hệ số tương quan R trong Multiple Linear Regression đo lường điều gì?",
        "options": ["Mức độ giải thích của mô hình", "Mối quan hệ tuyến tính giữa các biến độc lập và biến phụ thuộc", 
                    "Phương sai của sai số", "Độ chính xác của dự đoán"],
        "answer": "Mối quan hệ tuyến tính giữa các biến độc lập và biến phụ thuộc",
        "explanation": "R đo sức mạnh mối quan hệ tuyến tính giữa biến độc lập và phụ thuộc."
    },
    {
        "question": "Điều gì xảy ra với R2 khi thêm một biến độc lập không liên quan vào mô hình?",
        "options": ["R2 luôn giảm", "R2 không đổi", "R2 luôn tăng hoặc không giảm", "R2 trở thành âm"],
        "answer": "R2 luôn tăng hoặc không giảm",
        "explanation": "R2 không bao giờ giảm khi thêm biến, ngay cả khi biến không liên quan."
    },
    {
        "question": "R2 trong Multiple Linear Regression biểu thị điều gì?",
        "options": ["Tỷ lệ phương sai của biến phụ thuộc được giải thích bởi mô hình", 
                    "Độ chính xác của từng tham số trong mô hình", 
                    "Tổng sai số bình phương", 
                    "Số lượng biến độc lập trong mô hình"],
        "answer": "Tỷ lệ phương sai của biến phụ thuộc được giải thích bởi mô hình",
        "explanation": "R2 cho biết phần trăm biến thiên của y được giải thích bởi mô hình."
    },
    {
        "question": "Adjusted R2 khác với R2 ở điểm nào?",
        "options": ["Nó luôn lớn hơn R2", "Nó điều chỉnh theo số lượng biến độc lập trong mô hình", 
                    "Nó đo lường độ chính xác của dự đoán", "Nó không nằm trong khoảng [0, 1]"],
        "answer": "Nó điều chỉnh theo số lượng biến độc lập trong mô hình",
        "explanation": "Adjusted R2 điều chỉnh R2 dựa trên số biến để tránh tăng giả tạo."
    },
    {
        "question": "Metric nào thường được dùng để so sánh các mô hình Multiple Linear Regression với số lượng biến độc lập khác nhau?",
        "options": ["R", "R2", "Adjusted R2", "Standard Error của mô hình"],
        "answer": "Adjusted R2",
        "explanation": "Adjusted R2 phù hợp để so sánh mô hình với số biến khác nhau."
    },
    {
        "question": "Nếu R2=0.85, điều này có nghĩa là gì?",
        "options": ["85% sai số được giải thích bởi mô hình", 
                    "85% biến thiên của biến phụ thuộc được giải thích bởi các biến độc lập", 
                    "Mô hình có 85% độ chính xác dự đoán", 
                    "Có 85 biến độc lập trong mô hình"],
        "answer": "85% biến thiên của biến phụ thuộc được giải thích bởi các biến độc lập",
        "explanation": "R2=0.85 nghĩa là 85% biến thiên của y được mô hình giải thích."
    },
    {
        "question": "Standard Error (SE) của một hệ số trong Multiple Linear Regression đo lường điều gì?",
        "options": ["Độ lệch của dự đoán so với giá trị thực tế", "Độ không chắc chắn trong ước lượng hệ số", 
                    "Tổng sai số bình phương của mô hình", "Mức độ tương quan giữa các biến độc lập"],
        "answer": "Độ không chắc chắn trong ước lượng hệ số",
        "explanation": "SE đo độ tin cậy của ước lượng hệ số, SE nhỏ cho thấy độ chính xác cao."
    }
]

# Phần 8: Regularization (Ridge, Lasso, Elastic Net)
regularization_questions = [
    {
        "question": "Khi áp dụng Gradient Descent cho Ridge Regression, gradient có thêm thành phần nào?",
        "options": ["2λw", "λsign(w)", "λw^2", "−λw"],
        "answer": "2λw",
        "explanation": "Ridge thêm 2λw vào gradient để phạt các trọng số lớn (L2 penalty)."
    },
    {
        "question": "Gradient Descent trong Lasso Regression khác Ridge ở điểm nào?",
        "options": ["Dùng 2λw thay vì λsign(w)", "Dùng λsign(w) thay vì 2λw", 
                    "Không có thành phần penalty trong gradient", "Chỉ tính gradient trên một mẫu duy nhất"],
        "answer": "Dùng λsign(w) thay vì 2λw",
        "explanation": "Lasso dùng L1 penalty (λsign(w)), trong khi Ridge dùng L2 (2λw)."
    },
    {
        "question": "Nếu λ=0 trong Ridge Regression, điều gì xảy ra?",
        "options": ["Mô hình trở thành Linear Regression thông thường", "Tất cả tham số bị triệt tiêu về 0", 
                    "Mô hình không thể hội tụ", "Hình phạt L2 tăng vô hạn"],
        "answer": "Mô hình trở thành Linear Regression thông thường",
        "explanation": "Khi λ=0, không có phạt, Ridge trở thành hồi quy tuyến tính thông thường."
    },
    {
        "question": "Nếu l1_ratio=1 trong Elastic Net, mô hình trở thành gì?",
        "options": ["Ridge Regression", "Lasso Regression", "Linear Regression", "Polynomial Regression"],
        "answer": "Lasso Regression",
        "explanation": "l1_ratio=1 nghĩa là chỉ dùng L1 penalty, tức là Lasso Regression."
    },
    {
        "question": "Khi λ trong Lasso Regression rất lớn, điều gì xảy ra với các tham số?",
        "options": ["Các tham số tăng lên vô hạn", "Hầu hết các tham số bị thu nhỏ về 0", 
                    "Các tham số không thay đổi", "Chỉ tham số lớn nhất bị ảnh hưởng"],
        "answer": "Hầu hết các tham số bị thu nhỏ về 0",
        "explanation": "λ lớn trong Lasso tăng L1 penalty, đẩy nhiều tham số về 0."
    },
    {
        "question": "Hạn chế chính của Lasso Regression so với Ridge là gì?",
        "options": ["Không thể thu nhỏ tham số", "Không xử lý tốt khi các biến độc lập tương quan cao", 
                    "Luôn loại bỏ tất cả biến độc lập", "Không áp dụng được Gradient Descent"],
        "answer": "Không xử lý tốt khi các biến độc lập tương quan cao",
        "explanation": "Lasso chọn ngẫu nhiên một biến trong nhóm tương quan, không ổn định như Ridge."
    },
    {
        "question": "Lasso Regression phù hợp nhất trong trường hợp nào?",
        "options": ["Khi tất cả biến độc lập đều quan trọng", "Khi cần chọn lọc đặc trưng", 
                    "Khi dữ liệu có phương sai sai số không đổi", "Khi Gradient Descent không hội tụ"],
        "answer": "Khi cần chọn lọc đặc trưng",
        "explanation": "Lasso đưa một số hệ số về 0, phù hợp cho việc chọn đặc trưng."
    },
    {
        "question": "Mục đích chính của regularization trong hồi quy là gì?",
        "options": ["Tăng độ chính xác của dự đoán trên tập huấn luyện", 
                    "Giảm overfitting bằng cách kiểm soát độ lớn của tham số", 
                    "Loại bỏ tất cả các biến độc lập không quan trọng", 
                    "Tăng tốc độ hội tụ của Gradient Descent"],
        "answer": "Giảm overfitting bằng cách kiểm soát độ lớn của tham số",
        "explanation": "Regularization phạt các tham số lớn để ngăn mô hình quá khớp."
    },
    {
        "question": "Elastic Net có ưu điểm gì so với Lasso khi xử lý multicollinearity?",
        "options": ["Loại bỏ tất cả biến tương quan", "Kết hợp L2 để ổn định các tham số tương quan", 
                    "Tăng tốc độ tính toán Gradient Descent", "Không cần hệ số λ"],
        "answer": "Kết hợp L2 để ổn định các tham số tương quan",
        "explanation": "Elastic Net dùng cả L1 và L2, xử lý tốt hơn khi biến có tương quan cao."
    },
    {
        "question": "Elastic Net kết hợp những thành phần nào?",
        "options": ["L1 penalty và L2 penalty", "L2 penalty và hàm log", 
                    "L1 penalty và hàm bậc ba", "Chỉ L1 penalty với hệ số thay đổi"],
        "answer": "L1 penalty và L2 penalty",
        "explanation": "Elastic Net kết hợp cả L1 (Lasso) và L2 (Ridge) penalty."
    },
    {
        "question": "Ridge Regression thường được sử dụng trong trường hợp nào?",
        "options": ["Khi cần loại bỏ hoàn toàn các biến không quan trọng", 
                    "Khi các biến độc lập có tương quan cao", 
                    "Khi dữ liệu hoàn toàn không có nhiễu", 
                    "Khi số lượng biến độc lập ít hơn số mẫu"],
        "answer": "Khi các biến độc lập có tương quan cao",
        "explanation": "Ridge xử lý tốt đa cộng tuyến bằng cách thu nhỏ hệ số."
    },
    {
        "question": "Đặc điểm nổi bật của Lasso Regression so với Ridge Regression là gì?",
        "options": ["Thu nhỏ tất cả các tham số về giá trị trung bình", "Có thể đưa một số tham số về đúng 0", 
                    "Không cần hệ số điều chuẩn λ", "Chỉ áp dụng cho dữ liệu phi tuyến"],
        "answer": "Có thể đưa một số tham số về đúng 0",
        "explanation": "Lasso có khả năng chọn đặc trưng bằng cách đặt hệ số về 0, không như Ridge."
    }
]

# Phần 9: Logistic Regression
logistic_regression_questions = [
    {
        "question": "Trong bài toán phân loại nhị phân, nếu tỷ lệ lớp không cân bằng (ví dụ: 90% lớp 0, 10% lớp 1), Logistic Regression có thể gặp vấn đề gì?",
        "options": ["Dự đoán luôn là lớp 1", "Dự đoán luôn là lớp 0", "Không ảnh hưởng", "Mô hình không hội tụ"],
        "answer": "Dự đoán luôn là lớp 0",
        "explanation": "Khi lớp không cân bằng, mô hình có xu hướng dự đoán lớp đa số (lớp 0) để giảm lỗi."
    },
    {
        "question": "Trong Logistic Regression, quá trình huấn luyện nhằm mục đích gì?",
        "options": ["Tối đa hóa hàm likelihood", "Tối thiểu hóa hàm mất mát cross-entropy", "Cả hai", "Không phải A cũng không phải B"],
        "answer": "Cả hai",
        "explanation": "Huấn luyện Logistic Regression tối đa hóa likelihood, tương đương với tối thiểu hóa cross-entropy."
    },
    {
        "question": "Regularization trong Logistic Regression giúp ích gì?",
        "options": ["Tăng tốc độ huấn luyện", "Giảm overfitting", "Tăng độ chính xác trên tập huấn luyện", "Giảm số lượng đặc trưng"],
        "answer": "Giảm overfitting",
        "explanation": "Regularization ngăn mô hình quá khớp bằng cách phạt các trọng số lớn."
    },
    {
        "question": "Logistic Regression có thể được sử dụng cho bài toán đa lớp không?",
        "options": ["Có, nhưng chỉ với chiến lược One-vs-Rest", "Không, chỉ cho nhị phân", 
                    "Có, bằng cách sử dụng hàm sigmoid", "Không, phải dùng Softmax Regression"],
        "answer": "Có, nhưng chỉ với chiến lược One-vs-Rest",
        "explanation": "Logistic Regression cơ bản là nhị phân, nhưng có thể mở rộng đa lớp qua One-vs-Rest."
    },
    {
        "question": "Logistic Regression có thể được coi là một mô hình nào sau đây?",
        "options": ["Mô hình tuyến tính", "Mô hình phi tuyến", "Mô hình cây", "Mô hình dựa trên khoảng cách"],
        "answer": "Mô hình tuyến tính",
        "explanation": "Logistic Regression là mô hình tuyến tính trong không gian log-odds."
    },
    {
        "question": "Logistic Regression được sử dụng cho loại bài toán nào?",
        "options": ["Phân loại", "Hồi quy", "Cả hai", "Không phải phân loại cũng không phải hồi quy"],
        "answer": "Phân loại",
        "explanation": "Logistic Regression dùng để phân loại, không phải hồi quy giá trị liên tục."
    },
    {
        "question": "Cho một mẫu x với nhãn y=1, nếu mô hình dự đoán y^=0.9, giá trị mất mát cross-entropy là bao nhiêu?",
        "options": ["0.105", "0.9", "0.1", "1.0"],
        "answer": "0.105",
        "explanation": "Cross-entropy loss = -log(0.9) ≈ 0.105 cho y=1 và y^=0.9."
    },
    {
        "question": "Trong Logistic Regression, giá trị dự đoán y^ đại diện cho điều gì?",
        "options": ["Giá trị liên tục", "Nhãn lớp cụ thể", "Xác suất thuộc lớp 1", "Một vector xác suất"],
        "answer": "Xác suất thuộc lớp 1",
        "explanation": "y^ là xác suất dự đoán rằng mẫu thuộc lớp 1."
    },
    {
        "question": "Nếu tất cả các đặc trưng trong Logistic Regression đều có giá trị 0, dự đoán của mô hình sẽ là gì?",
        "options": ["0", "0.5", "1", "Không xác định"],
        "answer": "0.5",
        "explanation": "Khi đặc trưng bằng 0, w^T x = 0, sigmoid(0) = 0.5."
    },
    {
        "question": "Hàm nào được sử dụng để chuyển đổi giá trị đầu ra trong Logistic Regression?",
        "options": ["Hàm tuyến tính", "Hàm sigmoid", "Hàm softmax", "Hàm ReLU"],
        "answer": "Hàm sigmoid",
        "explanation": "Hàm sigmoid chuyển đổi đầu ra thành xác suất trong khoảng [0, 1]."
    },
    {
        "question": "Hàm mất mát thường dùng trong Logistic Regression là gì?",
        "options": ["Mean Squared Error", "Cross-Entropy Loss", "Hinge Loss", "Absolute Error"],
        "answer": "Cross-Entropy Loss",
        "explanation": "Cross-entropy là hàm mất mát chuẩn cho bài toán phân loại nhị phân."
    },
    {
        "question": "Điều kiện nào sau đây là giả định của Logistic Regression?",
        "options": ["Dữ liệu tuân theo phân phối Gaussian", "Các đặc trưng là độc lập", 
                    "Mối quan hệ tuyến tính giữa log-odds và đặc trưng", "Không có giả định nào"],
        "answer": "Mối quan hệ tuyến tính giữa log-odds và đặc trưng",
        "explanation": "Logistic Regression giả định log-odds tuyến tính với đặc trưng."
    },
    {
        "question": "Trong Logistic Regression, nếu giá trị w^T x rất lớn, xác suất y^ sẽ gần với giá trị nào?",
        "options": ["0", "0.5", "1", "Không xác định"],
        "answer": "1",
        "explanation": "Khi w^T x lớn, sigmoid(w^T x) tiến gần 1."
    }
]

# Phần 10: Softmax Regression
softmax_regression_questions = [
    {
        "question": "Trong Softmax Regression, tổng các xác suất dự đoán cho các lớp là bao nhiêu?",
        "options": ["0", "1", "Tùy vào dữ liệu", "Không xác định"],
        "answer": "1",
        "explanation": "Hàm softmax đảm bảo tổng xác suất các lớp bằng 1."
    },
    {
        "question": "Softmax Regression được sử dụng cho loại bài toán nào?",
        "options": ["Phân loại nhị phân", "Phân loại đa lớp", "Hồi quy", "Cả A và B"],
        "answer": "Phân loại đa lớp",
        "explanation": "Softmax Regression dùng cho phân loại đa lớp, mở rộng Logistic Regression."
    },
    {
        "question": "Trong bài toán phân loại đa lớp, nếu các lớp không cân bằng, Softmax Regression có thể gặp vấn đề gì?",
        "options": ["Dự đoán luôn là lớp đa số", "Không hội tụ", "Không ảnh hưởng", "Dự đoán luôn là lớp thiểu số"],
        "answer": "Dự đoán luôn là lớp đa số",
        "explanation": "Lớp không cân bằng khiến mô hình thiên về lớp đa số để giảm lỗi."
    },
    {
        "question": "Trong Softmax Regression, nếu tất cả các giá trị z_k bằng nhau, xác suất dự đoán cho mỗi lớp sẽ là gì?",
        "options": ["0", "1", "1/K, với K là số lớp", "Không xác định"],
        "answer": "1/K, với K là số lớp",
        "explanation": "Khi z_k bằng nhau, softmax chia đều xác suất: 1/K."
    },
    {
        "question": "Regularization trong Softmax Regression giúp ích gì?",
        "options": ["Tăng tốc độ huấn luyện", "Giảm overfitting", "Tăng độ chính xác trên tập huấn luyện", "Giảm số lượng đặc trưng"],
        "answer": "Giảm overfitting",
        "explanation": "Regularization giảm overfitting bằng cách phạt các trọng số lớn."
    },
    {
        "question": "Softmax Regression có thể được coi là trường hợp đặc biệt của Logistic Regression khi số lớp là bao nhiêu?",
        "options": ["1", "2", "3", "Không thể"],
        "answer": "2",
        "explanation": "Khi số lớp là 2, softmax tương đương với Logistic Regression."
    },
    {
        "question": "Trong Softmax Regression, quá trình huấn luyện nhằm mục đích gì?",
        "options": ["Tối đa hóa hàm likelihood", "Tối thiểu hóa hàm mất mát cross-entropy", "Cả hai", "Không phải A cũng không phải B"],
        "answer": "Cả hai",
        "explanation": "Tối đa hóa likelihood tương đương với tối thiểu hóa cross-entropy."
    },
    {
        "question": "Softmax Regression sử dụng ma trận trọng số như thế nào?",
        "options": ["Một vector trọng số cho tất cả các lớp", "Một vector trọng số cho mỗi lớp", 
                    "Một ma trận trọng số chung cho tất cả các lớp", "Không sử dụng trọng số"],
        "answer": "Một vector trọng số cho mỗi lớp",
        "explanation": "Mỗi lớp trong softmax có một vector trọng số riêng."
    },
    {
        "question": "Softmax Regression và Logistic Regression khác nhau ở điểm nào?",
        "options": ["Số lớp", "Hàm kích hoạt", "Cả hai", "Không có khác biệt"],
        "answer": "Cả hai",
        "explanation": "Softmax khác Logistic ở số lớp (đa lớp) và hàm kích hoạt (softmax vs sigmoid)."
    },
    {
        "question": "Softmax Regression có thể được coi là một mô hình nào sau đây?",
        "options": ["Mô hình tuyến tính", "Mô hình phi tuyến", "Mô hình cây", "Mô hình dựa trên khoảng cách"],
        "answer": "Mô hình tuyến tính",
        "explanation": "Softmax là mô hình tuyến tính trong không gian log-odds."
    },
    {
        "question": "Hàm nào được sử dụng để chuyển đổi đầu ra trong Softmax Regression?",
        "options": ["Hàm sigmoid", "Hàm softmax", "Hàm tuyến tính", "Hàm ReLU"],
        "answer": "Hàm softmax",
        "explanation": "Hàm softmax chuyển đổi đầu ra thành phân phối xác suất qua các lớp."
    },
    {
        "question": "Trong Softmax Regression, nếu giá trị z_k của một lớp rất lớn, xác suất dự đoán cho lớp đó sẽ gần với giá trị nào?",
        "options": ["0", "0.5", "1", "Không xác định"],
        "answer": "1",
        "explanation": "z_k lớn làm softmax gán xác suất gần 1 cho lớp đó."
    },
    {
        "question": "Hàm mất mát thường dùng trong Softmax Regression là gì?",
        "options": ["Mean Squared Error", "Cross-Entropy Loss", "Hinge Loss", "Absolute Error"],
        "answer": "Cross-Entropy Loss",
        "explanation": "Cross-entropy là hàm mất mát chuẩn cho phân loại đa lớp."
    }
]

# Phần 11: Decision Trees
decision_tree_questions = [
    {
        "question": "Khi xây dựng Decision Tree, thuộc tính nào được chọn để chia nhánh?",
        "options": ["Thuộc tính có Information Gain cao nhất", "Thuộc tính có giá trị trung bình lớn nhất", 
                    "Thuộc tính có số lượng giá trị duy nhất ít nhất", "Thuộc tính xuất hiện đầu tiên trong dữ liệu"],
        "answer": "Thuộc tính có Information Gain cao nhất",
        "explanation": "Decision Tree chọn thuộc tính có Information Gain cao nhất để tối ưu hóa việc chia."
    },
    {
        "question": "Ưu điểm nào sau đây KHÔNG phải của Decision Trees?",
        "options": ["Dễ hiểu và trực quan", "Không yêu cầu chuẩn hóa dữ liệu", 
                    "Hiệu quả với dữ liệu cao chiều", "Không nhạy cảm với giá trị thiếu"],
        "answer": "Không nhạy cảm với giá trị thiếu",
        "explanation": "Decision Trees nhạy cảm với giá trị thiếu và cần xử lý trước."
    },
    {
        "question": "Information Gain trong Decision Trees là gì?",
        "options": ["Sự giảm entropy sau khi chia nhánh", "Tổng entropy của tất cả các nút lá", 
                    "Độ sâu tối đa của cây", "Số lượng nút trong cây"],
        "answer": "Sự giảm entropy sau khi chia nhánh",
        "explanation": "Information Gain đo lường mức giảm độ hỗn loạn sau khi chia nhánh."
    },
    {
        "question": "Giá trị Entropy của một nút trong Decision Tree là bao nhiêu khi tất cả các mẫu trong nút thuộc cùng một lớp?",
        "options": ["0", "1", "∞", "-1"],
        "answer": "0",
        "explanation": "Entropy bằng 0 khi tất cả mẫu cùng lớp, tức là hoàn toàn tinh khiết."
    },
    {
        "question": "Nhược điểm chính của Decision Trees là gì?",
        "options": ["Dễ bị overfitting nếu không được cắt tỉa", "Không thể xử lý dữ liệu hạng mục", 
                    "Yêu cầu nhiều bộ nhớ", "Chỉ hoạt động với dữ liệu số"],
        "answer": "Dễ bị overfitting nếu không được cắt tỉa",
        "explanation": "Decision Trees dễ quá khớp nếu không giới hạn độ sâu hoặc cắt tỉa."
    },
    {
        "question": "Entropy và Gini Index có thể được sử dụng trong Decision Trees cho bài toán hồi quy không?",
        "options": ["Có, với các điều chỉnh phù hợp", "Không, chỉ dùng cho phân loại", 
                    "Chỉ Entropy có thể dùng cho hồi quy", "Chỉ Gini Index có thể dùng cho hồi quy"],
        "answer": "Không, chỉ dùng cho phân loại",
        "explanation": "Entropy và Gini dùng cho phân loại; hồi quy dùng các tiêu chí như MSE."
    },
    {
        "question": "Information Gain được tính như thế nào trong Decision Trees?",
        "options": ["Entropy của nút cha trừ Entropy trung bình có trọng số của các nút con", 
                    "Tổng Entropy của tất cả các nút lá", "Độ sâu của cây", "Số lượng nút trong cây"],
        "answer": "Entropy của nút cha trừ Entropy trung bình có trọng số của các nút con",
        "explanation": "Information Gain = Entropy trước chia - Entropy trung bình sau chia."
    },
    {
        "question": "Việc tối đa hóa Information Gain ở mỗi bước có thể dẫn đến điều gì?",
        "options": ["Underfitting", "Overfitting", "Không ảnh hưởng đến overfitting", "Giảm độ sâu của cây"],
        "answer": "Overfitting",
        "explanation": "Tối đa hóa quá mức có thể làm cây quá phức tạp, dẫn đến overfitting."
    },
    {
        "question": "Entropy trong Decision Trees đo lường điều gì?",
        "options": ["Độ tinh khiết của nhãn trong một nút", "Khoảng cách giữa các điểm dữ liệu", 
                    "Độ phức tạp của cây", "Tốc độ huấn luyện của mô hình"],
        "answer": "Độ tinh khiết của nhãn trong một nút",
        "explanation": "Entropy đo mức độ hỗn loạn hoặc không tinh khiết của nhãn trong nút."
    },
    {
        "question": "Sự khác biệt chính giữa Gini Index và Entropy trong Decision Trees là gì?",
        "options": ["Gini Index dùng cho phân loại, Entropy dùng cho hồi quy", 
                    "Gini Index nhạy cảm hơn với sự thay đổi nhỏ trong dữ liệu", 
                    "Entropy đo mức độ hỗn loạn, Gini Index đo độ tạp chất", 
                    "Không có sự khác biệt, chúng có thể thay thế cho nhau"],
        "answer": "Entropy đo mức độ hỗn loạn, Gini Index đo độ tạp chất",
        "explanation": "Cả hai đo độ tinh khiết, nhưng Entropy dựa trên lý thuyết thông tin, Gini đơn giản hơn."
    },
    {
        "question": "Trong bài toán hồi quy, giá trị dự đoán của một nút lá trong Decision Tree là gì?",
        "options": ["Giá trị trung bình của các mẫu trong nút đó", "Giá trị lớn nhất trong các mẫu", 
                    "Giá trị nhỏ nhất trong các mẫu", "Giá trị trung vị của các mẫu"],
        "answer": "Giá trị trung bình của các mẫu trong nút đó",
        "explanation": "Trong hồi quy, nút lá dự đoán bằng trung bình các giá trị trong nút."
    },
    {
        "question": "Mục đích chính của pruning (cắt tỉa) trong Decision Trees là gì?",
        "options": ["Giảm độ sâu của cây", "Tăng accuracy trên tập huấn luyện", 
                    "Ngăn chặn overfitting", "Tăng tốc độ huấn luyện"],
        "answer": "Ngăn chặn overfitting",
        "explanation": "Pruning giảm độ phức tạp của cây để tránh quá khớp."
    },
    {
        "question": "Trong thực tế, khi nào bạn nên ưu tiên sử dụng Gini Index thay vì Entropy?",
        "options": ["Khi cần tính toán nhanh hơn", "Khi dữ liệu có nhiều lớp", 
                    "Khi cần mô hình chính xác hơn", "Khi dữ liệu có nhiễu"],
        "answer": "Khi cần tính toán nhanh hơn",
        "explanation": "Gini Index tính toán nhanh hơn Entropy do công thức đơn giản hơn."
    },
    {
        "question": "Làm thế nào Decision Trees xử lý các thuộc tính liên tục?",
        "options": ["Chia thành các bin rời rạc", "Sử dụng ngưỡng để chia thành hai khoảng", 
                    "Bỏ qua các thuộc tính liên tục", "Chuyển đổi thành các giá trị hạng mục"],
        "answer": "Sử dụng ngưỡng để chia thành hai khoảng",
        "explanation": "Decision Trees chọn ngưỡng tối ưu để chia thuộc tính liên tục."
    },
    {
        "question": "Giá trị Gini Index của một nút trong đó tất cả các mẫu thuộc cùng một lớp là bao nhiêu?",
        "options": ["0", "1", "0.5", "∞"],
        "answer": "0",
        "explanation": "Gini Index bằng 0 khi nút hoàn toàn tinh khiết."
    },
    {
        "question": "Trong Decision Tree, thành phần nào đại diện cho điều kiện để chia dữ liệu?",
        "options": ["Root node (Nút gốc)", "Leaf node (Nút lá)", "Internal node (Nút bên trong)", "Branch (Nhánh)"],
        "answer": "Internal node (Nút bên trong)",
        "explanation": "Nút bên trong chứa điều kiện chia dữ liệu."
    },
    {
        "question": "Tiêu chí nào sau đây KHÔNG được sử dụng để đánh giá việc chia nhánh trong Decision Tree?",
        "options": ["Gini impurity", "Entropy", "Mean Squared Error (MSE)", "Correlation coefficient"],
        "answer": "Correlation coefficient",
        "explanation": "Correlation coefficient không được dùng để đánh giá chia nhánh."
    },
    {
        "question": "Decision Tree là một thuật toán học máy thuộc loại nào?",
        "options": ["Học có giám sát (Supervised Learning)", "Học không giám sát (Unsupervised Learning)", 
                    "Học bán giám sát (Semi-supervised Learning)", "Học tăng cường (Reinforcement Learning)"],
        "answer": "Học có giám sát (Supervised Learning)",
        "explanation": "Decision Tree cần dữ liệu có nhãn, thuộc học có giám sát."
    }
]

# Phần 12: k-Nearest Neighbors (k-NN)
knn_questions = [
    {
        "question": "Cho một tập dữ liệu với hai đặc trưng: chiều cao (cm) và thu nhập (đồng). Nếu không chuẩn hóa dữ liệu, điều gì có thể xảy ra khi áp dụng k-NN?",
        "options": ["Chiều cao sẽ có ảnh hưởng lớn hơn thu nhập", "Thu nhập sẽ có ảnh hưởng lớn hơn chiều cao", 
                    "Cả hai có ảnh hưởng như nhau", "Không ảnh hưởng đến kết quả"],
        "answer": "Thu nhập sẽ có ảnh hưởng lớn hơn chiều cao",
        "explanation": "Thu nhập có thang đo lớn hơn (đồng) sẽ chi phối khoảng cách nếu không chuẩn hóa."
    },
    {
        "question": "Điều gì xảy ra khi chọn giá trị k quá lớn trong thuật toán k-NN?",
        "options": ["Thuật toán có thể trở nên nhạy cảm với nhiễu", "Mô hình có thể bị overfitting", 
                    "Mô hình có thể bị underfitting", "Thuật toán k-NN không bị ảnh hưởng bởi giá trị"],
        "answer": "Mô hình có thể bị underfitting",
        "explanation": "k lớn làm mô hình quá tổng quát, dẫn đến underfitting."
    },
    {
        "question": "Thuật toán k-NN có phù hợp với dữ liệu lớn không?",
        "options": ["Có, vì nó dễ triển khai", "Có, vì nó có thời gian dự đoán nhanh", 
                    "Không, vì nó đòi hỏi tính toán khoảng cách với toàn bộ tập dữ liệu", 
                    "Không, vì nó yêu cầu dữ liệu phải tuyến tính"],
        "answer": "Không, vì nó đòi hỏi tính toán khoảng cách với toàn bộ tập dữ liệu",
        "explanation": "k-NN chậm với dữ liệu lớn do phải tính khoảng cách cho mọi điểm."
    },
    {
        "question": "Cho tập dữ liệu phân loại gồm 2 lớp A và B. Khi tăng giá trị k từ 3 lên 15, điều gì có khả năng xảy ra nhất?",
        "options": ["Mô hình có xu hướng overfitting", "Mô hình có xu hướng underfitting", 
                    "Không ảnh hưởng", "Độ chính xác luôn tăng"],
        "answer": "Mô hình có xu hướng underfitting",
        "explanation": "Tăng k làm mô hình dựa vào quá nhiều láng giềng, dẫn đến underfitting."
    },
    {
        "question": "Thuật toán k-NN có thể được sử dụng cho bài toán nào sau đây?",
        "options": ["Phân loại (Classification)", "Hồi quy (Regression)", "Cả hai Classification & Regression", "Không có phương án nào đúng"],
        "answer": "Cả hai Classification & Regression",
        "explanation": "k-NN dùng được cho cả phân loại (lớp phổ biến nhất) và hồi quy (trung bình)."
    },
    {
        "question": "Trong k-NN, nếu dữ liệu có nhiều đặc trưng (features), vấn đề gì có thể xảy ra?",
        "options": ["Hiệu suất của mô hình sẽ tăng lên", "Khoảng cách giữa các điểm sẽ trở nên kém phân biệt hơn", 
                    "k-NN hoạt động tốt hơn với dữ liệu có nhiều chiều", "Mô hình trở nên ít bị ảnh hưởng bởi giá trị k"],
        "answer": "Khoảng cách giữa các điểm sẽ trở nên kém phân biệt hơn",
        "explanation": "Nhiều đặc trưng gây ra 'lời nguyền chiều' làm khoảng cách mất ý nghĩa."
    },
    {
        "question": "Một tập dữ liệu có 1000 điểm trong không gian 2D. Khi tăng số chiều lên 100D, điều gì xảy ra với thuật toán k-NN?",
        "options": ["Hiệu suất tăng vì có thêm thông tin", "Khoảng cách giữa các điểm trở nên vô nghĩa", 
                    "k-NN hoạt động nhanh hơn", "Mô hình không bị ảnh hưởng"],
        "answer": "Khoảng cách giữa các điểm trở nên vô nghĩa",
        "explanation": "Chiều cao làm tất cả điểm xa nhau gần như đồng đều, mất ý nghĩa phân biệt."
    },
    {
        "question": "Cho hai điểm A(3,4) và B(7,1). Khoảng cách Manhattan giữa hai điểm này là?",
        "options": ["5", "6", "7", "8"],
        "answer": "7",
        "explanation": "Manhattan = |3-7| + |4-1| = 4 + 3 = 7."
    },
    {
        "question": "Ưu điểm chính của thuật toán k-NN là gì?",
        "options": ["Huấn luyện nhanh, chỉ cần tính toán một lần", "Đưa ra dự đoán nhanh ngay cả với tập dữ liệu lớn", 
                    "Không yêu cầu giả định về phân phối dữ liệu", "Không bị ảnh hưởng bởi số lượng đặc trưng"],
        "answer": "Không yêu cầu giả định về phân phối dữ liệu",
        "explanation": "k-NN không cần giả định phân phối, chỉ dựa trên khoảng cách."
    },
    {
        "question": "Khoảng cách phổ biến nào sau đây thường được sử dụng trong thuật toán k-NN?",
        "options": ["Khoảng cách Euclidean", "Khoảng cách Manhattan", "Khoảng cách Minkowski", "Tất cả các phương án trên"],
        "answer": "Tất cả các phương án trên",
        "explanation": "Euclidean, Manhattan, và Minkowski đều là các khoảng cách phổ biến trong k-NN."
    },
    {
        "question": "Giả sử ta dùng k-NN Regression với k = 3 để dự đoán giá trị y của điểm x=4. Tập dữ liệu (x, y) như sau: (2, 3), (5, 7), (7, 8), (3, 4). Giá trị dự đoán là bao nhiêu?",
        "options": ["4", "5", "6", "7"],
        "answer": "5",
        "explanation": "3 láng giềng gần nhất: (2, 3), (3, 4), (5, 7). Trung bình = (3+4+7)/3 = 5."
    },
    {
        "question": "Trong thuật toán k-NN, 'k' biểu thị điều gì?",
        "options": ["Số lượng lớp trong dữ liệu huấn luyện", "Số lượng hàng trong tập dữ liệu huấn luyện", 
                    "Số lượng láng giềng gần nhất được xem xét khi phân loại một điểm dữ liệu", 
                    "Số lần thuật toán lặp lại để tìm kiếm láng giềng tốt nhất"],
        "answer": "Số lượng láng giềng gần nhất được xem xét khi phân loại một điểm dữ liệu",
        "explanation": "k là số láng giềng gần nhất dùng để dự đoán."
    },
    {
        "question": "Thuật toán k-NN thuộc nhóm thuật toán nào trong Machine Learning?",
        "options": ["Học có giám sát (Supervised Learning)", "Học không giám sát (Unsupervised Learning)", 
                    "Học tăng cường (Reinforcement Learning)", "Học sâu (Deep Learning)"],
        "answer": "Học có giám sát (Supervised Learning)",
        "explanation": "k-NN cần dữ liệu có nhãn để dự đoán, thuộc học có giám sát."
    }
]

# Phần 13: Naive Bayes
naive_bayes_questions = [
    {
        "question": "Loại Naive Bayes nào sau đây thường được sử dụng cho dữ liệu văn bản?",
        "options": ["Gaussian Naive Bayes", "Bernoulli Naive Bayes", "Multinomial Naive Bayes", "K-Nearest Neighbors"],
        "answer": "Multinomial Naive Bayes",
        "explanation": "Multinomial Naive Bayes phù hợp với dữ liệu văn bản như tần suất từ."
    },
    {
        "question": "So với Logistic Regression, Naive Bayes có ưu điểm gì?",
        "options": ["Không yêu cầu giả định về phân phối dữ liệu", "Luôn cho độ chính xác cao hơn", 
                    "Không cần điều chỉnh siêu tham số", "Có thể xử lý dữ liệu phi tuyến tính"],
        "answer": "Không cần điều chỉnh siêu tham số",
        "explanation": "Naive Bayes đơn giản, không cần điều chỉnh nhiều tham số như Logistic."
    },
    {
        "question": "Một tập dữ liệu có 3 lớp C1, C2, C3 với xác suất tiên nghiệm: P(C1)=0.5, P(C2)=0.3, P(C3)=0.2. Một điểm dữ liệu X có xác suất có điều kiện: P(X|C1)=0.2, P(X|C2)=0.4, P(X|C3)=0.6. Hỏi điểm dữ liệu X thuộc lớp nào theo Naive Bayes?",
        "options": ["C1", "C2", "C3", "Không xác định"],
        "answer": "C3",
        "explanation": "P(C1|X) ∝ 0.5*0.2=0.1; P(C2|X) ∝ 0.3*0.4=0.12; P(C3|X) ∝ 0.2*0.6=0.12. C2 và C3 bằng nhau, nhưng thường chọn lớp có P(X|C) cao nhất nếu xét chuẩn hóa, cần tính P(X)."
    },
    {
        "question": "Gaussian Naive Bayes thường được sử dụng khi nào?",
        "options": ["Khi đặc trưng có phân phối nhị thức (Bernoulli)", "Khi đặc trưng có phân phối chuẩn (Gaussian)", 
                    "Khi đặc trưng có phân phối Poisson", "Khi đặc trưng có phân phối bất kỳ"],
        "answer": "Khi đặc trưng có phân phối chuẩn (Gaussian)",
        "explanation": "Gaussian Naive Bayes giả định đặc trưng tuân theo phân phối chuẩn."
    },
    {
        "question": "Cho một tập dữ liệu phân loại thư rác, trong đó: P(S)=0.3, P(N)=0.7, P(W|S)=0.8, P(W|N)=0.1. Tính xác suất một email là spam khi biết rằng nó chứa từ 'giảm giá' (P(S|W))?",
        "options": ["0.852", "0.774", "0.654", "0.981"],
        "answer": "0.774",
        "explanation": "P(S|W) = P(W|S)P(S) / P(W) = 0.8*0.3 / (0.8*0.3 + 0.1*0.7) = 0.24 / 0.31 ≈ 0.774."
    },
    {
        "question": "Điều nào sau đây là không đúng về Naive Bayes?",
        "options": ["Naive Bayes có thể hoạt động tốt ngay cả khi giả định độc lập không hoàn toàn đúng", 
                    "Naive Bayes có thể sử dụng cho cả bài toán phân loại nhị phân và đa lớp", 
                    "Naive Bayes có thể xử lý dữ liệu bị thiếu giá trị mà không cần xử lý trước", 
                    "Naive Bayes thường được dùng trong lọc thư rác (Spam Filtering)"],
        "answer": "Naive Bayes có thể xử lý dữ liệu bị thiếu giá trị mà không cần xử lý trước",
        "explanation": "Naive Bayes cần xử lý giá trị thiếu trước khi áp dụng."
    },
    {
        "question": "Naive Bayes thường không phù hợp trong trường hợp nào dưới đây?",
        "options": ["Khi dữ liệu có nhiều đặc trưng độc lập", "Khi dữ liệu có chứa đặc trưng liên quan chặt chẽ với nhau", 
                    "Khi tập dữ liệu có kích thước nhỏ", "Khi mô hình cần dự đoán nhanh trên dữ liệu lớn"],
        "answer": "Khi dữ liệu có chứa đặc trưng liên quan chặt chẽ với nhau",
        "explanation": "Giả định độc lập của Naive Bayes không đúng khi đặc trưng tương quan cao."
    },
    {
        "question": "Nhược điểm chính của Naive Bayes là gì?",
        "options": ["Yêu cầu dữ liệu lớn", "Giả định độc lập giữa các đặc trưng thường không đúng", 
                    "Tính toán phức tạp", "Không thể xử lý dữ liệu liên tục"],
        "answer": "Giả định độc lập giữa các đặc trưng thường không đúng",
        "explanation": "Giả định độc lập hiếm khi đúng trong thực tế, ảnh hưởng đến hiệu suất."
    },
    {
        "question": "Loại Naive Bayes nào phù hợp nhất cho dữ liệu liên tục?",
        "options": ["Gaussian Naive Bayes", "Multinomial Naive Bayes", "Bernoulli Naive Bayes", "Categorical Naive Bayes"],
        "answer": "Gaussian Naive Bayes",
        "explanation": "Gaussian Naive Bayes dùng cho dữ liệu liên tục với phân phối chuẩn."
    },
    {
        "question": "Xác suất hậu nghiệm P(C|X) trong Naive Bayes được tính như thế nào?",
        "options": ["P(C|X) = P(X|C)P(C) / P(X)", "P(C|X) = P(X|C)P(C)", "P(C|X) = P(C) / P(X)", "P(C|X) = P(X|C)"],
        "answer": "P(C|X) = P(X|C)P(C) / P(X)",
        "explanation": "Đây là công thức Bayes chuẩn để tính xác suất hậu nghiệm."
    },
    {
        "question": "Naive Bayes dựa trên nguyên tắc nào dưới đây?",
        "options": ["Các thuộc tính của dữ liệu là hoàn toàn độc lập", "Các thuộc tính của dữ liệu phụ thuộc tuyến tính vào nhau", 
                    "Các thuộc tính của dữ liệu có quan hệ phi tuyến tính", "Không có giả định nào về mối quan hệ giữa các thuộc tính"],
        "answer": "Các thuộc tính của dữ liệu là hoàn toàn độc lập",
        "explanation": "Naive Bayes giả định các đặc trưng độc lập với nhau."
    },
    {
        "question": "Vấn đề 'zero probability' trong Naive Bayes là gì?",
        "options": ["Xác suất của một lớp bằng 0", "Xác suất có điều kiện của một đặc trưng bằng 0", 
                    "Xác suất tiên nghiệm bằng 0", "Xác suất hậu nghiệm bằng 0"],
        "answer": "Xác suất có điều kiện của một đặc trưng bằng 0",
        "explanation": "Khi P(X|C)=0, xác suất hậu nghiệm thành 0, cần kỹ thuật như Laplace smoothing."
    },
    {
        "question": "Naive Bayes thường được sử dụng trong ứng dụng nào?",
        "options": ["Spam Filtering", "Image Classification", "Object Tracking", "Tất cả các phương án trên"],
        "answer": "Spam Filtering",
        "explanation": "Naive Bayes phổ biến trong lọc thư rác do tốc độ và hiệu quả."
    },
    {
        "question": "Một tập dữ liệu huấn luyện có hai lớp: Lớp A và Lớp B. Xác suất tiên nghiệm: P(A)=0.4, P(B)=0.6. Một điểm dữ liệu X có xác suất có điều kiện: P(X|A)=0.5, P(X|B)=0.2. Tính xác suất hậu nghiệm P(A|X)?",
        "options": ["0.625", "0.375", "0.5", "0.4"],
        "answer": "0.625",
        "explanation": "P(A|X) = 0.5*0.4 / (0.5*0.4 + 0.2*0.6) = 0.2 / 0.32 = 0.625."
    },
    {
        "question": "Naive Bayes thuộc loại thuật toán nào trong Machine Learning?",
        "options": ["Học có giám sát (Supervised Learning)", "Học không giám sát (Unsupervised Learning)", 
                    "Học tăng cường (Reinforcement Learning)", "Học sâu (Deep Learning)"],
        "answer": "Học có giám sát (Supervised Learning)",
        "explanation": "Naive Bayes cần dữ liệu có nhãn, thuộc học córoman sát."
    },
    {
        "question": "So với các thuật toán phân loại khác, Naive Bayes có ưu điểm nào?",
        "options": ["Tốc độ huấn luyện và dự đoán nhanh", "Yêu cầu nhiều dữ liệu hơn so với SVM", 
                    "Không cần giả định nào về dữ liệu", "Luôn có độ chính xác cao hơn Decision Tree"],
        "answer": "Tốc độ huấn luyện và dự đoán nhanh",
        "explanation": "Naive Bayes nhanh do tính toán đơn giản dựa trên xác suất."
    }
]

# Phần 14: Ứng dụng Thực tế
real_world_questions = [
    {
        "question": "Một nền tảng học trực tuyến muốn phát hiện sớm học viên có khả năng ngưng học. Họ nên làm gì?",
        "options": ["Xem tỷ lệ hoàn thành khóa học theo tháng", "Ước tính khả năng rời bỏ dựa trên hoạt động trước đó", 
                    "Gửi khảo sát đánh giá chất lượng bài giảng", "Thống kê lượt truy cập theo chuyên ngành"],
        "answer": "Ước tính khả năng rời bỏ dựa trên hoạt động trước đó",
        "explanation": "Dự đoán dựa trên hành vi trước đó là cách tiếp cận học máy hiệu quả."
    },
    {
        "question": "Một ứng dụng âm nhạc muốn đề xuất danh sách nhạc cá nhân hóa cho người dùng dựa trên sở thích và hành vi nghe trước đó. Cách tiếp cận nào phù hợp?",
        "options": ["Liệt kê bài hát phổ biến nhất trong tuần", "Sắp xếp nhạc theo thể loại", 
                    "Ước tính xu hướng nghe của người dùng cụ thể", "Thống kê số lần nhấn 'like' mỗi bài"],
        "answer": "Ước tính xu hướng nghe của người dùng cụ thể",
        "explanation": "Dự đoán sở thích cá nhân hóa dựa trên dữ liệu người dùng là tối ưu."
    },
    {
        "question": "Một hãng vận chuyển muốn nhóm các tuyến đường có điểm tương đồng để tối ưu hóa hoạt động. Họ nên làm gì?",
        "options": ["Đếm số lượt xe đi trong ngày", "Xếp tuyến theo vị trí địa lý của trạm", 
                    "Tìm các tuyến có đặc điểm tương tự nhau", "Thống kê doanh thu theo tuyến"],
        "answer": "Tìm các tuyến có đặc điểm tương tự nhau",
        "explanation": "Phân cụm các tuyến tương đồng là cách tiếp cận học không giám sát phù hợp."
    },
    {
        "question": "Một công ty truyền thông muốn tự động xác định các chủ đề chính trong hàng nghìn bài viết tin tức mỗi ngày mà không cần gán thẻ thủ công. Họ nên làm gì?",
        "options": ["Lọc các bài viết theo độ dài", "Tìm ra các chủ đề thường gặp dựa trên nội dung", 
                    "Xếp bài theo nguồn đăng tải", "Sắp xếp bài viết theo ngày đăng"],
        "answer": "Tìm ra các chủ đề thường gặp dựa trên nội dung",
        "explanation": "Topic modeling (như LDA) là cách học không giám sát để tìm chủ đề."
    },
    {
        "question": "Một ngân hàng muốn dự đoán khả năng một khách hàng mới sẽ hoàn trả khoản vay. Họ nên làm gì?",
        "options": ["Phân tích thời gian xử lý đơn vay", "Dự đoán hành vi dựa vào hồ sơ tương tự", 
                    "Tổng hợp danh sách khách hàng mới", "Xác định các khu vực có nhiều đơn vay"],
        "answer": "Dự đoán hành vi dựa vào hồ sơ tương tự",
        "explanation": "Dự đoán dựa trên hồ sơ tương tự là cách tiếp cận phân loại học máy."
    },
    {
        "question": "Một bệnh viện muốn dự đoán khả năng bệnh nhân quay lại tái khám trong 30 ngày tới. Cách nào hợp lý?",
        "options": ["Tạo báo cáo thống kê các loại thuốc dùng", "Tìm những người có lịch sử tương tự", 
                    "Ước lượng khả năng quay lại dựa trên hồ sơ", "Thống kê số bệnh nhân theo khu vực"],
        "answer": "Ước lượng khả năng quay lại dựa trên hồ sơ",
        "explanation": "Dự đoán dựa trên hồ sơ cá nhân là cách tiếp cận học máy hiệu quả."
    },
    {
        "question": "Một tổ chức bảo tồn động vật muốn phát hiện các nhóm loài vật có hành vi tương tự nhau từ dữ liệu theo dõi GPS mà không cần phân loại sẵn. Họ nên làm gì?",
        "options": ["Sắp xếp dữ liệu theo thời gian", "Tìm các mẫu di chuyển có tính chất gần nhau", 
                    "Ghi lại số lượng mẫu được theo dõi", "So sánh khoảng cách di chuyển mỗi ngày"],
        "answer": "Tìm các mẫu di chuyển có tính chất gần nhau",
        "explanation": "Phân cụm dữ liệu GPS là cách học không giám sát để tìm nhóm tương tự."
    },
    {
        "question": "Một công ty bảo hiểm muốn xây dựng hệ thống để ước tính khả năng xảy ra rủi ro tài chính đối với từng khách hàng trong năm tới. Cách tiếp cận nào phù hợp?",
        "options": ["Gộp khách hàng theo ngành nghề", "Tính tổng số yêu cầu bồi thường trong năm trước", 
                    "Dự báo nguy cơ dựa trên lịch sử từng cá nhân", "So sánh số lượng hợp đồng theo khu vực"],
        "answer": "Dự báo nguy cơ dựa trên lịch sử từng cá nhân",
        "explanation": "Dự đoán dựa trên lịch sử cá nhân là cách tiếp cận học máy phù hợp."
    },
    {
        "question": "Một công ty an ninh mạng muốn phát hiện hành vi truy cập bất thường từ dữ liệu chưa được lọc thủ công. Họ nên làm gì?",
        "options": ["Tạo bản đồ truy cập theo múi giờ", "Xác định mẫu khác biệt trong hoạt động", 
                    "Gộp các IP theo vị trí địa lý", "Tổng hợp số lượt đăng nhập theo ngày"],
        "answer": "Xác định mẫu khác biệt trong hoạt động",
        "explanation": "Phát hiện bất thường (anomaly detection) là cách học máy hiệu quả."
    },
    {
        "question": "Một siêu thị muốn tìm các kiểu mua sắm thường gặp giữa các khách hàng. Cách nào phù hợp?",
        "options": ["So sánh doanh thu giữa các tháng", "Tìm các mẫu hành vi mua giống nhau", 
                    "Lập danh sách sản phẩm giảm giá", "Thống kê thời gian cao điểm trong ngày"],
        "answer": "Tìm các mẫu hành vi mua giống nhau",
        "explanation": "Phân cụm hành vi mua sắm là cách học không giám sát phù hợp."
    }
]

# Phần 15: Reinforcement Learning
reinforcement_learning_questions = [
    {
        "question": "Trong mê cung, Environment trả về gì sau khi robot di chuyển sang phải?",
        "options": ["Action và Policy", "State và Reward", "Value Function", "Discount Factor"],
        "answer": "State và Reward",
        "explanation": "Environment trả về trạng thái mới và phần thưởng sau hành động."
    },
    {
        "question": "Tại sao thử và sai (trial and error) quan trọng trong RL?",
        "options": ["Để định nghĩa môi trường", "Để học các hành động tốt nhất qua kinh nghiệm", 
                    "Để tính toán phần thưởng chính xác", "Để lưu trữ tất cả trạng thái"],
        "answer": "Để học các hành động tốt nhất qua kinh nghiệm",
        "explanation": "Thử và sai giúp agent học chính sách tối ưu từ trải nghiệm."
    },
    {
        "question": "Phần thưởng (Reward) trong ví dụ mê cung đại diện cho điều gì?",
        "options": ["Phản hồi cho một hành động (ví dụ: +10 cho kho báu)", "Vị trí của robot trong lưới", 
                    "Chính sách chọn hành động", "Xác suất chuyển sang trạng thái mới"],
        "answer": "Phản hồi cho một hành động (ví dụ: +10 cho kho báu)",
        "explanation": "Reward là phản hồi định lượng cho hành động của agent."
    },
    {
        "question": "Chính sách (Policy) trong Reinforcement Learning là gì?",
        "options": ["Giá trị của một trạng thái", "Phần thưởng cho một hành động", 
                    "Chiến lược chọn hành động", "Quy tắc của môi trường"],
        "answer": "Chiến lược chọn hành động",
        "explanation": "Policy định nghĩa cách agent chọn hành động trong mỗi trạng thái."
    },
    {
        "question": "Vai trò của Value Function trong ví dụ mê cung là gì?",
        "options": ["Chọn hành động tiếp theo", "Ước tính con đường tốt nhất đến kho báu", 
                    "Xác định tường của mê cung", "Đặt giá trị phần thưởng"],
        "answer": "Ước tính con đường tốt nhất đến kho báu",
        "explanation": "Value Function ước lượng phần thưởng dài hạn từ mỗi trạng thái."
    },
    {
        "question": "Vai trò của Agent trong Reinforcement Learning là gì?",
        "options": ["Xác định quy tắc của môi trường", "Thực hiện hành động và học từ phần thưởng", 
                    "Cung cấp phần thưởng cho môi trường", "Lưu trữ tất cả trạng thái có thể"],
        "answer": "Thực hiện hành động và học từ phần thưởng",
        "explanation": "Agent tương tác với môi trường, học từ phản hồi để cải thiện."
    },
    {
        "question": "Hành động (Action) trong Reinforcement Learning là gì?",
        "options": ["Phần thưởng do môi trường cung cấp", "Giá trị của một trạng thái", 
                    "Lựa chọn của Agent (ví dụ: di chuyển sang phải)", "Phản hồi của môi trường"],
        "answer": "Lựa chọn của Agent (ví dụ: di chuyển sang phải)",
        "explanation": "Action là quyết định do agent đưa ra trong trạng thái hiện tại."
    },
    {
        "question": "Agent gửi gì đến Environment trong ví dụ mê cung?",
        "options": ["State", "Reward", "Action", "Policy"],
        "answer": "Action",
        "explanation": "Agent gửi hành động (như 'di chuyển sang phải') đến môi trường."
    },
    {
        "question": "RL khác với supervised learning như thế nào?",
        "options": ["Sử dụng dữ liệu có nhãn", "Học từ phần thưởng", "Yêu cầu tập dữ liệu cố định", "Dự đoán đầu ra trực tiếp"],
        "answer": "Học từ phần thưởng",
        "explanation": "RL học qua thử nghiệm và phần thưởng, không cần dữ liệu có nhãn."
    },
    {
        "question": "Thành phần nào kết nối Agent và Environment?",
        "options": ["Value Function", "State và Reward", "Policy", "Discount Factor"],
        "answer": "State và Reward",
        "explanation": "State và Reward là giao tiếp chính giữa agent và môi trường."
    },
    {
        "question": "Mục tiêu chính của Reinforcement Learning là gì?",
        "options": ["Dự đoán nhãn từ dữ liệu", "Tối đa hóa phần thưởng tích lũy", 
                    "Giảm thiểu thời gian tính toán", "Lưu trữ tất cả trạng thái"],
        "answer": "Tối đa hóa phần thưởng tích lũy",
        "explanation": "RL nhằm tối ưu hóa tổng phần thưởng dài hạn."
    },
    {
        "question": "Trong mê cung, điều gì cho thấy một Policy tốt?",
        "options": ["Tốc độ tính toán cao", "Đạt được kho báu với phần thưởng cao", 
                    "Tránh tất cả hành động", "Ghi nhớ tất cả trạng thái"],
        "answer": "Đạt được kho báu với phần thưởng cao",
        "explanation": "Policy tốt dẫn agent đến mục tiêu với phần thưởng tối ưu."
    },
    {
        "question": "Policy định nghĩa điều gì trong ví dụ mê cung?",
        "options": ["Bố cục của mê cung", "Phần thưởng khi đến (3,3)", "Hành động thực hiện trong mỗi trạng thái", "Xác suất va vào tường"],
        "answer": "Hành động thực hiện trong mỗi trạng thái",
        "explanation": "Policy xác định hành động agent chọn ở mỗi vị trí."
    },
    {
        "question": "Mục đích của Discount Factor (γ) là gì?",
        "options": ["Ưu tiên phần thưởng ngắn hạn so với dài hạn", "Chọn hành động tốt nhất", 
                    "Xác định trạng thái của môi trường", "Tính phần thưởng tức thời"],
        "answer": "Ưu tiên phần thưởng ngắn hạn so với dài hạn",
        "explanation": "γ điều chỉnh tầm quan trọng của phần thưởng tương lai."
    },
    {
        "question": "Trong ví dụ mê cung, Environment đại diện cho cái gì?",
        "options": ["Vị trí của robot", "Lưới 3x3 với tường và kho báu", "Hành động của robot", "Phần thưởng khi đến kho báu"],
        "answer": "Lưới 3x3 với tường và kho báu",
        "explanation": "Environment là không gian mà agent tương tác (lưới mê cung)."
    },
    {
        "question": "State trong ví dụ mê cung là gì?",
        "options": ["Hành động của robot (ví dụ: di chuyển sang phải)", "Chính sách chọn hành động", 
                    "Phần thưởng nhận được (ví dụ: -1)", "Vị trí của robot (ví dụ: (2,2))"],
        "answer": "Vị trí của robot (ví dụ: (2,2))",
        "explanation": "State là trạng thái hiện tại của agent, ở đây là tọa độ."
    },
    {
        "question": "Trong mê cung, điều gì xảy ra nếu robot va vào tường?",
        "options": ["Nhận +10 phần thưởng", "Chuyển sang trạng thái mới", "Nhận -5 phần thưởng", "Cập nhật chính sách"],
        "answer": "Nhận -5 phần thưởng",
        "explanation": "Va vào tường thường được phạt bằng phần thưởng âm."
    },
    {
        "question": "Value Function ước lượng điều gì?",
        "options": ["Xác suất của một hành động", "Chuyển đổi trạng thái của môi trường", 
                    "Phần thưởng tức thời cho một hành động", "Phần thưởng dài hạn của một trạng thái"],
        "answer": "Phần thưởng dài hạn của một trạng thái",
        "explanation": "Value Function dự đoán tổng phần thưởng tích lũy từ trạng thái."
    },
    {
        "question": "Điều gì xảy ra nếu Discount Factor (γ) được đặt bằng 0?",
        "options": ["Phần thưởng dài hạn được ưu tiên", "Chỉ phần thưởng tức thời được xem xét", 
                    "Không có phần thưởng nào được tính", "Hành động được chọn ngẫu nhiên"],
        "answer": "Chỉ phần thưởng tức thời được xem xét",
        "explanation": "γ=0 làm agent chỉ quan tâm đến phần thưởng hiện tại."
    },
    {
        "question": "Đặc điểm chính của Environment trong RL là gì?",
        "options": ["Nó học từ Agent", "Nó cung cấp State và Reward", "Nó chọn hành động", "Nó đặt Discount Factor"],
        "answer": "Nó cung cấp State và Reward",
        "explanation": "Environment cung cấp phản hồi (state, reward) cho hành động của agent."
    }
]

# ==== HÀM HIỂN THỊ QUIZ ====
def run_quiz(questions, session_key_prefix):
    with st.form(f"{session_key_prefix}_form"):
        for i, q in enumerate(questions):
            st.subheader(f"Câu {i+1}: {q['question']}")
            st.radio("Chọn một đáp án:", q['options'], key=f"{session_key_prefix}_q{i}")
            with st.expander("Giải thích câu hỏi"):
                st.info(q["explanation"])
        col1, col2 = st.columns(2)
        with col1:
            submitted = st.form_submit_button("Nộp bài")
        with col2:
            reset = st.form_submit_button("Làm lại")

    if submitted:
        answers = [st.session_state.get(f"{session_key_prefix}_q{i}", None) for i in range(len(questions))]
        st.session_state[f"{session_key_prefix}_answers"] = answers
        st.session_state[f"{session_key_prefix}_submitted"] = True
    elif reset:
        for i in range(len(questions)):
            key = f"{session_key_prefix}_q{i}"
            if key in st.session_state:
                del st.session_state[key]
        st.session_state[f"{session_key_prefix}_answers"] = [None] * len(questions)
        st.session_state[f"{session_key_prefix}_submitted"] = False

    if st.session_state.get(f"{session_key_prefix}_submitted", False):
        answers = st.session_state[f"{session_key_prefix}_answers"]
        not_answered = [i+1 for i, ans in enumerate(answers) if ans is None]
        if not_answered:
            st.warning(f"❗ Bạn chưa trả lời câu số: {', '.join(map(str, not_answered))}")
        else:
            score = sum(1 for i, ans in enumerate(answers) if ans == questions[i]["answer"])
            st.success(f"🎉 Bạn đã hoàn thành bài trắc nghiệm với {score}/{len(questions)} câu đúng!")
            st.write("### ❌ Các câu bạn trả lời sai:")
            any_wrong = False
            for i, q in enumerate(questions):
                if answers[i] != q["answer"]:
                    any_wrong = True
                    st.markdown(f"**Câu {i+1}:** {q['question']}")
                    st.write(f"- Đáp án bạn chọn: {answers[i]}")
                    st.write(f"- Đáp án đúng: {q['answer']}")
                    st.info(q["explanation"])
            if not any_wrong:
                st.success("🎉 Bạn đã trả lời đúng tất cả các câu hỏi!")

# ==== GỌI QUIZ DỰA TRÊN PHẦN ĐƯỢC CHỌN ====
if section == "Cơ bản về AI và Học Máy":
    run_quiz(ai_ml_questions, "ai_ml")
elif section == "Thuật toán Phân cụm":
    run_quiz(clustering_questions, "cluster")
elif section == "Giảm chiều dữ liệu":
    run_quiz(dimred_questions, "dimred")
elif section == "Hồi quy Tuyến tính và Normal Equation":
    run_quiz(linear_regression_questions, "linear_reg")
elif section == "Gradient Descent":
    run_quiz(gradient_descent_questions, "grad_desc")
elif section == "Mini-batch, Stochastic, Batch Gradient Descent":
    run_quiz(gradient_variants_questions, "grad_vars")
elif section == "Multiple Linear Regression":
    run_quiz(multiple_regression_questions, "multi_reg")
elif section == "Regularization (Ridge, Lasso, Elastic Net)":
    run_quiz(regularization_questions, "regularization")
elif section == "Logistic Regression":
    run_quiz(logistic_regression_questions, "logistic_reg")
elif section == "Softmax Regression":
    run_quiz(softmax_regression_questions, "softmax_reg")
elif section == "Decision Trees":
    run_quiz(decision_tree_questions, "decision_tree")
elif section == "k-Nearest Neighbors (k-NN)":
    run_quiz(knn_questions, "knn")
elif section == "Naive Bayes":
    run_quiz(naive_bayes_questions, "naive_bayes")
elif section == "Ứng dụng Thực tế":
    run_quiz(real_world_questions, "real_world")
elif section == "Reinforcement Learning":
    run_quiz(reinforcement_learning_questions, "rl")