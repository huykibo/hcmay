import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.stats import norm

# Tạo thư mục để lưu ảnh nếu chưa tồn tại
output_dir = "netw"
os.makedirs(output_dir, exist_ok=True)

# Bước 1: Lớp đầu vào (Input Layer)
def create_step1_image():
    np.random.seed(42)
    
    # Tạo một ảnh MNIST giả (28x28)
    mnist_sample = np.random.randint(0, 256, (28, 28))
    mnist_sample[10:18, 10:18] = 255  # Tạo một hình vuông trắng để minh họa số
    
    # Tạo vector 784 chiều
    flattened = mnist_sample.flatten()
    x = np.arange(784)
    
    # Vẽ hình
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    
    # Bên trái: Hình ảnh MNIST
    ax1.imshow(mnist_sample, cmap='gray')
    ax1.set_title("Ảnh MNIST (28x28)")
    ax1.axis('off')
    
    # Bên phải: Vector 784 chiều
    ax2.plot(x, flattened, 'b-', lw=1)
    ax2.set_title("Vector 784 chiều")
    ax2.set_xlabel("Index pixel")
    ax2.set_ylabel("Giá trị (0-255)")
    ax2.set_ylim(0, 300)
    
    fig.suptitle("Bước 1: Lớp đầu vào nhận dữ liệu từ ảnh MNIST", fontsize=12)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(os.path.join(output_dir, "nn_step_1.png"), dpi=300)
    plt.close()

# Bước 2: Lớp ẩn (Hidden Layers)
def create_step2_image():
    np.random.seed(42)
    
    # Minh họa các nơ-ron trong lớp ẩn
    input_nodes = 5  # Đại diện cho 784 đầu vào
    hidden_nodes = 4
    
    # Tạo vị trí cho các nơ-ron
    x_input = np.zeros(input_nodes)
    y_input = np.arange(input_nodes)
    x_hidden = np.ones(hidden_nodes)
    y_hidden = np.linspace(0, input_nodes-1, hidden_nodes)
    
    # Vẽ hình
    plt.figure(figsize=(6, 6))
    
    # Vẽ các nơ-ron
    plt.scatter(x_input, y_input, s=200, c='blue', label='Lớp đầu vào (784)')
    plt.scatter(x_hidden, y_hidden, s=200, c='orange', label='Lớp ẩn')
    
    # Vẽ các kết nối
    for i in range(input_nodes):
        for j in range(hidden_nodes):
            plt.plot([x_input[i], x_hidden[j]], [y_input[i], y_hidden[j]], 
                    'gray', alpha=0.3)
    
    # Thêm đường cong ReLU
    x = np.linspace(-1, 2, 100)
    relu = np.maximum(0, x)
    plt.plot(x + 1.5, relu + 2, 'r-', lw=2, label='Hàm ReLU')
    
    plt.title("Bước 2: Lớp ẩn trích xuất đặc trưng", fontsize=12)
    plt.xlabel("Layers")
    plt.ylabel("Neurons")
    plt.legend()
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "nn_step_2.png"), dpi=300)
    plt.close()

# Bước 3: Lớp đầu ra (Output Layer)
def create_step3_image():
    np.random.seed(42)
    
    # Minh họa lớp đầu ra với 10 nơ-ron
    hidden_nodes = 4
    output_nodes = 10
    
    # Tạo vị trí cho các nơ-ron
    x_hidden = np.zeros(hidden_nodes)
    y_hidden = np.linspace(0, output_nodes-1, hidden_nodes)
    x_output = np.ones(output_nodes)
    y_output = np.arange(output_nodes)
    
    # Giả lập xác suất Softmax
    softmax = np.random.rand(output_nodes)
    softmax = softmax / softmax.sum()
    
    # Vẽ hình
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    
    # Bên trái: Kết nối từ lớp ẩn đến lớp đầu ra
    ax1.scatter(x_hidden, y_hidden, s=200, c='orange', label='Lớp ẩn')
    ax1.scatter(x_output, y_output, s=200, c='green', label='Lớp đầu ra (0-9)')
    for i in range(hidden_nodes):
        for j in range(output_nodes):
            ax1.plot([x_hidden[i], x_output[j]], [y_hidden[i], y_output[j]], 
                    'gray', alpha=0.3)
    ax1.set_title("Kết nối đến lớp đầu ra")
    ax1.legend()
    ax1.axis('off')
    
    # Bên phải: Thanh xác suất Softmax
    ax2.bar(y_output, softmax, color='green', alpha=0.7)
    ax2.set_title("Xác suất Softmax")
    ax2.set_xlabel("Nhãn (0-9)")
    ax2.set_ylabel("Xác suất")
    
    fig.suptitle("Bước 3: Lớp đầu ra dự đoán nhãn với Softmax", fontsize=12)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(os.path.join(output_dir, "nn_step_3.png"), dpi=300)
    plt.close()

# Bước 4: Huấn luyện
def create_step4_image():
    np.random.seed(42)
    
    # Minh họa quá trình tối ưu hóa
    epochs = np.arange(1, 11)
    loss = 2 / (1 + 0.5 * epochs) + np.random.normal(0, 0.05, epochs.size)
    accuracy = 1 - 1 / (1 + 0.3 * epochs) + np.random.normal(0, 0.02, epochs.size)
    
    # Vẽ hình
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    
    # Bên trái: Loss giảm dần
    ax1.plot(epochs, loss, 'r-', label='Hàm mất mát')
    ax1.set_title("Hàm mất mát giảm")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.legend()
    
    # Bên phải: Accuracy tăng dần
    ax2.plot(epochs, accuracy, 'b-', label='Độ chính xác')
    ax2.set_title("Độ chính xác tăng")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.legend()
    
    fig.suptitle("Bước 4: Huấn luyện tối ưu hóa trọng số", fontsize=12)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(os.path.join(output_dir, "nn_step_4.png"), dpi=300)
    plt.close()

# Tạo tất cả các ảnh
if __name__ == "__main__":
    create_step1_image()
    create_step2_image()
    create_step3_image()
    create_step4_image()
    print(f"Đã tạo 4 ảnh minh họa cho Neural Network trong thư mục '{output_dir}'.")