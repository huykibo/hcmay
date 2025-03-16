import os
import numpy as np
import struct

# Định nghĩa hàm để đọc file .idx3-ubyte (hình ảnh)
def load_mnist_images(filename):
    """
    Đọc file hình ảnh MNIST từ định dạng .idx3-ubyte.
    Trả về: numpy array chứa các hình ảnh (số mẫu, chiều cao, chiều rộng).
    """
    with open(filename, 'rb') as f:
        # Đọc header
        magic, num, rows, cols = struct.unpack('>IIII', f.read(16))
        # Đọc dữ liệu hình ảnh
        images = np.fromfile(f, dtype=np.uint8).reshape(num, rows, cols)
    return images

# Định nghĩa hàm để đọc file .idx1-ubyte (nhãn)
def load_mnist_labels(filename):
    """
    Đọc file nhãn MNIST từ định dạng .idx1-ubyte.
    Trả về: numpy array chứa các nhãn (số mẫu,).
    """
    with open(filename, 'rb') as f:
        # Đọc header
        magic, num = struct.unpack('>II', f.read(8))
        # Đọc dữ liệu nhãn
        labels = np.fromfile(f, dtype=np.uint8)
    return labels

def load_mnist_data():
    """
    Tải toàn bộ dữ liệu MNIST từ các file .idx trong thư mục cụ thể.
    Trả về: train_images, train_labels, test_images, test_labels.
    """
    # Đường dẫn cụ thể
    dataset_path = r"D:\pyucen\dulieuumnist"
    train_images_path = os.path.join(dataset_path, "train-images-idx3-ubyte")
    train_labels_path = os.path.join(dataset_path, "train-labels-idx1-ubyte")
    test_images_path = os.path.join(dataset_path, "t10k-images-idx3-ubyte")
    test_labels_path = os.path.join(dataset_path, "t10k-labels-idx1-ubyte")

    # Kiểm tra sự tồn tại của file trước khi tải
    for path in [train_images_path, train_labels_path, test_images_path, test_labels_path]:
        if not os.path.exists(path):
            print(f"Lỗi: File không tồn tại tại đường dẫn: {path}")
            return None, None, None, None

    # Tải dữ liệu
    try:
        train_images = load_mnist_images(train_images_path)
        train_labels = load_mnist_labels(train_labels_path)
        test_images = load_mnist_images(test_images_path)
        test_labels = load_mnist_labels(test_labels_path)

        print(f"Đã tải dữ liệu huấn luyện: {train_images.shape} hình ảnh, {train_labels.shape} nhãn")
        print(f"Đã tải dữ liệu kiểm tra: {test_images.shape} hình ảnh, {test_labels.shape} nhãn")

        return train_images, train_labels, test_images, test_labels

    except Exception as e:
        print(f"Lỗi khi tải dữ liệu MNIST: {e}")
        return None, None, None, None

if __name__ == "__main__":
    # Gọi hàm để tải dữ liệu
    train_images, train_labels, test_images, test_labels = load_mnist_data()

    # Kiểm tra dữ liệu đã tải thành công
    if train_images is not None:
        print("\nVí dụ dữ liệu:")
        print(f" - Kích thước một hình ảnh huấn luyện: {train_images[0].shape}")
        print(f" - Nhãn đầu tiên: {train_labels[0]}")
        print(f" - Giá trị pixel đầu tiên (mẫu 0): {train_images[0][0]}")