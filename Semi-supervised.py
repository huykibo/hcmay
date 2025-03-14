import os
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, Ellipse, Circle

# Thiết lập font chuyên nghiệp
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = 12

# Tạo thư mục lưu ảnh
output_dir = "plnw"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Hàm vẽ mạng nơ-ron cải tiến
def draw_simple_network(ax, layers, title, arrows=True, arrow_color='#4682B4', backprop_arrows=False, annotations=None):
    colors = ['#FFB6C1', '#87CEEB', '#98FB98']  # Màu pastel: Đầu vào, Ẩn, Đầu ra
    neuron_positions = []
    for i, layer_size in enumerate(layers):
        offset = (layer_size - 1) / 2
        layer_pos = []
        for j in range(layer_size):
            y_pos = j - offset
            # Nơ-ron đơn giản hóa, không bóng đổ để tránh rối
            neuron = Circle((i, y_pos), 0.08, color=colors[i], edgecolor='black', linewidth=1)
            ax.add_patch(neuron)
            layer_pos.append(y_pos)
        neuron_positions.append(layer_pos)
    
    # Vẽ mũi tên mượt mà hơn
    if arrows:
        for i in range(len(layers) - 1):
            for j in range(layers[i]):
                for k in range(layers[i + 1]):
                    start = (i + 0.1, neuron_positions[i][j])
                    end = (i + 0.9, neuron_positions[i + 1][k])
                    arrow = FancyArrowPatch(start, end, mutation_scale=15, color=arrow_color, lw=1, alpha=0.8)
                    ax.add_patch(arrow)
    
    # Mũi tên lan truyền ngược
    if backprop_arrows:
        for i in range(1, len(layers)):
            for j in range(layers[i]):
                for k in range(layers[i - 1]):
                    start = (i - 0.1, neuron_positions[i][j])
                    end = (i - 0.9, neuron_positions[i - 1][k])
                    arrow = FancyArrowPatch(start, end, mutation_scale=15, color='#DC143C', lw=1, alpha=0.8)
                    ax.add_patch(arrow)
    
    # Chú thích đơn giản, rõ ràng
    if annotations:
        for (x, y, text) in annotations:
            ax.text(x, y, text, fontsize=10, ha='center', bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', boxstyle='round,pad=0.2'))

    # Thiết lập trục
    ax.set_xlim(-0.5, len(layers) - 0.5)
    ax.set_ylim(-max(layers) / 2 - 0.5, max(layers) / 2 + 0.5)
    ax.set_xticks(range(len(layers)))
    ax.set_xticklabels(['Đầu vào', 'Ẩn', 'Đầu ra'], fontsize=11)
    ax.set_yticks([])
    ax.set_title(title, fontsize=14, pad=10, fontweight='bold')
    ax.set_facecolor('#F8F9FA')  # Màu nền nhạt

# Bước 1: Khởi tạo mô hình
fig, ax = plt.subplots(figsize=(8, 4))
draw_simple_network(ax, [3, 3, 2], "1. Khởi tạo mạng nơ-ron", arrows=False, annotations=[(1, 1.5, "Cấu trúc mạng")])
plt.savefig(os.path.join(output_dir, "step1_init.png"), dpi=300, bbox_inches='tight')
plt.close()

# Bước 2: Lan truyền thuận
fig, ax = plt.subplots(figsize=(8, 4))
draw_simple_network(ax, [3, 3, 2], "2. Lan truyền thuận", arrow_color='#4682B4', annotations=[(1, 1.5, "Tính toán tiến")])
plt.savefig(os.path.join(output_dir, "step2_feedforward.png"), dpi=300, bbox_inches='tight')
plt.close()

# Bước 3: Tính hàm mất mát
fig, ax = plt.subplots(figsize=(6, 4))
x = np.arange(3)
y_true = [0, 1, 0]
y_pred = [0.2, 0.7, 0.1]
ax.bar(x - 0.15, y_true, 0.3, label='Thực tế', color='#87CEEB', edgecolor='black', linewidth=1)
ax.bar(x + 0.15, y_pred, 0.3, label='Dự đoán', color='#FFB6C1', edgecolor='black', linewidth=1)
ax.set_xticks(x)
ax.set_xticklabels(['0', '1', '2'], fontsize=11)
ax.set_title("3. Tính sai lệch", fontsize=14, pad=10, fontweight='bold')
ax.legend(fontsize=10, frameon=False)
ax.set_facecolor('#F8F9FA')
plt.savefig(os.path.join(output_dir, "step3_loss.png"), dpi=300, bbox_inches='tight')
plt.close()

# Bước 4: Lan truyền ngược
fig, ax = plt.subplots(figsize=(8, 4))
draw_simple_network(ax, [3, 3, 2], "4. Lan truyền ngược", arrows=False, backprop_arrows=True, annotations=[(1, 1.5, "Điều chỉnh ngược")])
plt.savefig(os.path.join(output_dir, "step4_backprop.png"), dpi=300, bbox_inches='tight')
plt.close()

# Bước 5: Cập nhật tham số
fig, ax = plt.subplots(figsize=(6, 4))
x = np.linspace(0, 5, 50)
y = np.exp(-x)
ax.plot(x, y, color='#4682B4', lw=2, label='Mất mát giảm')
ax.set_title("5. Cập nhật tham số", fontsize=14, pad=10, fontweight='bold')
ax.set_xlabel("Số lần học", fontsize=11)
ax.set_ylabel("Mất mát", fontsize=11)
ax.legend(fontsize=10, frameon=False)
ax.set_facecolor('#F8F9FA')
plt.savefig(os.path.join(output_dir, "step5_gradient.png"), dpi=300, bbox_inches='tight')
plt.close()

# Bước 6: Lặp lại
fig, ax = plt.subplots(figsize=(6, 4))
loop = Ellipse((0.5, 0.5), 0.8, 0.6, edgecolor='#228B22', linewidth=2, facecolor='#D4EDDA', alpha=0.9)
ax.add_patch(loop)
arrow = FancyArrowPatch((0.75, 0.5), (0.25, 0.5), mutation_scale=20, color='#228B22', lw=2, alpha=0.9)
ax.add_patch(arrow)
ax.text(0.5, 0.5, "Lặp lại", ha='center', va='center', fontsize=14, fontweight='bold', 
        bbox=dict(facecolor='white', edgecolor='none', boxstyle='round,pad=0.3', alpha=0.9))
ax.set_title("6. Lặp lại quá trình", fontsize=14, pad=10, fontweight='bold')
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.set_xticks([])
ax.set_yticks([])
ax.set_facecolor('#F8F9FA')
plt.savefig(os.path.join(output_dir, "step6_repeat_improved.png"), dpi=300, bbox_inches='tight')
plt.close()

print(f"Đã tạo 6 ảnh minh họa chuyên nghiệp và lưu vào thư mục '{output_dir}'!")