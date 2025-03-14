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
def draw_simple_network(ax, layers, title, arrows=True, arrow_color='#4B8BBE', backprop_arrows=False, annotations=None):
    colors = ['#FFB6C1', '#87CEEB', '#98FB98']  # Màu pastel: Đầu vào, Ẩn, Đầu ra
    neuron_positions = []
    for i, layer_size in enumerate(layers):
        offset = (layer_size - 1) / 2
        layer_pos = []
        for j in range(layer_size):
            y_pos = j - offset
            neuron = Circle((i, y_pos), 0.08, facecolor=colors[i], edgecolor='#333333', linewidth=1.5, zorder=3)
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
                    arrow = FancyArrowPatch(start, end, mutation_scale=25, color=arrow_color, lw=1.5, alpha=0.8, zorder=2)
                    ax.add_patch(arrow)
    
    # Mũi tên lan truyền ngược
    if backprop_arrows:
        for i in range(1, len(layers)):
            for j in range(layers[i]):
                for k in range(layers[i - 1]):
                    start = (i - 0.1, neuron_positions[i][j])
                    end = (i - 0.9, neuron_positions[i - 1][k])
                    arrow = FancyArrowPatch(start, end, mutation_scale=25, color='#FF8C69', lw=1.5, alpha=0.8, zorder=2)
                    ax.add_patch(arrow)
    
    # Chú thích cải tiến
    if annotations:
        for (x, y, text) in annotations:
            ax.text(x, y, text, fontsize=11, ha='center', va='center', 
                    bbox=dict(facecolor='#FFFFFF', edgecolor='#D3D3D3', boxstyle='round,pad=0.3', alpha=0.9))

    # Thiết lập trục
    ax.set_xlim(-0.5, len(layers) - 0.5)
    ax.set_ylim(-max(layers) / 2 - 0.5, max(layers) / 2 + 0.5)
    ax.set_xticks(range(len(layers)))
    ax.set_xticklabels(['Đầu vào', 'Ẩn', 'Đầu ra'], fontsize=11)
    ax.set_yticks([])
    ax.set_title(title, fontsize=14, pad=15, fontweight='bold')
    ax.set_facecolor('#F5F6F5')
    for spine in ['left', 'right', 'top', 'bottom']:
        ax.spines[spine].set_color('#D3D3D3')
        ax.spines[spine].set_linewidth(0.8)

# Bước 1: Khởi tạo mô hình
fig, ax = plt.subplots(figsize=(8, 4))
draw_simple_network(ax, [3, 3, 2], "1. Khởi tạo mạng nơ-ron", arrows=False, 
                    annotations=[(1, 1.5, "Xây dựng cấu trúc mạng\nvà khởi tạo tham số")])
plt.savefig(os.path.join(output_dir, "step1_init.png"), dpi=400, bbox_inches='tight')
plt.close()

# Bước 2: Lan truyền thuận
fig, ax = plt.subplots(figsize=(8, 4))
draw_simple_network(ax, [3, 3, 2], "2. Lan truyền thuận", arrow_color='#4B8BBE', 
                    annotations=[(1, 1.5, "Tính toán đầu ra từ\nđầu vào qua các lớp")])
plt.savefig(os.path.join(output_dir, "step2_feedforward.png"), dpi=400, bbox_inches='tight')
plt.close()

# Bước 3: Tính hàm mất mát
fig, ax = plt.subplots(figsize=(8, 4))
x = np.arange(3)
y_true = [0, 1, 0]
y_pred = [0.2, 0.7, 0.1]
ax.bar(x - 0.15, y_true, 0.3, label='Thực tế', color='#66B2FF', edgecolor='#333333', linewidth=1.5, zorder=3)
ax.bar(x + 0.15, y_pred, 0.3, label='Dự đoán', color='#FF9999', edgecolor='#333333', linewidth=1.5, zorder=3)
ax.set_xticks(x)
ax.set_xticklabels(['0', '1', '2'], fontsize=11)
ax.set_title("3. Tính hàm mất mát", fontsize=14, pad=15, fontweight='bold')
ax.text(1, 1.2, "Đánh giá sai lệch giữa\nthực tế và dự đoán", fontsize=11, ha='center', 
        bbox=dict(facecolor='#FFFFFF', edgecolor='#D3D3D3', boxstyle='round,pad=0.3', alpha=0.9))
ax.legend(fontsize=11, frameon=False, loc='upper right')
ax.grid(True, axis='y', linestyle='--', alpha=0.3, zorder=0)
ax.set_facecolor('#F5F6F5')
for spine in ['left', 'right', 'top', 'bottom']:
    ax.spines[spine].set_color('#D3D3D3')
    ax.spines[spine].set_linewidth(0.8)
plt.savefig(os.path.join(output_dir, "step3_loss.png"), dpi=400, bbox_inches='tight')
plt.close()

# Bước 4: Lan truyền ngược
fig, ax = plt.subplots(figsize=(8, 4))
draw_simple_network(ax, [3, 3, 2], "4. Lan truyền ngược", arrows=False, backprop_arrows=True, 
                    annotations=[(1, 1.5, "Tính gradient của\nhàm mất mát")])
plt.savefig(os.path.join(output_dir, "step4_backprop.png"), dpi=400, bbox_inches='tight')
plt.close()

# Bước 5: Cập nhật tham số
fig, ax = plt.subplots(figsize=(8, 4))
x = np.linspace(0, 5, 50)
y = np.exp(-x)
ax.plot(x, y, color='#4B8BBE', lw=2.5, label='Mất mát giảm', marker='o', markersize=5, markevery=10)
ax.set_title("5. Cập nhật tham số", fontsize=14, pad=15, fontweight='bold')
ax.text(2.5, 1.5, "Điều chỉnh trọng số\ndựa trên gradient", fontsize=11, ha='center', 
        bbox=dict(facecolor='#FFFFFF', edgecolor='#D3D3D3', boxstyle='round,pad=0.3', alpha=0.9))
ax.set_xlabel("Số lần học", fontsize=11)
ax.set_ylabel("Mất mát", fontsize=11)
ax.legend(fontsize=11, frameon=False, loc='upper right')
ax.grid(True, linestyle='--', alpha=0.3, zorder=0)
ax.set_facecolor('#F5F6F5')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
for spine in ['left', 'bottom']:
    ax.spines[spine].set_color('#D3D3D3')
    ax.spines[spine].set_linewidth(0.8)
plt.savefig(os.path.join(output_dir, "step5_gradient.png"), dpi=400, bbox_inches='tight')
plt.close()

# Bước 6: Lặp lại (Minh họa vòng lặp với các mũi tên chạy vòng)
fig, ax = plt.subplots(figsize=(8, 4))
# Vẽ vòng tròn trung tâm
circle = Circle((0.5, 0.5), 0.3, edgecolor='#77DD77', linewidth=2.5, facecolor='#D4EDDA', alpha=0.9, zorder=3)
ax.add_patch(circle)

# Thêm các mũi tên chạy vòng lặp
arrow1 = FancyArrowPatch((0.8, 0.5), (0.65, 0.65), mutation_scale=25, color='#77DD77', lw=2.5, alpha=0.9, zorder=4)
arrow2 = FancyArrowPatch((0.65, 0.65), (0.5, 0.8), mutation_scale=25, color='#77DD77', lw=2.5, alpha=0.9, zorder=4)
arrow3 = FancyArrowPatch((0.5, 0.8), (0.35, 0.65), mutation_scale=25, color='#77DD77', lw=2.5, alpha=0.9, zorder=4)
arrow4 = FancyArrowPatch((0.35, 0.65), (0.2, 0.5), mutation_scale=25, color='#77DD77', lw=2.5, alpha=0.9, zorder=4)
arrow5 = FancyArrowPatch((0.2, 0.5), (0.35, 0.35), mutation_scale=25, color='#77DD77', lw=2.5, alpha=0.9, zorder=4)
arrow6 = FancyArrowPatch((0.35, 0.35), (0.5, 0.2), mutation_scale=25, color='#77DD77', lw=2.5, alpha=0.9, zorder=4)
arrow7 = FancyArrowPatch((0.5, 0.2), (0.65, 0.35), mutation_scale=25, color='#77DD77', lw=2.5, alpha=0.9, zorder=4)
arrow8 = FancyArrowPatch((0.65, 0.35), (0.8, 0.5), mutation_scale=25, color='#77DD77', lw=2.5, alpha=0.9, zorder=4)
ax.add_patch(arrow1)
ax.add_patch(arrow2)
ax.add_patch(arrow3)
ax.add_patch(arrow4)
ax.add_patch(arrow5)
ax.add_patch(arrow6)
ax.add_patch(arrow7)
ax.add_patch(arrow8)

# Thêm ký hiệu "Bước 2" trên vòng lặp
ax.text(0.8, 0.65, "Bước 2", fontsize=12, ha='center', va='center', color='#77DD77', fontweight='bold',
        bbox=dict(facecolor='#FFFFFF', edgecolor='#77DD77', boxstyle='round,pad=0.3', alpha=0.9))

# Chú thích chính giữa
ax.text(0.5, 0.5, "Quay lại bước 2\nqua nhiều epoch\ncho đến khi L hội tụ", ha='center', va='center', fontsize=14, fontweight='bold', 
        bbox=dict(facecolor='#FFFFFF', edgecolor='#77DD77', boxstyle='round,pad=0.4', alpha=0.95))

ax.set_title("6. Lặp lại quá trình", fontsize=14, pad=15, fontweight='bold')
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.set_xticks([])
ax.set_yticks([])
ax.set_facecolor('#F5F6F5')
for spine in ['left', 'right', 'top', 'bottom']:
    ax.spines[spine].set_color('#D3D3D3')
    ax.spines[spine].set_linewidth(0.8)
plt.savefig(os.path.join(output_dir, "step6_repeat_improved.png"), dpi=400, bbox_inches='tight')
plt.close()

print(f"Đã tạo 6 ảnh minh họa chuyên nghiệp hơn và lưu vào thư mục '{output_dir}'!")