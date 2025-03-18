import matplotlib.pyplot as plt
import numpy as np
import os

# Tạo thư mục nếu chưa tồn tại
output_dir = "plnw"
if not os.path.exists(output_dir):
    os.makedirs(output_dir, exist_ok=True)  # Thêm exist_ok để tránh lỗi nếu thư mục đã tồn tại

# Cài đặt giao diện chuyên nghiệp
plt.rcParams.update({
    'font.size': 12,
    'font.family': 'sans-serif',
    'axes.titlesize': 18,
    'axes.labelsize': 14,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'figure.figsize': (12, 7),  # Tăng kích thước để chứa công thức
    'text.usetex': False,       # Sử dụng mathtext thay vì LaTeX
    'mathtext.fontset': 'stix', # Font toán học đẹp
})

# Định nghĩa màu sắc
colors = {
    'input': '#4CAF50',    # Xanh lá
    'hidden': '#2196F3',   # Xanh dương
    'output': '#FF9800',   # Cam
    'arrow': '#000000',    # Đen cho mũi tên
    'text': '#212121',     # Đen đậm
    'background': '#F5F5F5' # Xám nhạt
}

# Hàm vẽ mũi tên với công thức
def draw_arrow(ax, start, end, color=colors['arrow'], label=None, offset=(0, 0), lw=2):
    ax.annotate('', xy=end, xytext=start,
                arrowprops=dict(arrowstyle='->', color=color, lw=lw))
    if label:
        mid = ((start[0] + end[0]) / 2 + offset[0], (start[1] + end[1]) / 2 + offset[1])
        ax.text(mid[0], mid[1], label, ha='center', va='center', color=colors['text'], fontsize=12,
                bbox=dict(facecolor='white', edgecolor=colors['arrow'], alpha=0.8, pad=3))

# Bước 1: Khởi tạo mô hình
def draw_step1():
    fig, ax = plt.subplots()
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 7)
    ax.set_axis_off()
    fig.patch.set_facecolor(colors['background'])
    ax.set_title("Bước 1: Khởi tạo Mô hình", pad=20, color=colors['text'])

    # Vẽ node
    ax.add_patch(plt.Circle((2, 4), 0.5, color=colors['input'], label='Đầu Vào ($X$)', ec='black'))
    ax.add_patch(plt.Circle((5, 5), 0.5, color=colors['hidden'], label='Lớp Ẩn', ec='black'))
    ax.add_patch(plt.Circle((5, 3), 0.5, color=colors['hidden'], ec='black'))
    ax.add_patch(plt.Circle((8, 4), 0.5, color=colors['output'], label='Đầu Ra ($\hat{Y}$)', ec='black'))

    # Mũi tên
    draw_arrow(ax, (2.5, 4), (4.5, 5), color='green', label=r"$W^1, b^1$", offset=(0, 0.5))
    draw_arrow(ax, (2.5, 4), (4.5, 3), color='green')
    draw_arrow(ax, (5.5, 5), (7.5, 4), color='green', label=r"$W^2, b^2$", offset=(0, 0.5))
    draw_arrow(ax, (5.5, 3), (7.5, 4), color='green')

    # Chú thích công thức
    ax.text(5, 1.5, r"$W$: Trọng số, $b$: Độ lệch (khởi tạo ngẫu nhiên)", 
            ha='center', va='center', color=colors['text'], fontsize=14,
            bbox=dict(facecolor='white', edgecolor=colors['arrow'], alpha=0.8, pad=5))
    
    ax.legend(loc='upper right', fontsize=10, frameon=True, edgecolor=colors['arrow'])
    plt.savefig(os.path.join(output_dir, "step1_init.png"), dpi=300, bbox_inches='tight')
    plt.close()

# Bước 2: Lan truyền thuận
def draw_step2():
    fig, ax = plt.subplots()
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 7)
    ax.set_axis_off()
    fig.patch.set_facecolor(colors['background'])
    ax.set_title("Bước 2: Lan Truyền Thuận", pad=20, color=colors['text'])

    # Vẽ node
    ax.add_patch(plt.Circle((2, 4), 0.5, color=colors['input'], label='Đầu Vào ($X$)', ec='black'))
    ax.add_patch(plt.Circle((5, 5), 0.5, color=colors['hidden'], label='Lớp Ẩn', ec='black'))
    ax.add_patch(plt.Circle((5, 3), 0.5, color=colors['hidden'], ec='black'))
    ax.add_patch(plt.Circle((8, 4), 0.5, color=colors['output'], label='Đầu Ra ($\hat{Y}$)', ec='black'))

    # Mũi tên với công thức
    draw_arrow(ax, (2.5, 4), (4.5, 5), color='green', label=r"$Z^l = W^l \cdot X + b^l$", offset=(0, 0.5))
    draw_arrow(ax, (2.5, 4), (4.5, 3), color='green')
    draw_arrow(ax, (5.5, 5), (7.5, 4), color='green', label=r"$A^l = \sigma(Z^l)$", offset=(0, 0.5))
    draw_arrow(ax, (5.5, 3), (7.5, 4), color='green')

    ax.legend(loc='upper right', fontsize=10, frameon=True, edgecolor=colors['arrow'])
    plt.savefig(os.path.join(output_dir, "step2_feedforward.png"), dpi=300, bbox_inches='tight')
    plt.close()

# Bước 3: Tính hàm mất mát
def draw_step3():
    fig, ax = plt.subplots()
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 7)
    ax.set_axis_off()
    fig.patch.set_facecolor(colors['background'])
    ax.set_title("Bước 3: Tính Hàm Mất Mát", pad=20, color=colors['text'])

    # Vẽ node
    ax.add_patch(plt.Circle((2, 4), 0.5, color=colors['input'], label='Đầu Vào ($X$)', ec='black'))
    ax.add_patch(plt.Circle((5, 4), 0.5, color=colors['output'], label='Đầu Ra ($\hat{Y}$)', ec='black'))
    ax.add_patch(plt.Rectangle((7, 3.75), 1, 0.5, color='#F44336', label='Mất Mát ($L$)', ec='black'))

    # Mũi tên với công thức
    draw_arrow(ax, (2.5, 4), (4.5, 4), color='green')
    draw_arrow(ax, (5.5, 4), (7, 4), color='green', label=r"$L = -\sum y \log(\hat{y})$", offset=(0, 0.5))

    ax.legend(loc='upper right', fontsize=10, frameon=True, edgecolor=colors['arrow'])
    plt.savefig(os.path.join(output_dir, "step3_loss.png"), dpi=300, bbox_inches='tight')
    plt.close()

# Bước 4: Lan truyền ngược
def draw_step4():
    fig, ax = plt.subplots()
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 7)
    ax.set_axis_off()
    fig.patch.set_facecolor(colors['background'])
    ax.set_title("Bước 4: Lan Truyền Ngược", pad=20, color=colors['text'])

    # Vẽ node
    ax.add_patch(plt.Circle((2, 4), 0.5, color=colors['input'], label='Đầu Vào ($X$)', ec='black'))
    ax.add_patch(plt.Circle((5, 5), 0.5, color=colors['hidden'], label='Lớp Ẩn', ec='black'))
    ax.add_patch(plt.Circle((5, 3), 0.5, color=colors['hidden'], ec='black'))
    ax.add_patch(plt.Circle((8, 4), 0.5, color=colors['output'], label='Đầu Ra ($\hat{Y}$)', ec='black'))

    # Mũi tên ngược với công thức
    draw_arrow(ax, (7.5, 4), (5.5, 5), color='red', label=r"$\frac{\partial L}{\partial W^l}$", offset=(0, 0.5))
    draw_arrow(ax, (7.5, 4), (5.5, 3), color='red')
    draw_arrow(ax, (4.5, 5), (2.5, 4), color='red')
    draw_arrow(ax, (4.5, 3), (2.5, 4), color='red')

    # Chú thích
    ax.text(5, 1.5, r"Tính gradient: $\frac{\partial L}{\partial W^l}$, $\frac{\partial L}{\partial b^l}$", 
            ha='center', va='center', color=colors['text'], fontsize=14,
            bbox=dict(facecolor='white', edgecolor=colors['arrow'], alpha=0.8, pad=5))
    
    ax.legend(loc='upper right', fontsize=10, frameon=True, edgecolor=colors['arrow'])
    plt.savefig(os.path.join(output_dir, "step4_backprop.png"), dpi=300, bbox_inches='tight')
    plt.close()

# Bước 5: Cập nhật tham số
def draw_step5():
    fig, ax = plt.subplots()
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 7)
    ax.set_axis_off()
    fig.patch.set_facecolor(colors['background'])
    ax.set_title("Bước 5: Cập Nhật Tham Số", pad=20, color=colors['text'])

    # Vẽ node
    ax.add_patch(plt.Circle((2, 4), 0.5, color=colors['input'], label='Đầu Vào ($X$)', ec='black'))
    ax.add_patch(plt.Circle((5, 4), 0.5, color=colors['hidden'], label='Lớp Ẩn', ec='black'))
    ax.add_patch(plt.Circle((8, 4), 0.5, color=colors['output'], label='Đầu Ra ($\hat{Y}$)', ec='black'))

    # Mũi tên
    draw_arrow(ax, (2.5, 4), (4.5, 4), color='green')
    draw_arrow(ax, (5.5, 4), (7.5, 4), color='green')

    # Công thức cập nhật
    ax.text(5, 2, r"$W^l = W^l - \eta \cdot \frac{\partial L}{\partial W^l}$", 
            ha='center', va='center', color=colors['text'], fontsize=14,
            bbox=dict(facecolor='white', edgecolor=colors['arrow'], alpha=0.8, pad=5))
    ax.text(5, 1.2, r"$b^l = b^l - \eta \cdot \frac{\partial L}{\partial b^l}$", 
            ha='center', va='center', color=colors['text'], fontsize=14,
            bbox=dict(facecolor='white', edgecolor=colors['arrow'], alpha=0.8, pad=5))
    
    ax.legend(loc='upper right', fontsize=10, frameon=True, edgecolor=colors['arrow'])
    plt.savefig(os.path.join(output_dir, "step5_gradient.png"), dpi=300, bbox_inches='tight')
    plt.close()

# Bước 6: Lặp lại
def draw_step6():
    fig = plt.figure(figsize=(12, 7))
    
    # Điều chỉnh kích thước: bên trái lớn hơn (0.6), bên phải nhỏ hơn (0.25)
    ax = fig.add_subplot(121, position=[0.05, 0.1, 0.6, 0.8])  # Left: sơ đồ chính
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 7)
    ax.set_axis_off()
    fig.patch.set_facecolor(colors['background'])
    ax.set_title("Bước 6: Lặp Lại", pad=20, color=colors['text'])

    # Vẽ node
    ax.add_patch(plt.Circle((2, 4), 0.5, color=colors['input'], label='Đầu Vào ($X$)', ec='black'))
    ax.add_patch(plt.Circle((5, 5), 0.5, color=colors['hidden'], label='Lớp Ẩn', ec='black'))
    ax.add_patch(plt.Circle((5, 3), 0.5, color=colors['hidden'], ec='black'))
    ax.add_patch(plt.Circle((8, 4), 0.5, color=colors['output'], label='Đầu Ra ($\hat{Y}$)', ec='black'))

    # Mũi tên lan truyền thuận (màu xanh lá) - chạy song song ở mức trên
    offset_distance = 0.15  # Giảm khoảng cách giữa các mũi tên
    draw_arrow(ax, (2.5, 4 + offset_distance), (4.5, 5 + offset_distance), color='green', label=r"$Z^l = W^l \cdot X + b^l$", offset=(0, 0.5))
    draw_arrow(ax, (2.5, 4 - offset_distance), (4.5, 3 - offset_distance), color='green', offset=(0, -0.5))
    draw_arrow(ax, (5.5, 5 + offset_distance), (7.5, 4 + offset_distance), color='green', offset=(0, 0.5))
    draw_arrow(ax, (5.5, 3 - offset_distance), (7.5, 4 - offset_distance), color='green', offset=(0, -0.5))

    # Mũi tên lan truyền ngược (màu đỏ) - chạy song song ở mức dưới
    draw_arrow(ax, (7.5, 4 - offset_distance), (5.5, 5 - offset_distance), color='red', label=r"$\frac{\partial L}{\partial W^l}$", offset=(0, -0.5))
    draw_arrow(ax, (7.5, 4 + offset_distance), (5.5, 3 + offset_distance), color='red', offset=(0, 0.5))
    draw_arrow(ax, (4.5, 5 - offset_distance), (2.5, 4 - offset_distance), color='red', offset=(0, -0.5))
    draw_arrow(ax, (4.5, 3 + offset_distance), (2.5, 4 + offset_distance), color='red', offset=(0, 0.5))

    # Thêm mũi tên vòng lặp để biểu thị epoch
    ax.annotate('', xy=(9, 3), xytext=(9, 5),
                arrowprops=dict(arrowstyle='->', color='purple', lw=2, connectionstyle="arc3,rad=-0.5"))
    ax.text(9.5, 4, "Epochs", ha='center', va='center', color='purple', fontsize=10,
            bbox=dict(facecolor='white', edgecolor='purple', alpha=0.8, pad=3))

    # Chú thích
    ax.text(5, 1.5, "Lặp lại bước 2-5 qua nhiều epoch\nđể giảm hàm mất mát \(L\)", 
            ha='center', va='center', color=colors['text'], fontsize=14,
            bbox=dict(facecolor='white', edgecolor=colors['arrow'], alpha=0.8, pad=5))
    
    # Thêm chú thích cho mũi tên
    ax.plot([], [], color='green', label='Lan truyền thuận', linewidth=2)
    ax.plot([], [], color='red', label='Lan truyền ngược', linewidth=2)

    ax.legend(loc='upper right', fontsize=10, frameon=True, edgecolor=colors['arrow'])

    # Thêm subplot cho đường cong mất mát - nhỏ hơn
    ax2 = fig.add_subplot(122, position=[0.7, 0.2, 0.25, 0.6])  # Right: đồ thị loss
    epochs = np.arange(1, 11)
    loss = np.exp(-0.3 * epochs)  # Giả lập đường cong mất mát giảm dần
    ax2.plot(epochs, loss, color='red', label='Hàm Mất Mát ($L$)', linewidth=2)
    ax2.set_xlabel('Epoch', fontsize=10)
    ax2.set_ylabel('Mất Mát', fontsize=10)
    ax2.set_title('Giảm Mất Mát Qua Epoch', fontsize=12)
    ax2.grid(True, linestyle='--', alpha=0.7)
    ax2.legend(fontsize=8)

    plt.savefig(os.path.join(output_dir, "step6_repeat.png"), dpi=300, bbox_inches='tight')
    plt.close()

# Tạo tất cả các hình
if __name__ == "__main__":
    draw_step1()
    draw_step2()
    draw_step3()
    draw_step4()
    draw_step5()
    draw_step6()

    print("Đã tạo xong các hình minh họa chuyên nghiệp với công thức trong thư mục 'plnw'!")