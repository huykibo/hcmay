import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from matplotlib.patches import FancyArrowPatch, Rectangle, Circle
import os
from matplotlib.patheffects import withStroke

# Thiết lập giao diện chuyên nghiệp
sns.set_style("white")
plt.rcParams.update({
    "font.size": 14,
    "font.family": "Arial",
    "axes.titlesize": 18,
    "axes.titleweight": "bold",
    "axes.labelsize": 14,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "figure.figsize": (10, 6),
    "axes.facecolor": "#ffffff",
    "figure.facecolor": "#ffffff",
    "axes.edgecolor": "#333333",
    "text.color": "#333333"
})

# Tạo thư mục để lưu ảnh nếu chưa tồn tại
output_dir = "mhpersoudo"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Màu sắc chính cho các biểu đồ
colors = {
    "data": "#1f77b4",  # Xanh dương
    "labeled": "#2ca02c",  # Xanh lá
    "unlabeled": "#ff7f0e",  # Cam
    "pseudo": "#d62728",  # Đỏ
    "model": "#9467bd",  # Tím
    "arrow": "#555555",  # Xám
    "text": "#333333"  # Đen xám
}

# Hiệu ứng bóng mờ
shadow_effect = withStroke(linewidth=3, foreground="#cccccc")

# Hàm vẽ mũi tên với tỷ lệ chính xác
def draw_arrow(ax, start, end, color=colors["arrow"], lw=2.5):
    arrow = FancyArrowPatch(start, end, mutation_scale=25, color=color, lw=lw, arrowstyle="->")
    ax.add_patch(arrow)

# Bước 1: Chuẩn bị dữ liệu và chia tập train/test
def create_step1_illustration():
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)
    ax.axis("off")

    # Vẽ khối dữ liệu đầy đủ
    rect = Rectangle((2, 2), 2.5, 1.5, color=colors["data"], edgecolor="black", lw=1.5)
    rect.set_path_effects([shadow_effect])
    ax.add_patch(rect)
    ax.text(3.25, 2.75, "Dữ liệu đầy đủ\n(70,000 mẫu)", ha="center", va="center", color=colors["text"], fontsize=14)

    # Mũi tên chia
    draw_arrow(ax, (4.5, 2.75), (5.5, 2.75))

    # Vẽ tập Train
    rect = Rectangle((5.5, 2), 2, 1.5, color=colors["labeled"], edgecolor="black", lw=1.5)
    rect.set_path_effects([shadow_effect])
    ax.add_patch(rect)
    ax.text(6.5, 2.75, "Tập Train\n(80%)", ha="center", va="center", color=colors["text"], fontsize=14)

    # Vẽ tập Test
    rect = Rectangle((7.5, 2), 2, 1.5, color=colors["data"], alpha=0.6, edgecolor="black", lw=1.5)
    rect.set_path_effects([shadow_effect])
    ax.add_patch(rect)
    ax.text(8.5, 2.75, "Tập Test\n(20%)", ha="center", va="center", color=colors["text"], fontsize=14)

    # Tiêu đề
    ax.set_title("Bước 1: Chuẩn bị dữ liệu và chia tập Train/Test", pad=20, color=colors["text"])
    plt.savefig(os.path.join(output_dir, "pseudo_step1.png"), bbox_inches="tight", dpi=300)
    plt.close()

# Bước 2: Lấy 1% dữ liệu mỗi lớp làm tập ban đầu (điều chỉnh để không lệch)
def create_step2_illustration():
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)
    ax.axis("off")

    # Vẽ tập Train (80%)
    rect = Rectangle((1.5, 2), 3, 1.5, color=colors["labeled"], edgecolor="black", lw=1.5)
    rect.set_path_effects([shadow_effect])
    ax.add_patch(rect)
    ax.text(3, 2.75, "Tập Train\n(80%)", ha="center", va="center", color=colors["text"], fontsize=14)

    # Mũi tên chia
    draw_arrow(ax, (4.5, 2.75), (5.5, 2.75))

    # Vẽ tập Labeled (1%) - căn chỉnh ngang hàng
    rect = Rectangle((5.5, 2), 1.5, 1.5, color=colors["labeled"], edgecolor="black", lw=1.5)
    rect.set_path_effects([shadow_effect])
    ax.add_patch(rect)
    ax.text(6.25, 2.75, "Tập Labeled\n(1%)", ha="center", va="center", color=colors["text"], fontsize=14)

    # Vẽ tập Unlabeled (99%) - căn chỉnh ngang hàng với "Tập Train"
    rect = Rectangle((7, 2), 2.5, 1.5, color=colors["unlabeled"], edgecolor="black", lw=1.5)
    rect.set_path_effects([shadow_effect])
    ax.add_patch(rect)
    ax.text(8.25, 2.75, "Tập Unlabeled\n(99%)", ha="center", va="center", color=colors["text"], fontsize=14)

    # Tiêu đề
    ax.set_title("Bước 2: Lấy 1% dữ liệu mỗi lớp làm tập ban đầu", pad=20, color=colors["text"])
    plt.savefig(os.path.join(output_dir, "pseudo_step2.png"), bbox_inches="tight", dpi=300)
    plt.close()

# Bước 3: Huấn luyện mô hình Neural Network trên tập 1%
def create_step3_illustration():
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)
    ax.axis("off")

    # Vẽ tập Labeled (1%)
    rect = Rectangle((2, 2.5), 2, 1, color=colors["labeled"], edgecolor="black", lw=1.5)
    rect.set_path_effects([shadow_effect])
    ax.add_patch(rect)
    ax.text(3, 3, "Tập Labeled\n(1%)", ha="center", va="center", color=colors["text"], fontsize=14)

    # Mũi tên
    draw_arrow(ax, (4, 3), (5, 3))

    # Vẽ mô hình Neural Network
    circle = Circle((6, 3), 0.8, color=colors["model"], edgecolor="black", lw=1.5)
    circle.set_path_effects([shadow_effect])
    ax.add_patch(circle)
    ax.text(6, 3, "Neural\nNetwork", ha="center", va="center", color=colors["text"], fontsize=14)

    # Mũi tên
    draw_arrow(ax, (6.8, 3), (7.8, 3))

    # Vẽ mô hình đã huấn luyện
    circle = Circle((8.5, 3), 0.8, color=colors["model"], alpha=0.6, edgecolor="black", lw=1.5)
    circle.set_path_effects([shadow_effect])
    ax.add_patch(circle)
    ax.text(8.5, 3, "Mô hình\nĐã huấn luyện", ha="center", va="center", color=colors["text"], fontsize=14)

    # Tiêu đề
    ax.set_title("Bước 3: Huấn luyện Neural Network trên tập 1%", pad=20, color=colors["text"])
    plt.savefig(os.path.join(output_dir, "pseudo_step3.png"), bbox_inches="tight", dpi=300)
    plt.close()

# Bước 4: Dự đoán nhãn cho dữ liệu không có nhãn (99%)
def create_step4_illustration():
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)
    ax.axis("off")

    # Vẽ tập Unlabeled (99%)
    rect = Rectangle((2, 2.5), 2, 1, color=colors["unlabeled"], edgecolor="black", lw=1.5)
    rect.set_path_effects([shadow_effect])
    ax.add_patch(rect)
    ax.text(3, 3, "Tập Unlabeled\n(99%)", ha="center", va="center", color=colors["text"], fontsize=14)

    # Mũi tên
    draw_arrow(ax, (4, 3), (5, 3))

    # Vẽ mô hình đã huấn luyện
    circle = Circle((6, 3), 0.8, color=colors["model"], edgecolor="black", lw=1.5)
    circle.set_path_effects([shadow_effect])
    ax.add_patch(circle)
    ax.text(6, 3, "Mô hình\nĐã huấn luyện", ha="center", va="center", color=colors["text"], fontsize=14)

    # Mũi tên
    draw_arrow(ax, (6.8, 3), (7.8, 3))

    # Vẽ dự đoán
    rect = Rectangle((8, 2.5), 2, 1, color=colors["pseudo"], edgecolor="black", lw=1.5)
    rect.set_path_effects([shadow_effect])
    ax.add_patch(rect)
    ax.text(9, 3, "Dự đoán\nNhãn", ha="center", va="center", color=colors["text"], fontsize=14)

    # Tiêu đề
    ax.set_title("Bước 4: Dự đoán nhãn cho dữ liệu không có nhãn", pad=20, color=colors["text"])
    plt.savefig(os.path.join(output_dir, "pseudo_step4.png"), bbox_inches="tight", dpi=300)
    plt.close()

# Bước 5: Gán nhãn giả với ngưỡng tin cậy (threshold = 0.95)
def create_step5_illustration():
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)
    ax.axis("off")

    # Vẽ dự đoán
    rect = Rectangle((2, 2.5), 2, 1, color=colors["pseudo"], edgecolor="black", lw=1.5)
    rect.set_path_effects([shadow_effect])
    ax.add_patch(rect)
    ax.text(3, 3, "Dự đoán\nNhãn", ha="center", va="center", color=colors["text"], fontsize=14)

    # Mũi tên
    draw_arrow(ax, (4, 3), (5, 3))

    # Vẽ ngưỡng tin cậy
    rect = Rectangle((5, 2.5), 2, 1, color=colors["pseudo"], alpha=0.6, edgecolor="black", lw=1.5)
    rect.set_path_effects([shadow_effect])
    ax.add_patch(rect)
    ax.text(6, 3, "Ngưỡng\nTin cậy ≥ 0.95", ha="center", va="center", color=colors["text"], fontsize=14)

    # Mũi tên
    draw_arrow(ax, (7, 3), (8, 3))

    # Vẽ nhãn giả
    rect = Rectangle((8, 2.5), 2, 1, color=colors["labeled"], edgecolor="black", lw=1.5)
    rect.set_path_effects([shadow_effect])
    ax.add_patch(rect)
    ax.text(9, 3, "Nhãn giả", ha="center", va="center", color=colors["text"], fontsize=14)

    # Tiêu đề
    ax.set_title("Bước 5: Gán nhãn giả với ngưỡng tin cậy 0.95", pad=20, color=colors["text"])
    plt.savefig(os.path.join(output_dir, "pseudo_step5.png"), bbox_inches="tight", dpi=300)
    plt.close()

# Bước 6: Huấn luyện lại mô hình với tập dữ liệu mới
def create_step6_illustration():
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)
    ax.axis("off")

    # Vẽ tập Labeled (1%)
    rect = Rectangle((2, 3.5), 2, 1, color=colors["labeled"], edgecolor="black", lw=1.5)
    rect.set_path_effects([shadow_effect])
    ax.add_patch(rect)
    ax.text(3, 4, "Tập Labeled\n(1%)", ha="center", va="center", color=colors["text"], fontsize=14)

    # Vẽ nhãn giả
    rect = Rectangle((2, 2), 2, 1, color=colors["pseudo"], edgecolor="black", lw=1.5)
    rect.set_path_effects([shadow_effect])
    ax.add_patch(rect)
    ax.text(3, 2.5, "Nhãn giả", ha="center", va="center", color=colors["text"], fontsize=14)

    # Mũi tên kết hợp
    draw_arrow(ax, (4, 4), (5, 3))
    draw_arrow(ax, (4, 2.5), (5, 3))

    # Vẽ tập dữ liệu mới
    rect = Rectangle((5, 2.5), 2, 1, color=colors["labeled"], alpha=0.8, edgecolor="black", lw=1.5)
    rect.set_path_effects([shadow_effect])
    ax.add_patch(rect)
    ax.text(6, 3, "Tập dữ liệu mới", ha="center", va="center", color=colors["text"], fontsize=14)

    # Mũi tên
    draw_arrow(ax, (7, 3), (8, 3))

    # Vẽ mô hình
    circle = Circle((9, 3), 0.8, color=colors["model"], edgecolor="black", lw=1.5)
    circle.set_path_effects([shadow_effect])
    ax.add_patch(circle)
    ax.text(9, 3, "Neural\nNetwork", ha="center", va="center", color=colors["text"], fontsize=14)

    # Tiêu đề
    ax.set_title("Bước 6: Huấn luyện lại với tập dữ liệu mới", pad=20, color=colors["text"])
    plt.savefig(os.path.join(output_dir, "pseudo_step6.png"), bbox_inches="tight", dpi=300)
    plt.close()

# Bước 7: Lặp lại các bước 4-6
def create_step7_illustration():
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)
    ax.axis("off")

    # Vẽ tập Unlabeled
    rect = Rectangle((2, 2.5), 2, 1, color=colors["unlabeled"], edgecolor="black", lw=1.5)
    rect.set_path_effects([shadow_effect])
    ax.add_patch(rect)
    ax.text(3, 3, "Tập Unlabeled", ha="center", va="center", color=colors["text"], fontsize=14)

    # Mũi tên
    draw_arrow(ax, (4, 3), (5, 3))

    # Vẽ bước dự đoán
    circle = Circle((6, 3), 0.8, color=colors["model"], edgecolor="black", lw=1.5)
    circle.set_path_effects([shadow_effect])
    ax.add_patch(circle)
    ax.text(6, 3, "Dự đoán", ha="center", va="center", color=colors["text"], fontsize=14)

    # Mũi tên
    draw_arrow(ax, (6.8, 3), (7.8, 3))

    # Vẽ bước gán nhãn giả
    rect = Rectangle((8, 2.5), 2, 1, color=colors["pseudo"], edgecolor="black", lw=1.5)
    rect.set_path_effects([shadow_effect])
    ax.add_patch(rect)
    ax.text(9, 3, "Gán nhãn giả", ha="center", va="center", color=colors["text"], fontsize=14)

    # Vẽ đường lặp lại
    draw_arrow(ax, (9, 2.5), (9, 1.5))
    draw_arrow(ax, (9, 1.5), (6, 1.5))
    draw_arrow(ax, (6, 1.5), (6, 2.2))
    ax.text(7.5, 1.2, "Lặp lại", ha="center", va="center", color=colors["arrow"], fontsize=14)

    # Tiêu đề
    ax.set_title("Bước 7: Lặp lại các bước 4-6", pad=20, color=colors["text"])
    plt.savefig(os.path.join(output_dir, "pseudo_step7.png"), bbox_inches="tight", dpi=300)
    plt.close()

# Bước 8: Huấn luyện lần cuối và đánh giá
def create_step8_illustration():
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)
    ax.axis("off")

    # Vẽ tập dữ liệu đã gắn nhãn
    rect = Rectangle((2, 2.5), 2, 1, color=colors["labeled"], edgecolor="black", lw=1.5)
    rect.set_path_effects([shadow_effect])
    ax.add_patch(rect)
    ax.text(3, 3, "Tập dữ liệu\nĐã gắn nhãn", ha="center", va="center", color=colors["text"], fontsize=14)

    # Mũi tên
    draw_arrow(ax, (4, 3), (5, 3))

    # Vẽ mô hình
    circle = Circle((6, 3), 0.8, color=colors["model"], edgecolor="black", lw=1.5)
    circle.set_path_effects([shadow_effect])
    ax.add_patch(circle)
    ax.text(6, 3, "Neural\nNetwork", ha="center", va="center", color=colors["text"], fontsize=14)

    # Mũi tên
    draw_arrow(ax, (6.8, 3), (7.8, 3))

    # Vẽ đánh giá
    rect = Rectangle((8, 2.5), 2, 1, color=colors["data"], alpha=0.6, edgecolor="black", lw=1.5)
    rect.set_path_effects([shadow_effect])
    ax.add_patch(rect)
    ax.text(9, 3, "Đánh giá\nTập Test", ha="center", va="center", color=colors["text"], fontsize=14)

    # Tiêu đề
    ax.set_title("Bước 8: Huấn luyện lần cuối và đánh giá", pad=20, color=colors["text"])
    plt.savefig(os.path.join(output_dir, "pseudo_step8.png"), bbox_inches="tight", dpi=300)
    plt.close()

# Tạo tất cả các ảnh minh họa
create_step1_illustration()
create_step2_illustration()
create_step3_illustration()
create_step4_illustration()
create_step5_illustration()
create_step6_illustration()
create_step7_illustration()
create_step8_illustration()

print(f"Đã tạo tất cả ảnh minh họa và lưu tại thư mục: {output_dir}")