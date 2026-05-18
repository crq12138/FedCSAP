import numpy as np
import matplotlib.pyplot as plt

# =========================
# 1. 数据定义
# =========================
attack_types = ['TLF攻击', 'SF攻击', 'IPM攻击', 'DBA攻击']
datasets = ['CIFAR-10', 'PathMNIST', 'MNIST']

# 恶意攻击者数量 = 8
data_8 = np.array([
    [5, 5, 2, 5],   # CIFAR-10
    [5, 2, 7, 0],   # PathMNIST
    [6, 6, 7, 0]    # MNIST
])
baseline_8 = 33  # 无委员会选举机制

# 恶意攻击者数量 = 5
data_5 = np.array([
    [1, 5, 1, 0],   # CIFAR-10
    [0, 2, 0, 0],   # PathMNIST
    [0, 2, 0, 1]    # MNIST
])
baseline_5 = 7  # 无委员会选举机制


# =========================
# 2. 全局字体设置（适配中文）
# =========================
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False


# =========================
# 3. 绘图函数
# =========================
def plot_heatmap(data, row_labels, col_labels, malicious_num, baseline, save_prefix):
    fig, ax = plt.subplots(figsize=(8, 4.8))

    # 固定颜色范围，便于两张图直接比较
    im = ax.imshow(data, aspect='auto', vmin=0, vmax=7)

    # 坐标轴标签
    ax.set_xticks(np.arange(len(col_labels)))
    ax.set_yticks(np.arange(len(row_labels)))
    ax.set_xticklabels(col_labels, fontsize=12)
    ax.set_yticklabels(row_labels, fontsize=12)

    # 标题
    ax.set_title(
        f'委员会被占领次数热力图（恶意攻击者数量 = {malicious_num}）\n'
        f'无委员会选举机制下占领次数：{baseline}',
        fontsize=14,
        pad=14
    )

    # 网格线
    ax.set_xticks(np.arange(-0.5, len(col_labels), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(row_labels), 1), minor=True)
    ax.grid(which='minor', color='white', linestyle='-', linewidth=1.5)
    ax.tick_params(which='minor', bottom=False, left=False)

    # 数值标注
    max_val = np.max(data)
    threshold = max_val / 2.0

    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            value = data[i, j]
            text_color = 'white' if value > threshold else 'black'
            ax.text(
                j, i, f'{value}',
                ha='center', va='center',
                color=text_color,
                fontsize=12,
                fontweight='bold'
            )

    # 颜色条
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('委员会被占领次数', fontsize=12)

    # 坐标轴名称
    ax.set_xlabel('攻击方式', fontsize=12)
    ax.set_ylabel('数据集', fontsize=12)

    plt.tight_layout()

    # 同时保存 EPS 和 PNG
    plt.savefig(f'{save_prefix}.eps', format='eps', dpi=600, bbox_inches='tight')
    plt.savefig(f'{save_prefix}.png', format='png', dpi=600, bbox_inches='tight')

    plt.show()
    plt.close(fig)


# =========================
# 4. 绘制并保存两张热力图
# =========================
plot_heatmap(
    data=data_8,
    row_labels=datasets,
    col_labels=attack_types,
    malicious_num=8,
    baseline=baseline_8,
    save_prefix='committee_takeover_8'
)

plot_heatmap(
    data=data_5,
    row_labels=datasets,
    col_labels=attack_types,
    malicious_num=5,
    baseline=baseline_5,
    save_prefix='committee_takeover_5'
)