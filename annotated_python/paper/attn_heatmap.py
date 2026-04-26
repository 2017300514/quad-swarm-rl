import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

from plots.plot_utils import set_matplotlib_params

# 这个脚本生成论文中的注意力热力图。
# 它不读取训练日志，而是直接把论文分析阶段整理好的注意力权重矩阵硬编码进来，
# 用于对比“正常速度输入”和“速度清零”两种情况下，注意力模块如何在不同邻居之间分配权重。

set_matplotlib_params()

PAGE_WIDTH_INCHES = 8.2
FULL_PAGE_WIDTH = 1.4 * PAGE_WIDTH_INCHES
HALF_PAGE_WIDTH = FULL_PAGE_WIDTH / 2

plt.rcParams['figure.figsize'] = (HALF_PAGE_WIDTH, 2.1)


def main():
    # 每个矩阵的行表示 query agent，列表示它关注的其他 agent。
    # 对角线置 0，表示不讨论 agent 对自身的注意力。
    attention_scores = np.array([
        [0, 0.18558, 0.19735, 0.61707],
        [0.37036, 0, 0.29203, 0.33761],
        [0.37889, 0.30201, 0, 0.31910],
        [0.57469, 0.224756, 0.17775, 0],
    ])
    attention_scores_no_vel = np.array([
        [0, 0.39367, 0.29643, 0.30989],
        [0.36004, 0, 0.32138, 0.31858],
        [0.32372, 0.30816, 0, 0.36811],
        [0.34186, 0.33828, 0.31986, 0]
    ])

    # 两个子图共享同一颜色范围，便于直接比较速度信息对注意力分布的影响。
    fig, (ax1, ax2) = plt.subplots(1, 2)
    sns.heatmap(attention_scores, ax=ax1, linewidth=0.5, cmap="Reds", vmin=0, vmax=0.66)
    ax1.set_title("Attention weights")
    sns.heatmap(attention_scores_no_vel, ax=ax2, linewidths=0.5, cmap="Reds", vmin=0, vmax=0.66)
    ax2.set_title("Attention weights, velocity = 0")

    fig.tight_layout()
    axes = (ax1, ax2)
    plt.setp(
        axes,
        xticks=np.arange(4) + 0.5,
        xticklabels=['red', 'grey', 'green', 'blue'],
        yticks=np.arange(4) + 0.5,
        yticklabels=['red', 'grey', 'green', 'blue'],
    )

    # 输出到当前工作目录，供论文排版或附录直接引用。
    plt.savefig(os.path.join(os.getcwd(), 'attn_study.pdf'), format='pdf', bbox_inches='tight', pad_inches=0.02)


if __name__ == '__main__':
    main()
