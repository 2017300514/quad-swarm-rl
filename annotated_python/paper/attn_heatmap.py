# 中文注释副本；原始文件：paper/attn_heatmap.py
# 说明：为避免修改源码，本文件仅作为阅读辅助材料。

# 导入当前模块依赖。
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

# 导入当前模块依赖。
from plots.plot_utils import set_matplotlib_params

# 调用 `set_matplotlib_params` 执行当前处理。
set_matplotlib_params()

# 保存或更新 `PAGE_WIDTH_INCHES` 的值。
PAGE_WIDTH_INCHES = 8.2
# 保存或更新 `FULL_PAGE_WIDTH` 的值。
FULL_PAGE_WIDTH = 1.4 * PAGE_WIDTH_INCHES
# 保存或更新 `HALF_PAGE_WIDTH` 的值。
HALF_PAGE_WIDTH = FULL_PAGE_WIDTH / 2

# 保存或更新 `plt.rcParams[figure.figsize]` 的值。
plt.rcParams['figure.figsize'] = (HALF_PAGE_WIDTH, 2.1)  # (2.5, 2.0) 7.5， 4

# 定义函数 `main`。
def main():
    # 保存或更新 `attention_scores` 的值。
    attention_scores = np.array([
        [0, 0.18558, 0.19735, 0.61707],
        [0.37036, 0, 0.29203, 0.33761],
        [0.37889, 0.30201, 0, 0.31910],
        [0.57469, 0.224756, 0.17775, 0],
    ])
    # 保存或更新 `attention_scores_no_vel` 的值。
    attention_scores_no_vel = np.array([
        [0, 0.39367, 0.29643, 0.30989],
        [0.36004, 0, 0.32138, 0.31858],
        [0.32372, 0.30816, 0, 0.36811],
        [0.34186, 0.33828, 0.31986, 0]
    ])
    # 保存或更新 `cmap` 的值。
    cmap = sns.cm.rocket_r
    # 同时更新 `fig`, `(ax1, ax2)` 等变量。
    fig, (ax1, ax2) = plt.subplots(1, 2)
    # 保存或更新 `sns.heatmap(attention_scores, ax` 的值。
    sns.heatmap(attention_scores, ax=ax1, linewidth=0.5, cmap="Reds", vmin=0, vmax=0.66)
    # 调用 `set_title` 执行当前处理。
    ax1.set_title("Attention weights")
    # 保存或更新 `sns.heatmap(attention_scores_no_vel, ax` 的值。
    sns.heatmap(attention_scores_no_vel, ax=ax2, linewidths=0.5, cmap="Reds",vmin=0, vmax=0.66)
    # 保存或更新 `ax2.set_title(Attention weights, velocity` 的值。
    ax2.set_title("Attention weights, velocity = 0")
    # 调用 `tight_layout` 执行当前处理。
    fig.tight_layout()
    # 保存或更新 `axes` 的值。
    axes = (ax1, ax2)
    # 保存或更新 `plt.setp(axes, xticks` 的值。
    plt.setp(axes, xticks=np.arange(4) + 0.5, xticklabels=['red', 'grey', 'green', 'blue'], yticks=np.arange(4) + 0.5,
             yticklabels=['red', 'grey', 'green', 'blue'])
    # plt.show()
    # 保存或更新 `plt.savefig(os.path.join(os.getcwd(), fattn_study.pdf), format` 的值。
    plt.savefig(os.path.join(os.getcwd(), f'attn_study.pdf'), format='pdf', bbox_inches='tight', pad_inches=0.02)
# 根据条件决定是否进入当前分支。
if __name__ == '__main__':
    # 调用 `main` 执行当前处理。
    main()
