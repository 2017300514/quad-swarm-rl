import os

import matplotlib.pyplot as plt
import numpy as np

# 这个脚本画的是仿真吞吐量对比柱状图。
# 它直接使用预先整理好的 benchmark 数字，对比 `QuadSwarm` 和 `gym-pybullet-drones`
# 在不同无人机数量下每秒可生成多少仿真样本，用来支撑论文里的效率结论。

# 横轴是无人机数量，纵轴是对应实验设置下的 Simulation Samples Per Second。
x = [1, 8, 32, 128]
y_quad_swarm = [48589, 62042, 60241, 38449]
y_pybullet = [21883, 31539, 31457.28, 32522]

bar_width = 0.35
x_pos = np.arange(len(x))

fig, ax = plt.subplots()

# 采用并排柱状图，让两套仿真器在同一 agent 数下的吞吐量可以直接肉眼比较。
ax.bar(x_pos - bar_width / 2, y_pybullet, bar_width, label='gym-pybullet-drones')
ax.bar(x_pos + bar_width / 2, y_quad_swarm, bar_width, label='QuadSwarm')

ax.set_xlabel('Number of Quadrotors')
ax.set_ylabel('Simulation Samples Per Second (SPS)')
ax.set_xticks(x_pos)
ax.set_xticklabels(x)
lgd = ax.legend(
    bbox_to_anchor=(0.02, 0.95, 0.95, 0.17),
    loc='upper left',
    ncol=2,
    mode="expand",
    prop={'size': 12},
)
lgd.set_in_layout(True)

# 输出文件名虽然叫 `quads_train_setting.pdf`，实际内容是性能对比图。
plt.savefig(os.path.join(os.getcwd(), 'quads_train_setting.pdf'), format='pdf', bbox_inches='tight', pad_inches=0.01)
