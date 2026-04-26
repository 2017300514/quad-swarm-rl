# 中文注释副本；原始文件：paper/fps_compare.py
# 说明：为避免修改源码，本文件仅作为阅读辅助材料。

# 导入当前模块依赖。
import os

# 导入当前模块依赖。
import matplotlib.pyplot as plt
import numpy as np

# Define the data
# 保存或更新 `x` 的值。
x = [1, 8, 32, 128]
# 保存或更新 `y_quad_swarm` 的值。
y_quad_swarm = [48589, 62042, 60241, 38449]
# 保存或更新 `y_pybullet` 的值。
y_pybullet = [21883, 31539, 31457.28, 32522]

# Set the width of each bar and the positions of the x-ticks
# 保存或更新 `bar_width` 的值。
bar_width = 0.35
# 保存或更新 `x_pos` 的值。
x_pos = np.arange(len(x))

# Create a figure and axes object
# 同时更新 `fig`, `ax` 等变量。
fig, ax = plt.subplots()

# Plot the two groups as grouped bars
# 保存或更新 `rects1` 的值。
rects1 = ax.bar(x_pos - bar_width/2, y_pybullet, bar_width, label='gym-pybullet-drones')
# 保存或更新 `rects2` 的值。
rects2 = ax.bar(x_pos + bar_width/2, y_quad_swarm, bar_width, label='QuadSwarm')


# Add labels and legend
# 调用 `set_xlabel` 执行当前处理。
ax.set_xlabel('Number of Quadrotors')
# 调用 `set_ylabel` 执行当前处理。
ax.set_ylabel('Simulation Samples Per Second (SPS)')
# ax.set_title('Comparison of Quad Swarm and PyBullet')
# 调用 `set_xticks` 执行当前处理。
ax.set_xticks(x_pos)
# 调用 `set_xticklabels` 执行当前处理。
ax.set_xticklabels(x)
# 保存或更新 `lgd` 的值。
lgd = ax.legend(bbox_to_anchor=(0.02, 0.95, 0.95, 0.17), loc='upper left', ncol=2, mode="expand",
                 prop={'size': 12})
# 调用 `set_in_layout` 执行当前处理。
lgd.set_in_layout(True)

# Show the plot
# plt.show()
# 保存或更新 `plt.savefig(os.path.join(os.getcwd(), fquads_train_setting.pdf), format` 的值。
plt.savefig(os.path.join(os.getcwd(), f'quads_train_setting.pdf'), format='pdf', bbox_inches='tight', pad_inches=0.01)
