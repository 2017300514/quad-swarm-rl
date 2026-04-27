# Huang 障碍导航论文绘图与评估资产审计

本文件回答三件事：

1. 仓库里现在有哪些论文相关绘图/评估/渲染脚本  
2. 哪些图和指标已经能直接做，哪些还缺统计或缺 baseline  
3. 当前环境还缺哪些包或运行条件  

## 1. 当前已有的绘图与评估脚本

### 1.1 论文统计图脚本

| 文件 | 当前读取内容 | 输出 | 备注 |
| --- | --- | --- | --- |
| `paper/mean_std_plots_quad_obstacle.py` | `0_aux/avg_rewraw_pos` / `0_aux/avg_rewraw_crash` / `0_aux/avg_num_collisions_after_settle` / `0_aux/avg_num_collisions_obst_quad` | PDF | **不是** success/collision/distance 主图，而是 distance/flight performance/collision 两类辅助量 |
| `paper/mean_std_plots_quad_obstacle_ablation.py` | `metric/agent_success_rate` / `metric/agent_col_rate` / `o_random/distance_to_goal_1s` / `o_static_same_goal/distance_to_goal_1s` | PDF | 对应障碍消融 |
| `paper/mean_std_plots_quad_obstacle_num_agents.py` | 同上 | PDF | 对应不同 agent 数量 |
| `paper/mean_std_plots_quad_obstacle_compare_arch_neighbor.py` | 同上 | PDF | 对应不同 sensed neighbors |
| `paper/mean_std_plots_quad_obstacle_compare_arch_density.py` | 同上 | PDF | 对应不同 obstacle density |
| `paper/mean_std_plots_quad_baseline.py` | `0_aux/avg_reward` / `avg_rewraw_pos` / `avg_num_collisions_after_settle` / `avg_rewraw_crash` | PDF | 无障碍 baseline |
| `paper/mean_std_plots_quad_scale.py` | `0_aux/avg_rewraw_pos` / `avg_num_collisions_after_settle` | PDF | 无障碍 scale 图 |
| `paper/mean_std_plots_quad_annealing.py` | 训练过程指标 | PDF | annealing 对比 |
| `paper/fps_compare.py` | 硬编码 SPS 数组 | PDF | 不是从训练/测试日志自动算出来的 |
| `paper/attn_heatmap.py` | 硬编码 attention score 矩阵 | PDF | 静态示意图 |

### 1.2 评估与渲染入口

| 文件 | 作用 | 备注 |
| --- | --- | --- |
| `swarm_rl/enjoy.py` | 评估主入口 | 复用训练时的 parser 和 env/model 注册 |
| `swarm_rl/env_wrappers/v_value_map.py` | 把 critic value map 拼到 `rgb_array` 渲染帧右侧 | 依赖可加载 checkpoint |
| `gym_art/quadrotor_multi/quadrotor_multi.py:726-780` | 环境 render 主循环 | 生成多视角帧 |
| `gym_art/quadrotor_multi/quadrotor_multi_visualization.py` | 多机三维场景渲染 | 轨迹、障碍、相机视角都在这里 |
| `gym_art/quadrotor_multi/tests/plot_v_value_2d.py` | V-value heatmap 画图辅助 | 被 `V_ValueMapWrapper` 调用 |

## 2. 论文图和指标的支持情况

### 2.1 直接支持

| 类别 | 状态 | 证据 |
| --- | --- | --- |
| success rate | 直接支持 | `quadrotor_multi.py:703-718` 记录 `metric/agent_success_rate`，多个 obstacle plot 脚本直接消费 |
| collision rate | 直接支持 | 同上，`metric/agent_col_rate` |
| distance to goal (1s) | 直接支持 | `quadrotor_multi.py:649-660` 记录，obstacle ablation/neighbor/density/num_agents 脚本直接消费 |
| 多 seed 聚合 | 直接支持 | 所有 `paper/*.py` 基本都用 `EventAccumulator` 读 `tfevents` |
| 轨迹视频导出 | 直接支持 | `swarm_rl/enjoy.py` + Sample Factory `--save_video` |
| V-value 图 | 间接支持 | `V_ValueMapWrapper` + `plot_v_value_2d.py` |

### 2.2 只部分支持

| 类别 | 当前情况 | 缺口 |
| --- | --- | --- |
| obstacle 主实验曲线 | 有 `mean_std_plots_quad_obstacle.py` | 但它读的是 `0_aux/*`，并不是论文主结果常用的 success/collision/distance 那套 |
| inference time | 只有 `fps_compare.py` | 用的是硬编码数组，不是自动评估产物 |
| attention heatmap | 有 `attn_heatmap.py` | 也是硬编码，不是从训练模型提取 |
| V-value 图 | 代码链路完整 | 但必须先有 checkpoint，且渲染依赖显示/OpenGL |

### 2.3 当前不支持或缺失

| 类别 | 现状 |
| --- | --- |
| flight distance / path length | **没有发现**统一写入 TensorBoard 的字段 |
| GLAS baseline 复现代码 | **没有发现** |
| SBC baseline 复现代码 | **没有发现** |
| 从训练/评估自动统计 inference latency | **没有发现** |
| 现成 GIF 导出脚本 | **没有发现** |

## 3. 一个关键事实：主 obstacle plot 脚本并不等于论文主图

这是本次排查里最重要的新发现。

`paper/mean_std_plots_quad_obstacle.py:34-39` 读取的是：

- `0_aux/avg_rewraw_pos`
- `0_aux/avg_rewraw_crash`
- `0_aux/avg_num_collisions_after_settle`
- `0_aux/avg_num_collisions_obst_quad`

因此它更像：

1. 平均到目标距离  
2. 飞行保持率  
3. 机间碰撞频率  
4. 机-障碍碰撞频率  

而真正直接使用论文主指标 `success / collision / distance-to-goal` 的，是：

- `mean_std_plots_quad_obstacle_ablation.py`
- `mean_std_plots_quad_obstacle_num_agents.py`
- `mean_std_plots_quad_obstacle_compare_arch_neighbor.py`
- `mean_std_plots_quad_obstacle_compare_arch_density.py`

也就是说，**仓库里和“论文障碍主结果”最贴近的是这一批 compare/ablation 脚本，不是文件名最直观的 `mean_std_plots_quad_obstacle.py`。**

## 4. 当前日志里已经有、但脚本没有统一使用的指标

`gym_art/quadrotor_multi/quadrotor_multi.py:637-718` 在 episode 结束时会写入：

- `metric/agent_success_rate`
- `metric/agent_deadlock_rate`
- `metric/agent_col_rate`
- `metric/agent_neighbor_col_rate`
- `metric/agent_obst_col_rate`
- `{scenario}/distance_to_goal_1s`
- `{scenario}/distance_to_goal_3s`
- `{scenario}/distance_to_goal_5s`
- `num_collisions`
- `num_collisions_obst_quad`

这意味着当前日志已经足够支持：

1. success rate
2. total collision rate
3. obstacle collision rate
4. scenario-specific distance to goal

但没有足够支撑：

1. flight distance
2. inference latency

## 5. 缺失的统计项

### 5.1 flight distance

论文里有 `flight distance`，但仓库当前没有看到稳定写入 TensorBoard 的：

- `flight_distance`
- `path_length`
- `distance_traveled`

这意味着当前绘图层面无法直接从本地日志还原论文表格里的 `flying distance(m)`。

### 5.2 inference time

`paper/fps_compare.py` 只是：

- `y_quad_swarm = [48589, 62042, 60241, 38449]`
- `y_pybullet = [21883, 31539, 31457.28, 32522]`

它画的是静态比较图，不是从 enjoy/test 流程现场测出来的 per-checkpoint latency。  
如果想真正复现论文里的 inference time，需要在评估时对：

1. observation collection
2. policy forward
3. action emit

做真实计时。

### 5.3 plot 脚本里的时长常量和训练配置不完全一致

多个 plot 脚本把：

```python
EPISODE_DURATION = 16
```

写死在脚本里，但训练主配置 `quad_obstacle_baseline.py` 和环境参数默认值用的是：

```text
--quads_episode_duration=15.0
```

这会直接影响脚本里按“每分钟碰撞数”或“飞行保持率”做缩放时的系数，尤其是：

- `paper/mean_std_plots_quad_obstacle.py`
- `paper/mean_std_plots_quad_baseline.py`
- `paper/mean_std_plots_quad_scale.py`

因此当前仓库虽然能出图，但如果要追求和训练配置严格一致，**这些脚本里的 `EPISODE_DURATION=16` 应视为一个待校正偏差**。

## 6. baseline 方法复现情况

### 6.1 仓库里没有找到的 baseline

没有发现以下论文 baseline 的实现代码：

- GLAS
- SBC

本次代码搜索没有在仓库内找到对应实现、控制器或可调用入口。

### 6.2 仓库里实际存在的“baseline”

仓库里存在的是**同一套 RL 主线下的配置对比**，例如：

- obstacle / non-obstacle
- attention / no encoder / mean_embed / mlp
- replay / no replay
- different neighbor counts
- different densities / sizes

所以如果目标是完整复现论文中“与外部 baseline 方法对比”的图表，当前仓库还缺：

1. 外部 baseline 实现
2. 和当前环境对齐的评测脚本
3. 统一统计输出

## 7. 各脚本期望的数据目录布局

### 7.1 主 obstacle 聚合

`mean_std_plots_quad_obstacle.py:250-254` 会递归扫描：

```text
<path>/**/events.out.tfevents.*
```

因此最简单布局是：

```text
train_dir/paper_huang_obstacle/final/
  seed_0000/
    events.out.tfevents.*
  seed_1111/
    events.out.tfevents.*
  seed_2222/
    events.out.tfevents.*
  seed_3333/
    events.out.tfevents.*
```

### 7.2 消融

`mean_std_plots_quad_obstacle_ablation.py:429-433` 硬编码要求：

```text
1_default_posxy/
2_change_obs_octomap/
3_add_multi_head/
4_add_replay_buffer/
```

每个子目录下再放各 seed 的实验目录。

### 7.3 不同 agent 数

`mean_std_plots_quad_obstacle_num_agents.py:430-433` 硬编码要求：

```text
8_2/
16_2/
32_2/
```

### 7.4 不同邻居数

`mean_std_plots_quad_obstacle_compare_arch_neighbor.py:431-435` 硬编码要求：

```text
1/
2/
6/
16/
31/
```

### 7.5 不同 density

`mean_std_plots_quad_obstacle_compare_arch_density.py:426-433` 直接按目录名排序读取，脚本内部默认图例是：

```text
20%
40%
60%
80%
```

## 8. 当前环境包与运行约束

### 8.1 绘图主链路已经够用

已经确认可支持：

- matplotlib / seaborn / plotly
- tensorboard event 读取
- pyglet/OpenGL 渲染
- Sample Factory 训练与 enjoy

### 8.2 建议补的包

如果目标包含 GIF 导出，建议补：

- `imageio`
- `imageio-ffmpeg`

原因：

1. 论文静态统计图不需要它们  
2. 但帧序列转 GIF/MP4 更方便  

### 8.3 运行时限制

| 限制 | 影响 |
| --- | --- |
| X11 / DISPLAY | `human` 渲染和部分 `rgb_array` 图形路径可能受影响 |
| OpenGL framebuffer | 无头环境下可能需要虚拟显示或 Mesa 支持 |
| 无 checkpoint | V-value 图、视频评估都无法开始 |

## 9. 结论

当前仓库的真实状态是：

1. **训练日志聚合链路是存在的**
2. **论文大部分仿真图都有脚本雏形**
3. **主缺口不是画图脚本，而是本地训练产物**
4. **额外缺口是 flight distance、真实 inference latency、GLAS/SBC baseline 复现**

所以如果要完整复现论文图片，最合理顺序是：

1. 先跑四个 seed 的主 obstacle 训练
2. 再补齐不同 agent 数 / 邻居数 / density 的子实验
3. 用现有 `paper/*.py` 先出能直接出的图
4. 再单独补 flight distance 和 inference time 的统计
5. 最后决定是否补外部 baseline 方法
