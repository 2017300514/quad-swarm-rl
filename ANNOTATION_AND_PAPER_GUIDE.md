# 项目中文注释进度与论文复现源码说明

## 1. 生成结果总览

- 生成时间：`2026-04-26 15:26:32`
- 论文说明文件：`hybrid_auto/Huang 等 - Collision Avoidance and Navigation for a Quadrotor.md`
- Python 文件总数：`115`
- 已完成中文注释副本数量：`115/115`
- 注释输出目录：`annotated_python/`
- 说明：所有中文注释都写入独立副本，原始源码未被修改。

## 2. 注释进度统计

### 2.1 按目录统计

| 目录 | Python 文件数 |
| --- | ---: |
| `gym_art` | 70 |
| `swarm_rl` | 33 |
| `paper` | 11 |
| `repo_root` | 1 |

### 2.2 处理策略

- 对结构清晰且信息量较大的语句，采用逐行中文注释。
- 对 `import`、多行参数列表、简单返回语句等低信息密度片段，采用块级或语义合并注释，避免把文件注释得过碎。
- 注释副本保留原始代码顺序，便于和源码逐行对照。

## 3. 论文《Collision Avoidance and Navigation for a Quadrotor Swarm Using End-to-end Deep Reinforcement Learning》对应源码说明

### 3.1 训练入口与实验配置

- `swarm_rl/train.py`：训练主入口。负责注册 `quadrotor_multi` 环境、注册自定义模型，并调用 Sample Factory 的 `run_rl` 开始 APPO/IPPO 训练流程。
- `swarm_rl/env_wrappers/quadrotor_params.py`：集中定义论文实验会用到的命令行参数，包括无人机数量、邻居观测、障碍物观测、回放概率、碰撞奖励、房间大小等。
- `swarm_rl/runs/obstacles/quad_obstacle_baseline.py`：障碍环境基线配置，给出论文主实验的默认超参数，包括 `replay_buffer_sample_prob=0.75`、`quads_use_obstacles=True`、`quads_obstacle_obs_type=octomap`、`quads_use_downwash=True` 等。
- `swarm_rl/runs/obstacles/quads_multi_obstacles.py`：论文最终多机障碍实验配置。在基线之上启用 `quads_neighbor_visible_num=2`、`quads_neighbor_obs_type=pos_vel` 和 `quads_encoder_type=attention`，对应论文里的邻居感知和注意力网络。

### 3.2 环境、状态、动作与奖励

- `gym_art/quadrotor_multi/quadrotor_multi.py`：多机环境核心实现。负责多架无人机实例化、邻居选择、障碍生成、碰撞检测、奖励计算、回合重置、日志统计以及渲染调度。
- `gym_art/quadrotor_multi/quadrotor_single.py`：单架无人机底层环境与观测空间定义，包含自状态、姿态、角速度、房间相关观测等。
- `gym_art/quadrotor_multi/quadrotor_dynamics.py`、`quadrotor_control.py`、`sensor_noise.py`：动力学、控制和传感噪声实现，是论文“直接输出四旋翼推力”的物理基础。
- `swarm_rl/env_wrappers/reward_shaping.py`、`swarm_rl/env_wrappers/quad_utils.py`：把奖励系数、退火策略、环境包装和 Sample Factory 接口接起来；其中 `quadcol_bin`、`quadcol_bin_smooth_max`、`quadcol_bin_obst` 分别对应机间碰撞、平滑接近惩罚和障碍碰撞惩罚。

### 3.3 论文中的 SDF 障碍观测

- `gym_art/quadrotor_multi/obstacles/obstacles.py`：`MultiObstacles` 在 reset/step 时给每架无人机拼接 9 维障碍观测。
- `gym_art/quadrotor_multi/obstacles/utils.py`：`get_surround_sdfs` 用 `3x3` 局部网格和 `0.1m` 分辨率计算最近障碍距离，正对应论文里描述的 9 维、数量与排列无关的 SDF 风格障碍观测。
- `gym_art/quadrotor_multi/collisions/obstacles.py`：障碍碰撞后的法向与速度更新逻辑。

### 3.4 论文中的注意力模型

- `swarm_rl/models/quad_multi_model.py`：核心网络定义。
- `QuadMultiHeadAttentionEncoder`：当 `quads_encoder_type=attention` 时启用。它先分别编码 self / neighbor / obstacle 三类观测，再把 neighbor 和 obstacle embedding 拼成长度为 2 的 token 序列，送入多头注意力，再与 self embedding 融合输出。
- `QuadMultiEncoder`：非 attention 分支，对应论文早期/对照结构，使用 MLP 或其他邻居编码器。
- `swarm_rl/models/attention_layer.py`：多头注意力与单头注意力算子；其中单头版本用于 `sim2real` 小模型部署。

### 3.5 论文中的回放机制

- `gym_art/quadrotor_multi/quad_experience_replay.py`：论文回放机制的直接实现。
- 核心思路是：每隔一段时间保存环境检查点；一旦发生独特碰撞，就把“碰撞前约 1.5 秒”的状态存进 replay buffer；新 episode 开始时按概率采样这些历史碰撞片段重新训练。
- 这与论文中“放大碰撞事件、从裁剪后的碰撞片段继续训练”的描述一致，也是障碍密集环境下稳定训练的重要部分。

### 3.6 训练后分析、评估与复现实验辅助

- `swarm_rl/enjoy.py`：加载训练好的模型进行可视化评估。
- `swarm_rl/env_wrappers/v_value_map.py`、`paper/plot_v_value_*`：用于论文里的 V-value 可视化分析。
- `paper/*.py`：生成论文中的统计图、对比图和热力图。
- `swarm_rl/sim2real/`：面向部署的小模型与 C/torch 导出、测试逻辑，对应论文后半部分的板载部署讨论。

## 4. 训练流程与对应文件

1. 启动训练时，从 `swarm_rl/train.py` 进入，先注册环境和模型。
2. `parse_swarm_cfg()` 会结合 `swarm_rl/env_wrappers/quadrotor_params.py` 解析所有论文相关超参数。
3. `swarm_rl/env_wrappers/quad_utils.py` 根据配置构造 `QuadrotorEnvMulti`，并可选叠加 `ExperienceReplayWrapper`、奖励退火包装器和兼容层。
4. `gym_art/quadrotor_multi/quadrotor_multi.py` 在每个 step 中完成：动作施加、无人机动力学更新、机间/障碍/房间碰撞检测、奖励计算、障碍 SDF 观测拼接、统计信息更新。
5. `swarm_rl/models/quad_multi_model.py` 定义策略/价值网络的编码器；若启用论文主模型，则使用 `QuadMultiHeadAttentionEncoder`。
6. Sample Factory 的 APPO 训练循环根据这些环境与模型输出，完成 rollout、优势估计、PPO 更新、checkpoint 保存等工作。
7. 训练完成后，可用 `swarm_rl/enjoy.py` 做回放评估，也可用 `paper/*.py` 脚本生成论文中的统计图。

## 5. 论文复现时建议重点阅读的模块

- 环境主循环：`gym_art/quadrotor_multi/quadrotor_multi.py`
- 单机观测与动作接口：`gym_art/quadrotor_multi/quadrotor_single.py`
- 障碍观测与碰撞：`gym_art/quadrotor_multi/obstacles/obstacles.py`、`gym_art/quadrotor_multi/obstacles/utils.py`、`gym_art/quadrotor_multi/collisions/obstacles.py`
- 训练包装与奖励：`swarm_rl/env_wrappers/quad_utils.py`、`swarm_rl/env_wrappers/reward_shaping.py`、`swarm_rl/env_wrappers/quadrotor_params.py`
- 注意力模型：`swarm_rl/models/quad_multi_model.py`、`swarm_rl/models/attention_layer.py`
- 回放机制：`gym_art/quadrotor_multi/quad_experience_replay.py`
- 论文实验配置：`swarm_rl/runs/obstacles/quad_obstacle_baseline.py`、`swarm_rl/runs/obstacles/quads_multi_obstacles.py`

## 6. 注释副本清单

| 原始文件 | 状态 | 注释副本 |
| --- | --- | --- |
| `gym_art/__init__.py` | 已生成 | `annotated_python/gym_art/__init__.py` |
| `gym_art/quadrotor_multi/__init__.py` | 已生成 | `annotated_python/gym_art/quadrotor_multi/__init__.py` |
| `gym_art/quadrotor_multi/aerodynamics/__init__.py` | 已生成 | `annotated_python/gym_art/quadrotor_multi/aerodynamics/__init__.py` |
| `gym_art/quadrotor_multi/aerodynamics/downwash.py` | 已生成 | `annotated_python/gym_art/quadrotor_multi/aerodynamics/downwash.py` |
| `gym_art/quadrotor_multi/collisions/__init__.py` | 已生成 | `annotated_python/gym_art/quadrotor_multi/collisions/__init__.py` |
| `gym_art/quadrotor_multi/collisions/obstacles.py` | 已生成 | `annotated_python/gym_art/quadrotor_multi/collisions/obstacles.py` |
| `gym_art/quadrotor_multi/collisions/quadrotors.py` | 已生成 | `annotated_python/gym_art/quadrotor_multi/collisions/quadrotors.py` |
| `gym_art/quadrotor_multi/collisions/room.py` | 已生成 | `annotated_python/gym_art/quadrotor_multi/collisions/room.py` |
| `gym_art/quadrotor_multi/collisions/test/__init__.py` | 已生成 | `annotated_python/gym_art/quadrotor_multi/collisions/test/__init__.py` |
| `gym_art/quadrotor_multi/collisions/test/speed_test/__init__.py` | 已生成 | `annotated_python/gym_art/quadrotor_multi/collisions/test/speed_test/__init__.py` |
| `gym_art/quadrotor_multi/collisions/test/speed_test/quadrotor.py` | 已生成 | `annotated_python/gym_art/quadrotor_multi/collisions/test/speed_test/quadrotor.py` |
| `gym_art/quadrotor_multi/collisions/test/unit_test/__init__.py` | 已生成 | `annotated_python/gym_art/quadrotor_multi/collisions/test/unit_test/__init__.py` |
| `gym_art/quadrotor_multi/collisions/test/unit_test/obstacles.py` | 已生成 | `annotated_python/gym_art/quadrotor_multi/collisions/test/unit_test/obstacles.py` |
| `gym_art/quadrotor_multi/collisions/test/unit_test/quadrotor.py` | 已生成 | `annotated_python/gym_art/quadrotor_multi/collisions/test/unit_test/quadrotor.py` |
| `gym_art/quadrotor_multi/collisions/utils.py` | 已生成 | `annotated_python/gym_art/quadrotor_multi/collisions/utils.py` |
| `gym_art/quadrotor_multi/get_state.py` | 已生成 | `annotated_python/gym_art/quadrotor_multi/get_state.py` |
| `gym_art/quadrotor_multi/inertia.py` | 已生成 | `annotated_python/gym_art/quadrotor_multi/inertia.py` |
| `gym_art/quadrotor_multi/numba_utils.py` | 已生成 | `annotated_python/gym_art/quadrotor_multi/numba_utils.py` |
| `gym_art/quadrotor_multi/obstacles/__init__.py` | 已生成 | `annotated_python/gym_art/quadrotor_multi/obstacles/__init__.py` |
| `gym_art/quadrotor_multi/obstacles/obstacles.py` | 已生成 | `annotated_python/gym_art/quadrotor_multi/obstacles/obstacles.py` |
| `gym_art/quadrotor_multi/obstacles/test/__init__.py` | 已生成 | `annotated_python/gym_art/quadrotor_multi/obstacles/test/__init__.py` |
| `gym_art/quadrotor_multi/obstacles/test/speed_test.py` | 已生成 | `annotated_python/gym_art/quadrotor_multi/obstacles/test/speed_test.py` |
| `gym_art/quadrotor_multi/obstacles/test/unit_test.py` | 已生成 | `annotated_python/gym_art/quadrotor_multi/obstacles/test/unit_test.py` |
| `gym_art/quadrotor_multi/obstacles/utils.py` | 已生成 | `annotated_python/gym_art/quadrotor_multi/obstacles/utils.py` |
| `gym_art/quadrotor_multi/plots/__init__.py` | 已生成 | `annotated_python/gym_art/quadrotor_multi/plots/__init__.py` |
| `gym_art/quadrotor_multi/plots/plot_v_value_1d.py` | 已生成 | `annotated_python/gym_art/quadrotor_multi/plots/plot_v_value_1d.py` |
| `gym_art/quadrotor_multi/plots/plot_v_value_2d.py` | 已生成 | `annotated_python/gym_art/quadrotor_multi/plots/plot_v_value_2d.py` |
| `gym_art/quadrotor_multi/plots/plot_v_value_3d.py` | 已生成 | `annotated_python/gym_art/quadrotor_multi/plots/plot_v_value_3d.py` |
| `gym_art/quadrotor_multi/plots/plot_v_value_4d.py` | 已生成 | `annotated_python/gym_art/quadrotor_multi/plots/plot_v_value_4d.py` |
| `gym_art/quadrotor_multi/quad_experience_replay.py` | 已生成 | `annotated_python/gym_art/quadrotor_multi/quad_experience_replay.py` |
| `gym_art/quadrotor_multi/quad_models.py` | 已生成 | `annotated_python/gym_art/quadrotor_multi/quad_models.py` |
| `gym_art/quadrotor_multi/quad_utils.py` | 已生成 | `annotated_python/gym_art/quadrotor_multi/quad_utils.py` |
| `gym_art/quadrotor_multi/quadrotor_control.py` | 已生成 | `annotated_python/gym_art/quadrotor_multi/quadrotor_control.py` |
| `gym_art/quadrotor_multi/quadrotor_dynamics.py` | 已生成 | `annotated_python/gym_art/quadrotor_multi/quadrotor_dynamics.py` |
| `gym_art/quadrotor_multi/quadrotor_multi.py` | 已生成 | `annotated_python/gym_art/quadrotor_multi/quadrotor_multi.py` |
| `gym_art/quadrotor_multi/quadrotor_multi_visualization.py` | 已生成 | `annotated_python/gym_art/quadrotor_multi/quadrotor_multi_visualization.py` |
| `gym_art/quadrotor_multi/quadrotor_randomization.py` | 已生成 | `annotated_python/gym_art/quadrotor_multi/quadrotor_randomization.py` |
| `gym_art/quadrotor_multi/quadrotor_single.py` | 已生成 | `annotated_python/gym_art/quadrotor_multi/quadrotor_single.py` |
| `gym_art/quadrotor_multi/quadrotor_visualization.py` | 已生成 | `annotated_python/gym_art/quadrotor_multi/quadrotor_visualization.py` |
| `gym_art/quadrotor_multi/rendering3d.py` | 已生成 | `annotated_python/gym_art/quadrotor_multi/rendering3d.py` |
| `gym_art/quadrotor_multi/scenarios/__init__.py` | 已生成 | `annotated_python/gym_art/quadrotor_multi/scenarios/__init__.py` |
| `gym_art/quadrotor_multi/scenarios/base.py` | 已生成 | `annotated_python/gym_art/quadrotor_multi/scenarios/base.py` |
| `gym_art/quadrotor_multi/scenarios/dynamic_diff_goal.py` | 已生成 | `annotated_python/gym_art/quadrotor_multi/scenarios/dynamic_diff_goal.py` |
| `gym_art/quadrotor_multi/scenarios/dynamic_formations.py` | 已生成 | `annotated_python/gym_art/quadrotor_multi/scenarios/dynamic_formations.py` |
| `gym_art/quadrotor_multi/scenarios/dynamic_same_goal.py` | 已生成 | `annotated_python/gym_art/quadrotor_multi/scenarios/dynamic_same_goal.py` |
| `gym_art/quadrotor_multi/scenarios/ep_lissajous3D.py` | 已生成 | `annotated_python/gym_art/quadrotor_multi/scenarios/ep_lissajous3D.py` |
| `gym_art/quadrotor_multi/scenarios/ep_rand_bezier.py` | 已生成 | `annotated_python/gym_art/quadrotor_multi/scenarios/ep_rand_bezier.py` |
| `gym_art/quadrotor_multi/scenarios/mix.py` | 已生成 | `annotated_python/gym_art/quadrotor_multi/scenarios/mix.py` |
| `gym_art/quadrotor_multi/scenarios/obstacles/o_base.py` | 已生成 | `annotated_python/gym_art/quadrotor_multi/scenarios/obstacles/o_base.py` |
| `gym_art/quadrotor_multi/scenarios/obstacles/o_dynamic_same_goal.py` | 已生成 | `annotated_python/gym_art/quadrotor_multi/scenarios/obstacles/o_dynamic_same_goal.py` |
| `gym_art/quadrotor_multi/scenarios/obstacles/o_ep_rand_bezier.py` | 已生成 | `annotated_python/gym_art/quadrotor_multi/scenarios/obstacles/o_ep_rand_bezier.py` |
| `gym_art/quadrotor_multi/scenarios/obstacles/o_random.py` | 已生成 | `annotated_python/gym_art/quadrotor_multi/scenarios/obstacles/o_random.py` |
| `gym_art/quadrotor_multi/scenarios/obstacles/o_static_same_goal.py` | 已生成 | `annotated_python/gym_art/quadrotor_multi/scenarios/obstacles/o_static_same_goal.py` |
| `gym_art/quadrotor_multi/scenarios/obstacles/o_swap_goals.py` | 已生成 | `annotated_python/gym_art/quadrotor_multi/scenarios/obstacles/o_swap_goals.py` |
| `gym_art/quadrotor_multi/scenarios/run_away.py` | 已生成 | `annotated_python/gym_art/quadrotor_multi/scenarios/run_away.py` |
| `gym_art/quadrotor_multi/scenarios/static_diff_goal.py` | 已生成 | `annotated_python/gym_art/quadrotor_multi/scenarios/static_diff_goal.py` |
| `gym_art/quadrotor_multi/scenarios/static_same_goal.py` | 已生成 | `annotated_python/gym_art/quadrotor_multi/scenarios/static_same_goal.py` |
| `gym_art/quadrotor_multi/scenarios/swap_goals.py` | 已生成 | `annotated_python/gym_art/quadrotor_multi/scenarios/swap_goals.py` |
| `gym_art/quadrotor_multi/scenarios/swarm_vs_swarm.py` | 已生成 | `annotated_python/gym_art/quadrotor_multi/scenarios/swarm_vs_swarm.py` |
| `gym_art/quadrotor_multi/scenarios/test/__init__.py` | 已生成 | `annotated_python/gym_art/quadrotor_multi/scenarios/test/__init__.py` |
| `gym_art/quadrotor_multi/scenarios/test/o_test.py` | 已生成 | `annotated_python/gym_art/quadrotor_multi/scenarios/test/o_test.py` |
| `gym_art/quadrotor_multi/scenarios/utils.py` | 已生成 | `annotated_python/gym_art/quadrotor_multi/scenarios/utils.py` |
| `gym_art/quadrotor_multi/sensor_noise.py` | 已生成 | `annotated_python/gym_art/quadrotor_multi/sensor_noise.py` |
| `gym_art/quadrotor_multi/tests/__init__.py` | 已生成 | `annotated_python/gym_art/quadrotor_multi/tests/__init__.py` |
| `gym_art/quadrotor_multi/tests/plot_v_value.py` | 已生成 | `annotated_python/gym_art/quadrotor_multi/tests/plot_v_value.py` |
| `gym_art/quadrotor_multi/tests/plot_v_value_2d.py` | 已生成 | `annotated_python/gym_art/quadrotor_multi/tests/plot_v_value_2d.py` |
| `gym_art/quadrotor_multi/tests/plot_v_value_3d.py` | 已生成 | `annotated_python/gym_art/quadrotor_multi/tests/plot_v_value_3d.py` |
| `gym_art/quadrotor_multi/tests/plot_v_value_4d.py` | 已生成 | `annotated_python/gym_art/quadrotor_multi/tests/plot_v_value_4d.py` |
| `gym_art/quadrotor_multi/tests/test_multi_env.py` | 已生成 | `annotated_python/gym_art/quadrotor_multi/tests/test_multi_env.py` |
| `gym_art/quadrotor_multi/tests/test_numba_opt.py` | 已生成 | `annotated_python/gym_art/quadrotor_multi/tests/test_numba_opt.py` |
| `paper/attn_heatmap.py` | 已生成 | `annotated_python/paper/attn_heatmap.py` |
| `paper/fps_compare.py` | 已生成 | `annotated_python/paper/fps_compare.py` |
| `paper/mean_std_plots_quad_annealing.py` | 已生成 | `annotated_python/paper/mean_std_plots_quad_annealing.py` |
| `paper/mean_std_plots_quad_baseline.py` | 已生成 | `annotated_python/paper/mean_std_plots_quad_baseline.py` |
| `paper/mean_std_plots_quad_compare_arch.py` | 已生成 | `annotated_python/paper/mean_std_plots_quad_compare_arch.py` |
| `paper/mean_std_plots_quad_obstacle.py` | 已生成 | `annotated_python/paper/mean_std_plots_quad_obstacle.py` |
| `paper/mean_std_plots_quad_obstacle_ablation.py` | 已生成 | `annotated_python/paper/mean_std_plots_quad_obstacle_ablation.py` |
| `paper/mean_std_plots_quad_obstacle_compare_arch_density.py` | 已生成 | `annotated_python/paper/mean_std_plots_quad_obstacle_compare_arch_density.py` |
| `paper/mean_std_plots_quad_obstacle_compare_arch_neighbor.py` | 已生成 | `annotated_python/paper/mean_std_plots_quad_obstacle_compare_arch_neighbor.py` |
| `paper/mean_std_plots_quad_obstacle_num_agents.py` | 已生成 | `annotated_python/paper/mean_std_plots_quad_obstacle_num_agents.py` |
| `paper/mean_std_plots_quad_scale.py` | 已生成 | `annotated_python/paper/mean_std_plots_quad_scale.py` |
| `setup.py` | 已生成 | `annotated_python/setup.py` |
| `swarm_rl/__init__.py` | 已生成 | `annotated_python/swarm_rl/__init__.py` |
| `swarm_rl/enjoy.py` | 已生成 | `annotated_python/swarm_rl/enjoy.py` |
| `swarm_rl/env_wrappers/__init__.py` | 已生成 | `annotated_python/swarm_rl/env_wrappers/__init__.py` |
| `swarm_rl/env_wrappers/compatibility.py` | 已生成 | `annotated_python/swarm_rl/env_wrappers/compatibility.py` |
| `swarm_rl/env_wrappers/quad_utils.py` | 已生成 | `annotated_python/swarm_rl/env_wrappers/quad_utils.py` |
| `swarm_rl/env_wrappers/quadrotor_params.py` | 已生成 | `annotated_python/swarm_rl/env_wrappers/quadrotor_params.py` |
| `swarm_rl/env_wrappers/reward_shaping.py` | 已生成 | `annotated_python/swarm_rl/env_wrappers/reward_shaping.py` |
| `swarm_rl/env_wrappers/tests/__init__.py` | 已生成 | `annotated_python/swarm_rl/env_wrappers/tests/__init__.py` |
| `swarm_rl/env_wrappers/tests/test_quads.py` | 已生成 | `annotated_python/swarm_rl/env_wrappers/tests/test_quads.py` |
| `swarm_rl/env_wrappers/v_value_map.py` | 已生成 | `annotated_python/swarm_rl/env_wrappers/v_value_map.py` |
| `swarm_rl/models/__init__.py` | 已生成 | `annotated_python/swarm_rl/models/__init__.py` |
| `swarm_rl/models/attention_layer.py` | 已生成 | `annotated_python/swarm_rl/models/attention_layer.py` |
| `swarm_rl/models/quad_multi_model.py` | 已生成 | `annotated_python/swarm_rl/models/quad_multi_model.py` |
| `swarm_rl/models/weight_recycler.py` | 已生成 | `annotated_python/swarm_rl/models/weight_recycler.py` |
| `swarm_rl/runs/obstacles/obst_density_random.py` | 已生成 | `annotated_python/swarm_rl/runs/obstacles/obst_density_random.py` |
| `swarm_rl/runs/obstacles/obst_domain_random.py` | 已生成 | `annotated_python/swarm_rl/runs/obstacles/obst_domain_random.py` |
| `swarm_rl/runs/obstacles/obst_size_random.py` | 已生成 | `annotated_python/swarm_rl/runs/obstacles/obst_size_random.py` |
| `swarm_rl/runs/obstacles/pbt_quads_multi_obstacles.py` | 已生成 | `annotated_python/swarm_rl/runs/obstacles/pbt_quads_multi_obstacles.py` |
| `swarm_rl/runs/obstacles/quad_obstacle_baseline.py` | 已生成 | `annotated_python/swarm_rl/runs/obstacles/quad_obstacle_baseline.py` |
| `swarm_rl/runs/obstacles/quads_multi_obstacles.py` | 已生成 | `annotated_python/swarm_rl/runs/obstacles/quads_multi_obstacles.py` |
| `swarm_rl/runs/obstacles/quads_multi_obstacles_nei_encoder_search.py` | 已生成 | `annotated_python/swarm_rl/runs/obstacles/quads_multi_obstacles_nei_encoder_search.py` |
| `swarm_rl/runs/quad_multi_mix_baseline.py` | 已生成 | `annotated_python/swarm_rl/runs/quad_multi_mix_baseline.py` |
| `swarm_rl/runs/quad_multi_mix_baseline_attn_8.py` | 已生成 | `annotated_python/swarm_rl/runs/quad_multi_mix_baseline_attn_8.py` |
| `swarm_rl/runs/single_quad/__init__.py` | 已生成 | `annotated_python/swarm_rl/runs/single_quad/__init__.py` |
| `swarm_rl/runs/single_quad/baseline.py` | 已生成 | `annotated_python/swarm_rl/runs/single_quad/baseline.py` |
| `swarm_rl/runs/single_quad/single_quad.py` | 已生成 | `annotated_python/swarm_rl/runs/single_quad/single_quad.py` |
| `swarm_rl/sim2real/c_models/__init__.py` | 已生成 | `annotated_python/swarm_rl/sim2real/c_models/__init__.py` |
| `swarm_rl/sim2real/code_blocks.py` | 已生成 | `annotated_python/swarm_rl/sim2real/code_blocks.py` |
| `swarm_rl/sim2real/sim2real.py` | 已生成 | `annotated_python/swarm_rl/sim2real/sim2real.py` |
| `swarm_rl/sim2real/tests/unit_tests.py` | 已生成 | `annotated_python/swarm_rl/sim2real/tests/unit_tests.py` |
| `swarm_rl/sim2real/torch_models/__init__.py` | 已生成 | `annotated_python/swarm_rl/sim2real/torch_models/__init__.py` |
| `swarm_rl/train.py` | 已生成 | `annotated_python/swarm_rl/train.py` |
| `swarm_rl/utils.py` | 已生成 | `annotated_python/swarm_rl/utils.py` |
