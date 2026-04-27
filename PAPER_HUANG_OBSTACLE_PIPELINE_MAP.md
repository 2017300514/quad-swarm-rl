# Huang 障碍导航论文复现全流程数据流与代码框架

本文件把论文复现路径拆成三条主线：

1. 训练数据流  
2. 测试/评估/渲染数据流  
3. 绘图聚合数据流  

并列出关键 Python 文件、类、函数以及参数透传关系。

## 1. 总体数据流

```text
CLI flags
  -> swarm_rl/train.py / swarm_rl/enjoy.py
  -> quadrotor_params.py 注入 quad 专属参数
  -> quad_utils.py 构建 env + wrappers
  -> QuadrotorEnvMulti / QuadrotorSingle
  -> 邻居观测 + 障碍 SDF + 碰撞与奖励
  -> quad_multi_model.py encoder 前向
  -> Sample Factory APPO/IPPO 训练
  -> checkpoint / cfg.json / tfevents
  -> swarm_rl/enjoy.py 载入 checkpoint 做评估/视频/V-value
  -> paper/*.py 读取 tfevents 聚合成 pdf
```

## 2. 训练主线

### 2.1 入口层

| 文件 | 核心符号 | 作用 |
| --- | --- | --- |
| `swarm_rl/train.py` | `register_swarm_components()` | 注册 `quadrotor_multi` 环境和自定义 encoder |
| `swarm_rl/train.py` | `parse_swarm_cfg()` | 先拿 Sample Factory 参数，再注入 quad 参数，再做 full cfg |
| `swarm_rl/train.py` | `main()` | 把 cfg 送进 `sample_factory.train.run_rl` |
| `swarm_rl/enjoy.py` | `main()` | 复用同一套注册和解析逻辑进入 `sample_factory.enjoy` |

### 2.2 参数注入层

| 文件 | 核心符号 | 作用 |
| --- | --- | --- |
| `swarm_rl/env_wrappers/quadrotor_params.py` | `quadrotors_override_defaults()` | 覆写 Sample Factory 默认 encoder/rnn/env_frameskip |
| `swarm_rl/env_wrappers/quadrotor_params.py` | `add_quadrotors_env_args()` | 注入所有 quad 相关 CLI 参数 |

主要参数分组：

1. self observation：`quads_obs_repr`
2. neighbor：`quads_neighbor_visible_num` / `quads_neighbor_obs_type` / `quads_neighbor_encoder_type`
3. obstacle：`quads_use_obstacles` / `quads_obstacle_obs_type` / `quads_obst_density` / `quads_obst_size`
4. replay：`replay_buffer_sample_prob`
5. anneal：`anneal_collision_steps`
6. scenario：`quads_mode`
7. rendering：`quads_view_mode` / `quads_render` / `visualize_v_value`
8. sim2real：`quads_sim2real`

## 3. 环境构造与 wrapper 链

### 3.1 构造顺序

`swarm_rl/env_wrappers/quad_utils.py:20-118`

1. 创建 `QuadrotorEnvMulti`
2. 如果 `replay_buffer_sample_prob > 0`，包一层 `ExperienceReplayWrapper`
3. 创建 reward shaping scheme，并按需要设置 annealing schedule
4. 包 `QuadsRewardShapingWrapper`
5. 包 `QuadEnvCompatibility`
6. 如果 `visualize_v_value=True`，加载 checkpoint，再包 `V_ValueMapWrapper`

### 3.2 相关类

| 文件 | 类 / 函数 | 作用 |
| --- | --- | --- |
| `swarm_rl/env_wrappers/quad_utils.py` | `AnnealSchedule` | 保存某个 reward coeff 的退火目标 |
| `swarm_rl/env_wrappers/quad_utils.py` | `make_quadrotor_env_multi()` | 训练/评估主环境工厂 |
| `swarm_rl/env_wrappers/reward_shaping.py` | `QuadsRewardShapingWrapper` | 聚合 reward 分量，写 `true_reward` 和 `episode_extra_stats` |
| `swarm_rl/env_wrappers/compatibility.py` | `QuadEnvCompatibility` | 把旧 step API 转成 Gymnasium 兼容格式 |
| `swarm_rl/env_wrappers/v_value_map.py` | `V_ValueMapWrapper` | 渲染时追加 critic value map |

## 4. 环境内部主循环

### 4.1 核心环境

| 文件 | 类 / 函数 | 作用 |
| --- | --- | --- |
| `gym_art/quadrotor_multi/quadrotor_multi.py` | `QuadrotorEnvMulti.__init__` | 创建 N 个 `QuadrotorSingle`，初始化碰撞、日志、障碍、场景、render 状态 |
| `quadrotor_multi.py` | `reset()` | 重建场景、障碍地图、spawn/goal、拼 observation、清空 episode 统计 |
| `quadrotor_multi.py` | `step()` | 执行动作、计算碰撞/奖励/指标、推进 scenario、生成下一时刻 observation |
| `quadrotor_multi.py` | `render()` | 多相机渲染、轨迹可视化、障碍显示 |
| `quadrotor_multi.py` | `neighborhood_indices()` | 根据相对位置/速度决定 K 个可见邻居 |
| `quadrotor_multi.py` | `add_neighborhood_obs()` | 把邻居观测拼进 obs |
| `quadrotor_multi.py` | `obst_generation_given_density()` | 按 density 在 `8x8` 栅格上采样障碍位置 |

### 4.2 单机环境

`QuadrotorEnvMulti` 在 `__init__` 里循环创建 `QuadrotorSingle`：

- 文件：`gym_art/quadrotor_multi/quadrotor_single.py`
- 角色：单机动力学、控制、基础自观测和单机 reward 的来源

虽然本次没有逐段展开单机文件，但在论文复现主线上它是：

```text
动作 -> 电机/推力控制 -> 动力学积分 -> 单机基础观测与基础 reward
```

## 5. 障碍、场景与观测拼接

### 5.1 障碍地图生成

| 文件 | 类 / 函数 | 作用 |
| --- | --- | --- |
| `quadrotor_multi.py` | `obst_generation_given_density()` | 生成 `obst_map`、障碍中心坐标、cell centers |
| `gym_art/quadrotor_multi/obstacles/utils.py` | `get_cell_centers()` | 把 `8x8` 栅格映射到房间中心坐标 |

### 5.2 障碍观测

| 文件 | 类 / 函数 | 作用 |
| --- | --- | --- |
| `gym_art/quadrotor_multi/obstacles/obstacles.py` | `MultiObstacles.reset()` | episode 开始时计算 9 维 obstacle obs 并拼接 |
| `gym_art/quadrotor_multi/obstacles/obstacles.py` | `MultiObstacles.step()` | 每步更新 9 维 obstacle obs |
| `gym_art/quadrotor_multi/obstacles/utils.py` | `get_surround_sdfs()` | 对每台无人机计算 `3x3` 邻域最小障碍距离 |

参数透传路径：

```text
--quads_obst_density / --quads_obst_size / --quads_obstacle_obs_type
  -> add_quadrotors_env_args()
  -> cfg
  -> make_quadrotor_env_multi()
  -> QuadrotorEnvMulti(use_obstacles, obst_density, obst_size)
  -> obst_generation_given_density()
  -> MultiObstacles.reset/step()
  -> obs += 9D SDF obstacle features
```

### 5.3 场景

| 文件 | 类 / 函数 | 作用 |
| --- | --- | --- |
| `gym_art/quadrotor_multi/scenarios/mix.py` | `create_scenario()` | 按 `quads_mode` 动态构造具体场景类 |
| `scenarios/mix.py` | `Scenario_mix.reset()` | 每个 episode 随机挑选一个具体 scenario |
| `scenarios/mix.py` | `Scenario_mix.step()` | 把内部 scenario 的 goal/formation 变化回传给环境 |
| `scenarios/obstacles/o_base.py` | `Scenario_o_base` | 障碍场景基类，负责 free-space 采样和 obstacle-map 上的 spawn/goal 生成 |
| `scenarios/obstacles/o_random.py` | `Scenario_o_random` | 障碍随机起终点场景 |

参数透传路径：

```text
--quads_mode=mix
  -> create_scenario()
  -> Scenario_mix
  -> 某个 obstacle scenario
  -> reset(obst_map, cell_centers)
  -> 生成 spawn_points / goals
  -> QuadrotorSingle.reset()
```

## 6. replay 数据流

### 6.1 主要类

| 文件 | 类 / 函数 | 作用 |
| --- | --- | --- |
| `gym_art/quadrotor_multi/quad_experience_replay.py` | `ReplayBufferEvent` | 保存某次 replay 事件的 env 快照和 obs |
| 同上 | `ReplayBuffer` | 管理 collision 片段队列 |
| 同上 | `ExperienceReplayWrapper` | 在 env 外层保存 checkpoint、检测碰撞、随机回放片段 |

### 6.2 流程

```text
step()
  -> 每 0.5s save checkpoint
  -> 发生 collision 且满足 grace period
  -> 从 episode_checkpoints 取 collision 前 1.5s 的快照
  -> 写入 replay buffer
reset/new_episode()
  -> 以 replay_buffer_sample_prob 概率从 buffer 取历史片段
  -> 恢复 env 和 obs
```

对应关键代码：

- checkpoint 间隔：`quad_experience_replay.py:17-23`
- collision 前 `1.5s`：`quad_experience_replay.py:89-93`
- 从历史 checkpoint 取样：`quad_experience_replay.py:154-160`
- 以设定概率 replay：`quad_experience_replay.py:176-193`

## 7. 模型前向与参数流

### 7.1 模型注册

| 文件 | 符号 | 作用 |
| --- | --- | --- |
| `swarm_rl/models/quad_multi_model.py` | `register_models()` | 向 Sample Factory 注册 quad encoder 工厂 |
| 同上 | `make_quadmulti_encoder()` | 根据 `cfg.quads_encoder_type` 返回 attention 或普通 encoder |

### 7.2 主要类

| 类 | 文件 | 作用 |
| --- | --- | --- |
| `QuadMultiHeadAttentionEncoder` | `quad_multi_model.py` | 论文主路径 encoder：self / neighbor / obstacle 三段 embedding，再做 multi-head attention |
| `QuadSingleHeadAttentionEncoder_Sim2Real` | 同上 | sim2real 缩小版单头 attention |
| `QuadMultiEncoder` | 同上 | 非 attention 主干 |
| `QuadNeighborhoodEncoderAttention` | 同上 | 邻居分支 attention pooling |
| `QuadNeighborhoodEncoderDeepsets` | 同上 | 邻居分支 mean embedding |
| `QuadNeighborhoodEncoderMlp` | 同上 | 邻居分支 MLP |
| `MultiHeadAttention` | `swarm_rl/models/attention_layer.py` | 4-head attention |
| `OneHeadAttention` | 同上 | sim2real 单头 attention |

### 7.3 参数流

```text
--quads_obs_repr
--quads_neighbor_visible_num
--quads_neighbor_obs_type
--quads_obstacle_obs_type
--quads_encoder_type
--quads_sim2real
  -> cfg
  -> make_quadmulti_encoder()
  -> encoder 解析 obs 维度
  -> forward(obs_dict)
```

关键常量：

- `gym_art/quadrotor_multi/quad_utils.py:30-44`
  - `xyz_vxyz_R_omega_floor -> 19`
  - `pos_vel -> 6`
  - `octomap -> 9`

## 8. reward、碰撞与 metric 日志

### 8.1 环境内碰撞与指标

`gym_art/quadrotor_multi/quadrotor_multi.py:427-718`

环境 `step()` 内部依次做：

1. `calculate_collision_matrix()` 计算机间碰撞
2. `MultiObstacles.collision_detection()` 计算机-障碍碰撞
3. `calculate_room_collision()` 计算房间/地板/墙/天花板碰撞
4. 写 collision reward 和 proximity penalty
5. 更新 `distance_to_goal`
6. 在 episode 结束时把统计量写进 `info['episode_extra_stats']`

### 8.2 当前已经写入的论文相关字段

| 字段 | 代码位置 |
| --- | --- |
| `metric/agent_success_rate` | `quadrotor_multi.py:703-706` |
| `metric/agent_deadlock_rate` | `quadrotor_multi.py:707-709` |
| `metric/agent_col_rate` | `quadrotor_multi.py:710-712` |
| `metric/agent_neighbor_col_rate` | `quadrotor_multi.py:713-715` |
| `metric/agent_obst_col_rate` | `quadrotor_multi.py:716-718` |
| `{scenario}/distance_to_goal_1s` | `quadrotor_multi.py:656-660` |
| `{scenario}/distance_to_goal_3s` | `quadrotor_multi.py:658-660` |
| `{scenario}/distance_to_goal_5s` | `quadrotor_multi.py:660-661` |

### 8.3 reward shaping wrapper

`swarm_rl/env_wrappers/reward_shaping.py`

`QuadsRewardShapingWrapper.step()` 会：

1. 累积各个 `rew*` 分量
2. 在 episode 结束时生成 `true_reward`
3. 把 `rewraw_*`、`z_approx_total_training_steps`、动作均值/方差等写入 `episode_extra_stats`
4. 如果启用 annealing，按训练步数动态调整 collision reward coeff

## 9. checkpoint、评估与 V-value

### 9.1 checkpoint 产物

训练结束或过程中，Sample Factory 会在实验目录下产出：

- `cfg.json`
- `checkpoint_*`
- `best_*`
- `events.out.tfevents.*`

### 9.2 enjoy 流程

```text
python -m swarm_rl.enjoy
  -> register_swarm_components()
  -> parse_swarm_cfg(evaluation=True)
  -> sample_factory.enjoy(cfg)
```

这个流程和训练共用同一套 env/model 注册，因此：

1. 训练和评估的 obs/model 结构天然一致  
2. 只要 `cfg.json + checkpoint` 齐全，就能直接评估  

### 9.3 V-value 图

`swarm_rl/env_wrappers/v_value_map.py`

流程：

1. `make_quadrotor_env_multi()` 检测到 `visualize_v_value`
2. 通过 `Learner.get_checkpoints()` 和 `Learner.load_checkpoint()` 加载模型
3. `V_ValueMapWrapper.render()` 拿当前 obs
4. 在 `[-2,2] x [-2,2]` 局部网格上扰动 x/y
5. 调 `model.forward(..., values_only=True)` 得到 critic values
6. 用 `plot_v_value_2d()` 生成热图并和渲染帧横向拼接

## 10. 绘图聚合数据流

```text
episode_extra_stats
  -> Sample Factory TensorBoard scalar
  -> events.out.tfevents.*
  -> paper/*.py EventAccumulator.Reload().scalars
  -> 插值/平滑/均值方差
  -> pdf
```

### 10.1 关键绘图文件

| 文件 | 消费什么 |
| --- | --- |
| `paper/mean_std_plots_quad_obstacle_ablation.py` | success/collision/distance 主指标 |
| `paper/mean_std_plots_quad_obstacle_num_agents.py` | success/collision/distance 主指标 |
| `paper/mean_std_plots_quad_obstacle_compare_arch_neighbor.py` | success/collision/distance 主指标 |
| `paper/mean_std_plots_quad_obstacle_compare_arch_density.py` | success/collision/distance 主指标 |
| `paper/mean_std_plots_quad_obstacle.py` | `0_aux/*` 辅助指标 |
| `paper/fps_compare.py` | 硬编码 SPS |
| `paper/attn_heatmap.py` | 硬编码 attention 分数 |

## 11. 复现时最关键的文件清单

### 11.1 核心文件

1. `swarm_rl/train.py`
2. `swarm_rl/enjoy.py`
3. `swarm_rl/env_wrappers/quadrotor_params.py`
4. `swarm_rl/env_wrappers/quad_utils.py`
5. `swarm_rl/env_wrappers/reward_shaping.py`
6. `swarm_rl/env_wrappers/compatibility.py`
7. `swarm_rl/env_wrappers/v_value_map.py`
8. `swarm_rl/models/quad_multi_model.py`
9. `swarm_rl/models/attention_layer.py`
10. `gym_art/quadrotor_multi/quadrotor_multi.py`
11. `gym_art/quadrotor_multi/quadrotor_single.py`
12. `gym_art/quadrotor_multi/quad_experience_replay.py`
13. `gym_art/quadrotor_multi/obstacles/obstacles.py`
14. `gym_art/quadrotor_multi/obstacles/utils.py`
15. `gym_art/quadrotor_multi/scenarios/mix.py`
16. `gym_art/quadrotor_multi/scenarios/obstacles/o_base.py`
17. `gym_art/quadrotor_multi/scenarios/obstacles/o_random.py`
18. `swarm_rl/runs/obstacles/quad_obstacle_baseline.py`
19. `swarm_rl/runs/obstacles/quads_multi_obstacles.py`

### 11.2 出图/分析文件

1. `paper/mean_std_plots_quad_obstacle.py`
2. `paper/mean_std_plots_quad_obstacle_ablation.py`
3. `paper/mean_std_plots_quad_obstacle_num_agents.py`
4. `paper/mean_std_plots_quad_obstacle_compare_arch_neighbor.py`
5. `paper/mean_std_plots_quad_obstacle_compare_arch_density.py`
6. `paper/fps_compare.py`
7. `paper/attn_heatmap.py`
8. `gym_art/quadrotor_multi/tests/plot_v_value_2d.py`

## 12. 一句话总结

这条论文复现主线的本质是：

**CLI 参数通过 `train.py -> quadrotor_params.py -> quad_utils.py` 下沉到 `QuadrotorEnvMulti` 与 quad encoder；环境在 `step()` 里完成邻居观测、障碍 SDF、碰撞与 reward 统计；这些统计经 `episode_extra_stats -> tfevents` 流入 `paper/*.py`，而 checkpoint 又通过 `enjoy.py` 和 `V_ValueMapWrapper` 流入视频与 V-value 图。**
