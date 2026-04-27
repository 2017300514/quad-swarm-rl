# Huang 障碍导航论文训练复现分析

针对论文 `hybrid_auto/Huang 等 - Collision Avoidance and Navigation for a Quadrotor.md`，本文件把训练主入口、论文参数、代码默认值、4x4090 并行策略和可直接执行的训练方案整理到一起。

## 1. 训练主入口

主训练链路是：

```text
python -m swarm_rl.train
  -> swarm_rl/train.py:main()
  -> register_swarm_components()
  -> parse_swarm_cfg()
  -> sample_factory.train.run_rl(cfg)
  -> swarm_rl/env_wrappers/quad_utils.py:make_quadrotor_env_multi()
  -> gym_art/quadrotor_multi/quadrotor_multi.py:QuadrotorEnvMulti
  -> (可选) ExperienceReplayWrapper
  -> QuadsRewardShapingWrapper
  -> QuadEnvCompatibility
  -> Sample Factory actor-critic + quad encoder
```

对应代码位置：

- 训练入口：`swarm_rl/train.py:16-33`
- 配置注入：`swarm_rl/env_wrappers/quadrotor_params.py:4-122`
- 环境构造：`swarm_rl/env_wrappers/quad_utils.py:20-118`
- 多机环境：`gym_art/quadrotor_multi/quadrotor_multi.py:23-780`
- Replay：`gym_art/quadrotor_multi/quad_experience_replay.py:16-210`
- 模型注册与 encoder 选择：`swarm_rl/models/quad_multi_model.py:356-368`

## 2. 论文里明确给出的参数

| 类别 | 参数 | 论文值 | 代码落点 |
| --- | --- | --- | --- |
| 房间 | room size | `10m x 10m x 10m` | `swarm_rl/env_wrappers/quadrotor_params.py:99-101` |
| 障碍生成区域 | center area | `8m x 8m`，离散成 `64` 个 `1m^2` 栅格 | `quadrotor_multi.py:304-325` |
| 机器人数量 | base setting | `8` | `swarm_rl/runs/obstacles/quads_multi_obstacles.py:4-8` |
| 邻居数量 | sensed neighbors | `2` | `swarm_rl/runs/obstacles/quad_obstacle_baseline.py:16-21` |
| 障碍密度 | obstacle density | `20%` | `swarm_rl/runs/obstacles/quad_obstacle_baseline.py:19-21` |
| 障碍尺寸 | obstacle size | `0.6m` | `swarm_rl/runs/obstacles/quad_obstacle_baseline.py:19-21` |
| 训练种子 | seeds | `4` 个 | `swarm_rl/runs/obstacles/quads_multi_obstacles.py:4-8` |
| 观测 | obstacle obs | `3x3` SDF 风格局部障碍距离，分辨率 `0.1m` | `obstacles/utils.py:5-27`, `obstacles/obstacles.py:18-35` |
| 动作 | actions | `4` 个 rotor thrust，范围 `[0,1]` | 论文方法部分 |
| Replay | clip 窗口 | collision 前 `1.5s` | `quad_experience_replay.py:89-93,154-160` |
| 评估指标 | metrics | success rate / collision rate / distance to goal / flight distance / inference time | 论文实验部分 |

## 3. 代码里补出来的论文默认值

下面这些不是论文正文逐条明写，但仓库中的 obstacle baseline 已经把它们固定成了论文主实验的默认值。

| 类别 | 参数 | 当前值 | 来源 |
| --- | --- | --- | --- |
| RL 算法 | algo | `APPO` | `quad_obstacle_baseline.py:1-21` |
| 训练预算 | train_for_env_steps | `1_000_000_000` | 同上 |
| worker | `num_workers` | `36` | 同上 |
| 每 worker env 数 | `num_envs_per_worker` | `4` | 同上 |
| 学习率 | `learning_rate` | `1e-4` | 同上 |
| PPO value clip | `ppo_clip_value` | `5.0` | 同上 |
| rollout | `128` | 同上 |
| batch size | `1024` | 同上 |
| recurrence | `1` | 同上 |
| obs repr | `xyz_vxyz_R_omega_floor` | `19` 维 self obs | `quad_obstacle_baseline.py:13-21`, `gym_art/quadrotor_multi/quad_utils.py:30-44` |
| 邻居观测 | `pos_vel` | 每邻居 `6` 维 | `gym_art/quadrotor_multi/quad_utils.py:36-39` |
| 障碍观测 | `octomap` | `9` 维 | `gym_art/quadrotor_multi/quad_utils.py:41-44` |
| 主 encoder | `quads_encoder_type=attention` | multi-head attention 主干 | `quads_multi_obstacles.py:11-16`, `quad_multi_model.py:125-197,356-364` |
| 邻居 encoder | `quads_neighbor_encoder_type=no_encoder` | 主论文配置下邻居信息不单独走 `QuadNeighborhoodEncoderAttention` | `quad_obstacle_baseline.py:16-21` |
| replay 采样率 | `replay_buffer_sample_prob` | `0.75` | `quad_obstacle_baseline.py:9-10` |
| replay checkpoint 间隔 | `cp_step_size` | `0.5s` | `quad_experience_replay.py:17-23` |
| replay buffer 大小 | `20` | `20` | `quad_experience_replay.py:17-23` |
| replay 清理阈值 | 每条事件最多 replay `10` 次 | `quad_experience_replay.py:48-55` |
| 碰撞退火 | `anneal_collision_steps` | `300_000_000` | `quad_obstacle_baseline.py:8-10`, `quad_utils.py:78-90` |
| 机间碰撞惩罚 | `quads_collision_reward` | `5.0` | `quad_obstacle_baseline.py:16-18` |
| 机间接近平滑惩罚上限 | `quads_collision_smooth_max_penalty` | `4.0` | 同上 |
| 障碍碰撞惩罚 | `quads_obst_collision_reward` | `5.0` | `quad_obstacle_baseline.py:19-21` |
| 机间碰撞判定半径 | `quads_collision_hitbox_radius` | `2.0` 个 arm length | `quad_obstacle_baseline.py:16-18` |
| 平滑 falloff 半径 | `quads_collision_falloff_radius` | `4.0` 个 arm length | 同上 |
| 空气动力学 | `quads_use_downwash` | `True` | `quad_obstacle_baseline.py:19-21` |
| 加速 | `quads_use_numba` | `True` | `quad_obstacle_baseline.py:8-10` |
| 场景模式 | `quads_mode` | `mix` | `quad_obstacle_baseline.py:11-12`, `scenarios/mix.py:37-94` |
| 单 episode 时长 | `quads_episode_duration` | `15.0s` | `quad_obstacle_baseline.py:11-12` |

## 4. 论文与代码的关键对齐/歧义

### 4.1 已对齐

1. `8 agents + 2 neighbors + 20% density + 0.6m obstacle size` 和论文 base setting 一致。  
2. 障碍观测虽然 CLI 叫 `octomap`，但底层实现实际是 `3x3` 局部 SDF 距离网格，和论文表述一致。  
3. replay 机制确实是“碰撞前 1.5 秒回放”。  
4. 主模型里的 attention 确实是 neighbor/obstacle embedding 之间的 multi-head attention。  

### 4.2 需要明确写出来的歧义

1. `mean_embed/attention/mlp/no_encoder` 这些是 **neighbor encoder** 选项；论文里的主 attention 其实由 `--quads_encoder_type=attention` 触发，对应 `QuadMultiHeadAttentionEncoder`，不是 `QuadNeighborhoodEncoderAttention`。  
2. 论文正文讲的是 SDF obstacle observation，代码 flag 名字是 `octomap`，两者语义一致但命名不同。  
3. 论文提到 curriculum，但仓库里没有单独的 curriculum scheduler 参数；更接近的实现是 `mix` 场景 + collision penalty annealing + replay。  
4. 论文给出 `flight distance` 和 `inference time`，但训练主日志里没有统一的自动统计字段，后面绘图部分要单独说明。  
5. `quadrotor_params.py` 对 `--quads_obst_size` 的 help 写的是“obstacle radius”，但 `MultiObstacles` 实际把它除以 2 存成 `obstacle_radius`；结合论文“0.6m obstacle size”和实现逻辑，更合理的理解是：**CLI 里的 `quads_obst_size` 表示障碍直径/边宽意义上的 size，而非半径。**

## 5. 最接近论文的单 seed 命令

```bash
python -m swarm_rl.train \
  --env=quadrotor_multi \
  --algo=APPO \
  --train_for_env_steps=1000000000 \
  --use_rnn=False \
  --num_workers=36 \
  --num_envs_per_worker=4 \
  --learning_rate=0.0001 \
  --ppo_clip_value=5.0 \
  --recurrence=1 \
  --nonlinearity=tanh \
  --actor_critic_share_weights=False \
  --policy_initialization=xavier_uniform \
  --adaptive_stddev=False \
  --with_vtrace=False \
  --max_policy_lag=100000000 \
  --rnn_size=256 \
  --gae_lambda=1.00 \
  --max_grad_norm=5.0 \
  --exploration_loss_coeff=0.0 \
  --rollout=128 \
  --batch_size=1024 \
  --with_pbt=False \
  --normalize_input=False \
  --normalize_returns=False \
  --reward_clip=10 \
  --quads_use_numba=True \
  --save_milestones_sec=3600 \
  --anneal_collision_steps=300000000 \
  --replay_buffer_sample_prob=0.75 \
  --quads_mode=mix \
  --quads_episode_duration=15.0 \
  --quads_obs_repr=xyz_vxyz_R_omega_floor \
  --quads_neighbor_hidden_size=256 \
  --quads_neighbor_obs_type=pos_vel \
  --quads_collision_hitbox_radius=2.0 \
  --quads_collision_falloff_radius=4.0 \
  --quads_collision_reward=5.0 \
  --quads_collision_smooth_max_penalty=4.0 \
  --quads_neighbor_encoder_type=no_encoder \
  --quads_neighbor_visible_num=2 \
  --quads_use_obstacles=True \
  --quads_obst_spawn_area 8 8 \
  --quads_obst_density=0.2 \
  --quads_obst_size=0.6 \
  --quads_obst_collision_reward=5.0 \
  --quads_obstacle_obs_type=octomap \
  --quads_use_downwash=True \
  --quads_num_agents=8 \
  --seed=0000 \
  --device=gpu \
  --train_dir=/abs/path/to/train_dir/paper_huang_obstacle/final \
  --experiment=seed_0000
```

## 6. 四台 4090 的并行策略

### 6.1 推荐策略

**推荐一张卡跑一个 seed，一共四个独立训练进程。**

原因：

1. 论文本来就是 `4 seeds`，按 seed 粒度并行最自然。  
2. Sample Factory 在这个仓库里的 run scripts 也是按多 seed 实验组织的，README 还直接给了 `--max_parallel=4 --experiments_per_gpu=1 --num_gpus=4` 的例子。  
3. 后续 paper 脚本是按目录聚合多个 seed 的 `tfevents`，一 seed 一实验目录最清晰。  
4. 这个任务不是数据并行单模型训练，没必要把 4 卡绑成一次多 GPU 训练。  

### 6.2 为什么不优先用单实验多 GPU

仓库当前提供的是：

- `swarm_rl/runs/*` 里按 seed 组织的 `RunDescription`
- `README.md:60-67` 里按多实验多 GPU 调度的 launcher 用法

没有发现专门为一个 APPO run 做单实验多 GPU 同步训练的纸面主线配置。因此，论文复现时优先使用“4 个独立 seed 并行”而不是“1 个 seed 占 4 卡”。

### 6.3 CPU 现实约束

原始 baseline 是 **每个 seed `36 workers x 4 envs`**。  
如果四个 seed 全部原样并行，就是整机 `144 workers` 级别的 CPU 开销。

因此建议分两档：

1. **论文优先档**：`NUM_WORKERS_PER_SEED=36`，`NUM_ENVS_PER_WORKER=4`
2. **机器现实档**：只下调 worker 数，其他任务参数不变，例如 `NUM_WORKERS_PER_SEED=8~16`

优先保持不变的参数应是：

- 8 agents
- 2 neighbors
- density 0.2
- obstacle size 0.6
- replay 0.75
- attention encoder
- episode duration 15s

## 7. 本次给出的可执行脚本

已新增：

- `train_paper_obstacle_4x4090.sh`

特点：

1. 默认按 `0000/1111/2222/3333` 四个 seed，同时起四个进程。  
2. 默认一张 4090 对应一个 seed。  
3. 默认目录布局是：

```text
train_dir/paper_huang_obstacle/final/
  seed_0000/
  seed_1111/
  seed_2222/
  seed_3333/
```

这样后续可以直接：

```bash
python paper/mean_std_plots_quad_obstacle.py \
  --path /home/server2/sui_work_not_delete/quad-swarm-rl/train_dir/paper_huang_obstacle/final
```

脚本默认尽量保持论文参数，同时允许通过环境变量覆盖：

- `NUM_WORKERS_PER_SEED`
- `NUM_ENVS_PER_WORKER`
- `TRAIN_ROOT`
- `CONDA_ENV`

## 8. 仓库原生 launcher 的替代方案

如果你更想复用仓库原生 run description，可以使用：

```bash
python -m sample_factory.launcher.run \
  --run=swarm_rl.runs.obstacles.quads_multi_obstacles \
  --backend=processes \
  --max_parallel=4 \
  --pause_between=1 \
  --experiments_per_gpu=1 \
  --num_gpus=4
```

但这个路径有两个现实问题：

1. `quads_multi_obstacles.py` 默认带 `with_wandb=True`
2. 输出目录命名不如显式 bash 脚本那样方便对接 paper 聚合脚本

所以论文复现主线仍建议优先使用新加的 `train_paper_obstacle_4x4090.sh`。
