# 中文注释副本；原始文件：swarm_rl/runs/obstacles/quad_obstacle_baseline.py
# 这个文件不是训练逻辑，而是论文障碍环境主实验的一条“基础命令模板”。
# 上游没有更多输入；下游会被其它 `runs/obstacles/*.py` 继续拼接，形成 attention、PBT、
# domain randomization 等不同实验族的最终 CLI。

# `QUAD_BASELINE_CLI_8` 集中定义论文障碍实验的默认训练超参数。
# 它把 Sample Factory 的通用训练参数、四旋翼环境参数、障碍物参数和回放/退火参数
# 固化成一条长命令，后续各个实验脚本只在此基础上覆写少数关键项。
QUAD_BASELINE_CLI_8 = (
    'python -m swarm_rl.train --env=quadrotor_multi --train_for_env_steps=1000000000 --algo=APPO --use_rnn=False '
    '--num_workers=36 --num_envs_per_worker=4 --learning_rate=0.0001 --ppo_clip_value=5.0 --recurrence=1 '
    '--nonlinearity=tanh --actor_critic_share_weights=False --policy_initialization=xavier_uniform '
    '--adaptive_stddev=False --with_vtrace=False --max_policy_lag=100000000 --rnn_size=256 '
    '--gae_lambda=1.00 --max_grad_norm=5.0 --exploration_loss_coeff=0.0 --rollout=128 --batch_size=1024 '
    '--with_pbt=False --normalize_input=False --normalize_returns=False --reward_clip=10 '
    # 这里固定基础训练实现细节，例如 numba 加速、里程碑保存频率和碰撞惩罚退火时长。
    '--quads_use_numba=True --save_milestones_sec=3600 --anneal_collision_steps=300000000 '
    # 回放概率是论文障碍实验能稳定训练的关键开关之一。
    '--replay_buffer_sample_prob=0.75 '
    # 场景层默认使用 `mix`，表示训练时会在多种 obstacle task 之间切换。
    '--quads_mode=mix --quads_episode_duration=15.0 '
    # 自观测采用位置、速度、旋转矩阵、角速度与 floor 信息的组合。
    '--quads_obs_repr=xyz_vxyz_R_omega_floor '
    # baseline 先禁用邻居观测内容，只保留 hitbox / falloff / smooth penalty 等碰撞相关约束。
    '--quads_neighbor_hidden_size=256 --quads_neighbor_obs_type=none --quads_collision_hitbox_radius=2.0 '
    '--quads_collision_falloff_radius=4.0 --quads_collision_reward=5.0 --quads_collision_smooth_max_penalty=4.0 '
    '--quads_neighbor_encoder_type=no_encoder --quads_neighbor_visible_num=2 '
    # 这里给出论文障碍主实验的默认地图密度、障碍尺寸、障碍观测形式和 downwash 开关。
    '--quads_use_obstacles=True --quads_obst_spawn_area 8 8 --quads_obst_density=0.2 --quads_obst_size=0.6 '
    '--quads_obst_collision_reward=5.0 --quads_obstacle_obs_type=octomap --quads_use_downwash=True'
)
