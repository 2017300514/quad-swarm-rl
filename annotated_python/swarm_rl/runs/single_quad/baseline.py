# 中文注释副本；原始文件：swarm_rl/runs/single_quad/baseline.py
# 这个文件定义的是单机四旋翼实验的基础命令模板。
# 它和论文里的多机障碍主实验不同：这里把 `quads_num_agents` 固定成 1，
# 主要用于验证单机版本的控制、奖励和训练配置，而不是群体协同与避障能力。

from sample_factory.launcher.run_description import RunDescription, Experiment, ParamGrid

# 单机 baseline 这里只显式保留碰撞奖励一个扫描位，说明它更像“最小化配置模板”，
# 方便后续在本地或小规模调试时快速改动某个单项超参。
_params = ParamGrid([
    ('quads_collision_reward', [5.0]),
])

# `QUAD_BASELINE_CLI` 把单机训练需要的通用 Sample Factory 参数和环境参数串成一条命令。
# 关键变化是：
# 1. `quads_num_agents=1`，彻底退化成单机环境；
# 2. `quads_mode=static_same_goal`，场景最简单，便于看基础飞行是否稳定；
# 3. 邻居编码相关项全部关闭，因为单机时不存在 neighbor 观测。
QUAD_BASELINE_CLI = (
    'python -m swarm_rl.train --env=quadrotor_multi --train_for_env_steps=1000000000 --algo=APPO --use_rnn=False '
    '--num_workers=2 --num_envs_per_worker=8 --learning_rate=0.0001 --ppo_clip_value=5.0 --recurrence=1 '
    '--nonlinearity=tanh --actor_critic_share_weights=False --policy_initialization=xavier_uniform '
    '--adaptive_stddev=False --with_vtrace=False --max_policy_lag=100000000 --rnn_size=256 --with_pbt=False '
    '--gae_lambda=1.00 --max_grad_norm=5.0 --exploration_loss_coeff=0.0 --rollout=128 --batch_size=1024 '
    '--quads_use_numba=True --quads_num_agents=1 --quads_mode=static_same_goal --quads_episode_duration=15.0 '
    '--quads_neighbor_encoder_type=no_encoder --quads_neighbor_hidden_size=0 --quads_neighbor_obs_type=none '
    '--quads_neighbor_visible_num=0 --replay_buffer_sample_prob=0.75 --anneal_collision_steps=300000000 '
    '--normalize_input=False --normalize_returns=False --reward_clip=10.0 --save_milestones_sec=3600'
)

_experiment = Experiment(
    'quad_mix_baseline-8_mixed',
    QUAD_BASELINE_CLI,
    _params.generate_params(randomize=False),
)

# `RunDescription` 是 launcher 实际读取的入口，名称主要用于区分这一组单机 baseline 任务。
RUN_DESCRIPTION = RunDescription('quads_multi_mix_baseline_8a_local_v116', experiments=[_experiment])
