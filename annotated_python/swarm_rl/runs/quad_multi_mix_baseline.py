# 中文注释副本；原始文件：swarm_rl/runs/quad_multi_mix_baseline.py
# 说明：为避免修改源码，本文件仅作为阅读辅助材料。
# 这个脚本是早期 8-agent `mix` 场景训练的 launcher 配置。
# 它本质上把一长串 `swarm_rl.train` CLI 固化成一个可由 Sample Factory launcher 批量提交的实验描述，
# 上游没有复杂逻辑，核心就是把“混合场景 + 8 机 + attention 邻居编码”的一套基线超参打包起来。

from sample_factory.launcher.run_description import RunDescription, Experiment, ParamGrid

# 这里的参数网格其实只保留了一个 collision reward 值，所以它更像“统一接口下的单点实验”，
# 而不是像 PBT 或大规模 sweep 那样真正展开很多组合。
_params = ParamGrid([
    ('quads_collision_reward', [5.0]),
])

# 这一长串 CLI 就是实验的真实语义主体：
# 前半段是 APPO / Sample Factory 训练预算与并行度，中段是多机 `mix` 场景和注意力邻居观测配置，
# 后半段是碰撞惩罚、replay、anneal 和保存频率等训练细节。
QUAD_BASELINE_CLI_8 = (
    'python -m swarm_rl.train --env=quadrotor_multi --train_for_env_steps=1000000000 --algo=APPO --use_rnn=False '
    '--num_workers=36 --num_envs_per_worker=4 --learning_rate=0.0001 --ppo_clip_value=5.0 --recurrence=1 '
    '--nonlinearity=tanh --actor_critic_share_weights=False --policy_initialization=xavier_uniform '
    '--adaptive_stddev=False --with_vtrace=False --max_policy_lag=100000000 --rnn_size=256 --with_pbt=False '
    '--gae_lambda=1.00 --max_grad_norm=5.0 --exploration_loss_coeff=0.0 --rollout=128 --batch_size=1024 '
    '--quads_use_numba=True --quads_num_agents=8 --quads_mode=mix --quads_episode_duration=15.0 '
    '--quads_neighbor_encoder_type=attention --quads_neighbor_hidden_size=256 --quads_neighbor_obs_type=pos_vel '
    '--quads_collision_reward=5.0 --quads_collision_hitbox_radius=2.0 --quads_collision_falloff_radius=4.0 '
    '--quads_collision_smooth_max_penalty=10.0 --quads_neighbor_visible_num=6 '
    '--replay_buffer_sample_prob=0.75 --anneal_collision_steps=300000000 --normalize_input=False '
    '--normalize_returns=False --reward_clip=10.0 --save_milestones_sec=3600'
)

# `Experiment` 把“实验名 + CLI 模板 + 参数网格”绑成 launcher 可以重复展开的最小单元。
_experiment = Experiment(
    'quad_mix_baseline-8_mixed',
    QUAD_BASELINE_CLI_8,
    _params.generate_params(randomize=False),
)

# 最外层 `RunDescription` 只负责给这组 experiment 一个批次名，供 launcher / 日志目录使用。
RUN_DESCRIPTION = RunDescription('quads_multi_mix_baseline_8a_local_v116', experiments=[_experiment])
