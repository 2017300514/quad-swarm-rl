# 中文注释副本；原始文件：swarm_rl/runs/single_quad/baseline.py
# 说明：为避免修改源码，本文件仅作为阅读辅助材料。
# 该文件属于单机实验配置，用于验证基础飞行控制与奖励设计，而不是完整的多机协同避障。
# 其参数会流入训练入口，决定单机环境、模型和日志行为。

# 下面这组导入把当前模块会消费的环境组件、训练接口或数值工具集中拉进来；真正重要的是后续它们怎样参与数据流。
from sample_factory.launcher.run_description import RunDescription, Experiment, ParamGrid

_params = ParamGrid([
    ('quads_collision_reward', [5.0]),
])

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

RUN_DESCRIPTION = RunDescription('quads_multi_mix_baseline_8a_local_v116', experiments=[_experiment])
