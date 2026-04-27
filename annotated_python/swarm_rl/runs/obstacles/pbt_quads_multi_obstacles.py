# 中文注释副本；原始文件：swarm_rl/runs/obstacles/pbt_quads_multi_obstacles.py
# 这个脚本把障碍 attention 实验改造成 Population Based Training 版本。
# 上游同样依赖 `QUAD_BASELINE_CLI_8`；下游则生成一个启用 PBT 的超参搜索实验族，
# 用来在更长训练预算内自动扰动和淘汰策略。

from sample_factory.launcher.run_description import Experiment, ParamGrid, RunDescription
from swarm_rl.runs.obstacles.quad_obstacle_baseline import QUAD_BASELINE_CLI_8

_params = ParamGrid(
    [
        ("with_pbt", ["True"]),
    ]
)

OBSTACLE_MODEL_CLI = QUAD_BASELINE_CLI_8 + (
    # 这组参数控制 PBT 本身如何并行多策略、多久触发一次替换/变异，以及 gamma 是否参与优化。
    ' --num_policies=8 --pbt_mix_policies_in_one_env=True --pbt_period_env_steps=10000000 '
    '--pbt_start_mutation=50000000 --pbt_replace_reward_gap=0.2 --pbt_replace_reward_gap_absolute=3.0 '
    '--pbt_optimize_gamma=True --pbt_perturb_max=1.2 '
    # 这里顺便提高训练时长并关闭 collision annealing，给 PBT 充分的搜索空间。
    '--exploration_loss_coeff=0.0005 --max_entropy_coeff=0.0005 '
    '--anneal_collision_steps=0 --train_for_env_steps=10000000000 '
    # 由于同时训练多策略，worker 布局也相应改成 68x2。
    '--num_workers=68 --num_envs_per_worker=2 --quads_num_agents=8 '
    # 仍沿用论文障碍实验里的 attention + pos_vel 邻居观测主干。
    '--quads_neighbor_visible_num=6 --quads_neighbor_obs_type=pos_vel --quads_encoder_type=attention '
    '--with_wandb=True --wandb_project=Quad-Swarm-RL --wandb_user=multi-drones '
    '--wandb_group=pbt_obstacle_multi_attn_v2'
)

_experiment = Experiment(
    "pbt_obstacle_multi_attn_v2",
    OBSTACLE_MODEL_CLI,
    _params.generate_params(randomize=False),
)

RUN_DESCRIPTION = RunDescription("obstacles_multi", experiments=[_experiment])
