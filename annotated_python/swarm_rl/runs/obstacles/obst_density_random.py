# 中文注释副本；原始文件：swarm_rl/runs/obstacles/obst_density_random.py
# 这个配置脚本把论文障碍基线扩展成“障碍密度随机化”实验。
# 其作用不是改环境代码，而是通过 CLI 把 obstacle density 采样范围交给训练入口，
# 观察策略在不同障碍密度下的泛化能力。

from sample_factory.launcher.run_description import Experiment, ParamGrid, RunDescription
from swarm_rl.runs.obstacles.quad_obstacle_baseline import QUAD_BASELINE_CLI_8

_params = ParamGrid(
    [
        ("seed", [0000, 1111, 2222, 3333]),
    ]
)

OBSTACLE_MODEL_CLI = QUAD_BASELINE_CLI_8 + (
    ' --num_workers=36 --num_envs_per_worker=4 --quads_num_agents=8 '
    '--quads_neighbor_visible_num=6 --quads_neighbor_obs_type=pos_vel --quads_encoder_type=attention '
    '--with_wandb=True --wandb_project=Quad-Swarm-RL --wandb_user=multi-drones '
    # 这里打开 density randomization，并把 obstacle density 采样区间限制在 0.05 到 0.2。
    '--quads_domain_random=True --quads_obst_density_random=True '
    '--quads_obst_density_min=0.05 --quads_obst_density_max=0.2 '
    '--wandb_group=obst_density_random'
)

_experiment = Experiment(
    "obst_density_random",
    OBSTACLE_MODEL_CLI,
    _params.generate_params(randomize=False),
)

RUN_DESCRIPTION = RunDescription("obstacles_multi", experiments=[_experiment])
