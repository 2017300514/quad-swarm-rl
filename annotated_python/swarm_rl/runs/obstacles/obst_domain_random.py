# 中文注释副本；原始文件：swarm_rl/runs/obstacles/obst_domain_random.py
# 这个脚本是在障碍 attention 主配置上进一步同时随机化 obstacle density 与 obstacle size。
# 它服务的是更强的 domain randomization 版本，用来提升策略在不同障碍地图统计特征下的鲁棒性。

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
    # density 和 size 一起随机化，等于把障碍地图的“稠密程度”和“单体尺寸”同时打散。
    '--quads_domain_random=True --quads_obst_density_random=True --quads_obst_density_min=0.05 '
    '--quads_obst_density_max=0.2 --quads_obst_size_random=True --quads_obst_size_min=0.3 --quads_obst_size_max=0.6 '
    '--wandb_group=obst_domain_random'
)

_experiment = Experiment(
    "obst_domain_random",
    OBSTACLE_MODEL_CLI,
    _params.generate_params(randomize=False),
)

RUN_DESCRIPTION = RunDescription("obstacles_multi", experiments=[_experiment])
