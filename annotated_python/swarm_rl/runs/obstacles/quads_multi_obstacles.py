# 中文注释副本；原始文件：swarm_rl/runs/obstacles/quads_multi_obstacles.py
# 说明：为避免修改源码，本文件仅作为阅读辅助材料。

# 导入当前模块依赖。
from sample_factory.launcher.run_description import Experiment, ParamGrid, RunDescription
from swarm_rl.runs.obstacles.quad_obstacle_baseline import QUAD_BASELINE_CLI_8

# 保存或更新 `_params` 的值。
_params = ParamGrid(
    [
        ("seed", [0000, 1111, 2222, 3333]),
        ("quads_num_agents", [8]),
    ]
)

# 保存或更新 `OBSTACLE_MODEL_CLI` 的值。
OBSTACLE_MODEL_CLI = QUAD_BASELINE_CLI_8 + (
    ' --num_workers=36 --num_envs_per_worker=4 '
    '--quads_neighbor_visible_num=2 --quads_neighbor_obs_type=pos_vel --quads_encoder_type=attention '
    '--with_wandb=True --wandb_project=Quad-Swarm-RL --wandb_user=multi-drones '
    '--wandb_group=final'
)

# 保存或更新 `_experiment` 的值。
_experiment = Experiment(
    "final",
    OBSTACLE_MODEL_CLI,
    _params.generate_params(randomize=False),
)

# 保存或更新 `RUN_DESCRIPTION` 的值。
RUN_DESCRIPTION = RunDescription("obstacles_multi", experiments=[_experiment])
