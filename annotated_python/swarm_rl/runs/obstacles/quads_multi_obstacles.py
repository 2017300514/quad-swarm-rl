# 中文注释副本；原始文件：swarm_rl/runs/obstacles/quads_multi_obstacles.py
# 这个脚本把障碍环境最终主实验注册成 Sample Factory 的一个 `RunDescription`。
# 上游依赖 `QUAD_BASELINE_CLI_8` 这条基础命令模板；下游由 launcher 读取后，
# 展开出多 seed 的 attention 障碍实验。

from sample_factory.launcher.run_description import Experiment, ParamGrid, RunDescription
from swarm_rl.runs.obstacles.quad_obstacle_baseline import QUAD_BASELINE_CLI_8

# 这里只扫多 seed，不再扫其它结构超参，说明它服务的是论文最终版配置的重复实验。
_params = ParamGrid(
    [
        ("seed", [0000, 1111, 2222, 3333]),
        ("quads_num_agents", [8]),
    ]
)

# 在 baseline 模板上显式打开邻居位置速度观测和 attention 编码器，
# 这正是论文障碍主实验里“障碍 SDF + 邻居感知 + attention”那条完整配置。
OBSTACLE_MODEL_CLI = QUAD_BASELINE_CLI_8 + (
    ' --num_workers=36 --num_envs_per_worker=4 '
    '--quads_neighbor_visible_num=2 --quads_neighbor_obs_type=pos_vel --quads_encoder_type=attention '
    '--with_wandb=True --wandb_project=Quad-Swarm-RL --wandb_user=multi-drones '
    '--wandb_group=final'
)

_experiment = Experiment(
    "final",
    OBSTACLE_MODEL_CLI,
    _params.generate_params(randomize=False),
)

# `RunDescription` 是 launcher 真正消费的入口：名称决定实验族，`experiments` 给出展开后的任务集合。
RUN_DESCRIPTION = RunDescription("obstacles_multi", experiments=[_experiment])
