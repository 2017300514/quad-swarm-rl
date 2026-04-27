# 中文注释副本；原始文件：swarm_rl/runs/obstacles/quads_multi_obstacles_nei_encoder_search.py
# 这个脚本不是论文最终配置，而是为“邻居编码器怎么做更合适”准备的搜索入口。
# 它沿用障碍 baseline，再横向比较 `attention` / `mean_embed` / `mlp` 三种邻居编码方式，
# 同时对不同可见邻居数量做小规模网格搜索。

from sample_factory.launcher.run_description import Experiment, ParamGrid, RunDescription
from swarm_rl.runs.obstacles.quad_obstacle_baseline import QUAD_BASELINE_CLI_8

_params = ParamGrid(
    [
        ("seed", [0000, 3333]),
        ("quads_neighbor_visible_num", [2, 6]),
        ("quads_neighbor_encoder_type", ['attention', 'mean_embed', 'mlp']),
    ]
)

# 注意这里虽然网格里会扫 `quads_neighbor_encoder_type`，
# 但通用编码器分支仍固定为 `quads_encoder_type=attention`，说明实验关注的是“邻居子编码器”变化，
# 而不是整个总网络结构彻底改写。
OBSTACLE_MODEL_CLI = QUAD_BASELINE_CLI_8 + (
    ' --num_workers=36 --num_envs_per_worker=4 --quads_num_agents=8 '
    '--quads_neighbor_obs_type=pos_vel --quads_encoder_type=attention '
    '--with_wandb=True --wandb_project=Quad-Swarm-RL --wandb_user=multi-drones '
    '--wandb_group=test_neighbor_encoder'
)

_experiment = Experiment(
    "test_neighbor_encoder",
    OBSTACLE_MODEL_CLI,
    _params.generate_params(randomize=False),
)

RUN_DESCRIPTION = RunDescription("obstacles_multi", experiments=[_experiment])
