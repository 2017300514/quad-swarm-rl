# 中文注释副本；原始文件：swarm_rl/runs/obstacles/obst_size_random.py
# 说明：为避免修改源码，本文件仅作为阅读辅助材料。
# 该文件属于障碍场景实验配置，主要作用是把一组训练超参数打包成可复现实验入口。
# 这些配置本身不执行仿真，但会控制环境难度、观测结构、回放概率和模型结构选择。

# 下面这组导入把当前模块会消费的环境组件、训练接口或数值工具集中拉进来；真正重要的是后续它们怎样参与数据流。
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
    '--quads_domain_random=True --quads_obst_size_random=True '
    '--quads_obst_size_min=0.3 --quads_obst_size_max=0.6 '
    '--wandb_group=obst_size_random'
)

_experiment = Experiment(
    "obst_size_random",
    OBSTACLE_MODEL_CLI,
    _params.generate_params(randomize=False),
)

RUN_DESCRIPTION = RunDescription("obstacles_multi", experiments=[_experiment])
