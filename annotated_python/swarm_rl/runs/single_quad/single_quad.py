# 中文注释副本；原始文件：swarm_rl/runs/single_quad/single_quad.py
# 说明：为避免修改源码，本文件仅作为阅读辅助材料。
# 该文件属于单机实验配置，用于验证基础飞行控制与奖励设计，而不是完整的多机协同避障。
# 其参数会流入训练入口，决定单机环境、模型和日志行为。

# 下面这组导入把当前模块会消费的环境组件、训练接口或数值工具集中拉进来；真正重要的是后续它们怎样参与数据流。
from sample_factory.launcher.run_description import RunDescription, Experiment, ParamGrid
from swarm_rl.runs.single_quad.baseline import QUAD_BASELINE_CLI

_params = ParamGrid([
    ('seed', [0000, 1111, 2222, 3333]),
])

SINGLE_CLI = QUAD_BASELINE_CLI + (
    ' --with_wandb=False --wandb_project=Quad-Swarm-RL --wandb_group=single --wandb_user=multi-drones'
)

_experiment = Experiment(
    'single',
    SINGLE_CLI,
    _params.generate_params(randomize=False),
)

RUN_DESCRIPTION = RunDescription('paper_quads_multi_mix_baseline_8a_attn_v116', experiments=[_experiment])

# Command to use this script on local machine: Please change num_workers to the physical cores of your local machine
# python -m sample_factory.launcher.run --run=swarm_rl.runs.quad_multi_mix_baseline --backend=processes --max_parallel=4 --pause_between=1 --experiments_per_gpu=1 --num_gpus=4
