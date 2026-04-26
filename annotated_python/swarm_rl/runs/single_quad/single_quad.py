# 中文注释副本；原始文件：swarm_rl/runs/single_quad/single_quad.py
# 说明：为避免修改源码，本文件仅作为阅读辅助材料。

# 导入当前模块依赖。
from sample_factory.launcher.run_description import RunDescription, Experiment, ParamGrid
from swarm_rl.runs.single_quad.baseline import QUAD_BASELINE_CLI

# 保存或更新 `_params` 的值。
_params = ParamGrid([
    ('seed', [0000, 1111, 2222, 3333]),
])

# 保存或更新 `SINGLE_CLI` 的值。
SINGLE_CLI = QUAD_BASELINE_CLI + (
    ' --with_wandb=False --wandb_project=Quad-Swarm-RL --wandb_group=single --wandb_user=multi-drones'
)

# 保存或更新 `_experiment` 的值。
_experiment = Experiment(
    'single',
    SINGLE_CLI,
    _params.generate_params(randomize=False),
)

# 保存或更新 `RUN_DESCRIPTION` 的值。
RUN_DESCRIPTION = RunDescription('paper_quads_multi_mix_baseline_8a_attn_v116', experiments=[_experiment])

# Command to use this script on local machine: Please change num_workers to the physical cores of your local machine
# python -m sample_factory.launcher.run --run=swarm_rl.runs.quad_multi_mix_baseline --backend=processes --max_parallel=4 --pause_between=1 --experiments_per_gpu=1 --num_gpus=4
