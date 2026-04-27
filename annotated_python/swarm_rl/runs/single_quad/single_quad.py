# 中文注释副本；原始文件：swarm_rl/runs/single_quad/single_quad.py
# 这个脚本在单机 baseline 模板上补上多 seed 展开，形成真正可批量运行的单机实验入口。
# 上游依赖 `QUAD_BASELINE_CLI`；下游由 Sample Factory launcher 消费，
# 最终展开成 4 个随机种子的单机训练任务。

from sample_factory.launcher.run_description import RunDescription, Experiment, ParamGrid
from swarm_rl.runs.single_quad.baseline import QUAD_BASELINE_CLI

# 单机实验这里扫描的唯一维度是 seed，目的是看基础单机训练稳定性，而不是继续扫结构超参。
_params = ParamGrid([
    ('seed', [0000, 1111, 2222, 3333]),
])

# 在 baseline CLI 上只补充 WandB 相关标签。
# 这里把 `with_wandb=False` 固定掉，说明这份配置默认更偏本地复现或轻量验证，而非正式线上记录。
SINGLE_CLI = QUAD_BASELINE_CLI + (
    ' --with_wandb=False --wandb_project=Quad-Swarm-RL --wandb_group=single --wandb_user=multi-drones'
)

_experiment = Experiment(
    'single',
    SINGLE_CLI,
    _params.generate_params(randomize=False),
)

RUN_DESCRIPTION = RunDescription('paper_quads_multi_mix_baseline_8a_attn_v116', experiments=[_experiment])

# 这条注释保留的是源码作者给本地机器启动 launcher 的参考命令：
# 重点不是字面参数，而是提醒 `num_workers` 应按实际物理核心数调整。
# python -m sample_factory.launcher.run --run=swarm_rl.runs.quad_multi_mix_baseline --backend=processes --max_parallel=4 --pause_between=1 --experiments_per_gpu=1 --num_gpus=4
