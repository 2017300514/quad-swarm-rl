# 中文注释副本；原始文件：swarm_rl/runs/quad_multi_mix_baseline_attn_8.py
# 说明：为避免修改源码，本文件仅作为阅读辅助材料。
# 这个脚本不是重新定义一套 CLI，而是在 `quad_multi_mix_baseline.py` 的基础上只展开多 seed。
# 也就是说，真正的训练语义沿用前一个文件；这里新增的主要是“多次独立重复实验”和“带时间戳的运行名”。

from sample_factory.launcher.run_description import RunDescription, Experiment, ParamGrid
from swarm_rl.runs.quad_multi_mix_baseline import QUAD_BASELINE_CLI_8

from swarm_rl.utils import timeStamped

# 这里真正被 sweep 的只有随机 seed，目的是看同一套 `mix` 基线在不同初始化下的波动。
_params = ParamGrid([
    ('seed', [0000, 1111, 2222, 3333]),
])

# `Experiment` 继续复用基线 CLI，只让 launcher 生成四个 seed 版本。
_experiment = Experiment(
    'quad_mix_baseline-8_mixed_attn',
    QUAD_BASELINE_CLI_8,
    _params.generate_params(randomize=False),
)

# 运行名带时间戳，主要是为了避免本地反复试跑时目录名冲突。
run_name = timeStamped("test_anneal", fmt="{fname}_%Y%m%d_%H%M")

RUN_DESCRIPTION = RunDescription(run_name, experiments=[_experiment])

# 这个备注是在提醒以后做规模扩展时的并行度守恒关系：
# 改 `num_workers` / `num_envs_per_worker` / `quads_num_agents` 时，最好让三者乘积保持近似不变，
# 这样每个 rollout 周期里累计的 agent-step 规模不会突然跳变太多。
