# 中文注释副本；原始文件：swarm_rl/runs/quad_multi_mix_baseline_attn_8.py
# 说明：为避免修改源码，本文件仅作为阅读辅助材料。

# 导入当前模块依赖。
from sample_factory.launcher.run_description import RunDescription, Experiment, ParamGrid
from swarm_rl.runs.quad_multi_mix_baseline import QUAD_BASELINE_CLI_8

# 导入当前模块依赖。
from swarm_rl.utils import timeStamped

# 保存或更新 `_params` 的值。
_params = ParamGrid([
    ('seed', [0000, 1111, 2222, 3333]),
])

# 保存或更新 `_experiment` 的值。
_experiment = Experiment(
    'quad_mix_baseline-8_mixed_attn',
    QUAD_BASELINE_CLI_8,
    _params.generate_params(randomize=False),
)

# 保存或更新 `run_name` 的值。
run_name = timeStamped("test_anneal", fmt="{fname}_%Y%m%d_%H%M")

# 保存或更新 `RUN_DESCRIPTION` 的值。
RUN_DESCRIPTION = RunDescription(run_name, experiments=[_experiment])

# For scale, need to change
# num_workers / num_envs_per_worker && quads_num_agents
# num_workers * num_envs_per_worker * quads_num_agents should not change
