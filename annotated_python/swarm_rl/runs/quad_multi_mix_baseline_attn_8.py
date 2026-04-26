# 中文注释副本；原始文件：swarm_rl/runs/quad_multi_mix_baseline_attn_8.py
# 说明：为避免修改源码，本文件仅作为阅读辅助材料。
# 该文件属于强化学习训练侧逻辑，负责把环境、模型、配置或评估流程接到 Sample Factory 框架上。
# 这里产生的数据通常会继续流向训练循环、策略网络或实验分析脚本。

# 下面这组导入把当前模块会消费的环境组件、训练接口或数值工具集中拉进来；真正重要的是后续它们怎样参与数据流。
from sample_factory.launcher.run_description import RunDescription, Experiment, ParamGrid
from swarm_rl.runs.quad_multi_mix_baseline import QUAD_BASELINE_CLI_8

# 下面这组导入把当前模块会消费的环境组件、训练接口或数值工具集中拉进来；真正重要的是后续它们怎样参与数据流。
from swarm_rl.utils import timeStamped

_params = ParamGrid([
    ('seed', [0000, 1111, 2222, 3333]),
])

_experiment = Experiment(
    'quad_mix_baseline-8_mixed_attn',
    QUAD_BASELINE_CLI_8,
    _params.generate_params(randomize=False),
)

run_name = timeStamped("test_anneal", fmt="{fname}_%Y%m%d_%H%M")

RUN_DESCRIPTION = RunDescription(run_name, experiments=[_experiment])

# For scale, need to change
# num_workers / num_envs_per_worker && quads_num_agents
# num_workers * num_envs_per_worker * quads_num_agents should not change
