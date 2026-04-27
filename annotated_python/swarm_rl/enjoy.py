# 中文注释副本；原始文件：swarm_rl/enjoy.py
# 说明：为避免修改源码，本文件仅作为阅读辅助材料。
# 这是训练后的评估入口。它不会像 `train.py` 一样启动 APPO 更新，而是复用同一套环境/模型注册与参数解析逻辑，
# 然后把配置交给 Sample Factory 的 `enjoy()`，用 checkpoint 执行 rollout、渲染或人工验收。

import sys

from sample_factory.enjoy import enjoy

from swarm_rl.train import parse_swarm_cfg, register_swarm_components


def main():
    """Script entry point."""
    # 评估入口仍然必须先注册环境和自定义模型，否则 Sample Factory 在恢复 checkpoint 时找不到这些名字。
    register_swarm_components()

    # `evaluation=True` 会让配置解析走评估分支：重点变成加载已有实验、恢复策略并运行可视化/统计，
    # 而不是生成训练 worker 和优化器。
    cfg = parse_swarm_cfg(evaluation=True)

    # 下游 `sample_factory.enjoy.enjoy` 会消费这份配置去恢复 checkpoint、创建环境并执行推理阶段 rollout。
    status = enjoy(cfg)
    return status


if __name__ == '__main__':
    sys.exit(main())
