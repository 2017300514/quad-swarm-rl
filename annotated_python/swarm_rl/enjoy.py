# 中文注释副本；原始文件：swarm_rl/enjoy.py
# 说明：为避免修改源码，本文件仅作为阅读辅助材料。
# 该文件负责加载训练后的策略并驱动评估/可视化流程，用来检查训练出的多机避障与导航策略在环境中的实际表现。
# 它复用训练期的环境与模型注册逻辑，但下游不再执行参数更新，而是消费 checkpoint 做 rollout 和渲染。

# 下面这组导入把当前模块会消费的环境组件、训练接口或数值工具集中拉进来；真正重要的是后续它们怎样参与数据流。
import sys

# 下面这组导入把当前模块会消费的环境组件、训练接口或数值工具集中拉进来；真正重要的是后续它们怎样参与数据流。
from sample_factory.enjoy import enjoy

# 下面这组导入把当前模块会消费的环境组件、训练接口或数值工具集中拉进来；真正重要的是后续它们怎样参与数据流。
from swarm_rl.train import parse_swarm_cfg, register_swarm_components


# 这里串起训练脚本的顶层执行顺序：注册组件、解析配置、启动 RL 主循环。
# 如果任一步缺失，训练入口就无法把论文里的实验配置落到实际环境和模型上。
def main():
    # 下面的文档字符串通常由源码作者提供，用来补充模块职责、输入输出约束或使用方式。
    """Script entry point."""
    register_swarm_components()
    # 这里拿到的是训练或评估全流程共享的总配置对象，后续模型注册、环境创建和 PPO 超参数都会从中读取。
    cfg = parse_swarm_cfg(evaluation=True)
    # 训练返回状态汇总了 Sample Factory 主循环的执行结果，调用方用它判断本次任务是否正常结束。
    status = enjoy(cfg)
    # 这里把当前阶段整理好的结果交还给上层调用者；真正要理解的是返回值之后会进入哪条训练或仿真链路。
    return status


if __name__ == '__main__':
    sys.exit(main())
