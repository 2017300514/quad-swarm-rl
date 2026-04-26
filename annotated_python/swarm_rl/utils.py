# 中文注释副本；原始文件：swarm_rl/utils.py
# 说明：为避免修改源码，本文件仅作为阅读辅助材料。
# 该文件属于强化学习训练侧逻辑，负责把环境、模型、配置或评估流程接到 Sample Factory 框架上。
# 这里产生的数据通常会继续流向训练循环、策略网络或实验分析脚本。

# 下面这组导入把当前模块会消费的环境组件、训练接口或数值工具集中拉进来；真正重要的是后续它们怎样参与数据流。
import datetime
import random


# `timeStamped` 封装了当前模块中的一段独立流程，阅读时应重点关注它消费哪些状态、又把结果交给谁继续使用。
def timeStamped(fname, fmt='%Y-%m-%d-%H-%M-%S-{fname}'):
    # This creates a timestamped filename so we don't overwrite our good work
    # 这里把当前阶段整理好的结果交还给上层调用者；真正要理解的是返回值之后会进入哪条训练或仿真链路。
    return datetime.datetime.now().strftime(fmt).format(fname=fname)


# `generate_seeds` 封装了当前模块中的一段独立流程，阅读时应重点关注它消费哪些状态、又把结果交给谁继续使用。
def generate_seeds(num_seeds):
    # 这里把当前阶段整理好的结果交还给上层调用者；真正要理解的是返回值之后会进入哪条训练或仿真链路。
    return [random.randrange(0, 9999) for _ in range(num_seeds)]
