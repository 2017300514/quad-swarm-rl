# 中文注释副本；原始文件：swarm_rl/models/weight_recycler.py
# 说明：为避免修改源码，本文件仅作为阅读辅助材料。

# 导入当前模块依赖。
import torch
import numpy as np


# 定义函数 `estimate_neuron_score`。
def estimate_neuron_score(activation):
    # 下面开始文档字符串说明。
    """
    Calculates neuron score based on absolute value of activation.
    """
    # 保存或更新 `reduce_axes` 的值。
    reduce_axes = list(range(activation.dim() - 1))
    # 保存或更新 `score` 的值。
    score = torch.mean(torch.abs(activation), dim=reduce_axes)
    # score /= torch.mean(score) + 1e-9

    # 返回当前函数的结果。
    return score


