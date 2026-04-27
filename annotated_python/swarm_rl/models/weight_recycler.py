# 中文注释副本；原始文件：swarm_rl/models/weight_recycler.py
# 说明：为避免修改源码，本文件仅作为阅读辅助材料。
# 这个小工具函数服务于“哪些神经元长期没在工作”的判断逻辑。
# 输入是一层神经元在一个 batch 或时间窗里的激活张量；输出是按神经元维度聚合后的平均绝对激活，
# 供上层决定哪些通道值得保留、哪些可能应该被回收或重置。

import torch
import numpy as np


def estimate_neuron_score(activation):
    """
    Calculates neuron score based on absolute value of activation.
    """
    # 最后一维通常对应神经元/通道本身，前面的维度则是 batch、时间或空间位置。
    # 这里把前面这些样本维全部平均掉，只保留“每个神经元整体有多活跃”的单个分数。
    reduce_axes = list(range(activation.dim() - 1))
    score = torch.mean(torch.abs(activation), dim=reduce_axes)

    # 注释掉的归一化说明作者曾考虑把活跃度转成相对分数；
    # 当前保留绝对均值，方便上层直接用阈值比较不同神经元的沉寂程度。
    # score /= torch.mean(score) + 1e-9

    return score
