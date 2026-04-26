#!/usr/bin/env python
# 中文注释副本；原始文件：swarm_rl/models/attention_layer.py
# 说明：为避免修改源码，本文件仅作为阅读辅助材料。
# 该文件实现四旋翼模型里复用的注意力算子。
# 上游输入来自 `quad_multi_model.py` 已经编码好的 neighbor / obstacle token；
# 下游输出是经过 token 交互后的融合表示，以及可选的注意力权重矩阵。
# 训练主干和 sim2real 轻量分支都依赖这里，因此这里的张量形状约定会直接影响两条模型链路。

import torch
from torch import nn
import torch.nn.functional as F

"""
Implementation of Attention Block from https://github.com/jadore801120/attention-is-all-you-need-pytorch/blob/master/transformer/SubLayers.py
"""


class MultiHeadAttention(nn.Module):
    """ Multi-Head Attention module """

    def __init__(self, n_head, d_model, d_k, d_v):
        super().__init__()

        # `n_head` 决定并行注意力头数量；
        # `d_model / d_k / d_v` 决定每个 token 在进入注意力前后的特征宽度。
        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.w_qs = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_ks = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_vs = nn.Linear(d_model, n_head * d_v, bias=False)

        # 多头注意力的核心仍然是 scaled dot-product attention，
        # 这里只是把它包装成“多头投影 -> 逐头注意力 -> 拼接 -> 残差归一化”的标准结构。
        self.attention = ScaledDotProductAttention(temperature=d_k ** 0.5)
        self.fc = nn.Linear(n_head * d_v, d_model, bias=False)

        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)

    def forward(self, q, k, v, mask=None):
        # 输入张量约定为 `[batch, token_num, feature_dim]`。
        # 在本项目里，token_num 通常很小，例如只有 neighbor / obstacle 两个 token。
        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        size_b, len_q, len_k, len_v = q.size(0), q.size(1), k.size(1), v.size(1)

        residual = q

        # 先把每个 token 投影成多头版本的 q/k/v，再把 head 维度拆出来。
        q = self.w_qs(q).view(size_b, len_q, n_head, d_k)
        k = self.w_ks(k).view(size_b, len_k, n_head, d_k)
        v = self.w_vs(v).view(size_b, len_v, n_head, d_v)

        # 转置后就能按“每个 head 独立做注意力”的方式并行计算。
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

        q, attn = self.attention(q, k, v, mask=mask)

        # 逐头结果再拼回单个 token 表示，经过线性层、残差和 layer norm 得到最终输出。
        q = q.transpose(1, 2).contiguous().view(size_b, len_q, -1)
        q = self.fc(q)
        q += residual

        q = self.layer_norm(q)

        return q, attn


class OneHeadAttention(nn.Module):
    """ One-Head Attention module """

    def __init__(self, d_model):
        super().__init__()

        # 单头版本保留和多头版同样的 q/k/v -> residual -> norm 主线，
        # 但不再拆头，主要用于 sim2real 小模型。
        self.w_qs = nn.Linear(d_model, d_model, bias=False)
        self.w_ks = nn.Linear(d_model, d_model, bias=False)
        self.w_vs = nn.Linear(d_model, d_model, bias=False)

        self.fc = nn.Linear(d_model, d_model, bias=False)

        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)

        self.d_model = d_model

    def forward(self, q, k, v):
        # 这里直接在完整特征宽度上计算一次注意力，计算量更低，也更易部署。
        residual = q

        # Pass through the pre-attention projection: b x lq x dv
        q = self.w_qs(q)
        k = self.w_ks(k)
        v = self.w_vs(v)

        # 输出的 `attn` 仍然是 token 两两之间的权重矩阵，
        # 方便后续分析模型是在更关注邻居 token 还是障碍 token。
        attn = torch.matmul(q / (self.d_model ** 0.5), k.transpose(-1, -2))
        # attn /= torch.sqrt(self.d_model)
        attn = F.softmax(attn, dim=-1)
        q = torch.matmul(attn, v)

        q = self.fc(q)
        q += residual

        q = self.layer_norm(q)

        return q, attn


class ScaledDotProductAttention(nn.Module):
    """ Scaled Dot-Product Attention """

    def __init__(self, temperature):
        super().__init__()
        # `temperature` 一般取 `sqrt(d_k)`，用于稳定注意力打分尺度。
        self.temperature = temperature

    def forward(self, q, k, v, mask=None):
        # 这是最底层的注意力算子：
        # `softmax((QK^T) / sqrt(d_k)) V`。
        attn = torch.matmul(q / self.temperature, k.transpose(2, 3))

        if mask is not None:
            # 本项目主线里几乎不使用 mask，但保留这个接口可以兼容更一般的 token 屏蔽需求。
            attn = attn.masked_fill(mask == 0, -1e9)

        attn = F.softmax(attn, dim=-1)
        output = torch.matmul(attn, v)

        return output, attn
