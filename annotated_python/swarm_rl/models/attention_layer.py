# 中文注释副本；原始文件：swarm_rl/models/attention_layer.py
# 说明：为避免修改源码，本文件仅作为阅读辅助材料。

# 导入当前模块依赖。
import torch
from torch import nn
import torch.nn.functional as F

# 下面开始文档字符串说明。
"""
Implementation of Attention Block from https://github.com/jadore801120/attention-is-all-you-need-pytorch/blob/master/transformer/SubLayers.py
"""


# 定义类 `MultiHeadAttention`。
class MultiHeadAttention(nn.Module):
    # 下面的文档字符串用于说明当前模块或代码块。
    """ Multi-Head Attention module """

    # 定义函数 `__init__`。
    def __init__(self, n_head, d_model, d_k, d_v):
        # 调用 `super` 执行当前处理。
        super().__init__()

        # 保存或更新 `n_head` 的值。
        self.n_head = n_head
        # 保存或更新 `d_k` 的值。
        self.d_k = d_k
        # 保存或更新 `d_v` 的值。
        self.d_v = d_v

        # 保存或更新 `w_qs` 的值。
        self.w_qs = nn.Linear(d_model, n_head * d_k, bias=False)
        # 保存或更新 `w_ks` 的值。
        self.w_ks = nn.Linear(d_model, n_head * d_k, bias=False)
        # 保存或更新 `w_vs` 的值。
        self.w_vs = nn.Linear(d_model, n_head * d_v, bias=False)

        # 保存或更新 `attention` 的值。
        self.attention = ScaledDotProductAttention(temperature=d_k ** 0.5)
        # 保存或更新 `fc` 的值。
        self.fc = nn.Linear(n_head * d_v, d_model, bias=False)

        # 保存或更新 `layer_norm` 的值。
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)

    # 定义函数 `forward`。
    def forward(self, q, k, v, mask=None):
        # 同时更新 `d_k`, `d_v`, `n_head` 等变量。
        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        # 同时更新 `size_b`, `len_q`, `len_k`, `len_v` 等变量。
        size_b, len_q, len_k, len_v = q.size(0), q.size(1), k.size(1), v.size(1)

        # 保存或更新 `residual` 的值。
        residual = q

        # Pass through the pre-attention projection: b x lq x (n*dv)
        # Separate different heads: b x lq x n x dv
        # 保存或更新 `q` 的值。
        q = self.w_qs(q).view(size_b, len_q, n_head, d_k)
        # 保存或更新 `k` 的值。
        k = self.w_ks(k).view(size_b, len_k, n_head, d_k)
        # 保存或更新 `v` 的值。
        v = self.w_vs(v).view(size_b, len_v, n_head, d_v)

        # Transpose for attention dot product: b x n x lq x dv
        # 同时更新 `q`, `k`, `v` 等变量。
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

        # 同时更新 `q`, `attn` 等变量。
        q, attn = self.attention(q, k, v, mask=mask)

        # Transpose to move the head dimension back: b x lq x n x dv
        # Combine the last two dimensions to concatenate all the heads together: b x lq x (n*dv)
        # 保存或更新 `q` 的值。
        q = q.transpose(1, 2).contiguous().view(size_b, len_q, -1)
        # 保存或更新 `q` 的值。
        q = self.fc(q)
        # 保存或更新 `q` 的值。
        q += residual

        # 保存或更新 `q` 的值。
        q = self.layer_norm(q)

        # 返回当前函数的结果。
        return q, attn


# 定义类 `OneHeadAttention`。
class OneHeadAttention(nn.Module):
    # 下面的文档字符串用于说明当前模块或代码块。
    """ One-Head Attention module """

    # 定义函数 `__init__`。
    def __init__(self, d_model):
        # 调用 `super` 执行当前处理。
        super().__init__()

        # 保存或更新 `w_qs` 的值。
        self.w_qs = nn.Linear(d_model, d_model, bias=False)
        # 保存或更新 `w_ks` 的值。
        self.w_ks = nn.Linear(d_model, d_model, bias=False)
        # 保存或更新 `w_vs` 的值。
        self.w_vs = nn.Linear(d_model, d_model, bias=False)

        # 保存或更新 `fc` 的值。
        self.fc = nn.Linear(d_model, d_model, bias=False)

        # 保存或更新 `layer_norm` 的值。
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)

        # 保存或更新 `d_model` 的值。
        self.d_model = d_model

    # 定义函数 `forward`。
    def forward(self, q, k, v):
        # 保存或更新 `residual` 的值。
        residual = q

        # Pass through the pre-attention projection: b x lq x dv
        # 保存或更新 `q` 的值。
        q = self.w_qs(q)
        # 保存或更新 `k` 的值。
        k = self.w_ks(k)
        # 保存或更新 `v` 的值。
        v = self.w_vs(v)

        # Compute attention weights using queries and keys
        # 保存或更新 `attn` 的值。
        attn = torch.matmul(q / (self.d_model ** 0.5), k.transpose(-1, -2))
        # attn /= torch.sqrt(self.d_model)
        # 保存或更新 `attn` 的值。
        attn = F.softmax(attn, dim=-1)
        # 保存或更新 `q` 的值。
        q = torch.matmul(attn, v)

        # 保存或更新 `q` 的值。
        q = self.fc(q)
        # 保存或更新 `q` 的值。
        q += residual

        # 保存或更新 `q` 的值。
        q = self.layer_norm(q)

        # 返回当前函数的结果。
        return q, attn


# 定义类 `ScaledDotProductAttention`。
class ScaledDotProductAttention(nn.Module):
    # 下面的文档字符串用于说明当前模块或代码块。
    """ Scaled Dot-Product Attention """

    # 定义函数 `__init__`。
    def __init__(self, temperature):
        # 调用 `super` 执行当前处理。
        super().__init__()
        # 保存或更新 `temperature` 的值。
        self.temperature = temperature

    # 定义函数 `forward`。
    def forward(self, q, k, v, mask=None):
        # 保存或更新 `attn` 的值。
        attn = torch.matmul(q / self.temperature, k.transpose(2, 3))

        # 根据条件决定是否进入当前分支。
        if mask is not None:
            # 执行这一行逻辑。
            attn = attn.masked_fill(mask == 0, -1e9)

        # 保存或更新 `attn` 的值。
        attn = F.softmax(attn, dim=-1)
        # 保存或更新 `output` 的值。
        output = torch.matmul(attn, v)

        # 返回当前函数的结果。
        return output, attn
