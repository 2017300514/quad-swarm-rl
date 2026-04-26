#!/usr/bin/env python
# 中文注释副本；原始文件：swarm_rl/models/quad_multi_model.py
# 说明：为避免修改源码，本文件仅作为阅读辅助材料。
# 该文件定义四旋翼多机任务里的观测编码主干。
# 上游输入是环境已经按固定顺序拼好的观测向量：self、neighbor、obstacle 三段；
# 下游输出是 Sample Factory 策略头和价值头共用的隐表示。
# 论文里关于邻居编码、障碍编码和 attention 融合的核心实现，都集中在这里。

import torch
from torch import nn

from sample_factory.algo.utils.context import global_model_factory
from sample_factory.algo.utils.torch_utils import calc_num_elements
from sample_factory.model.encoder import Encoder
from sample_factory.model.model_utils import fc_layer, nonlinearity

from gym_art.quadrotor_multi.quad_utils import QUADS_OBS_REPR, QUADS_NEIGHBOR_OBS_TYPE, QUADS_OBSTACLE_OBS_TYPE

from swarm_rl.models.attention_layer import MultiHeadAttention, OneHeadAttention


class QuadNeighborhoodEncoder(nn.Module):
    # 这是邻居编码器的公共基类。
    # 它本身不实现具体聚合逻辑，只统一保存 self 维度、单邻居维度、隐藏宽度和可见邻居数量。
    def __init__(self, cfg, self_obs_dim, neighbor_obs_dim, neighbor_hidden_size, num_use_neighbor_obs):
        super().__init__()
        self.cfg = cfg
        self.self_obs_dim = self_obs_dim
        self.neighbor_obs_dim = neighbor_obs_dim
        self.neighbor_hidden_size = neighbor_hidden_size
        self.num_use_neighbor_obs = num_use_neighbor_obs


class QuadNeighborhoodEncoderDeepsets(QuadNeighborhoodEncoder):
    # 这一支对应“先独立编码每个邻居，再做均值聚合”的 DeepSets 风格邻居编码。
    def __init__(self, cfg, neighbor_obs_dim, neighbor_hidden_size, self_obs_dim, num_use_neighbor_obs):
        super().__init__(cfg, self_obs_dim, neighbor_obs_dim, neighbor_hidden_size, num_use_neighbor_obs)

        self.embedding_mlp = nn.Sequential(
            fc_layer(neighbor_obs_dim, neighbor_hidden_size),
            nonlinearity(cfg),
            fc_layer(neighbor_hidden_size, neighbor_hidden_size),
            nonlinearity(cfg)
        )

    def forward(self, self_obs, obs, all_neighbor_obs_size, batch_size):
        # 邻居观测在完整 observation 里是连续切片。
        # 这里先切出所有邻居槽位，再按“batch * neighbor_num”展平，逐邻居编码后做平均池化。
        obs_neighbors = obs[:, self.self_obs_dim:self.self_obs_dim + all_neighbor_obs_size]
        obs_neighbors = obs_neighbors.reshape(-1, self.neighbor_obs_dim)
        neighbor_embeds = self.embedding_mlp(obs_neighbors)
        neighbor_embeds = neighbor_embeds.reshape(batch_size, -1, self.neighbor_hidden_size)
        mean_embed = torch.mean(neighbor_embeds, dim=1)
        return mean_embed


class QuadNeighborhoodEncoderAttention(QuadNeighborhoodEncoder):
    # 这一支对应论文/相关工作里的邻居注意力编码：
    # 先给每个邻居做 embedding，再根据“该邻居 + 邻居总体上下文”算权重，最后做加权求和。
    def __init__(self, cfg, neighbor_obs_dim, neighbor_hidden_size, self_obs_dim, num_use_neighbor_obs):
        super().__init__(cfg, self_obs_dim, neighbor_obs_dim, neighbor_hidden_size, num_use_neighbor_obs)

        self.self_obs_dim = self_obs_dim

        # `embedding_mlp` 先把 self 与单邻居观测拼在一起，得到每个邻居的局部表示 `e_i`。
        self.embedding_mlp = nn.Sequential(
            fc_layer(self_obs_dim + neighbor_obs_dim, neighbor_hidden_size),
            nonlinearity(cfg),
            fc_layer(neighbor_hidden_size, neighbor_hidden_size),
            nonlinearity(cfg)
        )

        # `neighbor_value_mlp` 再把 `e_i` 映射成用于加权求和的 value 表示 `h_i`。
        self.neighbor_value_mlp = nn.Sequential(
            fc_layer(neighbor_hidden_size, neighbor_hidden_size),
            nonlinearity(cfg),
            fc_layer(neighbor_hidden_size, neighbor_hidden_size),
            nonlinearity(cfg),
        )

        # `attention_mlp` 根据单邻居表示和邻居均值上下文，给每个邻居打一个标量分数 `alpha_i`。
        self.attention_mlp = nn.Sequential(
            fc_layer(neighbor_hidden_size * 2, neighbor_hidden_size),
            # neighbor_hidden_size * 2 because we concat e_i and e_m
            nonlinearity(cfg),
            fc_layer(neighbor_hidden_size, neighbor_hidden_size),
            nonlinearity(cfg),
            fc_layer(neighbor_hidden_size, 1),
        )

    def forward(self, self_obs, obs, all_neighbor_obs_size, batch_size):
        obs_neighbors = obs[:, self.self_obs_dim:self.self_obs_dim + all_neighbor_obs_size]
        obs_neighbors = obs_neighbors.reshape(-1, self.neighbor_obs_dim)

        # 每个邻居的编码都显式拼上当前 agent 的 self 观测，
        # 这样注意力不是只看邻居自身状态，而是看“邻居相对当前无人机是否重要”。

        self_obs_repeat = self_obs.repeat(self.num_use_neighbor_obs, 1)
        mlp_input = torch.cat((self_obs_repeat, obs_neighbors), dim=1)
        neighbor_embeddings = self.embedding_mlp(mlp_input)  # e_i in the paper https://arxiv.org/pdf/1809.08835.pdf

        neighbor_values = self.neighbor_value_mlp(neighbor_embeddings)  # h_i in the paper

        neighbor_embeddings_mean_input = neighbor_embeddings.reshape(batch_size, -1, self.neighbor_hidden_size)
        # `e_m` 是所有邻居 embedding 的均值，用来提供“当前邻域整体长什么样”的上下文。
        neighbor_embeddings_mean = torch.mean(neighbor_embeddings_mean_input, dim=1)  # e_m in the paper
        neighbor_embeddings_mean_repeat = neighbor_embeddings_mean.repeat(self.num_use_neighbor_obs, 1)

        attention_mlp_input = torch.cat((neighbor_embeddings, neighbor_embeddings_mean_repeat), dim=1)
        attention_weights = self.attention_mlp(attention_mlp_input).view(batch_size, -1)  # alpha_i in the paper
        attention_weights_softmax = torch.nn.functional.softmax(attention_weights, dim=1)
        attention_weights_softmax = attention_weights_softmax.view(-1, 1)

        # softmax 后的权重把邻居 value 表示汇总成一个固定宽度向量，供策略后续与 self/obstacle 特征融合。
        final_neighborhood_embedding = attention_weights_softmax * neighbor_values
        final_neighborhood_embedding = final_neighborhood_embedding.view(batch_size, -1, self.neighbor_hidden_size)
        final_neighborhood_embedding = torch.sum(final_neighborhood_embedding, dim=1)

        return final_neighborhood_embedding


class QuadNeighborhoodEncoderMlp(QuadNeighborhoodEncoder):
    # 这一支最直接：把全部邻居槽位摊平成一个大向量，用 MLP 一次性压缩。
    # 它不具备排列不变性，主要作为对照或简化基线。
    def __init__(self, cfg, neighbor_obs_dim, neighbor_hidden_size, self_obs_dim, num_use_neighbor_obs):
        super().__init__(cfg, self_obs_dim, neighbor_obs_dim, neighbor_hidden_size, num_use_neighbor_obs)

        self.self_obs_dim = self_obs_dim

        self.neighbor_mlp = nn.Sequential(
            fc_layer(neighbor_obs_dim * num_use_neighbor_obs, neighbor_hidden_size),
            nonlinearity(cfg),
            fc_layer(neighbor_hidden_size, neighbor_hidden_size),
            nonlinearity(cfg),
            fc_layer(neighbor_hidden_size, neighbor_hidden_size),
            nonlinearity(cfg),
        )

    def forward(self, self_obs, obs, all_neighbor_obs_size, batch_size):
        obs_neighbors = obs[:, self.self_obs_dim:self.self_obs_dim + all_neighbor_obs_size]
        final_neighborhood_embedding = self.neighbor_mlp(obs_neighbors)
        return final_neighborhood_embedding


class QuadMultiHeadAttentionEncoder(Encoder):
    # 这是论文里更核心的编码器之一：
    # self、neighbor、obstacle 三段观测先各自编码，再把 neighbor 和 obstacle 当成两个 token 送入多头注意力融合。
    def __init__(self, cfg, obs_space):
        super().__init__(cfg)

        # self 观测维度直接由环境侧 `obs_repr` 决定。
        # 这一步必须和 `QuadrotorSingle.make_observation_space()` 保持一致，否则模型切片会错位。
        if cfg.quads_obs_repr in QUADS_OBS_REPR:
            self.self_obs_dim = QUADS_OBS_REPR[cfg.quads_obs_repr]
        else:
            raise NotImplementedError(f'Layer {cfg.quads_obs_repr} not supported!')

        self.neighbor_hidden_size = cfg.quads_neighbor_hidden_size
        self.use_obstacles = cfg.quads_use_obstacles

        if cfg.quads_neighbor_visible_num == -1:
            self.num_use_neighbor_obs = cfg.quads_num_agents - 1
        else:
            self.num_use_neighbor_obs = cfg.quads_neighbor_visible_num

        self.neighbor_obs_dim = QUADS_NEIGHBOR_OBS_TYPE[cfg.quads_neighbor_obs_type]

        self.all_neighbor_obs_dim = self.neighbor_obs_dim * self.num_use_neighbor_obs

        # 这里三条 embedding 支路分别吃三段观测：
        # 自身状态、邻居聚合输入、障碍物 9 维局部观测。
        fc_encoder_layer = cfg.rnn_size
        self.self_embed_layer = nn.Sequential(
            fc_layer(self.self_obs_dim, fc_encoder_layer),
            nonlinearity(cfg),
            fc_layer(fc_encoder_layer, fc_encoder_layer),
            nonlinearity(cfg)
        )
        self.neighbor_embed_layer = nn.Sequential(
            fc_layer(self.all_neighbor_obs_dim, fc_encoder_layer),
            nonlinearity(cfg),
            fc_layer(fc_encoder_layer, fc_encoder_layer),
            nonlinearity(cfg)
        )
        self.obstacle_obs_dim = QUADS_OBSTACLE_OBS_TYPE[cfg.quads_obstacle_obs_type]
        self.obstacle_embed_layer = nn.Sequential(
            fc_layer(self.obstacle_obs_dim, fc_encoder_layer),
            nonlinearity(cfg),
            fc_layer(fc_encoder_layer, fc_encoder_layer),
            nonlinearity(cfg)
        )

        # 注意力层这里只处理两个 token：neighbor token 和 obstacle token。
        # self token 不进入注意力，而是在最后和注意力输出直接拼接。
        self.attention_layer = MultiHeadAttention(4, cfg.rnn_size, cfg.rnn_size, cfg.rnn_size)

        # MLP Layer
        self.encoder_output_size = 2 * cfg.rnn_size
        self.feed_forward = nn.Sequential(fc_layer(3 * cfg.rnn_size, self.encoder_output_size),
                                          nn.Tanh())

    def forward(self, obs_dict):
        # 完整 observation 的切片顺序必须和环境拼接顺序一一对应：
        # `[self | neighbors | obstacles]`。
        obs = obs_dict['obs']
        batch_size = obs.shape[0]
        obs_self = obs[:, :self.self_obs_dim]
        obs_neighbor = obs[:, self.self_obs_dim: self.self_obs_dim + self.all_neighbor_obs_dim]
        obs_obstacle = obs[:, self.self_obs_dim + self.all_neighbor_obs_dim:]

        self_embed = self.self_embed_layer(obs_self)
        neighbor_embed = self.neighbor_embed_layer(obs_neighbor)
        obstacle_embed = self.obstacle_embed_layer(obs_obstacle)
        neighbor_embed = neighbor_embed.view(batch_size, 1, -1)
        obstacle_embed = obstacle_embed.view(batch_size, 1, -1)
        attn_embed = torch.cat((neighbor_embed, obstacle_embed), dim=1)

        # neighbor/obstacle 两个 token 在这里互相做注意力，
        # 让网络自行决定“当前决策更该关注邻居威胁还是障碍威胁”。
        attn_embed, attn_score = self.attention_layer(attn_embed, attn_embed, attn_embed)
        attn_embed = attn_embed.view(batch_size, -1)

        # 最终输出仍显式保留 self 表示，
        # 避免注意力只在外部实体之间流动后丢掉当前无人机自身动力学状态。
        embeddings = torch.cat((self_embed, attn_embed), dim=1)
        out = self.feed_forward(embeddings)

        return out

    def get_out_size(self):
        return self.encoder_output_size


class QuadSingleHeadAttentionEncoder_Sim2Real(QuadMultiHeadAttentionEncoder):
    # 这是 sim2real 分支的轻量版注意力编码器。
    # 结构更小、注意力头更少，目的是为部署约束让路，而不是追求训练期最大表达能力。
    def __init__(self, cfg, obs_space):
        super().__init__(cfg, obs_space)

        # Internal params
        if cfg.quads_obs_repr in QUADS_OBS_REPR:
            self.self_obs_dim = QUADS_OBS_REPR[cfg.quads_obs_repr]
        else:
            raise NotImplementedError(f'Layer {cfg.quads_obs_repr} not supported!')

        self.neighbor_hidden_size = cfg.quads_neighbor_hidden_size
        self.use_obstacles = cfg.quads_use_obstacles

        if cfg.quads_neighbor_visible_num == -1:
            self.num_use_neighbor_obs = cfg.quads_num_agents - 1
        else:
            self.num_use_neighbor_obs = cfg.quads_neighbor_visible_num

        self.neighbor_obs_dim = QUADS_NEIGHBOR_OBS_TYPE[cfg.quads_neighbor_obs_type]

        self.all_neighbor_obs_dim = self.neighbor_obs_dim * self.num_use_neighbor_obs

        # sim2real 版本把 embedding 支路压得更浅，便于后续导出和板载执行。
        fc_encoder_layer = cfg.rnn_size
        self.self_embed_layer = nn.Sequential(
            fc_layer(self.self_obs_dim, fc_encoder_layer),
            nonlinearity(cfg),
        )
        self.neighbor_embed_layer = nn.Sequential(
            fc_layer(self.all_neighbor_obs_dim, fc_encoder_layer),
            nonlinearity(cfg),
        )
        self.obstacle_obs_dim = QUADS_OBSTACLE_OBS_TYPE[cfg.quads_obstacle_obs_type]
        self.obstacle_embed_layer = nn.Sequential(
            fc_layer(self.obstacle_obs_dim, fc_encoder_layer),
            nonlinearity(cfg),
        )

        # 只保留单头注意力，减少推理复杂度。
        self.attention_layer = OneHeadAttention(cfg.rnn_size)

        # MLP Layer
        self.encoder_output_size = cfg.rnn_size
        self.feed_forward = nn.Sequential(fc_layer(3 * cfg.rnn_size, self.encoder_output_size),
                                          nn.Tanh())


class QuadMultiEncoder(Encoder):
    # Mean embedding encoder based on the DeepRL for Swarms Paper
    # 这是非 `quads_encoder_type=attention` 分支下的通用编码器。
    # 它支持多种邻居编码后端，再把 self / neighbor / obstacle 三段结果拼起来送入统一前馈层。
    def __init__(self, cfg, obs_space):
        super().__init__(cfg)

        self.self_obs_dim = QUADS_OBS_REPR[cfg.quads_obs_repr]
        self.use_obstacles = cfg.quads_use_obstacles

        # 邻居部分的真实输入宽度由“每个邻居多少维”乘以“保留多少个邻居槽位”决定。
        neighbor_hidden_size = cfg.quads_neighbor_hidden_size
        neighbor_obs_dim = QUADS_NEIGHBOR_OBS_TYPE[cfg.quads_neighbor_obs_type]

        if cfg.quads_neighbor_obs_type == 'none':
            num_use_neighbor_obs = 0
        else:
            if cfg.quads_neighbor_visible_num == -1:
                num_use_neighbor_obs = cfg.quads_num_agents - 1
            else:
                num_use_neighbor_obs = cfg.quads_neighbor_visible_num

        self.all_neighbor_obs_size = neighbor_obs_dim * num_use_neighbor_obs

        # 按实验配置选择邻居编码器实现。
        # 这里决定的是“邻居信息如何聚合”，不是环境里邻居看不看得见。
        neighbor_encoder_out_size = 0
        self.neighbor_encoder = None

        if num_use_neighbor_obs > 0:
            neighbor_encoder_type = cfg.quads_neighbor_encoder_type
            if neighbor_encoder_type == 'mean_embed':
                self.neighbor_encoder = QuadNeighborhoodEncoderDeepsets(
                    cfg=cfg, neighbor_obs_dim=neighbor_obs_dim, neighbor_hidden_size=neighbor_hidden_size,
                    self_obs_dim=self.self_obs_dim, num_use_neighbor_obs=num_use_neighbor_obs)
            elif neighbor_encoder_type == 'attention':
                self.neighbor_encoder = QuadNeighborhoodEncoderAttention(
                    cfg=cfg, neighbor_obs_dim=neighbor_obs_dim, neighbor_hidden_size=neighbor_hidden_size,
                    self_obs_dim=self.self_obs_dim, num_use_neighbor_obs=num_use_neighbor_obs)
            elif neighbor_encoder_type == 'mlp':
                self.neighbor_encoder = QuadNeighborhoodEncoderMlp(
                    cfg=cfg, neighbor_obs_dim=neighbor_obs_dim, neighbor_hidden_size=neighbor_hidden_size,
                    self_obs_dim=self.self_obs_dim, num_use_neighbor_obs=num_use_neighbor_obs)
            elif neighbor_encoder_type == 'no_encoder':
                # 明确关闭邻居编码，相当于策略只看 self 和可能存在的障碍观测。
                self.neighbor_encoder = None
            else:
                raise NotImplementedError

        if self.neighbor_encoder:
            neighbor_encoder_out_size = neighbor_hidden_size

        fc_encoder_layer = cfg.rnn_size
        # self 分支始终存在，因为每种策略至少都要知道本机位置、速度、姿态等自状态。
        self.self_encoder = nn.Sequential(
            fc_layer(self.self_obs_dim, fc_encoder_layer),
            nonlinearity(cfg),
            fc_layer(fc_encoder_layer, fc_encoder_layer),
            nonlinearity(cfg)
        )
        self_encoder_out_size = calc_num_elements(self.self_encoder, (self.self_obs_dim,))

        # 障碍分支只有在环境真的拼接了障碍观测时才启用。
        # 这一步的输入维度要和 `MultiObstacles` 输出的障碍观测切片一致。
        obstacle_encoder_out_size = 0
        if self.use_obstacles:
            obstacle_obs_dim = QUADS_OBSTACLE_OBS_TYPE[cfg.quads_obstacle_obs_type]
            obstacle_hidden_size = cfg.quads_obst_hidden_size
            self.obstacle_encoder = nn.Sequential(
                fc_layer(obstacle_obs_dim, obstacle_hidden_size),
                nonlinearity(cfg),
                fc_layer(obstacle_hidden_size, obstacle_hidden_size),
                nonlinearity(cfg),
            )
            obstacle_encoder_out_size = calc_num_elements(self.obstacle_encoder, (obstacle_obs_dim,))

        total_encoder_out_size = self_encoder_out_size + neighbor_encoder_out_size + obstacle_encoder_out_size

        # 这里的输出不会直接变成动作，而是继续流向 Sample Factory 的策略头/价值头。
        # 因此先统一投到 `2 * rnn_size`，给后续 actor-critic 参数化提供稳定宽度。
        self.feed_forward = nn.Sequential(
            fc_layer(total_encoder_out_size, 2 * cfg.rnn_size),
            nn.Tanh(),
        )

        self.encoder_out_size = 2 * cfg.rnn_size

    def forward(self, obs_dict):
        # 编码主线是：
        # self 先编码；如果启用邻居，则追加邻居聚合向量；如果启用障碍，则再追加障碍 embedding。
        obs = obs_dict['obs']
        obs_self = obs[:, :self.self_obs_dim]
        self_embed = self.self_encoder(obs_self)
        embeddings = self_embed
        batch_size = obs_self.shape[0]
        # Relative xyz and vxyz for the Entire Minibatch (batch dimension is batch_size * num_neighbors)
        if self.neighbor_encoder:
            neighborhood_embedding = self.neighbor_encoder(obs_self, obs, self.all_neighbor_obs_size, batch_size)
            embeddings = torch.cat((embeddings, neighborhood_embedding), dim=1)

        if self.use_obstacles:
            obs_obstacles = obs[:, self.self_obs_dim + self.all_neighbor_obs_size:]
            obstacle_embeds = self.obstacle_encoder(obs_obstacles)
            embeddings = torch.cat((embeddings, obstacle_embeds), dim=1)

        out = self.feed_forward(embeddings)
        return out

    def get_out_size(self) -> int:
        return self.encoder_out_size


def make_quadmulti_encoder(cfg, obs_space) -> Encoder:
    # 这是注册给 Sample Factory 的统一编码器工厂。
    # 它根据实验配置在 attention 主干和普通多分支编码器之间切换，并在 sim2real 时选轻量版 attention。
    if cfg.quads_encoder_type == "attention":
        if cfg.quads_sim2real:
            model = QuadSingleHeadAttentionEncoder_Sim2Real(cfg, obs_space)
        else:
            model = QuadMultiHeadAttentionEncoder(cfg, obs_space)
    else:
        model = QuadMultiEncoder(cfg, obs_space)
    return model


def register_models():
    # 训练入口调用这里后，Sample Factory 才能通过配置名字实例化本项目自定义编码器。
    global_model_factory().register_encoder_factory(make_quadmulti_encoder)
