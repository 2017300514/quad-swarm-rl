# 中文注释副本；原始文件：swarm_rl/models/quad_multi_model.py
# 说明：为避免修改源码，本文件仅作为阅读辅助材料。

# 导入当前模块依赖。
import torch
from torch import nn

# 导入当前模块依赖。
from sample_factory.algo.utils.context import global_model_factory
from sample_factory.algo.utils.torch_utils import calc_num_elements
from sample_factory.model.encoder import Encoder
from sample_factory.model.model_utils import fc_layer, nonlinearity

# 导入当前模块依赖。
from gym_art.quadrotor_multi.quad_utils import QUADS_OBS_REPR, QUADS_NEIGHBOR_OBS_TYPE, QUADS_OBSTACLE_OBS_TYPE

# 导入当前模块依赖。
from swarm_rl.models.attention_layer import MultiHeadAttention, OneHeadAttention


# 定义类 `QuadNeighborhoodEncoder`。
class QuadNeighborhoodEncoder(nn.Module):
    # 定义函数 `__init__`。
    def __init__(self, cfg, self_obs_dim, neighbor_obs_dim, neighbor_hidden_size, num_use_neighbor_obs):
        # 调用 `super` 执行当前处理。
        super().__init__()
        # 保存或更新 `cfg` 的值。
        self.cfg = cfg
        # 保存或更新 `self_obs_dim` 的值。
        self.self_obs_dim = self_obs_dim
        # 保存或更新 `neighbor_obs_dim` 的值。
        self.neighbor_obs_dim = neighbor_obs_dim
        # 保存或更新 `neighbor_hidden_size` 的值。
        self.neighbor_hidden_size = neighbor_hidden_size
        # 保存或更新 `num_use_neighbor_obs` 的值。
        self.num_use_neighbor_obs = num_use_neighbor_obs


# 定义类 `QuadNeighborhoodEncoderDeepsets`。
class QuadNeighborhoodEncoderDeepsets(QuadNeighborhoodEncoder):
    # 定义函数 `__init__`。
    def __init__(self, cfg, neighbor_obs_dim, neighbor_hidden_size, self_obs_dim, num_use_neighbor_obs):
        # 调用 `super` 执行当前处理。
        super().__init__(cfg, self_obs_dim, neighbor_obs_dim, neighbor_hidden_size, num_use_neighbor_obs)

        # 保存或更新 `embedding_mlp` 的值。
        self.embedding_mlp = nn.Sequential(
            fc_layer(neighbor_obs_dim, neighbor_hidden_size),
            nonlinearity(cfg),
            fc_layer(neighbor_hidden_size, neighbor_hidden_size),
            nonlinearity(cfg)
        )

    # 定义函数 `forward`。
    def forward(self, self_obs, obs, all_neighbor_obs_size, batch_size):
        # 保存或更新 `obs_neighbors` 的值。
        obs_neighbors = obs[:, self.self_obs_dim:self.self_obs_dim + all_neighbor_obs_size]
        # 保存或更新 `obs_neighbors` 的值。
        obs_neighbors = obs_neighbors.reshape(-1, self.neighbor_obs_dim)
        # 保存或更新 `neighbor_embeds` 的值。
        neighbor_embeds = self.embedding_mlp(obs_neighbors)
        # 保存或更新 `neighbor_embeds` 的值。
        neighbor_embeds = neighbor_embeds.reshape(batch_size, -1, self.neighbor_hidden_size)
        # 保存或更新 `mean_embed` 的值。
        mean_embed = torch.mean(neighbor_embeds, dim=1)
        # 返回当前函数的结果。
        return mean_embed


# 定义类 `QuadNeighborhoodEncoderAttention`。
class QuadNeighborhoodEncoderAttention(QuadNeighborhoodEncoder):
    # 定义函数 `__init__`。
    def __init__(self, cfg, neighbor_obs_dim, neighbor_hidden_size, self_obs_dim, num_use_neighbor_obs):
        # 调用 `super` 执行当前处理。
        super().__init__(cfg, self_obs_dim, neighbor_obs_dim, neighbor_hidden_size, num_use_neighbor_obs)

        # 保存或更新 `self_obs_dim` 的值。
        self.self_obs_dim = self_obs_dim

        # outputs e_i from the paper
        # 保存或更新 `embedding_mlp` 的值。
        self.embedding_mlp = nn.Sequential(
            fc_layer(self_obs_dim + neighbor_obs_dim, neighbor_hidden_size),
            nonlinearity(cfg),
            fc_layer(neighbor_hidden_size, neighbor_hidden_size),
            nonlinearity(cfg)
        )

        #  outputs h_i from the paper
        # 保存或更新 `neighbor_value_mlp` 的值。
        self.neighbor_value_mlp = nn.Sequential(
            fc_layer(neighbor_hidden_size, neighbor_hidden_size),
            nonlinearity(cfg),
            fc_layer(neighbor_hidden_size, neighbor_hidden_size),
            nonlinearity(cfg),
        )

        # outputs scalar score alpha_i for each neighbor i
        # 保存或更新 `attention_mlp` 的值。
        self.attention_mlp = nn.Sequential(
            fc_layer(neighbor_hidden_size * 2, neighbor_hidden_size),
            # neighbor_hidden_size * 2 because we concat e_i and e_m
            nonlinearity(cfg),
            fc_layer(neighbor_hidden_size, neighbor_hidden_size),
            nonlinearity(cfg),
            fc_layer(neighbor_hidden_size, 1),
        )

    # 定义函数 `forward`。
    def forward(self, self_obs, obs, all_neighbor_obs_size, batch_size):
        # 保存或更新 `obs_neighbors` 的值。
        obs_neighbors = obs[:, self.self_obs_dim:self.self_obs_dim + all_neighbor_obs_size]
        # 保存或更新 `obs_neighbors` 的值。
        obs_neighbors = obs_neighbors.reshape(-1, self.neighbor_obs_dim)

        # concatenate self observation with neighbor observation

        # 保存或更新 `self_obs_repeat` 的值。
        self_obs_repeat = self_obs.repeat(self.num_use_neighbor_obs, 1)
        # 保存或更新 `mlp_input` 的值。
        mlp_input = torch.cat((self_obs_repeat, obs_neighbors), dim=1)
        # 保存或更新 `neighbor_embeddings` 的值。
        neighbor_embeddings = self.embedding_mlp(mlp_input)  # e_i in the paper https://arxiv.org/pdf/1809.08835.pdf

        # 保存或更新 `neighbor_values` 的值。
        neighbor_values = self.neighbor_value_mlp(neighbor_embeddings)  # h_i in the paper

        # 保存或更新 `neighbor_embeddings_mean_input` 的值。
        neighbor_embeddings_mean_input = neighbor_embeddings.reshape(batch_size, -1, self.neighbor_hidden_size)
        # 保存或更新 `neighbor_embeddings_mean` 的值。
        neighbor_embeddings_mean = torch.mean(neighbor_embeddings_mean_input, dim=1)  # e_m in the paper
        # 保存或更新 `neighbor_embeddings_mean_repeat` 的值。
        neighbor_embeddings_mean_repeat = neighbor_embeddings_mean.repeat(self.num_use_neighbor_obs, 1)

        # 保存或更新 `attention_mlp_input` 的值。
        attention_mlp_input = torch.cat((neighbor_embeddings, neighbor_embeddings_mean_repeat), dim=1)
        # 保存或更新 `attention_weights` 的值。
        attention_weights = self.attention_mlp(attention_mlp_input).view(batch_size, -1)  # alpha_i in the paper
        # 保存或更新 `attention_weights_softmax` 的值。
        attention_weights_softmax = torch.nn.functional.softmax(attention_weights, dim=1)
        # 保存或更新 `attention_weights_softmax` 的值。
        attention_weights_softmax = attention_weights_softmax.view(-1, 1)

        # 保存或更新 `final_neighborhood_embedding` 的值。
        final_neighborhood_embedding = attention_weights_softmax * neighbor_values
        # 保存或更新 `final_neighborhood_embedding` 的值。
        final_neighborhood_embedding = final_neighborhood_embedding.view(batch_size, -1, self.neighbor_hidden_size)
        # 保存或更新 `final_neighborhood_embedding` 的值。
        final_neighborhood_embedding = torch.sum(final_neighborhood_embedding, dim=1)

        # 返回当前函数的结果。
        return final_neighborhood_embedding


# 定义类 `QuadNeighborhoodEncoderMlp`。
class QuadNeighborhoodEncoderMlp(QuadNeighborhoodEncoder):
    # 定义函数 `__init__`。
    def __init__(self, cfg, neighbor_obs_dim, neighbor_hidden_size, self_obs_dim, num_use_neighbor_obs):
        # 调用 `super` 执行当前处理。
        super().__init__(cfg, self_obs_dim, neighbor_obs_dim, neighbor_hidden_size, num_use_neighbor_obs)

        # 保存或更新 `self_obs_dim` 的值。
        self.self_obs_dim = self_obs_dim

        # 保存或更新 `neighbor_mlp` 的值。
        self.neighbor_mlp = nn.Sequential(
            fc_layer(neighbor_obs_dim * num_use_neighbor_obs, neighbor_hidden_size),
            nonlinearity(cfg),
            fc_layer(neighbor_hidden_size, neighbor_hidden_size),
            nonlinearity(cfg),
            fc_layer(neighbor_hidden_size, neighbor_hidden_size),
            nonlinearity(cfg),
        )

    # 定义函数 `forward`。
    def forward(self, self_obs, obs, all_neighbor_obs_size, batch_size):
        # 保存或更新 `obs_neighbors` 的值。
        obs_neighbors = obs[:, self.self_obs_dim:self.self_obs_dim + all_neighbor_obs_size]
        # 保存或更新 `final_neighborhood_embedding` 的值。
        final_neighborhood_embedding = self.neighbor_mlp(obs_neighbors)
        # 返回当前函数的结果。
        return final_neighborhood_embedding


# 定义类 `QuadMultiHeadAttentionEncoder`。
class QuadMultiHeadAttentionEncoder(Encoder):
    # 定义函数 `__init__`。
    def __init__(self, cfg, obs_space):
        # 调用 `super` 执行当前处理。
        super().__init__(cfg)

        # Internal params
        # 根据条件决定是否进入当前分支。
        if cfg.quads_obs_repr in QUADS_OBS_REPR:
            # 保存或更新 `self_obs_dim` 的值。
            self.self_obs_dim = QUADS_OBS_REPR[cfg.quads_obs_repr]
        # 当前置条件都不满足时，执行兜底分支。
        else:
            # 主动抛出异常以中止或提示错误。
            raise NotImplementedError(f'Layer {cfg.quads_obs_repr} not supported!')

        # 保存或更新 `neighbor_hidden_size` 的值。
        self.neighbor_hidden_size = cfg.quads_neighbor_hidden_size
        # 保存或更新 `use_obstacles` 的值。
        self.use_obstacles = cfg.quads_use_obstacles

        # 根据条件决定是否进入当前分支。
        if cfg.quads_neighbor_visible_num == -1:
            # 保存或更新 `num_use_neighbor_obs` 的值。
            self.num_use_neighbor_obs = cfg.quads_num_agents - 1
        # 当前置条件都不满足时，执行兜底分支。
        else:
            # 保存或更新 `num_use_neighbor_obs` 的值。
            self.num_use_neighbor_obs = cfg.quads_neighbor_visible_num

        # 保存或更新 `neighbor_obs_dim` 的值。
        self.neighbor_obs_dim = QUADS_NEIGHBOR_OBS_TYPE[cfg.quads_neighbor_obs_type]

        # 保存或更新 `all_neighbor_obs_dim` 的值。
        self.all_neighbor_obs_dim = self.neighbor_obs_dim * self.num_use_neighbor_obs

        # Embedding Layer
        # 保存或更新 `fc_encoder_layer` 的值。
        fc_encoder_layer = cfg.rnn_size
        # 保存或更新 `self_embed_layer` 的值。
        self.self_embed_layer = nn.Sequential(
            fc_layer(self.self_obs_dim, fc_encoder_layer),
            nonlinearity(cfg),
            fc_layer(fc_encoder_layer, fc_encoder_layer),
            nonlinearity(cfg)
        )
        # 保存或更新 `neighbor_embed_layer` 的值。
        self.neighbor_embed_layer = nn.Sequential(
            fc_layer(self.all_neighbor_obs_dim, fc_encoder_layer),
            nonlinearity(cfg),
            fc_layer(fc_encoder_layer, fc_encoder_layer),
            nonlinearity(cfg)
        )
        # 保存或更新 `obstacle_obs_dim` 的值。
        self.obstacle_obs_dim = QUADS_OBSTACLE_OBS_TYPE[cfg.quads_obstacle_obs_type]
        # 保存或更新 `obstacle_embed_layer` 的值。
        self.obstacle_embed_layer = nn.Sequential(
            fc_layer(self.obstacle_obs_dim, fc_encoder_layer),
            nonlinearity(cfg),
            fc_layer(fc_encoder_layer, fc_encoder_layer),
            nonlinearity(cfg)
        )

        # Attention Layer
        # 保存或更新 `attention_layer` 的值。
        self.attention_layer = MultiHeadAttention(4, cfg.rnn_size, cfg.rnn_size, cfg.rnn_size)

        # MLP Layer
        # 保存或更新 `encoder_output_size` 的值。
        self.encoder_output_size = 2 * cfg.rnn_size
        # 保存或更新 `feed_forward` 的值。
        self.feed_forward = nn.Sequential(fc_layer(3 * cfg.rnn_size, self.encoder_output_size),
                                          nn.Tanh())

    # 定义函数 `forward`。
    def forward(self, obs_dict):
        # 保存或更新 `obs` 的值。
        obs = obs_dict['obs']
        # 保存或更新 `batch_size` 的值。
        batch_size = obs.shape[0]
        # 保存或更新 `obs_self` 的值。
        obs_self = obs[:, :self.self_obs_dim]
        # 保存或更新 `obs_neighbor` 的值。
        obs_neighbor = obs[:, self.self_obs_dim: self.self_obs_dim + self.all_neighbor_obs_dim]
        # 保存或更新 `obs_obstacle` 的值。
        obs_obstacle = obs[:, self.self_obs_dim + self.all_neighbor_obs_dim:]

        # 保存或更新 `self_embed` 的值。
        self_embed = self.self_embed_layer(obs_self)
        # 保存或更新 `neighbor_embed` 的值。
        neighbor_embed = self.neighbor_embed_layer(obs_neighbor)
        # 保存或更新 `obstacle_embed` 的值。
        obstacle_embed = self.obstacle_embed_layer(obs_obstacle)
        # 保存或更新 `neighbor_embed` 的值。
        neighbor_embed = neighbor_embed.view(batch_size, 1, -1)
        # 保存或更新 `obstacle_embed` 的值。
        obstacle_embed = obstacle_embed.view(batch_size, 1, -1)
        # 保存或更新 `attn_embed` 的值。
        attn_embed = torch.cat((neighbor_embed, obstacle_embed), dim=1)

        # 同时更新 `attn_embed`, `attn_score` 等变量。
        attn_embed, attn_score = self.attention_layer(attn_embed, attn_embed, attn_embed)
        # 保存或更新 `attn_embed` 的值。
        attn_embed = attn_embed.view(batch_size, -1)

        # 保存或更新 `embeddings` 的值。
        embeddings = torch.cat((self_embed, attn_embed), dim=1)
        # 保存或更新 `out` 的值。
        out = self.feed_forward(embeddings)

        # 返回当前函数的结果。
        return out

    # 定义函数 `get_out_size`。
    def get_out_size(self):
        # 返回当前函数的结果。
        return self.encoder_output_size


# 定义类 `QuadSingleHeadAttentionEncoder_Sim2Real`。
class QuadSingleHeadAttentionEncoder_Sim2Real(QuadMultiHeadAttentionEncoder):
    # 定义函数 `__init__`。
    def __init__(self, cfg, obs_space):
        # 调用 `super` 执行当前处理。
        super().__init__(cfg, obs_space)

        # Internal params
        # 根据条件决定是否进入当前分支。
        if cfg.quads_obs_repr in QUADS_OBS_REPR:
            # 保存或更新 `self_obs_dim` 的值。
            self.self_obs_dim = QUADS_OBS_REPR[cfg.quads_obs_repr]
        # 当前置条件都不满足时，执行兜底分支。
        else:
            # 主动抛出异常以中止或提示错误。
            raise NotImplementedError(f'Layer {cfg.quads_obs_repr} not supported!')

        # 保存或更新 `neighbor_hidden_size` 的值。
        self.neighbor_hidden_size = cfg.quads_neighbor_hidden_size
        # 保存或更新 `use_obstacles` 的值。
        self.use_obstacles = cfg.quads_use_obstacles

        # 根据条件决定是否进入当前分支。
        if cfg.quads_neighbor_visible_num == -1:
            # 保存或更新 `num_use_neighbor_obs` 的值。
            self.num_use_neighbor_obs = cfg.quads_num_agents - 1
        # 当前置条件都不满足时，执行兜底分支。
        else:
            # 保存或更新 `num_use_neighbor_obs` 的值。
            self.num_use_neighbor_obs = cfg.quads_neighbor_visible_num

        # 保存或更新 `neighbor_obs_dim` 的值。
        self.neighbor_obs_dim = QUADS_NEIGHBOR_OBS_TYPE[cfg.quads_neighbor_obs_type]

        # 保存或更新 `all_neighbor_obs_dim` 的值。
        self.all_neighbor_obs_dim = self.neighbor_obs_dim * self.num_use_neighbor_obs

        # Embedding Layer
        # 保存或更新 `fc_encoder_layer` 的值。
        fc_encoder_layer = cfg.rnn_size
        # 保存或更新 `self_embed_layer` 的值。
        self.self_embed_layer = nn.Sequential(
            fc_layer(self.self_obs_dim, fc_encoder_layer),
            nonlinearity(cfg),
        )
        # 保存或更新 `neighbor_embed_layer` 的值。
        self.neighbor_embed_layer = nn.Sequential(
            fc_layer(self.all_neighbor_obs_dim, fc_encoder_layer),
            nonlinearity(cfg),
        )
        # 保存或更新 `obstacle_obs_dim` 的值。
        self.obstacle_obs_dim = QUADS_OBSTACLE_OBS_TYPE[cfg.quads_obstacle_obs_type]
        # 保存或更新 `obstacle_embed_layer` 的值。
        self.obstacle_embed_layer = nn.Sequential(
            fc_layer(self.obstacle_obs_dim, fc_encoder_layer),
            nonlinearity(cfg),
        )

        # Attention Layer
        # 保存或更新 `attention_layer` 的值。
        self.attention_layer = OneHeadAttention(cfg.rnn_size)

        # MLP Layer
        # 保存或更新 `encoder_output_size` 的值。
        self.encoder_output_size = cfg.rnn_size
        # 保存或更新 `feed_forward` 的值。
        self.feed_forward = nn.Sequential(fc_layer(3 * cfg.rnn_size, self.encoder_output_size),
                                          nn.Tanh())


# 定义类 `QuadMultiEncoder`。
class QuadMultiEncoder(Encoder):
    # Mean embedding encoder based on the DeepRL for Swarms Paper
    # 定义函数 `__init__`。
    def __init__(self, cfg, obs_space):
        # 调用 `super` 执行当前处理。
        super().__init__(cfg)

        # 保存或更新 `self_obs_dim` 的值。
        self.self_obs_dim = QUADS_OBS_REPR[cfg.quads_obs_repr]
        # 保存或更新 `use_obstacles` 的值。
        self.use_obstacles = cfg.quads_use_obstacles

        # Neighbor
        # 保存或更新 `neighbor_hidden_size` 的值。
        neighbor_hidden_size = cfg.quads_neighbor_hidden_size
        # 保存或更新 `neighbor_obs_dim` 的值。
        neighbor_obs_dim = QUADS_NEIGHBOR_OBS_TYPE[cfg.quads_neighbor_obs_type]

        # 根据条件决定是否进入当前分支。
        if cfg.quads_neighbor_obs_type == 'none':
            # 保存或更新 `num_use_neighbor_obs` 的值。
            num_use_neighbor_obs = 0
        # 当前置条件都不满足时，执行兜底分支。
        else:
            # 根据条件决定是否进入当前分支。
            if cfg.quads_neighbor_visible_num == -1:
                # 保存或更新 `num_use_neighbor_obs` 的值。
                num_use_neighbor_obs = cfg.quads_num_agents - 1
            # 当前置条件都不满足时，执行兜底分支。
            else:
                # 保存或更新 `num_use_neighbor_obs` 的值。
                num_use_neighbor_obs = cfg.quads_neighbor_visible_num

        # 保存或更新 `all_neighbor_obs_size` 的值。
        self.all_neighbor_obs_size = neighbor_obs_dim * num_use_neighbor_obs

        # # Neighbor Encoder
        # 保存或更新 `neighbor_encoder_out_size` 的值。
        neighbor_encoder_out_size = 0
        # 保存或更新 `neighbor_encoder` 的值。
        self.neighbor_encoder = None

        # 根据条件决定是否进入当前分支。
        if num_use_neighbor_obs > 0:
            # 保存或更新 `neighbor_encoder_type` 的值。
            neighbor_encoder_type = cfg.quads_neighbor_encoder_type
            # 根据条件决定是否进入当前分支。
            if neighbor_encoder_type == 'mean_embed':
                # 保存或更新 `neighbor_encoder` 的值。
                self.neighbor_encoder = QuadNeighborhoodEncoderDeepsets(
                    cfg=cfg, neighbor_obs_dim=neighbor_obs_dim, neighbor_hidden_size=neighbor_hidden_size,
                    self_obs_dim=self.self_obs_dim, num_use_neighbor_obs=num_use_neighbor_obs)
            # 当上一分支不满足时，继续判断新的条件。
            elif neighbor_encoder_type == 'attention':
                # 保存或更新 `neighbor_encoder` 的值。
                self.neighbor_encoder = QuadNeighborhoodEncoderAttention(
                    cfg=cfg, neighbor_obs_dim=neighbor_obs_dim, neighbor_hidden_size=neighbor_hidden_size,
                    self_obs_dim=self.self_obs_dim, num_use_neighbor_obs=num_use_neighbor_obs)
            # 当上一分支不满足时，继续判断新的条件。
            elif neighbor_encoder_type == 'mlp':
                # 保存或更新 `neighbor_encoder` 的值。
                self.neighbor_encoder = QuadNeighborhoodEncoderMlp(
                    cfg=cfg, neighbor_obs_dim=neighbor_obs_dim, neighbor_hidden_size=neighbor_hidden_size,
                    self_obs_dim=self.self_obs_dim, num_use_neighbor_obs=num_use_neighbor_obs)
            # 当上一分支不满足时，继续判断新的条件。
            elif neighbor_encoder_type == 'no_encoder':
                # Blind agent
                # 保存或更新 `neighbor_encoder` 的值。
                self.neighbor_encoder = None
            # 当前置条件都不满足时，执行兜底分支。
            else:
                # 主动抛出异常以中止或提示错误。
                raise NotImplementedError

        # 根据条件决定是否进入当前分支。
        if self.neighbor_encoder:
            # 保存或更新 `neighbor_encoder_out_size` 的值。
            neighbor_encoder_out_size = neighbor_hidden_size

        # 保存或更新 `fc_encoder_layer` 的值。
        fc_encoder_layer = cfg.rnn_size
        # Encode Self Obs
        # 保存或更新 `self_encoder` 的值。
        self.self_encoder = nn.Sequential(
            fc_layer(self.self_obs_dim, fc_encoder_layer),
            nonlinearity(cfg),
            fc_layer(fc_encoder_layer, fc_encoder_layer),
            nonlinearity(cfg)
        )
        # 保存或更新 `self_encoder_out_size` 的值。
        self_encoder_out_size = calc_num_elements(self.self_encoder, (self.self_obs_dim,))

        # Encode Obstacle Obs
        # 保存或更新 `obstacle_encoder_out_size` 的值。
        obstacle_encoder_out_size = 0
        # 根据条件决定是否进入当前分支。
        if self.use_obstacles:
            # 保存或更新 `obstacle_obs_dim` 的值。
            obstacle_obs_dim = QUADS_OBSTACLE_OBS_TYPE[cfg.quads_obstacle_obs_type]
            # 保存或更新 `obstacle_hidden_size` 的值。
            obstacle_hidden_size = cfg.quads_obst_hidden_size
            # 保存或更新 `obstacle_encoder` 的值。
            self.obstacle_encoder = nn.Sequential(
                fc_layer(obstacle_obs_dim, obstacle_hidden_size),
                nonlinearity(cfg),
                fc_layer(obstacle_hidden_size, obstacle_hidden_size),
                nonlinearity(cfg),
            )
            # 保存或更新 `obstacle_encoder_out_size` 的值。
            obstacle_encoder_out_size = calc_num_elements(self.obstacle_encoder, (obstacle_obs_dim,))

        # 保存或更新 `total_encoder_out_size` 的值。
        total_encoder_out_size = self_encoder_out_size + neighbor_encoder_out_size + obstacle_encoder_out_size

        # This is followed by another fully connected layer in the action parameterization, so we add a nonlinearity
        # here
        # 保存或更新 `feed_forward` 的值。
        self.feed_forward = nn.Sequential(
            fc_layer(total_encoder_out_size, 2 * cfg.rnn_size),
            nn.Tanh(),
        )

        # 保存或更新 `encoder_out_size` 的值。
        self.encoder_out_size = 2 * cfg.rnn_size

    # 定义函数 `forward`。
    def forward(self, obs_dict):
        # 保存或更新 `obs` 的值。
        obs = obs_dict['obs']
        # 保存或更新 `obs_self` 的值。
        obs_self = obs[:, :self.self_obs_dim]
        # 保存或更新 `self_embed` 的值。
        self_embed = self.self_encoder(obs_self)
        # 保存或更新 `embeddings` 的值。
        embeddings = self_embed
        # 保存或更新 `batch_size` 的值。
        batch_size = obs_self.shape[0]
        # Relative xyz and vxyz for the Entire Minibatch (batch dimension is batch_size * num_neighbors)
        # 根据条件决定是否进入当前分支。
        if self.neighbor_encoder:
            # 保存或更新 `neighborhood_embedding` 的值。
            neighborhood_embedding = self.neighbor_encoder(obs_self, obs, self.all_neighbor_obs_size, batch_size)
            # 保存或更新 `embeddings` 的值。
            embeddings = torch.cat((embeddings, neighborhood_embedding), dim=1)

        # 根据条件决定是否进入当前分支。
        if self.use_obstacles:
            # 保存或更新 `obs_obstacles` 的值。
            obs_obstacles = obs[:, self.self_obs_dim + self.all_neighbor_obs_size:]
            # 保存或更新 `obstacle_embeds` 的值。
            obstacle_embeds = self.obstacle_encoder(obs_obstacles)
            # 保存或更新 `embeddings` 的值。
            embeddings = torch.cat((embeddings, obstacle_embeds), dim=1)

        # 保存或更新 `out` 的值。
        out = self.feed_forward(embeddings)
        # 返回当前函数的结果。
        return out

    # 定义函数 `get_out_size`。
    def get_out_size(self) -> int:
        # 返回当前函数的结果。
        return self.encoder_out_size


# 定义函数 `make_quadmulti_encoder`。
def make_quadmulti_encoder(cfg, obs_space) -> Encoder:
    # 根据条件决定是否进入当前分支。
    if cfg.quads_encoder_type == "attention":
        # 根据条件决定是否进入当前分支。
        if cfg.quads_sim2real:
            # 保存或更新 `model` 的值。
            model = QuadSingleHeadAttentionEncoder_Sim2Real(cfg, obs_space)
        # 当前置条件都不满足时，执行兜底分支。
        else:
            # 保存或更新 `model` 的值。
            model = QuadMultiHeadAttentionEncoder(cfg, obs_space)
    # 当前置条件都不满足时，执行兜底分支。
    else:
        # 保存或更新 `model` 的值。
        model = QuadMultiEncoder(cfg, obs_space)
    # 返回当前函数的结果。
    return model


# 定义函数 `register_models`。
def register_models():
    # 调用 `global_model_factory` 执行当前处理。
    global_model_factory().register_encoder_factory(make_quadmulti_encoder)
