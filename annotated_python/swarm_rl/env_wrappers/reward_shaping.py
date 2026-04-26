# 中文注释副本；原始文件：swarm_rl/env_wrappers/reward_shaping.py
# 说明：为避免修改源码，本文件仅作为阅读辅助材料。
# 该文件位于“环境原始奖励”与“训练循环统计接口”之间。
# 它做的不是重新发明奖励函数，而是把环境内部已经算出的各项奖励系数写入运行时环境，
# 同时把逐步奖励累积成 episode 级统计，并在需要时按训练进度逐步抬高碰撞惩罚。
# 上游输入来自 `quad_utils.py` 整理好的 reward shaping 配置和退火计划；
# 下游输出是送给 Sample Factory 日志系统的 `true_reward` 与 `episode_extra_stats`。

import copy

import gymnasium as gym
import numpy as np
from sample_factory.envs.env_utils import TrainingInfoInterface, RewardShapingInterface

# 单机基础奖励表只包含“飞向目标、控制代价、姿态/旋转稳定”等与单机飞行本身相关的项。
# 这些值会被多机环境复用，再额外补上机间碰撞和障碍碰撞相关项。
DEFAULT_QUAD_REWARD_SHAPING_SINGLE = dict(
    quad_rewards=dict(
        pos=1.0, effort=0.05, spin=0.1, vel=0.0, crash=1.0, orient=1.0, yaw=0.0
    ),
)

# 多机版本在单机基础上增加三类安全相关奖励：
# `quadcol_bin` 是机间硬碰撞惩罚，
# `quadcol_bin_smooth_max` 是距离接近时的连续惩罚上界，
# `quadcol_bin_obst` 是撞障碍物惩罚。
DEFAULT_QUAD_REWARD_SHAPING = copy.deepcopy(DEFAULT_QUAD_REWARD_SHAPING_SINGLE)
DEFAULT_QUAD_REWARD_SHAPING['quad_rewards'].update(dict(
    quadcol_bin=0.0, quadcol_bin_smooth_max=0.0, quadcol_bin_obst=0.0
))


class QuadsRewardShapingWrapper(gym.Wrapper, TrainingInfoInterface, RewardShapingInterface):
    # 这个 wrapper 维护的是“训练视角下的奖励解释层”。
    # 环境本体负责按当前 `rew_coeff` 计算每一步奖励；
    # 这个 wrapper 负责在 reset/step 时把 reward shaping 配置同步到环境，
    # 汇总 episode 统计，并在训练中后期逐步提高安全相关奖励权重。
    def __init__(self, env, reward_shaping_scheme=None, annealing=None, with_pbt=False):
        gym.Wrapper.__init__(self, env)
        TrainingInfoInterface.__init__(self)
        if with_pbt:
            # 启用 PBT 时，外部训练系统可能会在训练过程中动态改 reward shaping。
            # 这里额外初始化 `RewardShapingInterface`，使 wrapper 能响应这类在线调整。
            RewardShapingInterface.__init__(self)

        # `reward_shaping_scheme` 来自 `quad_utils.py`，已经包含本次实验最终要使用的奖励系数。
        self.reward_shaping_scheme = reward_shaping_scheme

        # `cumulative_rewards` 记录一个 episode 内每个 agent 的奖励分项累计值，
        # 这样在 episode 结束时就能把 step 级 reward dict 汇总成日志统计。
        self.cumulative_rewards = None
        self.episode_actions = None

        # 多机环境下每个 agent 都要维护一份累计统计；如果底层不是多机环境，则退化成单 agent。
        self.num_agents = env.num_agents if hasattr(env, 'num_agents') else 1

        # 这个标志表示“reward shaping 配置需要重新写回底层环境”。
        # 它通常在初始化或 PBT 更新后为 True，随后在第一次 step 时同步给 `env.unwrapped.rew_coeff`。
        self.reward_shaping_updated = True

        # `annealing` 保存碰撞惩罚的退火计划表。
        # 如果为 None，说明本次实验直接使用最终奖励系数，不走逐步增强。
        self.annealing = annealing

    def get_default_reward_shaping(self):
        # PBT 接口要求 wrapper 能报告“默认 reward shaping 长什么样”。
        # 这里返回空壳结构，是因为项目真正的默认值由模块顶部的全局常量和环境初始配置共同决定。
        return dict(quad_rewards=dict())

    def get_current_reward_shaping(self, agent_idx: int):
        # 当前实现没有按 agent 区分 reward shaping，
        # 所有无人机共享同一套奖励权重，因此这里只返回统一结构。
        return dict(quad_rewards=dict())

    def set_reward_shaping(self, reward_shaping, unused_agent_idx):
        # 这个接口本来应该接收外部传入的新 reward shaping，
        # 但当前源码里把它重置成空结构，说明这里主要保留了框架兼容接口，
        # 并通过 `reward_shaping_updated` 触发下一次 step 时重新同步到底层环境。
        self.reward_shaping_scheme = dict(quad_rewards=dict())
        self.reward_shaping_updated = True

    def reset(self):
        # reset 时先让底层环境真正重置，再清空本 wrapper 自己维护的 episode 级缓存。
        obs = self.env.reset()

        # 每个 agent 都维护一份奖励累计字典，键是 `rew_*` / `rewraw_*` 这样的分项名字。
        self.cumulative_rewards = [dict() for _ in range(self.num_agents)]

        # 这里顺便重新开始记录整段 episode 的动作序列，
        # 后面结束时会把每个动作维度的均值和标准差写到日志里，辅助分析控制输出是否异常。
        self.episode_actions = []
        return obs

    def step(self, action):
        # 所有动作先记录下来，等 episode 结束后再统计每个控制通道的均值/方差。
        self.episode_actions.append(action)

        if self.reward_shaping_updated:
            # 这里把 wrapper 当前持有的 reward shaping 配置写回到底层环境真正参与奖励计算的 `rew_coeff`。
            # 也就是说，环境之后每一步产出的奖励，都会基于这里同步后的权重表。
            env_reward_shaping = self.env.unwrapped.rew_coeff
            for key, weight in self.reward_shaping_scheme['quad_rewards'].items():
                env_reward_shaping[key] = weight

            self.reward_shaping_updated = False

        obs, rewards, dones, infos = self.env.step(action)

        # 这个 wrapper 既兼容多机环境也兼容单机环境。
        # 为了统一后面的统计逻辑，这里把单机情况包装成长度为 1 的列表。
        if self.env.is_multiagent:
            infos_multi, dones_multi = infos, dones
        else:
            infos_multi, dones_multi = [infos], [dones]

        for i, info in enumerate(infos_multi):
            # `info['rewards']` 是底层环境给出的逐步奖励分解，
            # 例如位置奖励、坠毁惩罚、碰撞惩罚等都在这里按键拆开。
            rew_dict = info['rewards']

            # 这里只累计以 `rew` 开头的字段，意味着统计对象是奖励相关项，而不是 info 里的其他诊断信息。
            for key, value in rew_dict.items():
                if key.startswith('rew'):
                    if key not in self.cumulative_rewards[i]:
                        self.cumulative_rewards[i][key] = 0
                    self.cumulative_rewards[i][key] += value

            if dones_multi[i]:
                # episode 结束时，先取底层环境定义的主任务累计奖励。
                true_reward = self.cumulative_rewards[i]['rewraw_main']

                # 这里额外把机间碰撞乘以一个很大的权重叠加进 `true_reward`，
                # 反映的是项目评估口径：即使到达目标很好，只要发生碰撞，也不应被视为高质量 episode。
                true_reward_consider_collisions = True
                if true_reward_consider_collisions:
                    true_reward += 1000 * self.cumulative_rewards[i].get('rewraw_quadcol', 0)

                # `true_reward` 会被 Sample Factory 的日志和评估逻辑继续消费，
                # 所以这里实际上是在定义“实验报告里如何衡量一次 episode 的好坏”。
                info['true_reward'] = true_reward
                self.cumulative_rewards[i]['rewraw_main'] = true_reward

                if 'episode_extra_stats' not in info:
                    info['episode_extra_stats'] = dict()
                extra_stats = info['episode_extra_stats']

                # 把累积好的奖励分项整体塞进日志，便于后续区分：
                # 是位置表现差、控制代价高、还是碰撞项拖了后腿。
                extra_stats.update(self.cumulative_rewards[i])

                # `approx_total_training_steps` 由训练框架注入，是退火和日志对齐的关键时间轴。
                approx_total_training_steps = self.training_info.get('approx_total_training_steps', 0)
                extra_stats['z_approx_total_training_steps'] = approx_total_training_steps

                # 如果底层环境有场景对象，这里额外按场景名记录一些核心奖励，
                # 便于不同 scenario 的日志在同一训练里拆开看。
                if hasattr(self.env.unwrapped, 'scenario') and self.env.unwrapped.scenario:
                    scenario_name = self.env.unwrapped.scenario.name()
                    for rew_key in ['rew_pos', 'rew_crash']:
                        extra_stats[f'{scenario_name}/{rew_key}'] = self.cumulative_rewards[i][rew_key]

                # 这一段把整个 episode 的动作轨迹转成每个动作通道的均值/标准差，
                # 常用于判断策略是否饱和、抖动过大或长期偏向某个电机控制方向。
                episode_actions = np.array(self.episode_actions)
                episode_actions = episode_actions.transpose()
                for action_idx in range(episode_actions.shape[0]):
                    mean_action = np.mean(episode_actions[action_idx])
                    std_action = np.std(episode_actions[action_idx])
                    extra_stats[f'z_action{action_idx}_mean'] = mean_action
                    extra_stats[f'z_action{action_idx}_std'] = std_action

                # 当前 agent 的 episode 已结束，累计缓存要清空，避免污染下一个 episode。
                self.cumulative_rewards[i] = dict()

                if self.annealing:
                    # 如果启用了退火，这里按训练总步数把碰撞相关系数从 0 线性拉到目标值。
                    # 注意更新发生在 episode 结束处，因此 reward shaping 会以 episode 为单位逐步变严。
                    env_reward_shaping = self.env.unwrapped.rew_coeff
                    for anneal_schedule in self.annealing:
                        coeff_name = anneal_schedule.coeff_name
                        final_value = anneal_schedule.final_value
                        anneal_steps = anneal_schedule.anneal_env_steps
                        env_reward_shaping[coeff_name] = min(final_value * approx_total_training_steps / anneal_steps, final_value)
                        extra_stats[f'z_anneal_{coeff_name}'] = env_reward_shaping[coeff_name]

        if any(dones_multi):
            # 只要这一 step 里有任意 agent 结束，就把整段 episode 动作缓存清掉。
            # 这是因为这里记录的是“这一局”的动作统计，而不是跨局滚动窗口。
            self.episode_actions = []

        return obs, rewards, dones, infos
