# 中文注释副本；原始文件：swarm_rl/env_wrappers/reward_shaping.py
# 说明：为避免修改源码，本文件仅作为阅读辅助材料。

# 导入当前模块依赖。
import copy

# 导入当前模块依赖。
import gymnasium as gym
import numpy as np
from sample_factory.envs.env_utils import TrainingInfoInterface, RewardShapingInterface

# 保存或更新 `DEFAULT_QUAD_REWARD_SHAPING_SINGLE` 的值。
DEFAULT_QUAD_REWARD_SHAPING_SINGLE = dict(
    quad_rewards=dict(
        pos=1.0, effort=0.05, spin=0.1, vel=0.0, crash=1.0, orient=1.0, yaw=0.0
    ),
)

# 保存或更新 `DEFAULT_QUAD_REWARD_SHAPING` 的值。
DEFAULT_QUAD_REWARD_SHAPING = copy.deepcopy(DEFAULT_QUAD_REWARD_SHAPING_SINGLE)
# 执行这一行逻辑。
DEFAULT_QUAD_REWARD_SHAPING['quad_rewards'].update(dict(
    quadcol_bin=0.0, quadcol_bin_smooth_max=0.0, quadcol_bin_obst=0.0
))


# 定义类 `QuadsRewardShapingWrapper`。
class QuadsRewardShapingWrapper(gym.Wrapper, TrainingInfoInterface, RewardShapingInterface):
    # 定义函数 `__init__`。
    def __init__(self, env, reward_shaping_scheme=None, annealing=None, with_pbt=False):
        # 调用 `__init__` 执行当前处理。
        gym.Wrapper.__init__(self, env)
        # 调用 `__init__` 执行当前处理。
        TrainingInfoInterface.__init__(self)
        # 根据条件决定是否进入当前分支。
        if with_pbt:
            # 调用 `__init__` 执行当前处理。
            RewardShapingInterface.__init__(self)

        # 保存或更新 `reward_shaping_scheme` 的值。
        self.reward_shaping_scheme = reward_shaping_scheme
        # 保存或更新 `cumulative_rewards` 的值。
        self.cumulative_rewards = None
        # 保存或更新 `episode_actions` 的值。
        self.episode_actions = None

        # 保存或更新 `num_agents` 的值。
        self.num_agents = env.num_agents if hasattr(env, 'num_agents') else 1

        # 保存或更新 `reward_shaping_updated` 的值。
        self.reward_shaping_updated = True

        # 保存或更新 `annealing` 的值。
        self.annealing = annealing

    # 定义函数 `get_default_reward_shaping`。
    def get_default_reward_shaping(self):
        # 返回当前函数的结果。
        return dict(quad_rewards=dict())

    # 定义函数 `get_current_reward_shaping`。
    def get_current_reward_shaping(self, agent_idx: int):
        # 返回当前函数的结果。
        return dict(quad_rewards=dict())

    # 定义函数 `set_reward_shaping`。
    def set_reward_shaping(self, reward_shaping, unused_agent_idx):
        # 保存或更新 `reward_shaping_scheme` 的值。
        self.reward_shaping_scheme = dict(quad_rewards=dict())
        # 保存或更新 `reward_shaping_updated` 的值。
        self.reward_shaping_updated = True

    # 定义函数 `reset`。
    def reset(self):
        # 保存或更新 `obs` 的值。
        obs = self.env.reset()
        # 保存或更新 `cumulative_rewards` 的值。
        self.cumulative_rewards = [dict() for _ in range(self.num_agents)]
        # 保存或更新 `episode_actions` 的值。
        self.episode_actions = []
        # 返回当前函数的结果。
        return obs

    # 定义函数 `step`。
    def step(self, action):
        # 调用 `append` 执行当前处理。
        self.episode_actions.append(action)

        # 根据条件决定是否进入当前分支。
        if self.reward_shaping_updated:
            # set the updated reward shaping scheme
            # 保存或更新 `env_reward_shaping` 的值。
            env_reward_shaping = self.env.unwrapped.rew_coeff
            # 遍历当前序列或迭代器，逐项执行下面的逻辑。
            for key, weight in self.reward_shaping_scheme['quad_rewards'].items():
                # 保存或更新 `env_reward_shaping[key]` 的值。
                env_reward_shaping[key] = weight

            # 保存或更新 `reward_shaping_updated` 的值。
            self.reward_shaping_updated = False

        # 同时更新 `obs`, `rewards`, `dones`, `infos` 等变量。
        obs, rewards, dones, infos = self.env.step(action)
        # 根据条件决定是否进入当前分支。
        if self.env.is_multiagent:
            # 同时更新 `infos_multi`, `dones_multi` 等变量。
            infos_multi, dones_multi = infos, dones
        # 当前置条件都不满足时，执行兜底分支。
        else:
            # 同时更新 `infos_multi`, `dones_multi` 等变量。
            infos_multi, dones_multi = [infos], [dones]

        # 遍历当前序列或迭代器，逐项执行下面的逻辑。
        for i, info in enumerate(infos_multi):
            # 保存或更新 `rew_dict` 的值。
            rew_dict = info['rewards']

            # 遍历当前序列或迭代器，逐项执行下面的逻辑。
            for key, value in rew_dict.items():
                # 根据条件决定是否进入当前分支。
                if key.startswith('rew'):
                    # 根据条件决定是否进入当前分支。
                    if key not in self.cumulative_rewards[i]:
                        # 保存或更新 `cumulative_rewards[i][key]` 的值。
                        self.cumulative_rewards[i][key] = 0
                    # 保存或更新 `cumulative_rewards[i][key]` 的值。
                    self.cumulative_rewards[i][key] += value

            # 根据条件决定是否进入当前分支。
            if dones_multi[i]:
                # 保存或更新 `true_reward` 的值。
                true_reward = self.cumulative_rewards[i]['rewraw_main']
                # 保存或更新 `true_reward_consider_collisions` 的值。
                true_reward_consider_collisions = True
                # 根据条件决定是否进入当前分支。
                if true_reward_consider_collisions:
                    # we ideally want zero collisions, so collisions between quads are given very high weight
                    # 保存或更新 `true_reward` 的值。
                    true_reward += 1000 * self.cumulative_rewards[i].get('rewraw_quadcol', 0)

                # 保存或更新 `info[true_reward]` 的值。
                info['true_reward'] = true_reward
                # 保存或更新 `cumulative_rewards[i][rewraw_main]` 的值。
                self.cumulative_rewards[i]['rewraw_main'] = true_reward
                # 根据条件决定是否进入当前分支。
                if 'episode_extra_stats' not in info:
                    # 保存或更新 `info[episode_extra_stats]` 的值。
                    info['episode_extra_stats'] = dict()
                # 保存或更新 `extra_stats` 的值。
                extra_stats = info['episode_extra_stats']
                # 调用 `update` 执行当前处理。
                extra_stats.update(self.cumulative_rewards[i])

                # 保存或更新 `approx_total_training_steps` 的值。
                approx_total_training_steps = self.training_info.get('approx_total_training_steps', 0)
                # 保存或更新 `extra_stats[z_approx_total_training_steps]` 的值。
                extra_stats['z_approx_total_training_steps'] = approx_total_training_steps

                # 根据条件决定是否进入当前分支。
                if hasattr(self.env.unwrapped, 'scenario') and self.env.unwrapped.scenario:
                    # 保存或更新 `scenario_name` 的值。
                    scenario_name = self.env.unwrapped.scenario.name()
                    # 遍历当前序列或迭代器，逐项执行下面的逻辑。
                    for rew_key in ['rew_pos', 'rew_crash']:
                        # 保存或更新 `extra_stats[f{scenario_name}/{rew_key}]` 的值。
                        extra_stats[f'{scenario_name}/{rew_key}'] = self.cumulative_rewards[i][rew_key]

                # 保存或更新 `episode_actions` 的值。
                episode_actions = np.array(self.episode_actions)
                # 保存或更新 `episode_actions` 的值。
                episode_actions = episode_actions.transpose()
                # 遍历当前序列或迭代器，逐项执行下面的逻辑。
                for action_idx in range(episode_actions.shape[0]):
                    # 保存或更新 `mean_action` 的值。
                    mean_action = np.mean(episode_actions[action_idx])
                    # 保存或更新 `std_action` 的值。
                    std_action = np.std(episode_actions[action_idx])
                    # 保存或更新 `extra_stats[fz_action{action_idx}_mean]` 的值。
                    extra_stats[f'z_action{action_idx}_mean'] = mean_action
                    # 保存或更新 `extra_stats[fz_action{action_idx}_std]` 的值。
                    extra_stats[f'z_action{action_idx}_std'] = std_action

                # 保存或更新 `cumulative_rewards[i]` 的值。
                self.cumulative_rewards[i] = dict()

                # 根据条件决定是否进入当前分支。
                if self.annealing:
                    # 保存或更新 `env_reward_shaping` 的值。
                    env_reward_shaping = self.env.unwrapped.rew_coeff
                    # annealing from 0.0 to final value
                    # 遍历当前序列或迭代器，逐项执行下面的逻辑。
                    for anneal_schedule in self.annealing:
                        # 保存或更新 `coeff_name` 的值。
                        coeff_name = anneal_schedule.coeff_name
                        # 保存或更新 `final_value` 的值。
                        final_value = anneal_schedule.final_value
                        # 保存或更新 `anneal_steps` 的值。
                        anneal_steps = anneal_schedule.anneal_env_steps
                        # 保存或更新 `env_reward_shaping[coeff_name]` 的值。
                        env_reward_shaping[coeff_name] = min(final_value * approx_total_training_steps / anneal_steps, final_value)
                        # 保存或更新 `extra_stats[fz_anneal_{coeff_name}]` 的值。
                        extra_stats[f'z_anneal_{coeff_name}'] = env_reward_shaping[coeff_name]

        # 根据条件决定是否进入当前分支。
        if any(dones_multi):
            # 保存或更新 `episode_actions` 的值。
            self.episode_actions = []

        # 返回当前函数的结果。
        return obs, rewards, dones, infos
