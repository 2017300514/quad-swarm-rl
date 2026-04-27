# 中文注释副本；原始文件：swarm_rl/env_wrappers/v_value_map.py
# 说明：为避免修改源码，本文件仅作为阅读辅助材料。
# 这个 wrapper 不参与训练主路径的奖励或动力学，而是在评估/可视化时把 critic 的局部 V-value 地形图
# 拼到环境渲染帧旁边，帮助观察策略当前认为“往哪里移动更有价值”。
# 上游输入是环境刚输出的最新观测和已经训练好的模型；下游消费者是 `render()` 的 `rgb_array`
# 结果，以及依赖视频帧做论文展示或调试的人工分析流程。

import copy

import gymnasium as gym
import numpy as np
from sample_factory.algo.utils.rl_utils import prepare_and_normalize_obs

from gym_art.quadrotor_multi.tests.plot_v_value_2d import plot_v_value_2d


class V_ValueMapWrapper(gym.Wrapper):
    # 这里缓存两类长期状态：
    # 1. `curr_obs` 保存最近一个环境时刻的原始观测，供 render 阶段围绕当前 agent 位置做局部扫描。
    # 2. `model` 是已经训练好的 actor-critic；这里只读取其 value head，不改任何参数。
    def __init__(self, env, model, render_mode=None):
        """A wrapper that visualize V-value map at each time step"""
        gym.Wrapper.__init__(self, env)
        self._render_mode = render_mode
        self.curr_obs = None
        self.model = model

    # `reset` 与 `step` 都只做一件事：把最新观测缓存下来。
    # 这样后面的 `render()` 就能基于“当前时刻”而不是“上一次时刻”去询问 critic 的局部价值分布。
    def reset(self, **kwargs):
        obs, info = self.env.reset()
        self.curr_obs = obs
        return obs, info

    def step(self, action):
        obs, reward, info, terminated, truncated = self.env.step(action)
        self.curr_obs = obs
        return obs, reward, info, terminated, truncated

    # 渲染时，如果底层环境已经产出 RGB 帧，就在右侧再拼一张同尺寸的 value map。
    # 这样视频里左边仍是物理世界，右边则是 critic 在当前位置附近的二维价值地形。
    def render(self):
        if self._render_mode == 'rgb_array':
            frame = self.env.render()
            if frame is not None:
                width, height = frame.shape[0], frame.shape[1]
                v_value_map_2d = self.get_v_value_map_2d(width=width, height=height)
                frame = np.concatenate((frame, v_value_map_2d), axis=1)
            return frame
        else:
            return self.env.render()

    # 这里的核心不是重新跑环境，而是固定其它观测分量，只在当前 agent 的局部 x/y 平面上做 21x21 网格扰动。
    # 扰动后的观测先按 Sample Factory 训练时的同一套规则归一化，再直接走模型前向，只请求 `values_only=True`。
    # 这样得到的分数就是“如果 agent 瞬间处在附近这些相对位置，critic 会给出怎样的状态价值估计”。
    def get_v_value_map_2d(self, width=None, height=None):
        tmp_score = []
        idx = []
        idy = []
        rnn_states = None

        obs = dict(obs=np.array(self.curr_obs))
        normalized_obs = prepare_and_normalize_obs(self.model, obs)

        # 先记住当前归一化后的平面坐标，再围绕这个中心做相对偏移。
        init_x = copy.deepcopy(normalized_obs['obs'][0][0])
        init_y = copy.deepcopy(normalized_obs['obs'][0][1])

        for i in range(-10, 11):
            ti_score = []
            for j in range(-10, 11):
                normalized_obs['obs'][0][0] = init_x + i * 0.2
                normalized_obs['obs'][0][1] = init_y + j * 0.2

                # 这里直接复用整条模型 `forward`，因为 wrapper 的目标是得到和真实评估流程一致的 value 输出。
                result = self.model.forward(normalized_obs, rnn_states, values_only=True)

                ti_score.append(result['values'].item())
                idx.append(i * 0.2)
                idy.append(j * 0.2)

            tmp_score.append(ti_score)

        idx = np.array(idx)
        idy = np.array(idy)
        tmp_score = np.array(tmp_score)

        # 真正把数值栅格变成 RGB 图像的是 tests 目录里的这个 helper；
        # wrapper 本身只负责采样 critic，图像光栅化逻辑在下游脚本里。
        v_value_map_2d = plot_v_value_2d(idx, idy, tmp_score, width=width, height=height)
        return v_value_map_2d
