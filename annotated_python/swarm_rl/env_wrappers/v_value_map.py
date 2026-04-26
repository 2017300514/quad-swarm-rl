# 中文注释副本；原始文件：swarm_rl/env_wrappers/v_value_map.py
# 说明：为避免修改源码，本文件仅作为阅读辅助材料。

# 导入当前模块依赖。
import copy

# 导入当前模块依赖。
import gymnasium as gym
import numpy as np
from sample_factory.algo.utils.rl_utils import prepare_and_normalize_obs

# 导入当前模块依赖。
from gym_art.quadrotor_multi.tests.plot_v_value_2d import plot_v_value_2d


# 定义类 `V_ValueMapWrapper`。
class V_ValueMapWrapper(gym.Wrapper):
    # 定义函数 `__init__`。
    def __init__(self, env, model, render_mode=None):
        # 下面的文档字符串用于说明当前模块或代码块。
        """A wrapper that visualize V-value map at each time step"""
        # 调用 `__init__` 执行当前处理。
        gym.Wrapper.__init__(self, env)
        # 保存或更新 `_render_mode` 的值。
        self._render_mode = render_mode
        # 保存或更新 `curr_obs` 的值。
        self.curr_obs = None
        # 保存或更新 `model` 的值。
        self.model = model

    # 定义函数 `reset`。
    def reset(self, **kwargs):
        # 同时更新 `obs`, `info` 等变量。
        obs, info = self.env.reset()
        # 保存或更新 `curr_obs` 的值。
        self.curr_obs = obs
        # 返回当前函数的结果。
        return obs, info

    # 定义函数 `step`。
    def step(self, action):
        # 同时更新 `obs`, `reward`, `info`, `terminated` 等变量。
        obs, reward, info, terminated, truncated = self.env.step(action)
        # 保存或更新 `curr_obs` 的值。
        self.curr_obs = obs
        # 返回当前函数的结果。
        return obs, reward, info, terminated, truncated

    # 定义函数 `render`。
    def render(self):
        # 根据条件决定是否进入当前分支。
        if self._render_mode == 'rgb_array':
            # 保存或更新 `frame` 的值。
            frame = self.env.render()
            # 根据条件决定是否进入当前分支。
            if frame is not None:
                # 同时更新 `width`, `height` 等变量。
                width, height = frame.shape[0], frame.shape[1]
                # 保存或更新 `v_value_map_2d` 的值。
                v_value_map_2d = self.get_v_value_map_2d(width=width, height=height)
                # 保存或更新 `frame` 的值。
                frame = np.concatenate((frame, v_value_map_2d), axis=1)
            # 返回当前函数的结果。
            return frame
        # 当前置条件都不满足时，执行兜底分支。
        else:
            # 返回当前函数的结果。
            return self.env.render()

    # 定义函数 `get_v_value_map_2d`。
    def get_v_value_map_2d(self, width=None, height=None):
        # 保存或更新 `tmp_score` 的值。
        tmp_score = []
        # 保存或更新 `idx` 的值。
        idx = []
        # 保存或更新 `idy` 的值。
        idy = []
        # 保存或更新 `rnn_states` 的值。
        rnn_states = None
        # 保存或更新 `obs` 的值。
        obs = dict(obs=np.array(self.curr_obs))
        # 保存或更新 `normalized_obs` 的值。
        normalized_obs = prepare_and_normalize_obs(self.model, obs)
        # 同时更新 `init_x`, `init_y` 等变量。
        init_x, init_y = copy.deepcopy(normalized_obs['obs'][0][0]), copy.deepcopy(normalized_obs['obs'][0][1])
        # 遍历当前序列或迭代器，逐项执行下面的逻辑。
        for i in range(-10, 11):
            # 保存或更新 `ti_score` 的值。
            ti_score = []
            # 遍历当前序列或迭代器，逐项执行下面的逻辑。
            for j in range(-10, 11):
                # 保存或更新 `normalized_obs[obs][0][0]` 的值。
                normalized_obs['obs'][0][0] = init_x + i * 0.2
                # 保存或更新 `normalized_obs[obs][0][1]` 的值。
                normalized_obs['obs'][0][1] = init_y + j * 0.2

                # x = self.model.forward_head(self.curr_obs)
                # x, new_rnn_states = self.model.forward_core(x, rnn_states)
                # result = self.model.forward_tail(x, values_only=True, sample_actions=True)
                # 保存或更新 `result` 的值。
                result = self.model.forward(normalized_obs, rnn_states, values_only=True)

                # 调用 `append` 执行当前处理。
                ti_score.append(result['values'].item())
                # 调用 `append` 执行当前处理。
                idx.append(i * 0.2)
                # 调用 `append` 执行当前处理。
                idy.append(j * 0.2)

            # 调用 `append` 执行当前处理。
            tmp_score.append(ti_score)

        # 同时更新 `idx`, `idy`, `tmp_score` 等变量。
        idx, idy, tmp_score = np.array(idx), np.array(idy), np.array(tmp_score)
        # 保存或更新 `v_value_map_2d` 的值。
        v_value_map_2d = plot_v_value_2d(idx, idy, tmp_score, width=width, height=height)

        # 返回当前函数的结果。
        return v_value_map_2d
