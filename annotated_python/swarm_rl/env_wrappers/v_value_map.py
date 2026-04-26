# 中文注释副本；原始文件：swarm_rl/env_wrappers/v_value_map.py
# 说明：为避免修改源码，本文件仅作为阅读辅助材料。
# 该文件属于强化学习训练侧逻辑，负责把环境、模型、配置或评估流程接到 Sample Factory 框架上。
# 这里产生的数据通常会继续流向训练循环、策略网络或实验分析脚本。

# 下面这组导入把当前模块会消费的环境组件、训练接口或数值工具集中拉进来；真正重要的是后续它们怎样参与数据流。
import copy

# 下面这组导入把当前模块会消费的环境组件、训练接口或数值工具集中拉进来；真正重要的是后续它们怎样参与数据流。
import gymnasium as gym
import numpy as np
from sample_factory.algo.utils.rl_utils import prepare_and_normalize_obs

# 下面这组导入把当前模块会消费的环境组件、训练接口或数值工具集中拉进来；真正重要的是后续它们怎样参与数据流。
from gym_art.quadrotor_multi.tests.plot_v_value_2d import plot_v_value_2d


# `V_ValueMapWrapper` 是当前文件暴露的核心类型，它负责维护与该模块职责直接相关的长期状态。
class V_ValueMapWrapper(gym.Wrapper):
    # 初始化阶段会把实验配置翻译成环境内部状态，包括单机实例、观测裁剪边界、碰撞阈值、障碍物和日志缓存。
    # 这些状态会在后续每个 step 中被不断读取和更新，因此这里决定了环境运行时的数据布局。
    def __init__(self, env, model, render_mode=None):
        # 下面的文档字符串通常由源码作者提供，用来补充模块职责、输入输出约束或使用方式。
        """A wrapper that visualize V-value map at each time step"""
        gym.Wrapper.__init__(self, env)
        self._render_mode = render_mode
        self.curr_obs = None
        self.model = model

    # `reset` 封装了当前模块中的一段独立流程，阅读时应重点关注它消费哪些状态、又把结果交给谁继续使用。
    def reset(self, **kwargs):
        obs, info = self.env.reset()
        self.curr_obs = obs
        # 这里把当前阶段整理好的结果交还给上层调用者；真正要理解的是返回值之后会进入哪条训练或仿真链路。
        return obs, info

    # `step` 封装了当前模块中的一段独立流程，阅读时应重点关注它消费哪些状态、又把结果交给谁继续使用。
    def step(self, action):
        obs, reward, info, terminated, truncated = self.env.step(action)
        self.curr_obs = obs
        # 这里把当前阶段整理好的结果交还给上层调用者；真正要理解的是返回值之后会进入哪条训练或仿真链路。
        return obs, reward, info, terminated, truncated

    # `render` 封装了当前模块中的一段独立流程，阅读时应重点关注它消费哪些状态、又把结果交给谁继续使用。
    def render(self):
        if self._render_mode == 'rgb_array':
            frame = self.env.render()
            if frame is not None:
                width, height = frame.shape[0], frame.shape[1]
                v_value_map_2d = self.get_v_value_map_2d(width=width, height=height)
                # 这里执行观测拼接，把分散的物理特征重组为策略网络期望的固定顺序向量。
                frame = np.concatenate((frame, v_value_map_2d), axis=1)
            # 这里把当前阶段整理好的结果交还给上层调用者；真正要理解的是返回值之后会进入哪条训练或仿真链路。
            return frame
        else:
            # 这里把当前阶段整理好的结果交还给上层调用者；真正要理解的是返回值之后会进入哪条训练或仿真链路。
            return self.env.render()

    # `get_v_value_map_2d` 封装了当前模块中的一段独立流程，阅读时应重点关注它消费哪些状态、又把结果交给谁继续使用。
    def get_v_value_map_2d(self, width=None, height=None):
        tmp_score = []
        idx = []
        idy = []
        rnn_states = None
        # 这里构造的是环境默认奖励权重表，表示在没有实验覆盖时多机导航任务各个目标项的基准权重。
        obs = dict(obs=np.array(self.curr_obs))
        normalized_obs = prepare_and_normalize_obs(self.model, obs)
        init_x, init_y = copy.deepcopy(normalized_obs['obs'][0][0]), copy.deepcopy(normalized_obs['obs'][0][1])
        for i in range(-10, 11):
            ti_score = []
            for j in range(-10, 11):
                normalized_obs['obs'][0][0] = init_x + i * 0.2
                normalized_obs['obs'][0][1] = init_y + j * 0.2

                # x = self.model.forward_head(self.curr_obs)
                # x, new_rnn_states = self.model.forward_core(x, rnn_states)
                # result = self.model.forward_tail(x, values_only=True, sample_actions=True)
                result = self.model.forward(normalized_obs, rnn_states, values_only=True)

                ti_score.append(result['values'].item())
                idx.append(i * 0.2)
                idy.append(j * 0.2)

            tmp_score.append(ti_score)

        idx, idy, tmp_score = np.array(idx), np.array(idy), np.array(tmp_score)
        v_value_map_2d = plot_v_value_2d(idx, idy, tmp_score, width=width, height=height)

        # 这里把当前阶段整理好的结果交还给上层调用者；真正要理解的是返回值之后会进入哪条训练或仿真链路。
        return v_value_map_2d
