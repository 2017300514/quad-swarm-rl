# 中文注释副本；原始文件：swarm_rl/env_wrappers/compatibility.py
# 说明：为避免修改源码，本文件仅作为阅读辅助材料。
# 该文件属于强化学习训练侧逻辑，负责把环境、模型、配置或评估流程接到 Sample Factory 框架上。
# 这里产生的数据通常会继续流向训练循环、策略网络或实验分析脚本。

# 下面这组导入把当前模块会消费的环境组件、训练接口或数值工具集中拉进来；真正重要的是后续它们怎样参与数据流。
from typing import Any, Dict, Optional, Tuple

# 下面这组导入把当前模块会消费的环境组件、训练接口或数值工具集中拉进来；真正重要的是后续它们怎样参与数据流。
import gymnasium as gym
from gymnasium.core import ObsType
from gymnasium.utils.step_api_compatibility import convert_to_terminated_truncated_step_api

# Wrapper for compatibility with gym 0.26
# Mostly copied from gym.EnvCompatability
# Modified since swarm_rl does not have a seed, and is a vectorized env
# `QuadEnvCompatibility` 是当前文件暴露的核心类型，它负责维护与该模块职责直接相关的长期状态。
class QuadEnvCompatibility(gym.Wrapper):
    # 初始化阶段会把实验配置翻译成环境内部状态，包括单机实例、观测裁剪边界、碰撞阈值、障碍物和日志缓存。
    # 这些状态会在后续每个 step 中被不断读取和更新，因此这里决定了环境运行时的数据布局。
    def __init__(self, env, ):
        # 下面的文档字符串通常由源码作者提供，用来补充模块职责、输入输出约束或使用方式。
        """A wrapper which converts old-style envs to valid modern envs.

        Some information may be lost in the conversion, so we recommend updating your environment.

        Args:
            env (LegacyEnv): the env to wrap, implemented with the old API
        """
        gym.Wrapper.__init__(self, env)

    # `reset` 封装了当前模块中的一段独立流程，阅读时应重点关注它消费哪些状态、又把结果交给谁继续使用。
    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None) -> Tuple[ObsType, dict]:
        # 下面的文档字符串通常由源码作者提供，用来补充模块职责、输入输出约束或使用方式。
        """Resets the environment.

        Args:
            seed: the seed to reset the environment with
            options: the options to reset the environment with

        Returns:
            (observation, info)
        """
        # 这里把当前阶段整理好的结果交还给上层调用者；真正要理解的是返回值之后会进入哪条训练或仿真链路。
        return self.env.reset(), {}

    # `step` 封装了当前模块中的一段独立流程，阅读时应重点关注它消费哪些状态、又把结果交给谁继续使用。
    def step(self, action: Any) -> Tuple[Any, float, bool, bool, Dict]:
        # 下面的文档字符串通常由源码作者提供，用来补充模块职责、输入输出约束或使用方式。
        """Steps through the environment.

        Args:
            action: action to step through the environment with

        Returns:
            (observation, reward, terminated, truncated, info)
        """
        # For QuadMultiEnv, truncated is actually integrated in the env,
        # since the termination is tick > ep_len
        obs, reward, done, info = self.env.step(action)

        #convert_to_terminated_truncated_step_api treats done as an iterable if info is a dictionary, fails if it not iterable
        if isinstance(info, dict) and isinstance(done, bool):
            done = [done]

        # 这里把当前阶段整理好的结果交还给上层调用者；真正要理解的是返回值之后会进入哪条训练或仿真链路。
        return convert_to_terminated_truncated_step_api((obs, reward, done, info), is_vector_env=True)

    # `render` 封装了当前模块中的一段独立流程，阅读时应重点关注它消费哪些状态、又把结果交给谁继续使用。
    def render(self) -> Any:
        # 下面的文档字符串通常由源码作者提供，用来补充模块职责、输入输出约束或使用方式。
        """Renders the environment.
        Returns:
            The rendering of the environment, depending on the render mode
        """
        # 这里把当前阶段整理好的结果交还给上层调用者；真正要理解的是返回值之后会进入哪条训练或仿真链路。
        return self.env.render()
