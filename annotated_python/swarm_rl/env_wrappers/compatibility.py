# 中文注释副本；原始文件：swarm_rl/env_wrappers/compatibility.py
# 说明：为避免修改源码，本文件仅作为阅读辅助材料。
# 这个文件是 `swarm_rl` 接 Gymnasium 新 API 的兼容层。
# 上游还是 repo 里偏旧风格的四元组环境接口；下游则是 Sample Factory / Gymnasium 期望的
# `(obs, reward, terminated, truncated, info)` 形式。

from typing import Any, Dict, Optional, Tuple

import gymnasium as gym
from gymnasium.core import ObsType
from gymnasium.utils.step_api_compatibility import convert_to_terminated_truncated_step_api

# 这个 wrapper 基本照搬 Gym 的 compatibility wrapper，但按本仓库的向量化多机环境做了裁剪：
# 没有显式 seed 透传，且 `done` 需要按 vector env 语义改写。
class QuadEnvCompatibility(gym.Wrapper):
    def __init__(self, env, ):
        """A wrapper which converts old-style envs to valid modern envs.

        Some information may be lost in the conversion, so we recommend updating your environment.

        Args:
            env (LegacyEnv): the env to wrap, implemented with the old API
        """
        gym.Wrapper.__init__(self, env)

    # 老环境的 `reset()` 只返回 observation，这里补成 Gymnasium 要求的 `(obs, info)`。
    # 由于仓库里的旧 env 本身不消费 `seed/options`，这里直接忽略并返回空 info。
    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None) -> Tuple[ObsType, dict]:
        """Resets the environment.

        Args:
            seed: the seed to reset the environment with
            options: the options to reset the environment with

        Returns:
            (observation, info)
        """
        return self.env.reset(), {}

    # 核心兼容逻辑在这里：
    # 旧 env 给出 `(obs, reward, done, info)`，这里把它转换成 terminated/truncated 双标记版本。
    def step(self, action: Any) -> Tuple[Any, float, bool, bool, Dict]:
        """Steps through the environment.

        Args:
            action: action to step through the environment with

        Returns:
            (observation, reward, terminated, truncated, info)
        """
        # For QuadMultiEnv, truncated is actually integrated in the env,
        # since the termination is tick > ep_len
        obs, reward, done, info = self.env.step(action)

        # `convert_to_terminated_truncated_step_api` 在 `is_vector_env=True` 且 `info` 是 dict 时，
        # 会把 `done` 当成可迭代对象处理；单个 bool 会在这里出错，所以先包成长度 1 的列表。
        if isinstance(info, dict) and isinstance(done, bool):
            done = [done]

        return convert_to_terminated_truncated_step_api((obs, reward, done, info), is_vector_env=True)

    # 渲染不做任何兼容改写，直接透传到底层环境。
    def render(self) -> Any:
        """Renders the environment.
        Returns:
            The rendering of the environment, depending on the render mode
        """
        return self.env.render()
