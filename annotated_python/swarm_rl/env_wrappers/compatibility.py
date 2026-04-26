# 中文注释副本；原始文件：swarm_rl/env_wrappers/compatibility.py
# 说明：为避免修改源码，本文件仅作为阅读辅助材料。

# 导入当前模块依赖。
from typing import Any, Dict, Optional, Tuple

# 导入当前模块依赖。
import gymnasium as gym
from gymnasium.core import ObsType
from gymnasium.utils.step_api_compatibility import convert_to_terminated_truncated_step_api

# Wrapper for compatibility with gym 0.26
# Mostly copied from gym.EnvCompatability
# Modified since swarm_rl does not have a seed, and is a vectorized env
# 定义类 `QuadEnvCompatibility`。
class QuadEnvCompatibility(gym.Wrapper):
    # 定义函数 `__init__`。
    def __init__(self, env, ):
        # 下面的文档字符串用于说明当前模块或代码块。
        """A wrapper which converts old-style envs to valid modern envs.

        Some information may be lost in the conversion, so we recommend updating your environment.

        Args:
            env (LegacyEnv): the env to wrap, implemented with the old API
        """
        # 调用 `__init__` 执行当前处理。
        gym.Wrapper.__init__(self, env)

    # 定义函数 `reset`。
    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None) -> Tuple[ObsType, dict]:
        # 下面的文档字符串用于说明当前模块或代码块。
        """Resets the environment.

        Args:
            seed: the seed to reset the environment with
            options: the options to reset the environment with

        Returns:
            (observation, info)
        """
        # 返回当前函数的结果。
        return self.env.reset(), {}

    # 定义函数 `step`。
    def step(self, action: Any) -> Tuple[Any, float, bool, bool, Dict]:
        # 下面的文档字符串用于说明当前模块或代码块。
        """Steps through the environment.

        Args:
            action: action to step through the environment with

        Returns:
            (observation, reward, terminated, truncated, info)
        """
        # For QuadMultiEnv, truncated is actually integrated in the env,
        # since the termination is tick > ep_len
        # 同时更新 `obs`, `reward`, `done`, `info` 等变量。
        obs, reward, done, info = self.env.step(action)

        #convert_to_terminated_truncated_step_api treats done as an iterable if info is a dictionary, fails if it not iterable
        # 根据条件决定是否进入当前分支。
        if isinstance(info, dict) and isinstance(done, bool):
            # 保存或更新 `done` 的值。
            done = [done]

        # 返回当前函数的结果。
        return convert_to_terminated_truncated_step_api((obs, reward, done, info), is_vector_env=True)

    # 定义函数 `render`。
    def render(self) -> Any:
        # 下面的文档字符串用于说明当前模块或代码块。
        """Renders the environment.
        Returns:
            The rendering of the environment, depending on the render mode
        """
        # 返回当前函数的结果。
        return self.env.render()
