# 中文注释副本；原始文件：swarm_rl/env_wrappers/tests/test_quads.py
# 说明：为避免修改源码，本文件仅作为阅读辅助材料。

# 导入当前模块依赖。
import unittest
from unittest import TestCase

# 导入当前模块依赖。
from sample_factory.envs.create_env import create_env
from sample_factory.utils.timing import Timing
from sample_factory.utils.utils import log, is_module_available

# 导入当前模块依赖。
from swarm_rl.train import register_swarm_components, parse_swarm_cfg


# 定义函数 `numba_available`。
def numba_available():
    # 返回当前函数的结果。
    return is_module_available('numba')


# 定义函数 `run_multi_quadrotor_env`。
def run_multi_quadrotor_env(env_name, cfg):
    # 保存或更新 `env` 的值。
    env = create_env(env_name, cfg=cfg)
    # 调用 `reset` 执行当前处理。
    env.reset()
    # 遍历当前序列或迭代器，逐项执行下面的逻辑。
    for i in range(100):
        # 同时更新 `obs`, `r`, `term`, `trunc` 等变量。
        obs, r, term, trunc, info = env.step([env.action_space.sample() for _ in range(env.num_agents)])

    # 保存或更新 `n_frames` 的值。
    n_frames = 1000
    # 保存或更新 `env` 的值。
    env = create_env(env_name, cfg=cfg)
    # 调用 `reset` 执行当前处理。
    env.reset()

    # 保存或更新 `timing` 的值。
    timing = Timing()
    # 使用上下文管理器包裹后续资源操作。
    with timing.timeit('step'):
        # 遍历当前序列或迭代器，逐项执行下面的逻辑。
        for i in range(n_frames):
            # 同时更新 `obs`, `r`, `term`, `trunc` 等变量。
            obs, r, term, trunc, info = env.step([env.action_space.sample() for _ in range(env.num_agents)])

    # 调用 `debug` 执行当前处理。
    log.debug('Time %s, FPS %.1f', timing, n_frames * env.num_agents / timing.step)
    # 调用 `close` 执行当前处理。
    env.close()


# 定义类 `TestQuads`。
class TestQuads(TestCase):
    # 定义函数 `test_quad_multi_env`。
    def test_quad_multi_env(self):
        # 调用 `register_swarm_components` 执行当前处理。
        register_swarm_components()

        # 保存或更新 `env_name` 的值。
        env_name = 'quadrotor_multi'
        # 保存或更新 `experiment_name` 的值。
        experiment_name = 'test_multi'
        # 保存或更新 `cfg` 的值。
        cfg = parse_swarm_cfg(argv=["--algo=APPO", f"--env={env_name}", f"--experiment={experiment_name}"])
        # 保存或更新 `assertIsNotNone(create_env(env_name, cfg` 的值。
        self.assertIsNotNone(create_env(env_name, cfg=cfg))
        # 调用 `run_multi_quadrotor_env` 执行当前处理。
        run_multi_quadrotor_env(env_name, cfg)

    # 为下面的函数或方法附加装饰器行为。
    @unittest.skipUnless(numba_available(), 'Numba is not installed')
    # 定义函数 `test_quad_multi_env_with_numba`。
    def test_quad_multi_env_with_numba(self):
        # 调用 `register_swarm_components` 执行当前处理。
        register_swarm_components()

        # 保存或更新 `env_name` 的值。
        env_name = 'quadrotor_multi'
        # 保存或更新 `experiment_name` 的值。
        experiment_name = 'test_numba'
        # 保存或更新 `cfg` 的值。
        cfg = parse_swarm_cfg(argv=["--algo=APPO", f"--env={env_name}", f"--experiment={experiment_name}"])
        # 保存或更新 `cfg.quads_use_numba` 的值。
        cfg.quads_use_numba = True
        # 保存或更新 `assertIsNotNone(create_env(env_name, cfg` 的值。
        self.assertIsNotNone(create_env(env_name, cfg=cfg))
        # 调用 `run_multi_quadrotor_env` 执行当前处理。
        run_multi_quadrotor_env(env_name, cfg)
