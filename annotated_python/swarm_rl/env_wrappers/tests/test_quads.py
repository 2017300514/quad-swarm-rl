# 中文注释副本；原始文件：swarm_rl/env_wrappers/tests/test_quads.py
# 说明：为避免修改源码，本文件仅作为阅读辅助材料。
# 这个文件是 `swarm_rl` 训练包装层的冒烟测试。
# 它不关心策略效果，而是验证 “注册组件 -> 解析训练配置 -> 由 Sample Factory 创建环境 -> 多步 rollout”
# 这一整条入口链路有没有断。

import unittest
from unittest import TestCase

from sample_factory.envs.create_env import create_env
from sample_factory.utils.timing import Timing
from sample_factory.utils.utils import log, is_module_available

from swarm_rl.train import register_swarm_components, parse_swarm_cfg


# 这里只是给 `skipUnless` 用的小 helper，避免测试机没装 numba 时直接失败。
def numba_available():
    return is_module_available('numba')


# 统一封装一次短 rollout 和一次计时 rollout。
# 前 100 步先验证环境能正常 reset/step，后 1000 步再粗略看吞吐，主要是防止包装层改坏后出现明显性能退化。
def run_multi_quadrotor_env(env_name, cfg):
    env = create_env(env_name, cfg=cfg)
    env.reset()
    for i in range(100):
        obs, r, term, trunc, info = env.step([env.action_space.sample() for _ in range(env.num_agents)])

    n_frames = 1000
    env = create_env(env_name, cfg=cfg)
    env.reset()

    timing = Timing()
    with timing.timeit('step'):
        for i in range(n_frames):
            obs, r, term, trunc, info = env.step([env.action_space.sample() for _ in range(env.num_agents)])

    log.debug('Time %s, FPS %.1f', timing, n_frames * env.num_agents / timing.step)
    env.close()


# 这组测试覆盖两条训练入口：普通 python 版环境，以及打开 numba 优化后的环境。
class TestQuads(TestCase):
    # 验证标准 `quadrotor_multi` 训练入口能被完整拉起。
    def test_quad_multi_env(self):
        register_swarm_components()

        env_name = 'quadrotor_multi'
        experiment_name = 'test_multi'
        # 这里直接走和真实训练相同的配置解析路径，而不是手工拼一个假的 cfg。
        cfg = parse_swarm_cfg(argv=["--algo=APPO", f"--env={env_name}", f"--experiment={experiment_name}"])
        self.assertIsNotNone(create_env(env_name, cfg=cfg))
        run_multi_quadrotor_env(env_name, cfg)

    @unittest.skipUnless(numba_available(), 'Numba is not installed')
    # 这条测试和上面完全相同，只额外打开 `quads_use_numba`，确认加速路径也能从训练包装层正常进入。
    def test_quad_multi_env_with_numba(self):
        register_swarm_components()

        env_name = 'quadrotor_multi'
        experiment_name = 'test_numba'
        cfg = parse_swarm_cfg(argv=["--algo=APPO", f"--env={env_name}", f"--experiment={experiment_name}"])
        cfg.quads_use_numba = True
        self.assertIsNotNone(create_env(env_name, cfg=cfg))
        run_multi_quadrotor_env(env_name, cfg)
