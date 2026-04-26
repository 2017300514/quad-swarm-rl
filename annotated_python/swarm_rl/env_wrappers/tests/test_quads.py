# 中文注释副本；原始文件：swarm_rl/env_wrappers/tests/test_quads.py
# 说明：为避免修改源码，本文件仅作为阅读辅助材料。
# 该文件属于强化学习训练侧逻辑，负责把环境、模型、配置或评估流程接到 Sample Factory 框架上。
# 这里产生的数据通常会继续流向训练循环、策略网络或实验分析脚本。

# 下面这组导入把当前模块会消费的环境组件、训练接口或数值工具集中拉进来；真正重要的是后续它们怎样参与数据流。
import unittest
from unittest import TestCase

# 下面这组导入把当前模块会消费的环境组件、训练接口或数值工具集中拉进来；真正重要的是后续它们怎样参与数据流。
from sample_factory.envs.create_env import create_env
from sample_factory.utils.timing import Timing
from sample_factory.utils.utils import log, is_module_available

# 下面这组导入把当前模块会消费的环境组件、训练接口或数值工具集中拉进来；真正重要的是后续它们怎样参与数据流。
from swarm_rl.train import register_swarm_components, parse_swarm_cfg


# `numba_available` 封装了当前模块中的一段独立流程，阅读时应重点关注它消费哪些状态、又把结果交给谁继续使用。
def numba_available():
    # 这里把当前阶段整理好的结果交还给上层调用者；真正要理解的是返回值之后会进入哪条训练或仿真链路。
    return is_module_available('numba')


# `run_multi_quadrotor_env` 封装了当前模块中的一段独立流程，阅读时应重点关注它消费哪些状态、又把结果交给谁继续使用。
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


# `TestQuads` 是当前文件暴露的核心类型，它负责维护与该模块职责直接相关的长期状态。
class TestQuads(TestCase):
    # `test_quad_multi_env` 封装了当前模块中的一段独立流程，阅读时应重点关注它消费哪些状态、又把结果交给谁继续使用。
    def test_quad_multi_env(self):
        register_swarm_components()

        env_name = 'quadrotor_multi'
        experiment_name = 'test_multi'
        # 这里拿到的是训练或评估全流程共享的总配置对象，后续模型注册、环境创建和 PPO 超参数都会从中读取。
        cfg = parse_swarm_cfg(argv=["--algo=APPO", f"--env={env_name}", f"--experiment={experiment_name}"])
        self.assertIsNotNone(create_env(env_name, cfg=cfg))
        run_multi_quadrotor_env(env_name, cfg)

    # 这里通过装饰器把额外框架语义附着到下面的定义上，真正影响的是后续调用方式或注册行为。
    @unittest.skipUnless(numba_available(), 'Numba is not installed')
    # `test_quad_multi_env_with_numba` 封装了当前模块中的一段独立流程，阅读时应重点关注它消费哪些状态、又把结果交给谁继续使用。
    def test_quad_multi_env_with_numba(self):
        register_swarm_components()

        env_name = 'quadrotor_multi'
        experiment_name = 'test_numba'
        # 这里拿到的是训练或评估全流程共享的总配置对象，后续模型注册、环境创建和 PPO 超参数都会从中读取。
        cfg = parse_swarm_cfg(argv=["--algo=APPO", f"--env={env_name}", f"--experiment={experiment_name}"])
        cfg.quads_use_numba = True
        self.assertIsNotNone(create_env(env_name, cfg=cfg))
        run_multi_quadrotor_env(env_name, cfg)
